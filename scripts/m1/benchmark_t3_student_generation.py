#!/usr/bin/env python3
"""Benchmark a BF16 or NF4 student rollout copy on a frozen prompt window."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from run_t3_lora_rank_probe import (
    gpu_inventory,
    language_linear_targets,
    package_versions,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-pool", type=Path, required=True)
    parser.add_argument("--prompt-pool-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subbatch-size", type=int, default=64)
    parser.add_argument("--precision", choices=["bf16", "nf4"], required=True)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--total-length-cap", type=int, default=4096)
    parser.add_argument("--completion-length-cap", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--generation-eos-token-id", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.subbatch_size <= 0:
        raise ValueError("batch and subbatch sizes must be positive")
    if args.prompt_pool_offset < 0:
        raise ValueError("prompt pool offset must be non-negative")
    eos_ids = args.generation_eos_token_id or [1, 106, 50]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    receipt = {
        "schema_version": 1,
        "stage": "T3_student_generation_backend_benchmark",
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "inputs": {
            "model": str(args.model.resolve()),
            "prompt_pool": str(args.prompt_pool.resolve()),
            "prompt_pool_sha256": sha256_file(args.prompt_pool),
            "prompt_pool_offset": args.prompt_pool_offset,
        },
        "configuration": vars(args) | {"generation_eos_token_ids": eos_ids},
        "runtime": {
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "initial_gpu_inventory": gpu_inventory(),
        },
    }
    receipt["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in receipt["configuration"].items()
    }
    try:
        rows = (
            pq.read_table(args.prompt_pool)
            .sort_by("order_index")
            .slice(args.prompt_pool_offset, args.batch_size)
            .to_pylist()
        )
        if len(rows) != args.batch_size:
            raise ValueError("prompt pool does not contain a complete requested window")
        prompts = [torch.tensor(row["prompt_ids"], dtype=torch.long, device="cuda") for row in rows]
        prompt_width = max(int(prompt.numel()) for prompt in prompts)
        completion_cap = min(args.completion_length_cap, args.total_length_cap - prompt_width)
        if completion_cap <= 0:
            raise ValueError("prompt window leaves no completion room")

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = {
            "local_files_only": True,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        if args.precision == "nf4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
        targets = language_linear_targets(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.rank,
                lora_alpha=args.alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=targets,
                task_type="CAUSAL_LM",
                use_rslora=True,
                init_lora_weights=True,
            ),
            autocast_adapter_dtype=args.precision == "nf4",
        )
        model.eval()
        model.config.use_cache = True
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = True
        adapter_dtypes = {
            str(parameter.dtype)
            for name, parameter in model.named_parameters()
            if "lora_" in name
        }

        padded = torch.full(
            (args.batch_size, prompt_width),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device="cuda",
        )
        attention_mask = torch.zeros_like(padded)
        for index, prompt in enumerate(prompts):
            padded[index, -prompt.numel() :] = prompt
            attention_mask[index, -prompt.numel() :] = 1
        generation_config = GenerationConfig(**model.generation_config.to_dict())
        generation_config.do_sample = True
        generation_config.temperature = 1.0
        generation_config.top_p = 0.95
        generation_config.top_k = 0
        generation_config.max_new_tokens = completion_cap
        generation_config.eos_token_id = eos_ids
        generation_config.pad_token_id = tokenizer.pad_token_id

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        generated_blocks = []
        generation_started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, args.batch_size, args.subbatch_size):
                stop = min(start + args.subbatch_size, args.batch_size)
                generated_blocks.append(
                    model.generate(
                        input_ids=padded[start:stop],
                        attention_mask=attention_mask[start:stop],
                        generation_config=generation_config,
                    )[:, prompt_width:]
                )
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started
        generated = torch.cat(generated_blocks)
        eos_tensor = torch.tensor(eos_ids, device=generated.device)
        completion_lengths = []
        for row in generated:
            terminals = torch.isin(row, eos_tensor).nonzero(as_tuple=True)[0]
            completion_lengths.append(int(terminals[0].item()) + 1 if len(terminals) else row.numel())
        generated_tokens = sum(completion_lengths)
        generated_capacity_tokens = int(generated.numel())
        receipt.update(
            {
                "status": "PASS",
                "window": {
                    "prompts": args.batch_size,
                    "prompt_width": prompt_width,
                    "completion_cap": completion_cap,
                    "completion_lengths": completion_lengths,
                    "generated_tokens": generated_tokens,
                    "generated_capacity_tokens": generated_capacity_tokens,
                    "hit_cap_fraction": sum(x >= completion_cap for x in completion_lengths)
                    / args.batch_size,
                    "completion_ids_sha256": hashlib.sha256(
                        generated.detach().cpu().numpy().astype("<i8").tobytes()
                    ).hexdigest(),
                },
                "adapter_dtypes": sorted(adapter_dtypes),
                "target_module_count": len(targets),
                "generation_seconds": generation_seconds,
                "generated_tokens_per_second": generated_tokens / generation_seconds,
                "capacity_tokens_per_second": generated_capacity_tokens / generation_seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
        return_code = 0
    except torch.OutOfMemoryError as error:
        receipt.update(
            {
                "status": "FAIL_OOM",
                "error": str(error),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
        return_code = 2
    except Exception as error:
        receipt.update({"status": "FAIL", "error": repr(error)})
        return_code = 1
    receipt["finished_at"] = datetime.now().astimezone().isoformat()
    receipt["wall_seconds"] = time.perf_counter() - started
    receipt["runtime"]["final_gpu_inventory"] = gpu_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                key: receipt.get(key)
                for key in (
                    "status",
                    "window",
                    "adapter_dtypes",
                    "generation_seconds",
                    "generated_tokens_per_second",
                    "capacity_tokens_per_second",
                    "peak_allocated_gib",
                    "peak_reserved_gib",
                    "wall_seconds",
                    "error",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
