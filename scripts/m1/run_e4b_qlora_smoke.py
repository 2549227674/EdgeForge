#!/usr/bin/env python3
"""Run the M1 T0 Gemma 4 E4B QLoRA loss and memory smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(path: Path, max_length: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids = row["input_ids"][:max_length]
            labels = row["labels"][:max_length]
            if any(label != -100 for label in labels):
                samples.append({"uid": row["uid"], "input_ids": ids, "labels": labels})
    if not samples:
        raise ValueError("no sample retains supervised tokens after truncation")
    return samples


def train_steps(
    model: Any,
    optimizer: torch.optim.Optimizer,
    samples: list[dict[str, Any]],
    *,
    steps: int,
    use_cache: bool,
) -> dict[str, Any]:
    losses: list[float] = []
    tokens = 0
    started = time.perf_counter()
    model.train()
    model.config.use_cache = use_cache
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = use_cache
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        sample = samples[step % len(samples)]
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([sample["labels"]], dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=use_cache)
        loss = output.loss
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            losses.append(value)
            break
        loss.backward()
        optimizer.step()
        losses.append(value)
        tokens += input_ids.numel()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "requested_steps": steps,
        "completed_steps": len(losses),
        "use_cache": use_cache,
        "losses": losses,
        "all_finite": len(losses) == steps and all(math.isfinite(value) for value in losses),
        "elapsed_seconds": elapsed,
        "input_tokens": tokens,
        "input_tokens_per_second": tokens / elapsed if elapsed else None,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cache-false-probe-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    seed_everything(args.seed)
    template = args.model / "chat_template.jinja"
    template_sha = hashlib.sha256(template.read_bytes()).hexdigest()
    samples = load_samples(args.samples, args.max_length)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=2e-4)
    initial_adapter = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if "lora_" in name})

    cache_false = train_steps(
        model,
        optimizer,
        samples,
        steps=args.cache_false_probe_steps,
        use_cache=False,
    )
    model.load_state_dict(initial_adapter, strict=False)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=2e-4)
    seed_everything(args.seed)
    cache_true = train_steps(model, optimizer, samples, steps=args.steps, use_cache=True)

    report = {
        "seed": args.seed,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "attention_implementation": args.attn_implementation,
        "quantization": "bnb_nf4_double_quant_bf16_compute",
        "max_length": args.max_length,
        "sample_count": len(samples),
        "template_sha256": template_sha,
        "template_matches_frozen": template_sha == "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters_visible": sum(parameter.numel() for parameter in model.parameters()),
        "cache_false_probe": cache_false,
        "cache_true_20_step": cache_true,
    }
    report["pass"] = bool(
        report["bf16_supported"]
        and report["template_matches_frozen"]
        and cache_true["completed_steps"] == args.steps
        and cache_true["all_finite"]
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(rendered, encoding="utf-8")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
