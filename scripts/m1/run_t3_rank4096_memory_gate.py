#!/usr/bin/env python3
"""Run one real 4,096-token OPD update as a fail-closed LoRA memory gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

from run_t3_lora_rank_direct_gate import reverse_kl_top1_tail_loss, score_teacher
from run_t3_lora_rank_probe import gpu_inventory, language_linear_targets, package_versions, sha256_file, teacher_health


def select_sequence(plan: Path, prompt_length: int, completion_length: int) -> dict:
    if prompt_length <= 0 or completion_length <= 0:
        raise ValueError("prompt and completion lengths must both be positive")
    sequence_length = prompt_length + completion_length
    table = pq.read_table(plan, columns=["uid", "input_ids"])
    candidates = []
    for row in table.to_pylist():
        if len(row["input_ids"]) != sequence_length:
            continue
        score = hashlib.sha256(
            f"t3-rank-memory:{prompt_length}:{completion_length}:{row['uid']}".encode()
        ).hexdigest()
        candidates.append((score, row))
    if not candidates:
        raise ValueError(f"no {sequence_length:,}-token sequence exists in the frozen plan")
    row = min(candidates)[1]
    sequence = row["input_ids"]
    return {
        "uid": row["uid"],
        "prompt_length": prompt_length,
        "completion_length": completion_length,
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(torch.tensor(sequence, dtype=torch.int64).numpy().astype("<i8").tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prompt-length", type=int, default=3072)
    parser.add_argument("--completion-length", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    initial_health = teacher_health(args.teacher_url)
    initial_gpus = gpu_inventory()
    rollout = select_sequence(args.plan, args.prompt_length, args.completion_length)
    teacher_scores, teacher_latency = score_teacher([rollout], args.teacher_url)
    receipt = {
        "schema_version": 1,
        "stage": "T3_OPD_rank_4096_memory_gate",
        "started_at": started_at,
        "argv": sys.argv,
        "configuration": {
            "rank": args.rank,
            "alpha": args.alpha,
            "use_rslora": True,
            "scaling": args.alpha / math.sqrt(args.rank),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "quantization": "bnb_nf4_double_quant_bf16_compute",
            "attention": "sdpa",
            "activation_checkpointing": False,
            "use_cache": True,
            "target_scope": "language_model_linear_only",
        },
        "inputs": {
            "model": str(args.model.resolve()),
            "plan": str(args.plan.resolve()),
            "plan_sha256": sha256_file(args.plan),
            "rollout": {key: value for key, value in rollout.items() if key != "sequence"},
            "teacher_score_latency_seconds": teacher_latency,
        },
        "runtime": {
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "initial_gpu_inventory": initial_gpus,
            "teacher_initial_health": initial_health,
        },
    }
    exit_code = 1
    try:
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
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model.config.use_cache = True
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = True
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
        )
        trainables = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        outside = [name for name, _ in trainables if ".language_model." not in name]
        if outside or not trainables:
            raise RuntimeError(f"invalid trainable scope: {outside[:20]}")
        parameters = [parameter for _, parameter in trainables]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        loss = reverse_kl_top1_tail_loss(model, rollout, teacher_scores[0])
        loss_value = float(loss.detach().cpu())
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu())
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        if not math.isfinite(loss_value) or not math.isfinite(grad_norm):
            raise RuntimeError("non-finite loss or gradient norm")
        receipt.update(
            {
                "status": "PASS",
                "trainable_parameter_tensors": len(trainables),
                "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "target_module_count": len(targets),
                "loss": loss_value,
                "grad_norm_before_clipping": grad_norm,
                "step_seconds": step_seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }
        )
        exit_code = 0
    except torch.OutOfMemoryError as error:
        receipt.update(
            {
                "status": "FAIL_OOM",
                "error_type": type(error).__name__,
                "error": str(error),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "traceback": traceback.format_exc(),
            }
        )
        exit_code = 2
    except Exception as error:
        receipt.update(
            {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        exit_code = 1
    receipt["finished_at"] = datetime.now().astimezone().isoformat()
    receipt["wall_seconds"] = time.perf_counter() - started
    receipt["runtime"]["final_gpu_inventory"] = gpu_inventory()
    receipt["runtime"]["teacher_final_health"] = teacher_health(args.teacher_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: receipt.get(key) for key in ["status", "configuration", "loss", "grad_norm_before_clipping", "step_seconds", "peak_allocated_gib", "peak_reserved_gib", "error"]}, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
