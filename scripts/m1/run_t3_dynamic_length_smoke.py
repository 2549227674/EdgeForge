#!/usr/bin/env python3
"""Run native TRL T3 training with the dynamic 4K rollout policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from trl.experimental.server_distillation import ServerDistillationConfig

from run_t3_lora_rank_probe import (
    PretokenizedPromptCollator,
    gpu_inventory,
    language_linear_targets,
    package_versions,
    sha256_file,
    teacher_health,
    trainable_inventory,
)
from t3_dynamic_length import DynamicLengthServerDistillationTrainer


def select_prompts(plan: Path, targets: list[int], seed: int) -> list[dict[str, Any]]:
    table = pq.read_table(plan, columns=["uid", "source_uid", "family", "task_type", "input_ids", "labels"])
    candidates: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in table.to_pylist():
        if row["source_uid"] in seen_sources:
            continue
        first_supervised = next((i for i, token in enumerate(row["labels"]) if token != -100), None)
        if first_supervised is None or not 1 <= first_supervised <= max(targets):
            continue
        seen_sources.add(row["source_uid"])
        candidates.append(
            {
                "uid": row["uid"],
                "source_uid": row["source_uid"],
                "family": row["family"],
                "task_type": row["task_type"],
                "prompt_ids": row["input_ids"][:first_supervised],
                "prompt_length": first_supervised,
            }
        )

    selected = []
    used: set[str] = set()
    for target in targets:
        available = [row for row in candidates if row["source_uid"] not in used]
        if not available:
            raise ValueError(f"no prompt available for target length {target}")
        row = min(
            available,
            key=lambda item: (
                abs(int(item["prompt_length"]) - target),
                hashlib.sha256(f"{seed}:{target}:{item['uid']}".encode()).hexdigest(),
            ),
        )
        selected.append(row)
        used.add(str(row["source_uid"]))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompt-pool", type=Path)
    parser.add_argument("--prompt-pool-offset", type=int, default=0)
    parser.add_argument(
        "--prompt-pool-count",
        type=int,
        help="ordered prompt rows to load; default is one rollout buffer, 0 means all remaining rows",
    )
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--student-base-precision", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--total-length-cap", type=int, default=4096)
    parser.add_argument("--prompt-length-cap", type=int, default=3072)
    parser.add_argument("--completion-length-cap", type=int, default=3072)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--optimizer-batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--state-output-dir", type=Path)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--save-final-model", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--generation-prefill-quadratic-budget", type=int, default=275_000_000)
    parser.add_argument("--use-student-vllm", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.2)
    parser.add_argument("--vllm-enable-sleep-mode", action="store_true")
    parser.add_argument("--vllm-model-impl", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--target-prompt-length", type=int, action="append")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--generation-eos-token-id", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets = args.target_prompt_length or [128, 1024, 1536, 3072]
    generation_eos_token_ids = args.generation_eos_token_id or [1, 106, 50]
    if args.generation_batch_size <= 0:
        raise ValueError("generation batch size must be positive")
    if args.optimizer_batch_size <= 0:
        raise ValueError("optimizer batch size must be positive")
    if args.generation_batch_size % args.optimizer_batch_size:
        raise ValueError("generation batch size must be divisible by optimizer batch size")
    if args.max_steps <= 0:
        raise ValueError("max steps must be positive")
    if args.prompt_pool_offset < 0:
        raise ValueError("prompt pool offset must be non-negative")
    if args.prompt_pool_count is not None and args.prompt_pool_count < 0:
        raise ValueError("prompt pool count must be non-negative")
    if args.warmup_steps < 0 or args.warmup_steps > args.max_steps:
        raise ValueError("warmup steps must be in [0, max steps]")
    if args.save_steps < 0:
        raise ValueError("save steps must be non-negative")
    if args.save_steps and args.save_steps % (args.generation_batch_size // args.optimizer_batch_size):
        raise ValueError("save steps must end on a rollout-window boundary")
    if args.prompt_pool is None and len(targets) != args.generation_batch_size:
        raise ValueError("target prompt count must equal generation batch size")
    if args.prompt_pool is None and max(targets) > args.prompt_length_cap:
        raise ValueError("target prompt exceeds prompt-length cap")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "T3_OPD_dynamic_length_native_trainer_smoke",
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "argv": sys.argv,
        "inputs": {
            "model": str(args.model.resolve()),
            "plan": str(args.plan.resolve()),
            "plan_sha256": sha256_file(args.plan),
        },
        "configuration": {
            "trainer": "t3_dynamic_length.DynamicLengthServerDistillationTrainer",
            "rank": args.rank,
            "alpha": args.alpha,
            "use_rslora": True,
            "scaling": args.alpha / math.sqrt(args.rank),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "student_base_precision": args.student_base_precision,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": args.optimizer_batch_size,
            "generation_batch_size": args.optimizer_batch_size,
            "rollout_buffer_size": args.generation_batch_size,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler_type": args.lr_scheduler_type,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "num_generations": 1,
            "total_length_cap": args.total_length_cap,
            "prompt_length_cap": args.prompt_length_cap,
            "completion_length_cap": args.completion_length_cap,
            "completion_rule": "min(completion_length_cap,total_length_cap-real_prompt_width)",
            "generation_prefill_quadratic_budget": args.generation_prefill_quadratic_budget,
            "generation_eos_token_ids": generation_eos_token_ids,
            "student_generation_backend": "vllm_colocate" if args.use_student_vllm else "transformers",
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization
            if args.use_student_vllm
            else None,
            "vllm_enable_sleep_mode": args.vllm_enable_sleep_mode if args.use_student_vllm else None,
            "vllm_model_impl": args.vllm_model_impl if args.use_student_vllm else None,
        },
        "runtime": {
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "initial_gpu_inventory": gpu_inventory(),
            "teacher_initial_health": teacher_health(args.teacher_url),
        },
    }
    exit_code = 1
    try:
        if args.prompt_pool is None:
            prompts = select_prompts(args.plan, targets, args.seed)
        else:
            ordered_prompt_table = pq.read_table(args.prompt_pool).sort_by("order_index")
            prompt_pool_count = (
                args.generation_batch_size
                if args.prompt_pool_count is None
                else args.prompt_pool_count
            )
            if prompt_pool_count == 0:
                prompt_pool_count = len(ordered_prompt_table) - args.prompt_pool_offset
            prompt_table = ordered_prompt_table.slice(args.prompt_pool_offset, prompt_pool_count)
            prompts = [
                {
                    "uid": row["plan_uid"],
                    "source_uid": row["source_uid"],
                    "family": row["family"],
                    "task_type": row["task_type"],
                    "prompt_ids": row["prompt_ids"],
                    "prompt_length": row["prompt_length"],
                }
                for row in prompt_table.to_pylist()
            ]
            if len(prompts) != prompt_pool_count:
                raise ValueError(
                    f"prompt pool must contain {prompt_pool_count} ordered prompts after offset"
                )
            if len(prompts) % args.generation_batch_size:
                raise ValueError("loaded prompt count must be divisible by generation batch size")
            required_prompts = args.max_steps * args.optimizer_batch_size
            if required_prompts != len(prompts):
                raise ValueError(
                    "formal schedule must consume every loaded prompt exactly once: "
                    f"max_steps*optimizer_batch_size={required_prompts}, prompts={len(prompts)}"
                )
            receipt["inputs"]["prompt_pool"] = str(args.prompt_pool.resolve())
            receipt["inputs"]["prompt_pool_sha256"] = sha256_file(args.prompt_pool)
            receipt["inputs"]["prompt_pool_offset"] = args.prompt_pool_offset
            receipt["inputs"]["prompt_pool_count"] = len(prompts)
        receipt["inputs"]["prompts"] = [
            {
                key: value
                for key, value in row.items()
                if key != "prompt_ids"
            }
            | {
                "prompt_ids_sha256": hashlib.sha256(
                    np.asarray(row["prompt_ids"], dtype="<i8").tobytes()
                ).hexdigest()
            }
            for row in prompts
        ]
        dataset = Dataset.from_list(
            [{"uid": row["uid"], "prompt_ids": row["prompt_ids"]} for row in prompts]
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        if args.student_base_precision == "nf4":
            model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_load_kwargs)
        if args.student_base_precision == "nf4":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        model.config.use_cache = True
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = True
        target_modules = language_linear_targets(model)
        state_output_dir = args.state_output_dir or (
            args.output.parent / "dynamic-length-native-smoke-state"
        )
        trainer_args = ServerDistillationConfig(
            output_dir=str(state_output_dir),
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.optimizer_batch_size,
            generation_batch_size=args.optimizer_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            optim="adamw_torch",
            max_grad_norm=1.0,
            bf16=True,
            fp16=False,
            tf32=True,
            gradient_checkpointing=False,
            use_cache=True,
            max_length=args.total_length_cap,
            max_prompt_length=args.prompt_length_cap,
            max_completion_length=args.completion_length_cap,
            temperature=1.0,
            top_p=0.95,
            top_k=0,
            beta=1.0,
            num_generations=1,
            use_vllm=args.use_student_vllm,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_length=args.total_length_cap,
            vllm_model_impl=args.vllm_model_impl,
            vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
            teacher_model_server_url=args.teacher_url,
            loss_top_k=1,
            loss_add_tail=True,
            reverse_kl_top_1_mode="sampled",
            disable_dropout=True,
            seed=args.seed,
            data_seed=args.seed,
            logging_strategy="steps",
            logging_steps=1,
            logging_first_step=True,
            save_strategy="steps" if args.save_steps else "no",
            save_steps=args.save_steps or 500,
            save_total_limit=args.save_total_limit,
            report_to="none",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        trainer = DynamicLengthServerDistillationTrainer(
            model=model,
            args=trainer_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=PretokenizedPromptCollator(tokenizer.pad_token_id),
            peft_config=LoraConfig(
                r=args.rank,
                lora_alpha=args.alpha,
                lora_dropout=0.0,
                bias="none",
                target_modules=target_modules,
                task_type="CAUSAL_LM",
                use_rslora=True,
                init_lora_weights=True,
            ),
            dynamic_total_length_cap=args.total_length_cap,
            dynamic_completion_length_cap=args.completion_length_cap,
            generation_eos_token_ids=generation_eos_token_ids,
            generation_prefill_quadratic_budget=args.generation_prefill_quadratic_budget,
            rollout_buffer_size=args.generation_batch_size,
        )
        inventory = trainable_inventory(trainer.model)
        if not inventory["all_trainables_inside_language_model"]:
            raise RuntimeError(f"trainables escaped language model: {inventory}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        torch.cuda.synchronize()
        events = trainer.dynamic_length_events
        expected_trained_slices = args.max_steps * args.optimizer_batch_size
        expected_rollout_windows = math.ceil(expected_trained_slices / args.generation_batch_size)
        if len(events) != expected_rollout_windows * args.generation_batch_size or any(
            event["padded_sequence_width"] > args.total_length_cap for event in events
        ):
            raise RuntimeError(f"invalid dynamic length events: {events}")
        trained_slices = trainer.dynamic_trained_slice_indices
        expected_slices = [
            slice_index
            for _ in range(expected_rollout_windows)
            for slice_index in range(args.generation_batch_size)
        ][:expected_trained_slices]
        if trained_slices != expected_slices:
            raise RuntimeError(
                f"rollout slices were repeated, skipped, or reordered: expected={expected_slices}, got={trained_slices}"
            )
        loss_rows = [row for row in trainer.state.log_history if "loss" in row]
        if trainer.state.global_step != args.max_steps or not loss_rows or not math.isfinite(float(loss_rows[-1]["loss"])):
            raise RuntimeError(f"native trainer did not finish a finite optimizer step: {trainer.state.log_history}")
        if args.save_final_model:
            trainer.save_model(str(state_output_dir / "final_adapter"))
            trainer.save_state()
        receipt.update(
            {
                "status": "PASS",
                "completed_global_steps": trainer.state.global_step,
                "dynamic_length_events": events,
                "dynamic_generation_batches": trainer.dynamic_generation_batches,
                "dynamic_trained_slice_indices": trained_slices,
                "trainable_inventory": inventory,
                "target_module_count": len(target_modules),
                "state_output_dir": str(state_output_dir.resolve()),
                "final_adapter": str((state_output_dir / "final_adapter").resolve())
                if args.save_final_model
                else None,
                "train_metrics": result.metrics,
                "log_history": trainer.state.log_history,
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
                "traceback": traceback.format_exc(),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
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
    print(
        json.dumps(
            {
                key: receipt.get(key)
                for key in (
                    "status",
                    "completed_global_steps",
                    "dynamic_length_events",
                    "dynamic_generation_batches",
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
