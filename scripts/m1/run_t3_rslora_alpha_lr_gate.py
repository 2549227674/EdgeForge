#!/usr/bin/env python3
"""Select the T3 rsLoRA alpha/LR pair on shared frozen OPD supervision."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from run_t3_lora_rank_probe import (
    gpu_inventory,
    language_linear_targets,
    package_versions,
    select_frozen_prompts,
    sha256_file,
    teacher_health,
)
from run_t3_lora_rank_rslora_gate import (
    evaluate,
    generate_frozen_rollouts,
    paired_differences,
    reverse_kl_top1_tail_loss,
    score_teacher,
    trainable_inventory,
)


RANK = 256
ALPHAS = (8, 16, 32, 64, 128)
LEARNING_RATES = (2.5e-5, 5e-5, 1e-4)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def arm_key(alpha: int, learning_rate: float) -> str:
    learning_rate_micro = round(learning_rate * 1_000_000)
    if not math.isclose(learning_rate, learning_rate_micro / 1_000_000, rel_tol=0, abs_tol=1e-15):
        raise ValueError(f"learning rate cannot be represented by an integer micro-unit: {learning_rate}")
    return f"alpha{alpha}_lr_micro{learning_rate_micro}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-prompts", type=int, default=32)
    parser.add_argument("--eval-prompts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--min-prompt-length", type=int, default=96)
    parser.add_argument("--max-prompt-length", type=int, default=448)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.steps > args.train_prompts:
        raise ValueError("steps cannot exceed train prompts in this no-repeat direct gate")

    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    seed_everything(args.seed)
    initial_health = teacher_health(args.teacher_url)
    initial_gpus = gpu_inventory()
    selected, prompt_manifest = select_frozen_prompts(
        args.plan,
        count=args.train_prompts + args.eval_prompts,
        seed=args.seed + 11,
        min_prompt_length=args.min_prompt_length,
        max_prompt_length=args.max_prompt_length,
    )
    train_prompts = selected[: args.train_prompts]
    eval_prompts = selected[args.train_prompts :]
    train_uids = {row["uid"] for row in train_prompts}
    eval_uids = {row["uid"] for row in eval_prompts}
    if train_uids & eval_uids:
        raise RuntimeError("train and eval prompts overlap")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
    base_model.config.use_cache = True
    if hasattr(base_model.config, "text_config"):
        base_model.config.text_config.use_cache = True
    targets = language_linear_targets(base_model)

    train_rollouts = generate_frozen_rollouts(
        base_model,
        train_prompts,
        tokenizer,
        max_completion_length=args.max_completion_length,
        seed=args.seed + 12,
    )
    eval_rollouts = generate_frozen_rollouts(
        base_model,
        eval_prompts,
        tokenizer,
        max_completion_length=args.max_completion_length,
        seed=args.seed + 13,
    )
    train_scores, train_teacher_seconds = score_teacher(train_rollouts, args.teacher_url)
    eval_scores, eval_teacher_seconds = score_teacher(eval_rollouts, args.teacher_url)
    baseline_eval = evaluate(base_model, eval_rollouts, eval_scores)

    arms = [(alpha, lr) for alpha in ALPHAS for lr in LEARNING_RATES]
    arm_results: dict[str, Any] = {}
    model = None
    previous_key = None
    torch.cuda.reset_peak_memory_stats()
    for alpha, learning_rate in arms:
        key = arm_key(alpha, learning_rate)
        seed_everything(args.seed)
        config = LoraConfig(
            r=RANK,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
            use_rslora=True,
            init_lora_weights=True,
        )
        if model is None:
            model = get_peft_model(base_model, config, adapter_name=key)
        else:
            model.delete_adapter(previous_key)
            model.add_adapter(key, config)
            model.set_adapter(key)
        inventory = trainable_inventory(model, key)
        if not inventory["valid"]:
            raise RuntimeError(f"invalid trainable scope for {key}: {inventory}")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )
        model.train()
        losses = []
        grad_norms = []
        arm_started = time.perf_counter()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = reverse_kl_top1_tail_loss(model, train_rollouts[step], train_scores[step])
            value = float(loss.detach().cpu())
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu())
            optimizer.step()
            if not math.isfinite(value) or not math.isfinite(grad_norm):
                raise RuntimeError(f"non-finite {key} step {step}")
            losses.append(value)
            grad_norms.append(grad_norm)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - arm_started
        train_eval = evaluate(model, train_rollouts, train_scores)
        heldout_eval = evaluate(model, eval_rollouts, eval_scores)
        arm_results[key] = {
            "rank": RANK,
            "alpha": alpha,
            "learning_rate": learning_rate,
            "scaling": alpha / math.sqrt(RANK),
            "scaled_learning_rate": learning_rate * alpha / math.sqrt(RANK),
            "use_rslora": True,
            "weight_decay": args.weight_decay,
            "trainable_inventory": inventory,
            "step_losses": losses,
            "step_grad_norms_before_clipping": grad_norms,
            "all_finite": True,
            "train_seconds": train_seconds,
            "post_train_on_frozen_train_rollouts": train_eval,
            "post_train_on_heldout_frozen_rollouts": heldout_eval,
            "heldout_delta_vs_unadapted": heldout_eval["sequence_mean"] - baseline_eval["sequence_mean"],
        }
        del optimizer
        torch.cuda.empty_cache()
        previous_key = key

    best_key = min(
        arm_results,
        key=lambda key: arm_results[key]["post_train_on_heldout_frozen_rollouts"]["sequence_mean"],
    )
    best_eval = arm_results[best_key]["post_train_on_heldout_frozen_rollouts"]
    comparisons = {
        key: paired_differences(result["post_train_on_heldout_frozen_rollouts"], best_eval)
        for key, result in arm_results.items()
    }
    tolerance = max(best_eval["sequence_mean"] * 0.01, 1e-4)
    eligible = [
        key
        for key, result in arm_results.items()
        if comparisons[key]["mean"] <= max(comparisons[key]["se"], tolerance)
        and result["post_train_on_heldout_frozen_rollouts"]["sequence_mean"]
        <= best_eval["sequence_mean"] + tolerance
    ]
    # The one-SE rule avoids selecting a more aggressive update when its measured
    # advantage is indistinguishable. This is a model-selection rule, not a fallback.
    selected_key = min(
        eligible or [best_key],
        key=lambda key: (
            arm_results[key]["scaled_learning_rate"],
            arm_results[key]["learning_rate"],
            arm_results[key]["alpha"],
        ),
    )
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "T3_OPD_rsLoRA_alpha_lr_direct_gate",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "argv": sys.argv,
        "method": {
            "design": "direct independent alpha/LR arms on identical frozen base-student rollouts and teacher scores",
            "rank": RANK,
            "alphas": list(ALPHAS),
            "learning_rates": list(LEARNING_RATES),
            "controls": {
                "use_rslora": True,
                "weight_decay": args.weight_decay,
                "optimizer": "AdamW",
                "max_grad_norm": 1.0,
                "steps": args.steps,
                "target_scope": "language_model_linear_only",
                "train_rollouts_shared": True,
                "eval_rollouts_shared": True,
                "teacher_scores_shared": True,
            },
            "limitation": "short frozen-rollout hyperparameter gate; formal OPD remains on-policy and retains normal validation gates",
        },
        "inputs": {
            "model": str(args.model.resolve()),
            "plan": str(args.plan.resolve()),
            "plan_sha256": sha256_file(args.plan),
            "prompt_manifest": prompt_manifest,
            "train_eval_uid_intersection": sorted(train_uids & eval_uids),
            "train_rollouts": train_rollouts,
            "eval_rollouts": eval_rollouts,
        },
        "teacher": {
            "url": args.teacher_url,
            "initial_health": initial_health,
            "final_health": teacher_health(args.teacher_url),
            "train_score_latency_seconds": train_teacher_seconds,
            "eval_score_latency_seconds": eval_teacher_seconds,
        },
        "adapter": {"target_module_count": len(targets), "target_modules": targets},
        "unadapted_heldout_eval": baseline_eval,
        "arm_results": arm_results,
        "selection": {
            "empirical_best_arm": best_key,
            "empirical_best_heldout_sequence_mean": best_eval["sequence_mean"],
            "paired_differences_vs_best": comparisons,
            "relative_loss_tolerance": tolerance,
            "eligible_arms": eligible,
            "rule": "least aggressive scaled LR within both 1% heldout loss and one paired SE of empirical best",
            "selected_arm": selected_key,
            "selected_rank": RANK,
            "selected_alpha": arm_results[selected_key]["alpha"],
            "selected_learning_rate": arm_results[selected_key]["learning_rate"],
            "selected_scaling": arm_results[selected_key]["scaling"],
            "selected_scaled_learning_rate": arm_results[selected_key]["scaled_learning_rate"],
            "use_rslora": True,
            "weight_decay": args.weight_decay,
        },
        "runtime": {
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "initial_gpu_inventory": initial_gpus,
            "final_gpu_inventory": gpu_inventory(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "baseline_heldout": baseline_eval["sequence_mean"],
                "heldout_by_arm": {
                    key: result["post_train_on_heldout_frozen_rollouts"]["sequence_mean"]
                    for key, result in arm_results.items()
                },
                "selection": receipt["selection"],
                "peak_allocated_gib": receipt["runtime"]["peak_allocated_gib"],
                "receipt": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
