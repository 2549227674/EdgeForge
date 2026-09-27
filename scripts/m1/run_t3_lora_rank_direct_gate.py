#!/usr/bin/env python3
"""Directly compare T3 LoRA ranks on identical frozen OPD supervision.

One unadapted C01-S instance generates disjoint train/eval rollouts.  The BF16
teacher scores those sequences exactly once.  Rank 8/16/32/64/128 adapters are
then trained independently with identical sequences, teacher logprobs,
``alpha / r == 2``, optimizer, step count, and language-model target modules.
This isolates rank without paying for five full on-policy runs.
"""

from __future__ import annotations

import argparse
import hashlib
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
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from trl.generation.vllm_client import VLLMClient

from run_t3_lora_rank_probe import (
    gpu_inventory,
    language_linear_targets,
    package_versions,
    select_frozen_prompts,
    sha256_file,
    teacher_health,
)


RANKS = (8, 16, 32, 64, 128)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_frozen_rollouts(
    model: Any,
    prompts: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_completion_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    seed_everything(seed)
    model.eval()
    generation_config = GenerationConfig(
        max_new_tokens=max_completion_length,
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    rollouts = []
    for row in prompts:
        prompt = torch.tensor([row["prompt_ids"]], dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(prompt)
        generated = model.generate(
            input_ids=prompt,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )[0]
        completion = generated[prompt.shape[1] :]
        if completion.numel() == 0:
            raise RuntimeError(f"empty completion for {row['uid']}")
        sequence = generated.tolist()
        prompt_length = prompt.shape[1]
        payload = np.asarray(sequence, dtype="<i8").tobytes()
        rollouts.append(
            {
                "uid": row["uid"],
                "family": row["family"],
                "task_type": row["task_type"],
                "prompt_length": prompt_length,
                "completion_length": completion.numel(),
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rollouts


def score_teacher(rollouts: list[dict[str, Any]], teacher_url: str) -> tuple[list[list[float]], float]:
    client = VLLMClient(base_url=teacher_url, connection_timeout=60.0)
    started = time.perf_counter()
    result = client.get_sequence_logprobs(
        sequences=[row["sequence"] for row in rollouts],
        prompt_lengths=[row["prompt_length"] for row in rollouts],
        top_logprobs=1,
        temperature=1.0,
    )
    latency = time.perf_counter() - started
    scores = []
    for row, actual in zip(rollouts, result["actual_logprobs"], strict=True):
        values = [float(item[0]) for item in actual]
        if len(values) != row["completion_length"]:
            raise RuntimeError(
                f"teacher length mismatch for {row['uid']}: {len(values)} != {row['completion_length']}"
            )
        if not all(math.isfinite(value) and value <= 0 for value in values):
            raise RuntimeError(f"invalid teacher logprobs for {row['uid']}")
        scores.append(values)
    return scores, latency


def reverse_kl_top1_tail_loss(
    model: Any,
    rollout: dict[str, Any],
    teacher_actual_logprobs: list[float],
) -> torch.Tensor:
    """Exact beta=1, sampled top-1 plus tail-bucket divergence used by TRL."""

    sequence = torch.tensor([rollout["sequence"]], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(sequence)
    output = model(input_ids=sequence, attention_mask=attention_mask, use_cache=True)
    start = rollout["prompt_length"] - 1
    length = rollout["completion_length"]
    logits = output.logits[:, start : start + length, :]
    actual_tokens = sequence[:, rollout["prompt_length"] : rollout["prompt_length"] + length]
    actual_logits = logits.gather(-1, actual_tokens.unsqueeze(-1)).squeeze(-1).float()
    student_logprob = actual_logits - torch.logsumexp(logits.float(), dim=-1)
    teacher_logprob = torch.tensor(
        teacher_actual_logprobs, dtype=torch.float32, device=student_logprob.device
    ).unsqueeze(0)
    student_logprob = student_logprob.clamp(max=-1e-7)
    teacher_logprob = teacher_logprob.clamp(max=-1e-7)
    student_probability = student_logprob.exp()
    teacher_probability = teacher_logprob.exp()
    student_tail_logprob = torch.log(-torch.expm1(student_logprob))
    teacher_tail_logprob = torch.log(-torch.expm1(teacher_logprob))
    loss_per_token = student_probability * (student_logprob - teacher_logprob) + (
        1 - student_probability
    ) * (student_tail_logprob - teacher_tail_logprob)
    loss = loss_per_token.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite divergence for {rollout['uid']}")
    return loss


@torch.no_grad()
def evaluate(
    model: Any,
    rollouts: list[dict[str, Any]],
    scores: list[list[float]],
) -> dict[str, Any]:
    model.eval()
    per_sequence = []
    token_weighted_sum = 0.0
    tokens = 0
    for rollout, teacher_scores in zip(rollouts, scores, strict=True):
        loss = float(reverse_kl_top1_tail_loss(model, rollout, teacher_scores).cpu())
        per_sequence.append({"uid": rollout["uid"], "loss": loss, "tokens": rollout["completion_length"]})
        token_weighted_sum += loss * rollout["completion_length"]
        tokens += rollout["completion_length"]
    values = [row["loss"] for row in per_sequence]
    return {
        "sequence_mean": float(np.mean(values)),
        "sequence_se": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
        "sequence_median": float(np.median(values)),
        "token_weighted_mean": token_weighted_sum / tokens,
        "tokens": tokens,
        "per_sequence": per_sequence,
    }


def trainable_inventory(model: Any, adapter_name: str) -> dict[str, Any]:
    rows = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    outside = [name for name, _ in rows if ".language_model." not in name]
    wrong_adapter = [name for name, _ in rows if f".{adapter_name}." not in name]
    return {
        "parameter_tensors": len(rows),
        "parameters": sum(parameter.numel() for _, parameter in rows),
        "dtypes": sorted({str(parameter.dtype) for _, parameter in rows}),
        "outside_language_model": outside[:20],
        "wrong_active_adapter": wrong_adapter[:20],
        "valid": bool(rows) and not outside and not wrong_adapter,
    }


def paired_differences(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float]:
    candidate_by_uid = {row["uid"]: row["loss"] for row in candidate["per_sequence"]}
    reference_by_uid = {row["uid"]: row["loss"] for row in reference["per_sequence"]}
    differences = np.asarray(
        [candidate_by_uid[uid] - reference_by_uid[uid] for uid in sorted(candidate_by_uid)], dtype=np.float64
    )
    return {
        "mean": float(differences.mean()),
        "se": float(differences.std(ddof=1) / math.sqrt(len(differences))) if len(differences) > 1 else 0.0,
        "min": float(differences.min()),
        "max": float(differences.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-prompts", type=int, default=16)
    parser.add_argument("--eval-prompts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
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
        seed=args.seed + 1,
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
        seed=args.seed + 2,
    )
    eval_rollouts = generate_frozen_rollouts(
        base_model,
        eval_prompts,
        tokenizer,
        max_completion_length=args.max_completion_length,
        seed=args.seed + 3,
    )
    train_scores, train_teacher_seconds = score_teacher(train_rollouts, args.teacher_url)
    eval_scores, eval_teacher_seconds = score_teacher(eval_rollouts, args.teacher_url)
    baseline_eval = evaluate(base_model, eval_rollouts, eval_scores)

    rank_results: dict[str, Any] = {}
    model = None
    torch.cuda.reset_peak_memory_stats()
    for rank in RANKS:
        adapter_name = f"rank{rank}"
        seed_everything(args.seed)
        config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
            use_rslora=False,
            init_lora_weights=True,
        )
        if model is None:
            model = get_peft_model(base_model, config, adapter_name=adapter_name)
        else:
            model.add_adapter(adapter_name, config)
            model.set_adapter(adapter_name)
        inventory = trainable_inventory(model, adapter_name)
        if not inventory["valid"]:
            raise RuntimeError(f"invalid trainable scope for {adapter_name}: {inventory}")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
        model.train()
        losses = []
        grad_norms = []
        rank_started = time.perf_counter()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = reverse_kl_top1_tail_loss(model, train_rollouts[step], train_scores[step])
            value = float(loss.detach().cpu())
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu())
            optimizer.step()
            if not math.isfinite(value) or not math.isfinite(grad_norm):
                raise RuntimeError(f"non-finite rank {rank} step {step}")
            losses.append(value)
            grad_norms.append(grad_norm)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - rank_started
        train_eval = evaluate(model, train_rollouts, train_scores)
        heldout_eval = evaluate(model, eval_rollouts, eval_scores)
        rank_results[str(rank)] = {
            "rank": rank,
            "alpha": rank * 2,
            "scaling": 2.0,
            "use_rslora": False,
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

    best_rank = min(RANKS, key=lambda rank: rank_results[str(rank)]["post_train_on_heldout_frozen_rollouts"]["sequence_mean"])
    best_eval = rank_results[str(best_rank)]["post_train_on_heldout_frozen_rollouts"]
    comparisons = {
        str(rank): paired_differences(
            rank_results[str(rank)]["post_train_on_heldout_frozen_rollouts"], best_eval
        )
        for rank in RANKS
    }
    # Prefer a smaller adapter only if its paired mean is statistically indistinguishable
    # (within one paired SE) and within 1% relative loss of the empirical best.
    tolerance = max(best_eval["sequence_mean"] * 0.01, 1e-4)
    eligible = [
        rank
        for rank in RANKS
        if comparisons[str(rank)]["mean"] <= max(comparisons[str(rank)]["se"], tolerance)
        and rank_results[str(rank)]["post_train_on_heldout_frozen_rollouts"]["sequence_mean"]
        <= best_eval["sequence_mean"] + tolerance
    ]
    selected_rank = min(eligible) if eligible else best_rank
    final_health = teacher_health(args.teacher_url)
    final_gpus = gpu_inventory()
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "stage": "T3_OPD_LoRA_rank_direct_gate",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "argv": sys.argv,
        "method": {
            "design": "direct independent adapters on identical frozen base-student rollouts and teacher scores",
            "ranks": list(RANKS),
            "rank_only_controls": {
                "standard_lora_alpha_over_rank": 2.0,
                "use_rslora": False,
                "learning_rate": args.learning_rate,
                "optimizer": "AdamW",
                "max_grad_norm": 1.0,
                "steps": args.steps,
                "target_scope": "language_model_linear_only",
                "train_rollouts_shared": True,
                "eval_rollouts_shared": True,
                "teacher_scores_shared": True,
            },
            "limitation": "short frozen-rollout capacity gate; formal OPD remains on-policy and requires normal holdout/fast-tier gates",
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
            "final_health": final_health,
            "train_score_latency_seconds": train_teacher_seconds,
            "eval_score_latency_seconds": eval_teacher_seconds,
        },
        "adapter": {
            "target_module_count": len(targets),
            "target_modules": targets,
        },
        "unadapted_heldout_eval": baseline_eval,
        "rank_results": rank_results,
        "selection": {
            "empirical_best_rank": best_rank,
            "empirical_best_heldout_sequence_mean": best_eval["sequence_mean"],
            "paired_differences_vs_best": comparisons,
            "relative_loss_tolerance": tolerance,
            "eligible_ranks": eligible,
            "rule": "smallest rank within both 1% heldout loss and one paired SE of empirical best",
            "selected_rank": selected_rank,
            "selected_alpha": selected_rank * 2,
            "selected_scaling": 2.0,
            "use_rslora": False,
        },
        "runtime": {
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "initial_gpu_inventory": initial_gpus,
            "final_gpu_inventory": final_gpus,
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
                "heldout_by_rank": {
                    rank: rank_results[str(rank)]["post_train_on_heldout_frozen_rollouts"]["sequence_mean"]
                    for rank in RANKS
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
