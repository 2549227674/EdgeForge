#!/usr/bin/env python3
"""Select the T3 OPD LoRA rank with one real high-rank training probe.

The probe deliberately trains a high-rank adapter once, using the same effective
``alpha / r == 2`` scaling as the T2 rank-8/alpha-16 adapter.  It then computes
the compact SVD of every learned language-model update and evaluates rank
8/16/32/64/128 truncations on an independently generated, held-out prompt
batch. This keeps rank as the only adapter variable and avoids five complete
training runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from trl.experimental.server_distillation import ServerDistillationConfig, ServerDistillationTrainer


PROBE_RANK = 128
PROBE_ALPHA = 256
CANDIDATE_RANKS = (8, 16, 32, 64, 128)
EXPECTED_TEACHER_ARTIFACT_SET_SHA256 = "d36ce1b459f12c53e47ffbf66af8f660fefea422a7c3d50ece850a11e04e9ee6"
EXPECTED_STUDENT_ARTIFACT_SET_SHA256 = "1eebd3e1302a502ce41a9416267219e70cec63ac69ea20429dd97bd47ba9a133"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = [
        "accelerate",
        "bitsandbytes",
        "datasets",
        "numpy",
        "peft",
        "pyarrow",
        "torch",
        "transformers",
        "trl",
        "vllm",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def teacher_health(url: str) -> dict[str, Any]:
    started = time.perf_counter()
    with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=10) as response:
        body = response.read()
        return {
            "status": response.status,
            "latency_seconds": time.perf_counter() - started,
            "body": body.decode("utf-8", errors="replace"),
        }


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    rows = []
    for line in output.splitlines():
        index, name, uuid, total, used, utilization = [part.strip() for part in line.split(",", 5)]
        rows.append(
            {
                "index": int(index),
                "name": name,
                "uuid": uuid,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": int(utilization),
            }
        )
    return rows


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty list")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * probability))]


def select_frozen_prompts(
    plan_path: Path,
    *,
    count: int,
    seed: int,
    min_prompt_length: int,
    max_prompt_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = pq.read_table(
        plan_path,
        columns=["uid", "source_uid", "family", "task_type", "input_ids", "labels"],
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    eligible_count = 0
    for row in table.to_pylist():
        labels = row["labels"]
        try:
            prompt_length = next(index for index, label in enumerate(labels) if label != -100)
        except StopIteration:
            continue
        if not min_prompt_length <= prompt_length <= max_prompt_length:
            continue
        eligible_count += 1
        uid = row["uid"]
        score = hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()
        grouped[(row["family"], row["task_type"])].append(
            {
                "uid": uid,
                "source_uid": row["source_uid"],
                "family": row["family"],
                "task_type": row["task_type"],
                "prompt_ids": row["input_ids"][:prompt_length],
                "prompt_length": prompt_length,
                "selection_score": score,
            }
        )

    for rows in grouped.values():
        rows.sort(key=lambda item: item["selection_score"])
    selected: list[dict[str, Any]] = []
    group_keys = sorted(grouped, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    cursor = 0
    while len(selected) < count:
        advanced = False
        for key in group_keys:
            rows = grouped[key]
            if cursor < len(rows):
                selected.append(rows[cursor])
                advanced = True
                if len(selected) == count:
                    break
        if not advanced:
            break
        cursor += 1
    if len(selected) != count:
        raise ValueError(f"only selected {len(selected)} prompts, expected {count}")

    manifest_rows = [
        {key: value for key, value in row.items() if key not in {"prompt_ids", "selection_score"}}
        | {"prompt_ids_sha256": hashlib.sha256(np.asarray(row["prompt_ids"], dtype="<i8").tobytes()).hexdigest()}
        for row in selected
    ]
    manifest_payload = json.dumps(manifest_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "selection": "family_task_type_round_robin_then_seeded_sha256",
        "seed": seed,
        "eligible_count": eligible_count,
        "selected_count": len(selected),
        "min_prompt_length": min_prompt_length,
        "max_prompt_length": max_prompt_length,
        "selected_prompt_tokens": sum(row["prompt_length"] for row in selected),
        "selected_rows": manifest_rows,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
    return selected, manifest


class PretokenizedPromptCollator:
    """Left-pad already-rendered T2 prompt prefixes without re-rendering them."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompt_tensors = [torch.tensor(example["prompt_ids"], dtype=torch.long) for example in examples]
        width = max(tensor.numel() for tensor in prompt_tensors)
        input_ids = torch.full((len(examples), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
        for index, prompt in enumerate(prompt_tensors):
            input_ids[index, -prompt.numel() :] = prompt
            attention_mask[index, -prompt.numel() :] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.full_like(input_ids, -100),
            "prompts": input_ids.clone(),
            "prompt_attention_mask": attention_mask.clone(),
        }


def language_linear_targets(model: nn.Module) -> list[str]:
    output_embeddings = model.get_output_embeddings()
    targets = []
    for name, module in model.named_modules():
        if ".language_model." not in f".{name}." or module is output_embeddings:
            continue
        if isinstance(module, nn.Linear) or module.__class__.__name__ == "Linear4bit":
            targets.append(name)
    if not targets:
        raise RuntimeError("no language-model linear targets were found")
    return sorted(targets)


def trainable_inventory(model: nn.Module) -> dict[str, Any]:
    by_branch: dict[str, int] = defaultdict(int)
    names = []
    dtypes: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        dtypes[str(parameter.dtype)] += parameter.numel()
        branch = "language_model" if ".language_model." in name else "outside_language_model"
        by_branch[branch] += parameter.numel()
    return {
        "parameter_tensors": len(names),
        "parameters": sum(by_branch.values()),
        "by_branch": dict(by_branch),
        "dtype_numel": dict(dtypes),
        "all_trainables_inside_language_model": bool(names)
        and by_branch.get("outside_language_model", 0) == 0,
        "first_names": names[:12],
    }


def adapter_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    modules = []
    for name, module in model.named_modules():
        if ".language_model." not in name:
            continue
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        if "default" not in module.lora_B:
            continue
        modules.append((name, module))
    return modules


def build_update_spectra(model: nn.Module) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Factor every B@A through a rank-sized core, never materializing B@A."""

    factors: list[dict[str, Any]] = []
    module_metrics = []
    aggregate_energy = {rank: 0.0 for rank in CANDIDATE_RANKS}
    total_energy = 0.0
    for name, module in adapter_modules(model):
        a = module.lora_A["default"].weight.detach().float()
        b = module.lora_B["default"].weight.detach().float()
        qb, rb = torch.linalg.qr(b, mode="reduced")
        qa, ra = torch.linalg.qr(a.T, mode="reduced")
        u, singular_values, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
        left = (qb @ u).cpu()
        right = (vh @ qa.T).cpu()
        singular_values_cpu = singular_values.cpu()
        energy = singular_values_cpu.square()
        energy_total = float(energy.sum())
        if not math.isfinite(energy_total) or energy_total <= 0:
            raise RuntimeError(f"invalid learned update energy for {name}: {energy_total}")
        cumulative = torch.cumsum(energy, dim=0) / energy_total
        capture = {rank: float(cumulative[min(rank, cumulative.numel()) - 1]) for rank in CANDIDATE_RANKS}
        for rank, value in capture.items():
            aggregate_energy[rank] += energy_total * value
        total_energy += energy_total
        probabilities = singular_values_cpu / singular_values_cpu.sum()
        effective_rank = float(torch.exp(-(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()))
        module_metrics.append(
            {
                "name": name,
                "shape": [b.shape[0], a.shape[1]],
                "frobenius_energy": energy_total,
                "effective_rank": effective_rank,
                "energy_capture": {str(rank): value for rank, value in capture.items()},
                "singular_values": singular_values_cpu.tolist(),
            }
        )
        factors.append(
            {
                "name": name,
                "module": module,
                "left": left,
                "right": right,
                "singular_values": singular_values_cpu,
                "original_a": a.cpu(),
                "original_b": b.cpu(),
            }
        )
        del a, b, qb, rb, qa, ra, u, vh, left, right, singular_values, energy

    if not factors:
        raise RuntimeError("no learned language adapter modules found")
    summary = {
        "module_count": len(module_metrics),
        "total_frobenius_energy": total_energy,
        "aggregate_energy_capture": {
            str(rank): aggregate_energy[rank] / total_energy for rank in CANDIDATE_RANKS
        },
        "module_capture_distribution": {
            str(rank): {
                "min": min(row["energy_capture"][str(rank)] for row in module_metrics),
                "p10": quantile([row["energy_capture"][str(rank)] for row in module_metrics], 0.10),
                "p50": quantile([row["energy_capture"][str(rank)] for row in module_metrics], 0.50),
                "p90": quantile([row["energy_capture"][str(rank)] for row in module_metrics], 0.90),
                "fraction_at_least_95_percent": sum(
                    row["energy_capture"][str(rank)] >= 0.95 for row in module_metrics
                )
                / len(module_metrics),
            }
            for rank in CANDIDATE_RANKS
        },
        "effective_rank": {
            "min": min(row["effective_rank"] for row in module_metrics),
            "p10": quantile([row["effective_rank"] for row in module_metrics], 0.10),
            "p50": quantile([row["effective_rank"] for row in module_metrics], 0.50),
            "p90": quantile([row["effective_rank"] for row in module_metrics], 0.90),
            "max": max(row["effective_rank"] for row in module_metrics),
        },
        "modules": module_metrics,
    }
    return factors, summary


@torch.no_grad()
def apply_truncated_updates(factors: list[dict[str, Any]], rank: int) -> None:
    for factor in factors:
        module = factor["module"]
        a = module.lora_A["default"].weight
        b = module.lora_B["default"].weight
        singular_values = factor["singular_values"][:rank].to(a.device)
        root = singular_values.sqrt()
        left = factor["left"][:, :rank].to(a.device)
        right = factor["right"][:rank, :].to(a.device)
        a.zero_()
        b.zero_()
        a[:rank, :].copy_((root[:, None] * right).to(a.dtype))
        b[:, :rank].copy_((left * root[None, :]).to(b.dtype))


@torch.no_grad()
def restore_original_updates(factors: list[dict[str, Any]]) -> None:
    for factor in factors:
        module = factor["module"]
        a = module.lora_A["default"].weight
        b = module.lora_B["default"].weight
        a.copy_(factor["original_a"].to(device=a.device, dtype=a.dtype))
        b.copy_(factor["original_b"].to(device=b.device, dtype=b.dtype))


def clone_tensor_dict(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in inputs.items()}


@torch.no_grad()
def fixed_batch_loss(trainer: ServerDistillationTrainer, inputs: dict[str, Any]) -> float:
    trainer.model.eval()
    loss = trainer.compute_loss(trainer.model, inputs)
    value = float(loss.detach().cpu())
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite fixed-batch loss: {value}")
    return value


def choose_rank(spectrum: dict[str, Any], fixed_losses: dict[str, float]) -> dict[str, Any]:
    """Apply a preregistered, scale-free capacity gate."""

    full_loss = fixed_losses[str(PROBE_RANK)]
    loss_tolerance = max(abs(full_loss) * 0.01, 1e-4)
    decisions = {}
    selected = PROBE_RANK
    for rank in CANDIDATE_RANKS:
        aggregate = spectrum["aggregate_energy_capture"][str(rank)]
        module_fraction = spectrum["module_capture_distribution"][str(rank)]["fraction_at_least_95_percent"]
        loss_delta = fixed_losses[str(rank)] - full_loss
        eligible = aggregate >= 0.99 and module_fraction >= 0.95 and loss_delta <= loss_tolerance
        decisions[str(rank)] = {
            "aggregate_energy_at_least_99_percent": aggregate >= 0.99,
            "at_least_95_percent_modules_capture_95_percent_energy": module_fraction >= 0.95,
            "fixed_batch_loss_within_one_percent_or_1e-4": loss_delta <= loss_tolerance,
            f"loss_delta_vs_rank{PROBE_RANK}": loss_delta,
            "eligible": eligible,
        }
        if eligible and selected == PROBE_RANK:
            selected = rank
    return {
        "rule": (
            "smallest rank with >=99% aggregate update energy, >=95% of modules individually retaining >=95% "
            f"energy, and held-out fixed-batch teacher divergence no more than max(1%,1e-4) above rank{PROBE_RANK}"
        ),
        "loss_tolerance": loss_tolerance,
        "decisions": decisions,
        "selected_rank": selected,
        "selected_standard_lora_alpha": selected * 2,
        "selected_scaling": 2.0,
        "use_rslora": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--eval-prompts", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-prompt-length", type=int, default=96)
    parser.add_argument("--max-prompt-length", type=int, default=448)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    if PROBE_ALPHA / PROBE_RANK != 2:
        raise AssertionError("probe scaling drifted from the T2 alpha/r == 2 baseline")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with bf16 support is required")
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    seed_everything(args.seed)
    initial_health = teacher_health(args.teacher_url)
    initial_gpus = gpu_inventory()
    prompts, prompt_manifest = select_frozen_prompts(
        args.plan,
        count=args.steps + args.eval_prompts,
        seed=args.seed,
        min_prompt_length=args.min_prompt_length,
        max_prompt_length=args.max_prompt_length,
    )
    train_prompts = prompts[: args.steps]
    eval_prompts = prompts[args.steps :]
    train_uids = {row["uid"] for row in train_prompts}
    eval_uids = {row["uid"] for row in eval_prompts}
    if train_uids & eval_uids or len(eval_prompts) != args.eval_prompts:
        raise RuntimeError("train/eval prompt split is not disjoint and complete")
    dataset = Dataset.from_list(
        [{"uid": row["uid"], "prompt_ids": row["prompt_ids"]} for row in train_prompts]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    target_modules = language_linear_targets(model)
    peft_config = LoraConfig(
        r=PROBE_RANK,
        lora_alpha=PROBE_ALPHA,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        use_rslora=False,
        init_lora_weights=True,
    )
    trainer_args = ServerDistillationConfig(
        output_dir=str(args.output.parent / "rank-probe-trainer-state"),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        generation_batch_size=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=False,
        use_cache=True,
        max_length=args.max_prompt_length + args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=0.95,
        top_k=0,
        beta=1.0,
        num_generations=1,
        use_vllm=False,
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
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    collator = PretokenizedPromptCollator(tokenizer.pad_token_id)
    trainer = ServerDistillationTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=peft_config,
    )
    inventory = trainable_inventory(trainer.model)
    if not inventory["all_trainables_inside_language_model"]:
        raise RuntimeError(f"trainables escaped language_model: {inventory}")
    actual_adapter_modules = len(adapter_modules(trainer.model))
    if actual_adapter_modules != len(target_modules):
        raise RuntimeError(
            f"adapter module count {actual_adapter_modules} != target count {len(target_modules)}"
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train_started = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    heldout_batch = collator(
        [{"uid": row["uid"], "prompt_ids": row["prompt_ids"]} for row in eval_prompts]
    )
    heldout_batch = {
        key: value.to(trainer.accelerator.device) if torch.is_tensor(value) else value
        for key, value in heldout_batch.items()
    }
    trainer.model.eval()
    trainer._buffered_inputs = [None]
    trainer._buffered_text_logs = [None]
    with torch.no_grad():
        trainer._generate_student_completions([heldout_batch], [0])
    if trainer._buffered_inputs[0] is None:
        raise RuntimeError("trainer did not generate the held-out fixed batch")
    fixed_inputs = clone_tensor_dict(trainer._buffered_inputs[0])
    fixed_completion_tokens = int((fixed_inputs["labels"] != -100).sum())
    if fixed_completion_tokens <= 0:
        raise RuntimeError("final generated batch contains no completion tokens")

    full_original_loss = fixed_batch_loss(trainer, fixed_inputs)
    factors, spectrum = build_update_spectra(trainer.model)
    fixed_losses = {}
    for rank in CANDIDATE_RANKS[:-1]:
        apply_truncated_updates(factors, rank)
        fixed_losses[str(rank)] = fixed_batch_loss(trainer, fixed_inputs)
    apply_truncated_updates(factors, PROBE_RANK)
    reconstructed_rank64_loss = fixed_batch_loss(trainer, fixed_inputs)
    reconstruction_delta = abs(reconstructed_rank64_loss - full_original_loss)
    restore_original_updates(factors)
    fixed_losses[str(PROBE_RANK)] = fixed_batch_loss(trainer, fixed_inputs)
    original_restore_delta = abs(fixed_losses[str(PROBE_RANK)] - full_original_loss)
    if original_restore_delta > max(abs(full_original_loss) * 1e-6, 1e-7):
        raise RuntimeError(
            f"restoring original rank-{PROBE_RANK} factors changed loss by {original_restore_delta}"
        )
    selection = choose_rank(spectrum, fixed_losses)
    log_history = [
        row
        for row in trainer.state.log_history
        if "loss" in row or "train_loss" in row or "train_runtime" in row
    ]
    step_losses = [float(row["loss"]) for row in log_history if "loss" in row]
    all_losses_finite = len(step_losses) == args.steps and all(math.isfinite(value) for value in step_losses)
    final_health = teacher_health(args.teacher_url)
    final_gpus = gpu_inventory()
    receipt = {
        "schema_version": 1,
        "status": "PASS" if all_losses_finite else "FAIL",
        "stage": "T3_OPD_LoRA_rank_probe",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "argv": sys.argv,
        "method": {
            "design": f"single rank-{PROBE_RANK} real-OPD pilot followed by compact-SVD truncation",
            "probe_rank": PROBE_RANK,
            "probe_alpha": PROBE_ALPHA,
            "scaling": PROBE_ALPHA / PROBE_RANK,
            "use_rslora": False,
            "candidate_ranks": list(CANDIDATE_RANKS),
            "controls": (
                "same C01-S, prompts, generated completion batch, teacher logprobs, alpha/r scaling, target "
                "modules, optimizer and dtype; only learned update rank is truncated"
            ),
        },
        "models": {
            "student": {
                "path": str(args.model.resolve()),
                "expected_artifact_set_sha256": EXPECTED_STUDENT_ARTIFACT_SET_SHA256,
                "class": trainer.model.get_base_model().__class__.__name__,
                "quantization": "bnb_nf4_double_quant_bf16_compute",
                "attention": "sdpa",
                "multimodal_checkpoint_retained": True,
                "trainable_scope": "language_model_linear_only",
            },
            "teacher": {
                "url": args.teacher_url,
                "expected_artifact_set_sha256": EXPECTED_TEACHER_ARTIFACT_SET_SHA256,
                "dtype": "bfloat16",
                "multimodal_checkpoint_retained": True,
                "request_scope": "text_token_ids",
                "initial_health": initial_health,
                "final_health": final_health,
            },
        },
        "inputs": {
            "plan": str(args.plan.resolve()),
            "plan_sha256": sha256_file(args.plan),
            "prompt_manifest": prompt_manifest,
            "train_prompt_uids": [row["uid"] for row in train_prompts],
            "heldout_eval_prompt_uids": [row["uid"] for row in eval_prompts],
            "train_eval_uid_intersection": sorted(train_uids & eval_uids),
        },
        "trainer": {
            "class": trainer.__class__.__module__ + "." + trainer.__class__.__name__,
            "steps": args.steps,
            "completed_global_steps": trainer.state.global_step,
            "learning_rate": args.learning_rate,
            "scheduler": "constant",
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "beta": 1.0,
            "loss_top_k": 1,
            "loss_add_tail": True,
            "reverse_kl_top_1_mode": "sampled",
            "generation_batch_size": 1,
            "num_generations": 1,
            "log_history": log_history,
            "step_losses": step_losses,
            "all_step_losses_finite": all_losses_finite,
            "train_seconds": train_seconds,
            "train_result_metrics": train_result.metrics,
            "heldout_fixed_batch_prompt_count": len(eval_prompts),
            "heldout_fixed_batch_completion_tokens": fixed_completion_tokens,
            "full_original_heldout_fixed_batch_loss": full_original_loss,
        },
        "adapter": {
            "target_module_count": len(target_modules),
            "target_modules": target_modules,
            "trainable_inventory": inventory,
        },
        "rank_spectrum": spectrum,
        "truncated_fixed_batch_losses": fixed_losses,
        f"reconstructed_rank{PROBE_RANK}_heldout_fixed_batch_loss": reconstructed_rank64_loss,
        f"rank{PROBE_RANK}_reconstruction_loss_delta": reconstruction_delta,
        f"original_rank{PROBE_RANK}_restore_loss_delta": original_restore_delta,
        "selection": selection,
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "packages": package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "initial_gpu_inventory": initial_gpus,
            "final_gpu_inventory": final_gpus,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "selected_rank": selection["selected_rank"],
                "selected_alpha": selection["selected_standard_lora_alpha"],
                "step_losses": step_losses,
                "aggregate_energy_capture": spectrum["aggregate_energy_capture"],
                "fixed_losses": fixed_losses,
                "peak_allocated_gib": peak_allocated,
                "peak_reserved_gib": peak_reserved,
                "receipt": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
