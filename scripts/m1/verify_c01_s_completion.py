#!/usr/bin/env python3
"""Verify the frozen R2.2 C01-S training and merged bf16 artifact."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
from typing import Any

from safetensors import safe_open
import torch
from transformers import AutoModelForImageTextToText


PLAN_SHA256 = "61ece8bad05f4156b4b20fa734c2b7e1f804fc879aa50d7a28cceb790042820e"
HOLDOUT_SHA256 = "0a28a7f7b52a01f7a73ccdf927f98ac21902f8037ef236db4f86d5024a67d73b"
TEMPLATE_SHA256 = "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
EXPECTED_STEPS = 4096
EXPECTED_RECORDS = 81757
EXPECTED_INPUT_TOKENS = 148425311
EXPECTED_SUPERVISED_TOKENS = 88486262
EXPECTED_EVALUATION_STEPS = [0, 1024, 2048, 3072, 4096]
EXPECTED_TRAINABLE_PARAMETERS = 25249792
EXPECTED_ADAPTER_TENSORS = 1184
EXPECTED_OPTIMIZER_MOMENT_NUMEL = 19439616
EXPECTED_INTERMEDIATE_CHECKPOINT_STEPS = list(range(512, 4096, 512))
EXPECTED_PROCESSOR_ASSET_NAMES = [
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
EXPECTED_MERGED_PARAMETER_TENSORS = 1106
EXPECTED_MERGED_PARAMETER_NUMEL = 7941100832


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_set_sha256(files: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def inspect_safetensors(root: Path) -> dict[str, Any]:
    dtype_numel: Counter[str] = Counter()
    tensor_count = 0
    shard_count = 0
    lora_names: list[str] = []
    for path in sorted(root.glob("*.safetensors")):
        shard_count += 1
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor_count += 1
                tensor_slice = handle.get_slice(name)
                dtype_numel[str(tensor_slice.get_dtype())] += math.prod(
                    tensor_slice.get_shape()
                )
                if "lora_" in name.lower():
                    lora_names.append(name)
    result = {
        "shard_count": shard_count,
        "tensor_count": tensor_count,
        "dtype_numel": dict(sorted(dtype_numel.items())),
        "all_tensors_bf16": set(dtype_numel) == {"BF16"},
        "lora_tensor_count": len(lora_names),
    }
    if not result["all_tensors_bf16"]:
        raise ValueError(f"candidate is not uniformly bf16: {result}")
    if lora_names:
        raise ValueError(f"candidate retains LoRA tensors: {lora_names[:8]}")
    return result


def verify_model_reload(candidate_dir: Path) -> dict[str, Any]:
    """Prove the sharded candidate can be reconstructed by Transformers."""
    model = AutoModelForImageTextToText.from_pretrained(
        candidate_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    dtype_numel: Counter[str] = Counter()
    parameter_count = 0
    lora_names = []
    for name, parameter in model.named_parameters():
        parameter_count += 1
        dtype_numel[str(parameter.dtype)] += parameter.numel()
        if "lora_" in name.lower():
            lora_names.append(name)
    result = {
        "model_class": type(model).__name__,
        "parameter_tensors": parameter_count,
        "parameter_dtype_numel": dict(dtype_numel),
        "lora_parameter_count": len(lora_names),
    }
    expected_dtype = Counter(
        {"torch.bfloat16": EXPECTED_MERGED_PARAMETER_NUMEL}
    )
    if type(model).__name__ != "Gemma4ForConditionalGeneration":
        raise ValueError(f"candidate reload model class mismatch: {result}")
    if parameter_count != EXPECTED_MERGED_PARAMETER_TENSORS:
        raise ValueError(f"candidate reload parameter count mismatch: {result}")
    if dtype_numel != expected_dtype:
        raise ValueError(f"candidate reload dtype/numel mismatch: {result}")
    if lora_names:
        raise ValueError(f"candidate reload retains LoRA parameters: {lora_names[:8]}")
    return result


def inspect_checkpoint_numeric_path(
    checkpoint_dir: Path, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    """Verify the owner-accepted FP32 adapter/optimizer checkpoint path."""
    ready_path = checkpoint_dir / "READY"
    adapter_dir = checkpoint_dir / "adapter"
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    for path in (ready_path, adapter_config_path, adapter_path):
        require_file(path)
    if ready_path.read_text(encoding="utf-8") != "complete\n":
        raise ValueError(f"checkpoint READY marker is invalid: {checkpoint_dir}")

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    expected_config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
    }
    for key, value in expected_config.items():
        if adapter_config.get(key) != value:
            raise ValueError(
                f"checkpoint adapter {key}: "
                f"{adapter_config.get(key)!r} != {value!r}"
            )

    adapter_dtype_numel: Counter[str] = Counter()
    adapter_tensor_count = 0
    adapter_nonfinite = 0
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            adapter_tensor_count += 1
            adapter_dtype_numel[str(tensor.dtype)] += tensor.numel()
            adapter_nonfinite += int((~torch.isfinite(tensor)).sum().item())
    if adapter_tensor_count != EXPECTED_ADAPTER_TENSORS:
        raise ValueError(
            f"checkpoint adapter tensor count: "
            f"{adapter_tensor_count} != {EXPECTED_ADAPTER_TENSORS}"
        )
    if adapter_dtype_numel != Counter(
        {"torch.float32": EXPECTED_TRAINABLE_PARAMETERS}
    ):
        raise ValueError(
            f"checkpoint adapter dtype/numel mismatch: {adapter_dtype_numel}"
        )
    if adapter_nonfinite:
        raise ValueError(
            f"checkpoint adapter contains {adapter_nonfinite} non-finite values"
        )

    optimizer_state = (checkpoint.get("optimizer") or {}).get("state") or {}
    if len(optimizer_state) != 686:
        raise ValueError(
            f"checkpoint optimizer state count: {len(optimizer_state)} != 686"
        )
    moment_dtype_numel: dict[str, Counter[str]] = {
        "exp_avg": Counter(),
        "exp_avg_sq": Counter(),
    }
    moment_nonfinite = {"exp_avg": 0, "exp_avg_sq": 0}
    for parameter_state in optimizer_state.values():
        for name in moment_dtype_numel:
            tensor = parameter_state.get(name)
            if tensor is None:
                continue
            moment_dtype_numel[name][str(tensor.dtype)] += tensor.numel()
            moment_nonfinite[name] += int((~torch.isfinite(tensor)).sum().item())
    expected_moments = Counter(
        {"torch.float32": EXPECTED_OPTIMIZER_MOMENT_NUMEL}
    )
    for name, inventory in moment_dtype_numel.items():
        if inventory != expected_moments:
            raise ValueError(
                f"checkpoint optimizer {name} dtype/numel mismatch: {inventory}"
            )
        if moment_nonfinite[name]:
            raise ValueError(
                f"checkpoint optimizer {name} contains "
                f"{moment_nonfinite[name]} non-finite values"
            )

    return {
        "ready_marker": "complete",
        "adapter_config": expected_config,
        "adapter_tensor_count": adapter_tensor_count,
        "adapter_dtype_numel": dict(adapter_dtype_numel),
        "adapter_nonfinite_values": adapter_nonfinite,
        "optimizer_state_entries": len(optimizer_state),
        "optimizer_moment_dtype_numel": {
            name: dict(inventory)
            for name, inventory in moment_dtype_numel.items()
        },
        "optimizer_moment_nonfinite_values": moment_nonfinite,
    }


def verify_checkpoint(
    checkpoint_dir: Path,
    expected_step: int,
    *,
    inspect_numeric_path: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one checkpoint and prove its frozen identity and state completeness."""
    ready_path = checkpoint_dir / "READY"
    continuation_path = checkpoint_dir / "continuation_state.pt"
    require_file(ready_path)
    require_file(continuation_path)
    if ready_path.read_text(encoding="utf-8") != "complete\n":
        raise ValueError(f"checkpoint READY marker is invalid: {checkpoint_dir}")

    checkpoint = torch.load(continuation_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state") or {}
    expected_state = {
        "schema_version": 2,
        "run_kind": "full",
        "recipe": "family_80_20",
        "plan_sha256": PLAN_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "completed_steps": expected_step,
        "full_schedule_steps": EXPECTED_STEPS,
    }
    for key, value in expected_state.items():
        if state.get(key) != value:
            raise ValueError(
                f"checkpoint {expected_step} state {key}: "
                f"{state.get(key)!r} != {value!r}"
            )
    if (checkpoint.get("scheduler") or {}).get("last_epoch") != expected_step:
        raise ValueError(f"checkpoint {expected_step} scheduler epoch mismatch")

    expected_evaluations = [
        step for step in EXPECTED_EVALUATION_STEPS if step <= expected_step
    ]
    actual_evaluations = [
        item.get("optimizer_step") for item in (state.get("evaluations") or [])
    ]
    if actual_evaluations != expected_evaluations:
        raise ValueError(
            f"checkpoint {expected_step} evaluation schedule: "
            f"{actual_evaluations} != {expected_evaluations}"
        )
    rank_audits = state.get("rank_audits") or []
    if sorted(item.get("rank") for item in rank_audits) != [0, 1]:
        raise ValueError(f"checkpoint {expected_step} rank audit is incomplete")
    rng_ranks = sorted(
        item.get("rank") for item in (checkpoint.get("rank_rng_states") or [])
    )
    if rng_ranks != [0, 1]:
        raise ValueError(f"checkpoint {expected_step} RNG state is incomplete")

    files = []
    checkpoint_paths = sorted(
        [ready_path, continuation_path]
        + [path for path in (checkpoint_dir / "adapter").rglob("*") if path.is_file()]
    )
    actual_relative_paths = [
        str(path.relative_to(checkpoint_dir)) for path in checkpoint_paths
    ]
    expected_relative_paths = [
        "READY",
        "adapter/README.md",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "continuation_state.pt",
    ]
    if actual_relative_paths != expected_relative_paths:
        raise ValueError(
            f"checkpoint {expected_step} file set differs: "
            f"{actual_relative_paths} != {expected_relative_paths}"
        )
    run_directory_entries = None
    if expected_step == EXPECTED_STEPS:
        run_directory_entries = sorted(path.name for path in checkpoint_dir.iterdir())
        expected_run_directory_entries = [
            "READY",
            "adapter",
            "checkpoints",
            "continuation_state.pt",
            "report.json",
            "train_log.jsonl",
        ]
        if run_directory_entries != expected_run_directory_entries:
            raise ValueError(
                "final run directory entries differ: "
                f"{run_directory_entries} != {expected_run_directory_entries}"
            )
    else:
        all_checkpoint_files = sorted(
            str(path.relative_to(checkpoint_dir))
            for path in checkpoint_dir.rglob("*")
            if path.is_file()
        )
        if all_checkpoint_files != expected_relative_paths:
            raise ValueError(
                f"checkpoint {expected_step} recursive file set differs: "
                f"{all_checkpoint_files} != {expected_relative_paths}"
            )
    for path in checkpoint_paths:
        files.append(
            {
                "path": str(path.relative_to(checkpoint_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    numeric_path = (
        inspect_checkpoint_numeric_path(checkpoint_dir, checkpoint)
        if inspect_numeric_path
        else None
    )
    evidence = {
        "path": str(checkpoint_dir),
        "completed_steps": expected_step,
        "evaluation_steps": actual_evaluations,
        "rank_audits": rank_audits,
        "rng_ranks": rng_ranks,
        "scheduler_last_epoch": expected_step,
        "files": files,
        "artifact_set_sha256": artifact_set_sha256(files),
    }
    if run_directory_entries is not None:
        evidence["run_directory_entries"] = run_directory_entries
    if numeric_path is not None:
        evidence["numeric_path_inventory"] = numeric_path
    return state, evidence


def verify_frozen_inputs(receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the independent offsets/labels audit used for exact rank totals."""
    require_file(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_identities = {
        "plan_sha256": PLAN_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "holdout_manifest_sha256": "c6ae96e93474b639ee4217a0d14506e383d1f406320b01a0b0aa1ec75c1b9374",
    }
    if receipt.get("status") != "PASS":
        raise ValueError("frozen input reverification did not pass")
    if receipt.get("identities") != expected_identities:
        raise ValueError("frozen input reverification identities differ")
    plan = receipt.get("plan") or {}
    expected_plan = {
        "records": EXPECTED_RECORDS,
        "unique_uids": EXPECTED_RECORDS,
        "input_tokens_from_offsets": EXPECTED_INPUT_TOKENS,
        "shifted_supervised_tokens_from_labels": EXPECTED_SUPERVISED_TOKENS,
        "optimizer_steps": EXPECTED_STEPS,
        "plan_index_contiguous": True,
        "rows_per_step_min": 10,
        "rows_per_step_max": 35,
        "step_supervised_tokens_min": 7102,
        "declared_step_supervision_matches_labels": True,
        "family_group_records": {"anthropic": 65362, "other": 16395},
    }
    for key, value in expected_plan.items():
        if plan.get(key) != value:
            raise ValueError(
                f"frozen input reverification plan {key}: "
                f"{plan.get(key)!r} != {value!r}"
            )
    expected_rank_audits = [
        {
            "rank": 0,
            "records": 41901,
            "input_tokens": 75994420,
            "supervised_tokens": 45422524,
        },
        {
            "rank": 1,
            "records": 39856,
            "input_tokens": 72430891,
            "supervised_tokens": 43063738,
        },
    ]
    if plan.get("rank_audits") != expected_rank_audits:
        raise ValueError("frozen input reverification rank allocation differs")
    holdout = receipt.get("holdout") or {}
    expected_holdout = {
        "records": 1948,
        "unique_uids": 1948,
        "input_tokens_from_offsets": 3483799,
        "shifted_supervised_tokens_from_labels": 1964016,
    }
    if holdout != expected_holdout:
        raise ValueError("frozen input reverification holdout totals differ")
    if receipt.get("training_holdout_intersection") != 0:
        raise ValueError("frozen input reverification found holdout leakage")
    if not receipt.get("gates") or set(receipt["gates"].values()) != {"PASS"}:
        raise ValueError("frozen input reverification contains a failed gate")

    input_verifier = Path("scripts/m1/verify_c01_s_frozen_inputs.py")
    require_file(input_verifier)
    if receipt.get("verifier_sha256") != sha256(input_verifier):
        raise ValueError("frozen input reverification verifier SHA differs")
    evidence = {
        "path": str(receipt_path),
        "sha256": sha256(receipt_path),
        "verifier_sha256": receipt["verifier_sha256"],
        "rank_audits": expected_rank_audits,
        "plan_payload_totals": {
            "records": plan["records"],
            "input_tokens": plan["input_tokens_from_offsets"],
            "supervised_tokens": plan["shifted_supervised_tokens_from_labels"],
        },
        "holdout_payload_totals": expected_holdout,
        "training_holdout_intersection": 0,
    }
    return receipt, evidence


def verify_live_process_identity(receipt_path: Path) -> dict[str, Any]:
    """Revalidate the fail-closed receipt captured from the live two-rank run."""
    require_file(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or receipt.get("candidate") != "C01-S":
        raise ValueError("live process identity receipt did not pass for C01-S")
    if receipt.get("rank_pids") != [4612, 4613]:
        raise ValueError("live process identity rank PID set differs")
    expected_shape = {
        "model_type": "gemma4",
        "num_hidden_layers": 42,
        "hidden_size": 2560,
        "sliding_window": 512,
    }
    if receipt.get("model_shape") != expected_shape:
        raise ValueError("live process identity model shape differs")
    expected_packages = {
        "torch": "2.11.0+cu128",
        "transformers": "5.14.1",
        "peft": "0.20.0",
        "bitsandbytes": "0.50.0",
        "accelerate": "1.14.0",
        "pyarrow": "25.0.1",
        "safetensors": "0.8.0",
    }
    if receipt.get("packages") != expected_packages:
        raise ValueError("live process identity package set differs")
    gpus = receipt.get("gpus") or []
    if len(gpus) != 2 or any("RTX PRO 6000 Blackwell Server Edition" not in row for row in gpus):
        raise ValueError("live process identity GPU set differs")
    identities = {
        item.get("path"): item.get("sha256")
        for item in (receipt.get("file_identities") or [])
    }
    expected_identities = {
        "scripts/m1/train_recipe_race.py": sha256(
            Path("scripts/m1/train_recipe_race.py")
        ),
        "results/t1/family_80_20_half_plan_r1/family_80_20.parquet": PLAN_SHA256,
        "results/t1/family_80_20_half_plan_r1/holdout.parquet": HOLDOUT_SHA256,
        "models/gemma-4-E4B-it/chat_template.jinja": TEMPLATE_SHA256,
        "models/gemma-4-E4B-it/config.json": sha256(
            Path("models/gemma-4-E4B-it/config.json")
        ),
    }
    if identities != expected_identities:
        raise ValueError("live process identity file hashes differ")
    gates = receipt.get("gates") or {}
    if not gates or any(not str(value).startswith("PASS") for value in gates.values()):
        raise ValueError("live process identity contains a failed gate")
    train_log = receipt.get("train_log") or {}
    if (
        train_log.get("last_optimizer_step") != 1074
        or train_log.get("validation_steps") != [0, 1024]
        or train_log.get("checkpoint_steps") != [512, 1024]
        or train_log.get("all_observed_losses_finite") is not True
    ):
        raise ValueError("live process identity log snapshot differs")
    generator = Path("scripts/m1/capture_c01_s_live_process_identity.py")
    require_file(generator)
    if receipt.get("generator_sha256") != sha256(generator):
        raise ValueError("live process identity generator SHA differs")
    return {
        "path": str(receipt_path),
        "sha256": sha256(receipt_path),
        "captured_at_utc": receipt["captured_at_utc"],
        "rank_pids": receipt["rank_pids"],
        "model_shape": expected_shape,
        "packages": expected_packages,
        "gpus": gpus,
        "file_identities": expected_identities,
        "train_log_snapshot": train_log,
        "scope": receipt["scope"],
        "limitation": receipt["limitation"],
        "generator_sha256": receipt["generator_sha256"],
    }


def expected_learning_rate(step: int) -> float:
    base_learning_rate = 2e-4
    warmup_steps = 123
    if step <= warmup_steps:
        return base_learning_rate * step / warmup_steps
    progress = (step - warmup_steps) / (EXPECTED_STEPS - warmup_steps)
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def inspect_train_log(run_dir: Path, expected_completed_steps: int) -> dict[str, Any]:
    """Verify contiguous finite losses and the exact frozen scheduler/event trace."""
    log_path = run_dir / "train_log.jsonl"
    require_file(log_path)
    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_events = [item for item in events if item.get("event") == "train_step"]
    validation_events = [item for item in events if item.get("event") == "validation"]
    checkpoint_events = [item for item in events if item.get("event") == "checkpoint"]
    unknown_events = [
        item.get("event")
        for item in events
        if item.get("event") not in {"train_step", "validation", "checkpoint"}
    ]
    if unknown_events:
        raise ValueError(f"training log contains unknown events: {unknown_events[:8]}")

    steps = [item.get("optimizer_step") for item in train_events]
    if steps != list(range(1, expected_completed_steps + 1)):
        raise ValueError("training log optimizer steps are not contiguous")
    losses = []
    previous_elapsed = -math.inf
    max_lr_error = 0.0
    for item in train_events:
        step = item["optimizer_step"]
        loss = item.get("weighted_loss", math.nan)
        elapsed = item.get("elapsed_seconds", math.nan)
        learning_rate = item.get("learning_rate", math.nan)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError(f"training log has invalid loss at step {step}: {loss}")
        if not math.isfinite(elapsed) or elapsed <= previous_elapsed:
            raise ValueError(f"training log elapsed time is invalid at step {step}")
        expected_lr = expected_learning_rate(step)
        if not math.isfinite(learning_rate):
            raise ValueError(
                f"training log has non-finite learning rate at step {step}"
            )
        error = abs(learning_rate - expected_lr)
        max_lr_error = max(max_lr_error, error)
        if not math.isclose(learning_rate, expected_lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"training log learning rate mismatch at step {step}: "
                f"{learning_rate} != {expected_lr}"
            )
        losses.append(loss)
        previous_elapsed = elapsed

    expected_validation_steps = [
        step
        for step in EXPECTED_EVALUATION_STEPS
        if step <= expected_completed_steps
    ]
    if [item.get("optimizer_step") for item in validation_events] != expected_validation_steps:
        raise ValueError("training log validation event schedule mismatch")
    expected_checkpoint_steps = [
        step
        for step in EXPECTED_INTERMEDIATE_CHECKPOINT_STEPS
        if step <= expected_completed_steps
    ]
    if [item.get("optimizer_step") for item in checkpoint_events] != expected_checkpoint_steps:
        raise ValueError("training log checkpoint event schedule mismatch")
    for item in checkpoint_events:
        step = item["optimizer_step"]
        expected_path = str(run_dir / "checkpoints" / f"step-{step:05d}")
        if item.get("path") != expected_path:
            raise ValueError(f"training log checkpoint path mismatch at step {step}")

    expected_event_trace = [("validation", 0)]
    for step in range(1, expected_completed_steps + 1):
        expected_event_trace.append(("train_step", step))
        if step in expected_validation_steps and step != 0:
            expected_event_trace.append(("validation", step))
        if step in expected_checkpoint_steps:
            expected_event_trace.append(("checkpoint", step))
    actual_event_trace = [
        (item.get("event"), item.get("optimizer_step")) for item in events
    ]
    if actual_event_trace != expected_event_trace:
        raise ValueError("training log event ordering differs from the frozen trace")

    last_100 = losses[-100:]
    return {
        "path": str(log_path),
        "sha256": sha256(log_path),
        "event_count": len(events),
        "train_step_count": len(train_events),
        "first_optimizer_step": steps[0],
        "last_optimizer_step": steps[-1],
        "validation_steps": expected_validation_steps,
        "checkpoint_steps": expected_checkpoint_steps,
        "all_losses_finite_nonnegative": True,
        "mean_weighted_loss": statistics.fmean(losses),
        "last_100_mean_weighted_loss": statistics.fmean(last_100),
        "last_100_median_weighted_loss": statistics.median(last_100),
        "min_weighted_loss": min(losses),
        "max_weighted_loss": max(losses),
        "elapsed_seconds_at_last_step": previous_elapsed,
        "learning_rate_at_last_step": train_events[-1]["learning_rate"],
        "expected_learning_rate_at_last_step": expected_learning_rate(
            expected_completed_steps
        ),
        "maximum_absolute_learning_rate_error": max_lr_error,
        "validation_events": validation_events,
    }


def verify_training(
    run_dir: Path,
    frozen_input_receipt: Path,
    live_process_receipt: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_inputs, frozen_input_evidence = verify_frozen_inputs(frozen_input_receipt)
    live_process_evidence = verify_live_process_identity(live_process_receipt)
    report_path = run_dir / "report.json"
    require_file(run_dir / "READY")
    require_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "run_kind": "full",
        "recipe": "family_80_20",
        "base_seed": 20260811,
        "world_size": 2,
        "race_steps": EXPECTED_STEPS,
        "completed_steps": EXPECTED_STEPS,
        "full_schedule_steps": EXPECTED_STEPS,
        "warmup_steps": 123,
        "learning_rate": 2e-4,
        "records": EXPECTED_RECORDS,
        "input_tokens": EXPECTED_INPUT_TOKENS,
        "expected_input_tokens": EXPECTED_INPUT_TOKENS,
        "supervised_tokens": EXPECTED_SUPERVISED_TOKENS,
        "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "plan_sha256": PLAN_SHA256,
        "holdout_sha256": HOLDOUT_SHA256,
        "template_sha256": TEMPLATE_SHA256,
        "rank_seeds": [20260811, 20260812],
        "attention_implementation": "sdpa",
        "use_cache": True,
        "activation_checkpointing": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"training report {key}: {report.get(key)!r} != {value!r}")
    evaluations = report.get("evaluations") or []
    if [item.get("optimizer_step") for item in evaluations] != EXPECTED_EVALUATION_STEPS:
        raise ValueError("training report does not contain the frozen evaluation schedule")
    for item in evaluations:
        if (
            item.get("records") != 1948
            or item.get("input_tokens") != 3483799
            or item.get("supervised_tokens") != 1964016
            or item.get("world_size") != 2
        ):
            raise ValueError(f"invalid holdout identity at step {item.get('optimizer_step')}")
        for key in (
            "sample_mean_loss",
            "sample_loss_standard_error",
            "token_weighted_loss",
            "elapsed_seconds",
        ):
            value = item.get(key, math.nan)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"non-finite {key} at step {item.get('optimizer_step')}")
    if report.get("final_validation") != evaluations[-1]:
        raise ValueError("final_validation does not identify the step-4096 evaluation")
    train_log_evidence = inspect_train_log(run_dir, EXPECTED_STEPS)
    logged_evaluations = [
        {key: value for key, value in item.items() if key != "event"}
        for item in train_log_evidence.pop("validation_events")
    ]
    if logged_evaluations != evaluations:
        raise ValueError("training log validation events differ from the report")
    rank_audits = report.get("rank_audits") or []
    if sorted(item.get("rank") for item in rank_audits) != [0, 1]:
        raise ValueError("rank audit must contain exactly ranks 0 and 1")
    expected_rank_audits = frozen_inputs["plan"]["rank_audits"]
    for expected_rank in expected_rank_audits:
        actual_rank = next(
            item for item in rank_audits if item.get("rank") == expected_rank["rank"]
        )
        expected_identity = {
            **expected_rank,
            "local_rank": expected_rank["rank"],
            "seed": 20260811 + expected_rank["rank"],
        }
        for key, value in expected_identity.items():
            if actual_rank.get(key) != value:
                raise ValueError(
                    f"rank {expected_rank['rank']} audit {key}: "
                    f"{actual_rank.get(key)!r} != {value!r}"
                )
    for key, total in (
        ("records", EXPECTED_RECORDS),
        ("input_tokens", EXPECTED_INPUT_TOKENS),
        ("supervised_tokens", EXPECTED_SUPERVISED_TOKENS),
    ):
        if sum(item.get(key, -1) for item in rank_audits) != total:
            raise ValueError(f"rank audit {key} total mismatch")
    for item in rank_audits:
        allocated = item.get("peak_allocated_gib", math.nan)
        reserved = item.get("peak_reserved_gib", math.nan)
        if (
            not math.isfinite(allocated)
            or not math.isfinite(reserved)
            or allocated <= 0
            or reserved < allocated
            or reserved >= 96
        ):
            raise ValueError(f"rank {item.get('rank')} peak memory audit is invalid")
    elapsed_seconds = report.get("elapsed_seconds", math.nan)
    throughput = report.get("input_tokens_per_second", math.nan)
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("training report elapsed_seconds is invalid")
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("training report input_tokens_per_second is invalid")
    expected_throughput = EXPECTED_INPUT_TOKENS / elapsed_seconds
    if not math.isclose(throughput, expected_throughput, rel_tol=1e-12):
        raise ValueError("training report input_tokens_per_second is inconsistent")
    expected_allocated = [item["peak_allocated_gib"] for item in rank_audits]
    expected_reserved = [item["peak_reserved_gib"] for item in rank_audits]
    if report.get("peak_allocated_gib_by_rank") != expected_allocated:
        raise ValueError("training report peak allocated memory differs by rank")
    if report.get("peak_reserved_gib_by_rank") != expected_reserved:
        raise ValueError("training report peak reserved memory differs by rank")

    checkpoints_root = run_dir / "checkpoints"
    expected_checkpoint_names = [
        f"step-{step:05d}" for step in EXPECTED_INTERMEDIATE_CHECKPOINT_STEPS
    ]
    actual_checkpoint_names = sorted(
        path.name for path in checkpoints_root.iterdir() if path.is_dir()
    )
    if actual_checkpoint_names != expected_checkpoint_names:
        raise ValueError(
            f"intermediate checkpoint schedule: {actual_checkpoint_names} "
            f"!= {expected_checkpoint_names}"
        )
    intermediate_checkpoints = []
    for step in EXPECTED_INTERMEDIATE_CHECKPOINT_STEPS:
        _, checkpoint_evidence = verify_checkpoint(
            checkpoints_root / f"step-{step:05d}",
            step,
            inspect_numeric_path=step == 512,
        )
        intermediate_checkpoints.append(checkpoint_evidence)

    final_checkpoint_state, final_checkpoint = verify_checkpoint(
        run_dir, EXPECTED_STEPS, inspect_numeric_path=True
    )
    if final_checkpoint_state != report:
        raise ValueError("final continuation state and training report differ")
    evidence = {
        "report_path": str(report_path),
        "report_sha256": sha256(report_path),
        "train_log": train_log_evidence,
        "frozen_input_reverification": frozen_input_evidence,
        "live_process_identity": live_process_evidence,
        "numeric_path": {
            "base_compute_dtype": "bfloat16",
            "adapter_trainable_dtype": "float32",
            "optimizer_moment_dtype": "float32",
            "final_merge_required_dtype": "bfloat16",
            "owner_decision_date": "2026-08-12",
        },
        "intermediate_checkpoint_steps": EXPECTED_INTERMEDIATE_CHECKPOINT_STEPS,
        "intermediate_checkpoints": intermediate_checkpoints,
        "final_checkpoint": final_checkpoint,
        "evaluation_steps": EXPECTED_EVALUATION_STEPS,
        "rank_audits": rank_audits,
    }
    return report, evidence


def verify_candidate(
    candidate_dir: Path, training: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = candidate_dir / "artifact_manifest.json"
    require_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 3,
        "candidate": "C01-S",
        "format": "huggingface_bf16_safetensors",
        "base_model": "models/gemma-4-E4B-it",
        "adapter": "results/t2/c01_s_family_80_20_half_r1/adapter",
        "training_report": "results/t2/c01_s_family_80_20_half_r1/report.json",
        "training_plan_sha256": PLAN_SHA256,
        "training_completed_steps": EXPECTED_STEPS,
        "training_input_tokens": EXPECTED_INPUT_TOKENS,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"candidate manifest {key}: {manifest.get(key)!r} != {value!r}")
    if training["plan_sha256"] != manifest["training_plan_sha256"]:
        raise ValueError("candidate and training plan identities diverge")
    merge_script = Path("scripts/m1/merge_lora_candidate.py")
    require_file(merge_script)
    if manifest.get("merge_script_sha256") != sha256(merge_script):
        raise ValueError("candidate merge script SHA differs")
    report_path = Path(manifest["training_report"])
    require_file(report_path)
    if manifest.get("training_report_sha256") != sha256(report_path):
        raise ValueError("candidate training report SHA differs")
    adapter_dir = Path(manifest["adapter"])
    declared_adapter_files = manifest.get("adapter_files") or []
    declared_adapter_paths = [item.get("path") for item in declared_adapter_files]
    expected_adapter_paths = [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    if declared_adapter_paths != expected_adapter_paths:
        raise ValueError("candidate source adapter file set differs")
    actual_adapter_paths = sorted(
        str(path.relative_to(adapter_dir))
        for path in adapter_dir.rglob("*")
        if path.is_file()
    )
    if actual_adapter_paths != declared_adapter_paths:
        raise ValueError("candidate source adapter files differ from the manifest")
    for item in declared_adapter_files:
        path = adapter_dir / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"candidate source adapter identity mismatch: {item['path']}")
    adapter_identity = artifact_set_sha256(declared_adapter_files)
    if manifest.get("adapter_artifact_set_sha256") != adapter_identity:
        raise ValueError("candidate source adapter artifact identity differs")

    declared_files = manifest.get("files") or []
    declared_paths = [item.get("path") for item in declared_files]
    if declared_paths != sorted(declared_paths) or len(set(declared_paths)) != len(declared_paths):
        raise ValueError("candidate manifest paths are not unique and sorted")
    actual_paths = sorted(
        str(path.relative_to(candidate_dir))
        for path in candidate_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if declared_paths != actual_paths:
        raise ValueError("candidate manifest does not cover the complete artifact file set")
    for item in declared_files:
        path = candidate_dir / item["path"]
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"candidate size mismatch: {item['path']}")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"candidate SHA mismatch: {item['path']}")
    if sum(item["bytes"] for item in declared_files) != manifest.get("total_bytes"):
        raise ValueError("candidate total_bytes mismatch")
    identity = artifact_set_sha256(declared_files)
    if identity != manifest.get("artifact_set_sha256"):
        raise ValueError("candidate artifact_set_sha256 mismatch")

    processor_assets = manifest.get("processor_assets") or []
    processor_paths = [item.get("path") for item in processor_assets]
    if processor_paths != EXPECTED_PROCESSOR_ASSET_NAMES:
        raise ValueError(f"candidate processor asset set is invalid: {processor_paths}")
    declared_by_path = {item["path"]: item for item in declared_files}
    base_model = Path(manifest["base_model"])
    base_weights = base_model / "model.safetensors"
    require_file(base_weights)
    actual_base_weights_sha256 = sha256(base_weights)
    if manifest.get("base_model_safetensors_sha256") != actual_base_weights_sha256:
        raise ValueError("candidate base model weights SHA differs from the source")
    for item in processor_assets:
        path = item["path"]
        source = base_model / path
        require_file(source)
        actual_source_sha256 = sha256(source)
        if item.get("source_sha256") != actual_source_sha256:
            raise ValueError(f"candidate processor asset source SHA mismatch: {path}")
        if actual_source_sha256 != item.get("candidate_sha256"):
            raise ValueError(f"candidate processor asset source mismatch: {path}")
        if item.get("candidate_sha256") != declared_by_path[path]["sha256"]:
            raise ValueError(f"candidate processor asset manifest mismatch: {path}")
        if item.get("bytes") != declared_by_path[path]["bytes"]:
            raise ValueError(f"candidate processor asset size mismatch: {path}")

    config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("dtype") != "bfloat16":
        raise ValueError(f"candidate config dtype is not bfloat16: {config.get('dtype')!r}")
    inventory = inspect_safetensors(candidate_dir)
    if inventory != manifest.get("tensor_inventory"):
        raise ValueError("candidate tensor inventory differs from the merge-time inventory")
    reload_evidence = verify_model_reload(candidate_dir)
    evidence = {
        "path": str(candidate_dir),
        "merge_script_sha256": manifest["merge_script_sha256"],
        "training_report_sha256": manifest["training_report_sha256"],
        "adapter_artifact_set_sha256": adapter_identity,
        "base_model": str(base_model),
        "base_model_safetensors_sha256": actual_base_weights_sha256,
        "manifest_sha256": sha256(manifest_path),
        "artifact_set_sha256": identity,
        "total_bytes": manifest["total_bytes"],
        "file_count_excluding_manifest": len(declared_files),
        "tensor_inventory": inventory,
        "cpu_reload": reload_evidence,
        "processor_assets": processor_assets,
    }
    return manifest, evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--frozen-input-receipt",
        type=Path,
        default=Path("results/t2/c01_s_frozen_inputs_reverification.json"),
    )
    parser.add_argument(
        "--live-process-receipt",
        type=Path,
        default=Path("results/m1/c01_s_live_process_identity_receipt.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    incomplete = args.output.with_name(args.output.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete audit output already exists: {incomplete}")

    training, training_evidence = verify_training(
        args.run_dir, args.frozen_input_receipt, args.live_process_receipt
    )
    _, candidate_evidence = verify_candidate(args.candidate_dir, training)
    audit = {
        "schema_version": 1,
        "milestone": "M1",
        "candidate": "C01-S",
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "verifier": str(Path(__file__)),
        "verifier_sha256": sha256(Path(__file__)),
        "training": training_evidence,
        "candidate_artifact": candidate_evidence,
        "gates": {
            "family_80_20_half_plan_identity": "PASS",
            "plan_offsets_labels_and_rank_allocation": "PASS",
            "two_rank_4096_step_training": "PASS",
            "live_two_rank_runtime_identity": "PASS",
            "contiguous_finite_train_log_and_exact_lr_schedule": "PASS",
            "exact_148425311_input_tokens": "PASS",
            "exact_88486262_supervised_tokens": "PASS",
            "five_finite_full_holdout_evaluations": "PASS",
            "step_00512_checkpoint_integrity_and_loadability": "PASS",
            "all_512_step_intermediate_checkpoints": "PASS",
            "final_4096_checkpoint": "PASS",
            "end_to_end_resume_smoke": "WAIVED_BY_OWNER",
            "fp32_lora_and_optimizer_dtype": "PASS_OWNER_ACCEPTED",
            "rank_consumption_audit": "PASS",
            "complete_candidate_file_hashes": "PASS",
            "all_merged_tensors_bf16": "PASS",
            "zero_residual_lora_tensors": "PASS",
            "merged_model_cpu_reload": "PASS",
            "processor_assets_byte_identical": "PASS",
        },
    }
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    incomplete.rename(args.output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
