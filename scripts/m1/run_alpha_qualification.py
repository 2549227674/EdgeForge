#!/usr/bin/env python3
"""Run the pre-registered M1 alpha qualification experiments.

Each command writes a self-contained JSON audit record.  GPU-heavy commands
are intentionally separate so a failed/OOM process cannot destroy evidence
from an earlier gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig


FROZEN_TEMPLATE_SHA256 = "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
MODEL_WEIGHTS_SHA256 = "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def environment_record(model_path: Path, command: str) -> dict[str, Any]:
    template = model_path / "chat_template.jinja"
    config = model_path / "config.json"
    weights = model_path / "model.safetensors"
    return {
        "command": command,
        "cwd": os.getcwd(),
        "hostname": platform.node(),
        "seed_determinism": "recorded_seeds_only",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "template_sha256": sha256_file(template),
        "template_matches_frozen": sha256_file(template) == FROZEN_TEMPLATE_SHA256,
        "model_weights_bytes": weights.stat().st_size,
        "model_weights_sha256": sha256_file(weights),
        "model_weights_matches_frozen": sha256_file(weights) == MODEL_WEIGHTS_SHA256,
        "model_config_sha256": sha256_file(config),
        "model_config": json.loads(config.read_text(encoding="utf-8")),
    }


def read_jsonl_samples(path: Path, max_length: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids = [int(value) for value in row["input_ids"]]
            labels = [int(value) for value in row["labels"]]
            if max_length is not None:
                ids = ids[:max_length]
                labels = labels[:max_length]
            if len(ids) != len(labels):
                raise ValueError(f"input/label length mismatch for {row.get('uid')}")
            if ids and any(label != -100 for label in labels):
                samples.append({"uid": str(row["uid"]), "input_ids": ids, "labels": labels})
    if not samples:
        raise ValueError(f"no usable samples in {path}")
    return samples


def sample_manifest(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "uid": row["uid"],
            "length": len(row["input_ids"]),
            "supervised_tokens": sum(value != -100 for value in row["labels"]),
            "token_label_sha256": canonical_hash([row["input_ids"], row["labels"]]),
        }
        for row in samples
    ]


def tensorize(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda")
    labels = torch.tensor([sample["labels"]], dtype=torch.long, device="cuda")
    return input_ids, labels, torch.ones_like(input_ids)


def load_model(model_path: Path, seed: int, *, adapter_path: Path | None = None) -> Any:
    seed_everything(seed)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    if adapter_path is None:
        model = get_peft_model(
            base,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
    else:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    return model


def configure_checkpointing(model: Any, enabled: bool) -> None:
    if enabled:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False


def adapter_snapshot(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def restore_adapter(model: Any, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameters[name].data.copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))
    model.zero_grad(set_to_none=True)


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_train_steps(
    model: Any,
    samples: list[dict[str, Any]],
    *,
    steps: int,
    seed: int,
    optimizer_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], torch.optim.Optimizer]:
    seed_everything(seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=2e-4)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    model.train()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    uids: list[str] = []
    tokens = 0
    stage = "start"
    started = time.perf_counter()
    try:
        for index in range(steps):
            sample = samples[index % len(samples)]
            input_ids, labels, attention_mask = tensorize(sample)
            stage = f"step_{index + 1}_forward"
            optimizer.zero_grad(set_to_none=True)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            stage = f"step_{index + 1}_loss"
            loss = output.loss
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite loss at step {index + 1}: {value}")
            stage = f"step_{index + 1}_backward"
            loss.backward()
            stage = f"step_{index + 1}_optimizer"
            optimizer.step()
            losses.append(value)
            uids.append(sample["uid"])
            tokens += input_ids.numel()
            del output, loss, input_ids, labels, attention_mask
        stage = "synchronize"
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "status": "completed",
            "last_stage": stage,
            "requested_steps": steps,
            "completed_steps": len(losses),
            "losses": losses,
            "all_finite": len(losses) == steps and all(math.isfinite(value) for value in losses),
            "sample_uids": uids,
            "input_tokens": tokens,
            "elapsed_seconds": elapsed,
            "input_tokens_per_second": tokens / elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "final_adapter_sha256": tensor_state_hash(adapter_snapshot(model)),
        }, optimizer
    except Exception as error:
        elapsed = time.perf_counter() - started
        report = {
            "status": "failed",
            "last_stage": stage,
            "requested_steps": steps,
            "completed_steps": len(losses),
            "losses": losses,
            "all_finite": False,
            "sample_uids": uids,
            "input_tokens": tokens,
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        return report, optimizer


def command_pair(args: argparse.Namespace) -> int:
    samples = read_jsonl_samples(args.samples, args.max_length)
    model = load_model(args.model, args.seed)
    initial = adapter_snapshot(model)
    initial_hash = tensor_state_hash(initial)
    configure_checkpointing(model, False)
    off, optimizer = run_train_steps(model, samples, steps=args.steps, seed=args.seed)
    del optimizer
    restore_adapter(model, initial)
    gc.collect()
    torch.cuda.empty_cache()
    configure_checkpointing(model, True)
    on, optimizer = run_train_steps(model, samples, steps=args.steps, seed=args.seed)
    del optimizer

    differences: list[float] = []
    if off["status"] == "completed" and on["status"] == "completed":
        differences = [
            abs(left - right) / max(abs(left), 1e-12)
            for left, right in zip(off["losses"], on["losses"], strict=True)
        ]
    loss_gate = bool(
        len(differences) == args.steps
        and all(value <= args.loss_relative_tolerance for value in differences)
    )
    memory_gate = bool(
        args.max_length != 4096
        or (on["status"] == "completed" and on["peak_allocated_gib"] <= args.memory_gate_gib)
    )
    report = {
        "schema_version": 1,
        "experiment": "alpha_checkpoint_pair",
        "environment": environment_record(args.model, " ".join(os.sys.argv)),
        "seed": args.seed,
        "checkpointing": {"implementation": "non_reentrant", "use_reentrant": False, "use_cache": False},
        "max_length": args.max_length,
        "steps": args.steps,
        "samples_file": str(args.samples),
        "samples_file_sha256": sha256_file(args.samples),
        "samples": sample_manifest(samples),
        "initial_adapter_sha256": initial_hash,
        "checkpoint_off": off,
        "checkpoint_on": on,
        "loss_relative_differences": differences,
        "max_loss_relative_difference": max(differences) if differences else None,
        "throughput_ratio_on_over_off": (
            on["input_tokens_per_second"] / off["input_tokens_per_second"]
            if on["status"] == off["status"] == "completed"
            else None
        ),
        "gates": {
            "loss_relative_tolerance": args.loss_relative_tolerance,
            "loss_equivalence_pass": loss_gate,
            "memory_gate_gib": args.memory_gate_gib if args.max_length == 4096 else None,
            "memory_pass": memory_gate,
        },
        "pass": loss_gate and memory_gate,
    }
    write_json(args.out, report)
    return 0 if report["pass"] else 2


def extract_token_fingerprint(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, indices = torch.max(logits.float(), dim=-1)
    log_normalizer = torch.logsumexp(logits.float(), dim=-1)
    return indices.detach().cpu(), (values - log_normalizer).detach().cpu()


def fingerprints_for_model(model: Any, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    model.train()
    for sample in samples:
        input_ids, _, attention_mask = tensorize(sample)
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        top1, logprobs = extract_token_fingerprint(output.logits)
        fingerprints.append({"uid": sample["uid"], "top1": top1, "logprobs": logprobs})
        del output, top1, logprobs, input_ids, attention_mask
        torch.cuda.empty_cache()
    return fingerprints


def compare_fingerprints(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    equal = 0
    total = 0
    delta_sum = 0.0
    per_sample: list[dict[str, Any]] = []
    for left, right in zip(reference, candidate, strict=True):
        if left["uid"] != right["uid"]:
            raise AssertionError("fingerprint UID order mismatch")
        same = left["top1"].eq(right["top1"])
        delta = (left["logprobs"] - right["logprobs"]).abs()
        count = same.numel()
        equal_count = int(same.sum())
        equal += equal_count
        total += count
        delta_sum += float(delta.sum())
        per_sample.append(
            {
                "uid": left["uid"],
                "tokens": count,
                "top1_agreement": equal_count / count,
                "mean_abs_delta_logprob": float(delta.mean()),
                "reference_top1_sha256": hashlib.sha256(left["top1"].numpy().tobytes()).hexdigest(),
                "candidate_top1_sha256": hashlib.sha256(right["top1"].numpy().tobytes()).hexdigest(),
            }
        )
    return {
        "tokens": total,
        "top1_equal_tokens": equal,
        "top1_agreement": equal / total,
        "mean_abs_delta_logprob": delta_sum / total,
        "per_sample": per_sample,
    }


def command_fingerprint(args: argparse.Namespace) -> int:
    samples = read_jsonl_samples(args.samples, args.max_length)[: args.count]
    if len(samples) != args.count:
        raise ValueError(f"expected {args.count} fingerprint samples, found {len(samples)}")
    model = load_model(args.model, args.seed)
    configure_checkpointing(model, False)
    torch.cuda.reset_peak_memory_stats()
    reference = fingerprints_for_model(model, samples)
    configure_checkpointing(model, True)
    candidate = fingerprints_for_model(model, samples)
    comparison = compare_fingerprints(reference, candidate)
    report = {
        "schema_version": 1,
        "experiment": "alpha_logits_fingerprint",
        "environment": environment_record(args.model, " ".join(os.sys.argv)),
        "seed": args.seed,
        "checkpointing": {"implementation": "non_reentrant", "use_reentrant": False, "use_cache": False},
        "samples_file": str(args.samples),
        "samples_file_sha256": sha256_file(args.samples),
        "samples": sample_manifest(samples),
        "comparison": comparison,
        "gates": {
            "min_top1_agreement": args.min_top1_agreement,
            "max_mean_abs_delta_logprob": args.max_mean_abs_delta_logprob,
        },
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    report["pass"] = bool(
        comparison["top1_agreement"] >= args.min_top1_agreement
        and comparison["mean_abs_delta_logprob"] <= args.max_mean_abs_delta_logprob
    )
    write_json(args.out, report)
    return 0 if report["pass"] else 2


def command_prepare_fingerprints(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    selected: list[dict[str, Any]] = []
    for path in sorted(args.rendered_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64, columns=["uid", "input_ids", "labels"]):
            for row in batch.to_pylist():
                ids = [int(value) for value in row["input_ids"][: args.max_length]]
                labels = [int(value) for value in row["labels"][: args.max_length]]
                if ids and any(value != -100 for value in labels):
                    selected.append({"uid": str(row["uid"]), "input_ids": ids, "labels": labels})
                if len(selected) == args.count:
                    break
            if len(selected) == args.count:
                break
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise ValueError(f"could select only {len(selected)} of {args.count} samples")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "source": str(args.rendered_dir),
        "selection": "sorted parquet shards then row order; prefix truncation",
        "count": args.count,
        "max_length": args.max_length,
        "output": str(args.out),
        "output_sha256": sha256_file(args.out),
        "samples": sample_manifest(selected),
    }
    write_json(args.manifest_out, manifest)
    return 0


def attention_kernel_records(profile: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in profile.key_averages(group_by_input_shape=True):
        if "scaled_dot_product" not in event.key:
            continue
        records.append(
            {
                "operator": event.key,
                "count": event.count,
                "input_shapes": event.input_shapes,
                "self_cuda_time_us": event.self_device_time_total,
                "cuda_time_us": event.device_time_total,
                "self_cuda_memory_bytes": event.self_device_memory_usage,
                "cuda_memory_bytes": event.device_memory_usage,
            }
        )
    return records


def top_memory_records(profile: Any, limit: int = 40) -> list[dict[str, Any]]:
    events = sorted(
        profile.key_averages(group_by_input_shape=True),
        key=lambda event: event.self_device_memory_usage,
        reverse=True,
    )
    return [
        {
            "operator": event.key,
            "count": event.count,
            "input_shapes": event.input_shapes,
            "self_cuda_memory_bytes": event.self_device_memory_usage,
            "cuda_memory_bytes": event.device_memory_usage,
            "self_cuda_time_us": event.self_device_time_total,
        }
        for event in events[:limit]
    ]


def command_profile(args: argparse.Namespace) -> int:
    samples = read_jsonl_samples(args.samples, args.max_length)
    sample = samples[0]
    model = load_model(args.model, args.seed)
    configure_checkpointing(model, args.gradient_checkpointing)
    model.train()
    input_ids, labels, attention_mask = tensorize(sample)
    stage = "setup"
    snapshot_written = False
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "alpha_memory_attention_profile",
        "environment": environment_record(args.model, " ".join(os.sys.argv)),
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "checkpointing": {"implementation": "non_reentrant", "use_reentrant": False, "use_cache": False},
        "samples_file": str(args.samples),
        "samples_file_sha256": sha256_file(args.samples),
        "sample": sample_manifest([sample])[0],
        "snapshot_path": str(args.snapshot_out),
    }
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="all", max_entries=args.memory_history_entries
        )
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profile:
            stage = "forward"
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss_value = float(output.loss.detach().cpu())
            stage = "backward"
            output.loss.backward()
            stage = "synchronize"
            torch.cuda.synchronize()
        args.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(args.snapshot_out))
        snapshot_written = True
        report.update(
            {
                "status": "completed",
                "last_stage": stage,
                "loss": loss_value,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "attention_kernels": attention_kernel_records(profile),
                "top_cuda_memory_operators": top_memory_records(profile),
            }
        )
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "last_stage": stage,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        try:
            args.snapshot_out.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(args.snapshot_out))
            snapshot_written = True
        except Exception as snapshot_error:
            report["snapshot_error"] = f"{type(snapshot_error).__name__}: {snapshot_error}"
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    if snapshot_written:
        report["snapshot_bytes"] = args.snapshot_out.stat().st_size
        report["snapshot_sha256"] = sha256_file(args.snapshot_out)
    write_json(args.out, report)
    return 0 if report["status"] == "completed" else 2


def standard_loss_backward(model: Any, sample: dict[str, Any]) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    input_ids, labels, attention_mask = tensorize(sample)
    started = time.perf_counter()
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    value = float(output.loss.detach().cpu())
    output.loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "loss": value,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    del output, input_ids, labels, attention_mask
    return result


def chunked_loss_backward(model: Any, sample: dict[str, Any], chunk_size: int) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    input_ids, labels, attention_mask = tensorize(sample)
    base = model.get_base_model()
    started = time.perf_counter()
    outputs = base.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    hidden = outputs.last_hidden_state[:, :-1, :]
    shifted_labels = labels[:, 1:]
    supervised = int(shifted_labels.ne(-100).sum())
    if not supervised:
        raise ValueError("chunked CE sample has no shifted supervised token")
    loss_parts: list[torch.Tensor] = []
    softcap = base.config.get_text_config().final_logit_softcapping

    def chunk_sum(chunk_hidden: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
        logits = base.lm_head(chunk_hidden)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            chunk_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    for start in range(0, hidden.shape[1], chunk_size):
        stop = min(start + chunk_size, hidden.shape[1])
        loss_parts.append(
            checkpoint(
                chunk_sum,
                hidden[:, start:stop, :],
                shifted_labels[:, start:stop],
                use_reentrant=False,
            )
        )
    loss = torch.stack(loss_parts).sum() / supervised
    value = float(loss.detach().cpu())
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "loss": value,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "chunk_size": chunk_size,
        "chunks": len(loss_parts),
        "supervised_tokens": supervised,
    }
    del outputs, hidden, shifted_labels, loss_parts, loss, input_ids, labels, attention_mask
    return result


def command_ce_probe(args: argparse.Namespace) -> int:
    samples = read_jsonl_samples(args.samples, args.max_length)
    sample = samples[0]
    model = load_model(args.model, args.seed)
    configure_checkpointing(model, args.gradient_checkpointing)
    model.train()
    standard = standard_loss_backward(model, sample)
    chunked = chunked_loss_backward(model, sample, args.chunk_size)
    delta = abs(standard["loss"] - chunked["loss"])
    report = {
        "schema_version": 1,
        "experiment": "alpha_chunked_ce_probe",
        "environment": environment_record(args.model, " ".join(os.sys.argv)),
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "sample": sample_manifest([sample])[0],
        "standard": standard,
        "chunked": chunked,
        "absolute_loss_difference": delta,
        "peak_allocated_delta_gib": standard["peak_allocated_gib"] - chunked["peak_allocated_gib"],
        "loss_equivalent_at_1e_4": delta <= 1e-4,
    }
    write_json(args.out, report)
    return 0


@torch.no_grad()
def eval_fingerprints(model: Any, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model.eval()
    result: list[dict[str, Any]] = []
    for sample in samples:
        input_ids, _, attention_mask = tensorize(sample)
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        top1, logprobs = extract_token_fingerprint(output.logits)
        result.append({"uid": sample["uid"], "top1": top1, "logprobs": logprobs})
        del output, input_ids, attention_mask, top1, logprobs
    return result


def command_ladder(args: argparse.Namespace) -> int:
    samples = read_jsonl_samples(args.samples, args.max_length)
    sample = samples[0]
    fingerprints = read_jsonl_samples(args.fingerprint_samples, args.fingerprint_max_length)[:32]
    output_dir = args.out.parent / f"checkpoint_{args.max_length}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model, args.seed)
    configure_checkpointing(model, True)
    training, optimizer = run_train_steps(model, [sample], steps=args.steps, seed=args.seed)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "alpha_checkpointed_length_ladder",
        "environment": environment_record(args.model, " ".join(os.sys.argv)),
        "seed": args.seed,
        "max_length": args.max_length,
        "sample": sample_manifest([sample])[0],
        "training": training,
        "checkpoint_dir": str(output_dir),
        "resume": {"status": "not_run"},
        "merge_fingerprint": {"status": "not_run"},
    }
    if training["status"] != "completed":
        write_json(args.out, report)
        return 2

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    continuation = {
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }
    continuation_path = output_dir / "continuation_state.pt"
    torch.save(continuation, continuation_path)
    checkpoint_files = [path for path in sorted(output_dir.rglob("*")) if path.is_file()]
    report["checkpoint_files"] = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in checkpoint_files
    ]
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    model = load_model(args.model, args.seed, adapter_path=adapter_dir)
    configure_checkpointing(model, True)
    state = torch.load(continuation_path, map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    resumed, optimizer = run_train_steps(
        model,
        [sample],
        steps=args.resume_steps,
        seed=args.seed,
        optimizer_state=state["optimizer"],
    )
    report["resume"] = resumed
    if resumed["status"] != "completed":
        write_json(args.out, report)
        return 2

    pre_merge = eval_fingerprints(model, fingerprints)
    merged = model.merge_and_unload(safe_merge=True)
    post_merge = eval_fingerprints(merged, fingerprints)
    merge_comparison = compare_fingerprints(pre_merge, post_merge)
    report["merge_fingerprint"] = {"status": "completed", **merge_comparison}
    report["pass"] = bool(
        training["peak_allocated_gib"] <= args.max_peak_gib
        and training["input_tokens_per_second"] >= args.min_throughput
        and resumed["all_finite"]
        and merge_comparison["top1_agreement"] >= args.merge_min_top1_agreement
    )
    report["gates"] = {
        "max_peak_gib": args.max_peak_gib,
        "min_input_tokens_per_second": args.min_throughput,
        "resume_steps": args.resume_steps,
        "merge_min_top1_agreement": args.merge_min_top1_agreement,
    }
    write_json(args.out, report)
    return 0 if report["pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-fingerprints")
    prepare.add_argument("--rendered-dir", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--manifest-out", type=Path, required=True)
    prepare.add_argument("--count", type=int, default=32)
    prepare.add_argument("--max-length", type=int, default=512)
    prepare.set_defaults(function=command_prepare_fingerprints)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--samples", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--seed", type=int, default=20260811)
        command.add_argument("--max-length", type=int, required=True)

    pair = subparsers.add_parser("pair")
    common(pair)
    pair.add_argument("--steps", type=int, required=True)
    pair.add_argument("--loss-relative-tolerance", type=float, default=1e-3)
    pair.add_argument("--memory-gate-gib", type=float, default=40.0)
    pair.set_defaults(function=command_pair)

    fingerprint = subparsers.add_parser("fingerprint")
    common(fingerprint)
    fingerprint.add_argument("--count", type=int, default=32)
    fingerprint.add_argument("--min-top1-agreement", type=float, default=0.999)
    fingerprint.add_argument("--max-mean-abs-delta-logprob", type=float, default=5e-3)
    fingerprint.set_defaults(function=command_fingerprint)

    profile = subparsers.add_parser("profile")
    common(profile)
    profile.add_argument("--snapshot-out", type=Path, required=True)
    profile.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    profile.add_argument("--memory-history-entries", type=int, default=200_000)
    profile.set_defaults(function=command_profile)

    ce_probe = subparsers.add_parser("ce-probe")
    common(ce_probe)
    ce_probe.add_argument("--chunk-size", type=int, default=128)
    ce_probe.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ce_probe.set_defaults(function=command_ce_probe)

    ladder = subparsers.add_parser("ladder")
    common(ladder)
    ladder.add_argument("--steps", type=int, default=20)
    ladder.add_argument("--resume-steps", type=int, default=3)
    ladder.add_argument("--fingerprint-samples", type=Path, required=True)
    ladder.add_argument("--fingerprint-max-length", type=int, default=512)
    ladder.add_argument("--max-peak-gib", type=float, default=88.0)
    ladder.add_argument("--min-throughput", type=float, default=1000.0)
    ladder.add_argument("--merge-min-top1-agreement", type=float, default=0.999)
    ladder.set_defaults(function=command_ladder)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
