#!/usr/bin/env python3
"""Run the frozen full-population lm_head lockstep C02 reconnection probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any

import torch
import torch.nn.functional as F

import run_k3_reconnect_r4 as shared
import run_op5_lm_head_deliverability_probe_r1 as delivery
from load_smoke_nf4 import get_submodule, memory_snapshot
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, summarize


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/operator/op5_lm_head_reconnect_r1.json"
DEFAULT_PREREG = (
    ROOT
    / "results/operator/pool/op5_lm_head_reconnect_r1_prereg_receipt_20260916.json"
)
DEFAULT_RESULT = ROOT / "results/operator/pool/op5_lm_head_reconnect_r1_20260916.json"
DEFAULT_RECEIPT = (
    ROOT / "results/operator/pool/op5_lm_head_reconnect_r1_receipt_20260916.json"
)
DEFAULT_PROGRESS = (
    ROOT / "results/operator/pool/op5_lm_head_reconnect_r1_progress_20260916.json"
)
DEFAULT_TELEMETRY = (
    ROOT / "results/operator/pool/op5_lm_head_reconnect_r1_telemetry_20260916.jsonl"
)


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_or_none(values: list[float]) -> dict[str, Any] | None:
    return summarize(values) if values else None


def validate_inputs(config: dict[str, Any], prereg_path: Path) -> dict[str, Any]:
    if config.get("stage") != "OP5_LM_HEAD_RECONNECT_R1":
        raise AssertionError("unexpected lm_head reconnection identity")
    if not config.get("frozen_before_formal_replay"):
        raise AssertionError("reconnection protocol was not frozen before replay")
    if config.get("authority", {}).get("sha_verification") != "SKIPPED_BY_OWNER_DIRECTION":
        raise AssertionError("reconnection must retain the owner no-SHA direction")
    if config.get("no_sha_policy", {}).get("calculate_or_compare_sha") is not False:
        raise AssertionError("reconnection no-SHA policy is not explicit")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise AssertionError("reconnection requires the frozen sm_89 CUDA device")

    prereg = load_json(prereg_path)
    if prereg.get("status") != "FROZEN_BEFORE_FORMAL_REPLAY":
        raise AssertionError("reconnection preregistration is not frozen")
    if prereg.get("formal_replay_started") is not False:
        raise AssertionError("preregistration was not captured before replay")
    if prereg.get("custom_candidate_implemented_or_timed") is not False:
        raise AssertionError("custom lm_head work is outside this reconnection")

    paths = {name: absolute(path) for name, path in config["inputs"].items()}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing input {name}: {path}")

    expected = config["authority"]["upstream_selection_status"]
    selection_receipt = load_json(paths["selection_receipt"])
    selection_result = load_json(paths["selection_result"])
    if selection_receipt.get("status") != expected or selection_result.get("status") != expected:
        raise AssertionError("upstream selection did not authorize mirror reconnection")
    if not selection_receipt.get("probe_valid") or not selection_receipt.get(
        "existing_path_eligible"
    ):
        raise AssertionError("upstream existing CUDA path is not eligible")
    if selection_receipt.get("bottleneck_gate") != "FAIL":
        raise AssertionError("custom-kernel bottleneck gate must remain closed")
    if selection_receipt.get("custom_candidate_implemented_or_timed") is not False:
        raise AssertionError("upstream selection unexpectedly contains a custom candidate")

    deliverability = load_json(paths["deliverability_receipt"])
    if deliverability.get("status") != "LM_HEAD_DELIVERABILITY_HOLD_LIFTED_SELECTION_AUTHORIZED":
        raise AssertionError("deliverability authorization is not intact")

    inventory = shared.validate_model_inventory_without_sha(paths["model"])
    tapes = shared.load_tapes_without_sha(paths["tapes_manifest"])
    census = load_json(paths["shape_census"])
    if census.get("status") != "PASS":
        raise AssertionError("shape census is not PASS")
    lm_head = census["op_families"]["lm_head"]
    projected = sum(int(row["invocations"]) for row in lm_head["shapes"])
    if projected != int(config["replay"]["expected_completion_tokens"]):
        raise AssertionError("lm_head census population drift")
    if float(lm_head["final_logit_softcap"]) != 30.0:
        raise AssertionError("lm_head softcap drift")
    archived = shared.load_qualified_baseline(paths["archived_baseline_metadata"])
    return {
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "preregistration": {"path": str(prereg_path), "status": prereg["status"]},
        "selection": {
            "result": str(paths["selection_result"]),
            "receipt": str(paths["selection_receipt"]),
            "status": expected,
            "bottleneck_gate": selection_receipt["bottleneck_gate"],
        },
        "deliverability": {
            "receipt": str(paths["deliverability_receipt"]),
            "status": deliverability["status"],
        },
        "model_inventory": inventory,
        "census": {
            "path": str(paths["shape_census"]),
            "projected_calls": projected,
            "shapes": lm_head["shapes"],
        },
        "paths": paths,
        "tapes": tapes,
        "census_full": census,
        "archived": archived,
    }


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = now_iso()
        self.state = "INITIALIZING"
        self.completed: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.executed_tokens = 0

    def write(self, state: str | None = None) -> None:
        if state is not None:
            self.state = state
        atomic_write_json(
            self.path,
            {
                "schema_version": 1,
                "stage": "OP5_LM_HEAD_RECONNECT_R1",
                "status": self.state,
                "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
                "started_at": self.started_at,
                "updated_at": now_iso(),
                "executed_lockstep_tokens": self.executed_tokens,
                "expected_lockstep_tokens": 21705,
                "completed_requests": len(self.completed),
                "expected_requests": 11,
                "current": self.current,
                "completed": self.completed,
            },
        )

    def token_progress(self, value: dict[str, Any]) -> None:
        self.current = value
        self.executed_tokens = int(value["global_executed_tokens"])
        self.write("RUNNING_LOCKSTEP")

    def complete(self, record: dict[str, Any]) -> None:
        self.executed_tokens += int(record["executed_tokens"]) - int(
            record["last_reported_global_delta"]
        )
        self.completed.append(
            {
                "tape_id": record["tape_id"],
                "request_id": record["request_id"],
                "requested_tokens": record["completion_tokens"],
                "executed_tokens": record["executed_tokens"],
                "full_budget": record["full_budget"],
                "token_exact": record["token_comparison"]["identical"],
                "first_divergence": record["token_comparison"]["first_divergence"],
                "completed_at": now_iso(),
            }
        )
        self.current = None
        self.write("RUNNING_LOCKSTEP")


def _head_token_cpu(
    output_head: Any, hidden_gpu: torch.Tensor, softcap: float
) -> tuple[int, torch.Tensor, torch.Tensor, float]:
    started = time.perf_counter()
    hidden_cpu = hidden_gpu.to("cpu")
    logits = output_head(hidden_cpu)
    capped = torch.tanh(logits / softcap) * softcap
    token = int(capped[0, -1].argmax().item())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return token, hidden_cpu, capped, elapsed_ms


def _head_token_cuda(
    mirror: torch.Tensor, hidden_gpu: torch.Tensor, softcap: float
) -> tuple[int, torch.Tensor, float]:
    started = time.perf_counter()
    logits = F.linear(hidden_gpu, mirror)
    capped = torch.tanh(logits / softcap) * softcap
    token = int(capped[0, -1].argmax().item())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return token, capped, elapsed_ms


def run_request_lockstep(
    model: Any,
    mirror: torch.Tensor,
    request: dict[str, Any],
    *,
    device: int,
    seed: int,
    progress: ProgressWriter,
    progress_interval: int,
    global_start_index: int,
    attention_modules: dict[int, Any],
    fail_fast: bool,
) -> dict[str, Any]:
    text_model = get_submodule(model, "model.language_model")
    token_embedding = text_model.embed_tokens
    per_layer_embedding = text_model.embed_tokens_per_layer
    output_head = get_submodule(model, "lm_head")
    cuda_device = torch.device(f"cuda:{device}")
    num_layers = int(text_model.config.num_hidden_layers)
    per_layer_width = int(text_model.config.hidden_size_per_layer_input)
    softcap = float(text_model.config.final_logit_softcapping)
    completion_tokens = int(request["completion_tokens"])
    prompt_ids = request["rendered_token_ids"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
    attention_mask = torch.ones_like(input_ids, device="cpu")
    current_ids = input_ids
    past_key_values = None
    generated_a: list[int] = []
    generated_b: list[int] = []
    common_ms: list[float] = []
    cpu_head_ms: list[float] = []
    cuda_head_ms: list[float] = []
    embedding_cuda_deltas: list[int] = []
    eos_positions: list[int] = []
    order_counts = {"cpu_first": 0, "cuda_first": 0}
    divergence: dict[str, Any] | None = None
    raw_eos_ids = text_model.config.eos_token_id
    eos_ids = (
        {int(raw_eos_ids)}
        if isinstance(raw_eos_ids, int)
        else {int(value) for value in raw_eos_ids}
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    request_started = time.perf_counter()
    last_reported_emitted = 0
    for token_index in range(completion_tokens):
        torch.cuda.synchronize(device)
        common_started = time.perf_counter()
        allocation_before_embedding = torch.cuda.memory_allocated(device)
        inputs_embeds_cpu = token_embedding(current_ids)
        per_layer_inputs_cpu = per_layer_embedding(current_ids).reshape(
            *current_ids.shape, num_layers, per_layer_width
        )
        allocation_after_embedding = torch.cuda.memory_allocated(device)
        embedding_delta = allocation_after_embedding - allocation_before_embedding
        embedding_cuda_deltas.append(int(embedding_delta))
        if embedding_delta != 0:
            raise AssertionError(
                f"CPU embedding lookup changed CUDA allocation: {embedding_delta}"
            )
        inputs_embeds = inputs_embeds_cpu.to(cuda_device)
        per_layer_inputs = per_layer_inputs_cpu.to(cuda_device)
        step_attention_mask = attention_mask.to(cuda_device)
        del inputs_embeds_cpu, per_layer_inputs_cpu
        with torch.inference_mode():
            output = text_model(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
            )
        past_key_values = output.past_key_values
        torch.cuda.synchronize(device)
        common_elapsed_ms = (time.perf_counter() - common_started) * 1000.0
        hidden_gpu = output.last_hidden_state[:, -1:, :]
        if tuple(hidden_gpu.shape) != (1, 1, 2560):
            raise AssertionError(f"hidden shape drift: {tuple(hidden_gpu.shape)}")
        if hidden_gpu.dtype != torch.bfloat16 or hidden_gpu.device.type != "cuda":
            raise AssertionError("hidden dtype/device drift")
        if not bool(torch.isfinite(hidden_gpu).all().item()):
            raise FloatingPointError("non-finite final hidden state")

        cpu_first = (global_start_index + token_index) % 2 == 0
        with torch.inference_mode():
            if cpu_first:
                order_counts["cpu_first"] += 1
                token_a, hidden_cpu, capped_a, elapsed_a = _head_token_cpu(
                    output_head, hidden_gpu, softcap
                )
                token_b, capped_b, elapsed_b = _head_token_cuda(
                    mirror, hidden_gpu, softcap
                )
            else:
                order_counts["cuda_first"] += 1
                token_b, capped_b, elapsed_b = _head_token_cuda(
                    mirror, hidden_gpu, softcap
                )
                token_a, hidden_cpu, capped_a, elapsed_a = _head_token_cpu(
                    output_head, hidden_gpu, softcap
                )

        expected_logits_shape = (1, 1, 262144)
        if tuple(capped_a.shape) != expected_logits_shape or tuple(capped_b.shape) != expected_logits_shape:
            raise AssertionError("lm_head output shape drift")
        if capped_a.dtype != torch.bfloat16 or capped_a.device.type != "cpu":
            raise AssertionError("CPU lm_head dtype/device drift")
        if capped_b.dtype != torch.bfloat16 or capped_b.device.type != "cuda":
            raise AssertionError("CUDA lm_head dtype/device drift")
        if not bool(torch.isfinite(capped_a).all().item()) or not bool(
            torch.isfinite(capped_b).all().item()
        ):
            raise FloatingPointError("non-finite lm_head output")

        common_ms.append(common_elapsed_ms)
        cpu_head_ms.append(elapsed_a)
        cuda_head_ms.append(elapsed_b)
        generated_a.append(token_a)
        generated_b.append(token_b)
        emitted = token_index + 1
        if token_a in eos_ids:
            eos_positions.append(token_index)

        if token_a != token_b:
            start = max(0, token_index - 8)
            divergence = {
                "generated_index_zero_based": token_index,
                "absolute_context_position_zero_based": len(prompt_ids) + token_index,
                "cpu_token_id": token_a,
                "cuda_token_id": token_b,
                "cpu_prefix_context": generated_a[start:token_index],
                "cuda_prefix_context": generated_b[start:token_index],
            }

        del inputs_embeds, per_layer_inputs, step_attention_mask
        del output, hidden_gpu, hidden_cpu, capped_a, capped_b

        global_executed = global_start_index + emitted
        if (
            emitted == 1
            or emitted % progress_interval == 0
            or emitted == completion_tokens
            or divergence is not None
        ):
            progress.token_progress(
                {
                    "tape_id": request["tape_id"],
                    "request_id": request["request_id"],
                    "executed_tokens": emitted,
                    "completion_tokens": completion_tokens,
                    "global_executed_tokens": global_executed,
                    "latest_common_ms": common_ms[-1],
                    "latest_cpu_head_ms": cpu_head_ms[-1],
                    "latest_cuda_head_ms": cuda_head_ms[-1],
                    "token_exact": divergence is None,
                    "first_divergence": divergence,
                    "cache_sequence_length": int(past_key_values.get_seq_length()),
                    "captured_at": now_iso(),
                }
            )
            last_reported_emitted = emitted
            log(
                f"lm_head lockstep {request['request_id']} token "
                f"{emitted}/{completion_tokens}; A={token_a} B={token_b}; "
                f"common={common_ms[-1]:.3f} ms, cpu={cpu_head_ms[-1]:.3f} ms, "
                f"cuda={cuda_head_ms[-1]:.3f} ms"
            )

        if divergence is not None and fail_fast:
            break
        current_ids = torch.tensor([[token_a]], dtype=torch.long, device="cpu")
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype),
            ),
            dim=1,
        )

    executed = len(generated_a)
    final_sequence_length = len(prompt_ids) + executed - 1
    if int(past_key_values.get_seq_length()) != final_sequence_length:
        raise AssertionError("final cache sequence length mismatch")
    cache = shared.cache_structure(
        past_key_values, attention_modules, final_sequence_length
    )
    full_budget = executed == completion_tokens
    exact = divergence is None and full_budget and generated_a == generated_b
    record = {
        "tape_id": request["tape_id"],
        "task_id": request["task_id"],
        "request_number": request["request_number"],
        "request_id": request["request_id"],
        "seed": seed,
        "greedy": True,
        "eos_does_not_truncate": True,
        "rendered_prompt_tokens": len(prompt_ids),
        "completion_tokens": completion_tokens,
        "executed_tokens": executed,
        "full_budget": full_budget,
        "generated_token_ids": {"A_cpu": generated_a, "B_cuda": generated_b},
        "eos_token_ids": sorted(eos_ids),
        "eos_generated_positions_zero_based": eos_positions,
        "token_comparison": {
            "comparison": "same-load same-hidden-state CPU tied head vs eager CUDA BF16 mirror",
            "identical": exact,
            "matching_prefix_tokens": executed if divergence is None else int(
                divergence["generated_index_zero_based"]
            ),
            "first_divergence": divergence,
        },
        "head_dispatch": {
            "A_cpu_calls": executed,
            "B_cuda_calls": executed,
            "prefill_calls_per_path": 1 if executed else 0,
            "decode_calls_per_path": max(0, executed - 1),
            "order_counts": order_counts,
        },
        "timing_steps_ms": {
            "common": common_ms,
            "A_cpu_head": cpu_head_ms,
            "B_cuda_head": cuda_head_ms,
        },
        "timing_summary_ms": {
            "common": summarize_or_none(common_ms),
            "A_cpu_head": summarize_or_none(cpu_head_ms),
            "B_cuda_head": summarize_or_none(cuda_head_ms),
            "A_counterfactual": summarize_or_none(
                [common + head for common, head in zip(common_ms, cpu_head_ms, strict=True)]
            ),
            "B_counterfactual": summarize_or_none(
                [common + head for common, head in zip(common_ms, cuda_head_ms, strict=True)]
            ),
        },
        "measured_lockstep_wall_seconds": time.perf_counter() - request_started,
        "embedding_cuda_allocation_delta_max_bytes": max(
            embedding_cuda_deltas, default=0
        ),
        "final_cache": cache,
        "memory": memory_snapshot(torch, device),
        "last_reported_global_delta": last_reported_emitted,
    }
    del current_ids, input_ids, attention_mask, past_key_values
    torch.cuda.empty_cache()
    return record


def aggregate_counterfactual(
    records: list[dict[str, Any]], head_key: str
) -> dict[str, Any]:
    common = [
        value
        for record in records
        for value in record["timing_steps_ms"]["common"]
    ]
    head = [
        value
        for record in records
        for value in record["timing_steps_ms"][head_key]
    ]
    paired = [left + right for left, right in zip(common, head, strict=True)]
    ttft = [
        record["timing_steps_ms"]["common"][0]
        + record["timing_steps_ms"][head_key][0]
        for record in records
        if record["executed_tokens"] > 0
    ]
    incremental = [
        common_value + head_value
        for record in records
        for common_value, head_value in zip(
            record["timing_steps_ms"]["common"][1:],
            record["timing_steps_ms"][head_key][1:],
            strict=True,
        )
    ]
    incremental_seconds = sum(incremental) / 1000.0
    return {
        "requests_started": len(records),
        "token_decisions": len(paired),
        "ttft_ms": summarize_or_none(ttft),
        "common_ms": summarize_or_none(common),
        "head_ms": summarize_or_none(head),
        "counterfactual_step_ms": summarize_or_none(paired),
        "incremental_tokens": len(incremental),
        "aggregate_incremental_seconds": incremental_seconds,
        "aggregate_incremental_tpot_ms": (
            sum(incremental) / len(incremental) if incremental else None
        ),
        "aggregate_incremental_tokens_per_second": (
            len(incremental) / incremental_seconds
            if incremental and incremental_seconds > 0
            else None
        ),
        "counterfactual_total_seconds": sum(paired) / 1000.0,
        "peak_cuda_allocated_bytes": max(
            (int(record["memory"]["peak_allocated_bytes"]) for record in records),
            default=0,
        ),
        "peak_cuda_reserved_bytes": max(
            (int(record["memory"]["peak_reserved_bytes"]) for record in records),
            default=0,
        ),
        "max_nvml_device_used_bytes": max(
            (int(record["memory"]["device_used_bytes"]) for record in records),
            default=0,
        ),
    }


def run_formal(
    config: dict[str, Any],
    config_path: Path,
    prereg_path: Path,
    progress: ProgressWriter,
    telemetry_path: Path,
    device: int,
) -> dict[str, Any]:
    started_at = now_iso()
    wall_started = time.perf_counter()
    progress.write("VALIDATING_WITHOUT_SHA")
    validated = validate_inputs(config, prereg_path)
    paths = validated.pop("paths")
    tapes = validated.pop("tapes")
    census = validated.pop("census_full")
    archived = validated.pop("archived")

    progress.write("LOADING_MODEL")
    model, tokenizer, carrier = shared.load_carrier_without_sha(config, device)
    rendered = shared.render_requests(tokenizer, tapes, census, archived)
    text_model = get_submodule(model, "model.language_model")
    output_head = get_submodule(model, "lm_head")
    token_embedding = get_submodule(model, "model.language_model.embed_tokens")
    if output_head.weight.data_ptr() != token_embedding.weight.data_ptr():
        raise AssertionError("lm_head is not tied to the CPU token embedding")
    if output_head.weight.device.type != "cpu":
        raise AssertionError("original lm_head is not CPU-resident")
    structure_before = shared.model_structure_snapshot(model)
    head_before = delivery.tensor_identity(output_head.weight)
    embedding_before = delivery.tensor_identity(token_embedding.weight)
    attention_modules = {
        index: get_submodule(text_model, f"layers.{index}.self_attn")
        for index in range(int(text_model.config.num_hidden_layers))
    }
    if len(attention_modules) != 42:
        raise AssertionError("formal carrier does not expose 42 attention modules")

    monitor = NvidiaSmiMonitor(
        interval_seconds=float(config["replay"]["telemetry_interval_seconds"])
    )
    monitor.set_phase("lm_head_lockstep_setup")
    monitor.start()
    mirror: torch.Tensor | None = None
    records: list[dict[str, Any]] = []
    try:
        mirror = output_head.weight.detach().to(
            device=torch.device(f"cuda:{device}"), dtype=torch.bfloat16, copy=True
        ).contiguous()
        if mirror.data_ptr() == output_head.weight.data_ptr():
            raise AssertionError("CUDA mirror aliases the original CPU storage")
        mirror_identity = delivery.tensor_identity(mirror)
        memory_after_mirror = memory_snapshot(torch, device)

        monitor.set_phase("lm_head_lockstep_warmup")
        softcap = float(text_model.config.final_logit_softcapping)
        hidden_cpu = torch.zeros((1, 1, 2560), dtype=torch.bfloat16, device="cpu")
        hidden_cuda = hidden_cpu.to(device)
        with torch.inference_mode():
            for _ in range(int(config["measurement"]["cpu_head_warmup"])):
                logits = output_head(hidden_cpu)
                (torch.tanh(logits / softcap) * softcap).argmax(dim=-1)
            for _ in range(int(config["measurement"]["cuda_head_warmup"])):
                logits = F.linear(hidden_cuda, mirror)
                (torch.tanh(logits / softcap) * softcap).argmax(dim=-1)
        torch.cuda.synchronize(device)
        del logits, hidden_cpu, hidden_cuda
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        progress.write("RUNNING_LOCKSTEP")
        global_index = 0
        for request in rendered:
            monitor.set_phase(f"lockstep:{request['request_id']}")
            record = run_request_lockstep(
                model,
                mirror,
                request,
                device=device,
                seed=int(config["replay"]["seed"]),
                progress=progress,
                progress_interval=int(config["replay"]["progress_interval_tokens"]),
                global_start_index=global_index,
                attention_modules=attention_modules,
                fail_fast=bool(config["replay"]["fail_fast_on_first_token_divergence"]),
            )
            records.append(record)
            global_index += int(record["executed_tokens"])
            progress.complete(record)
            if record["token_comparison"]["first_divergence"] is not None:
                log(
                    "lm_head lockstep fail-fast: first unequal token recorded at "
                    f"{record['request_id']} index "
                    f"{record['token_comparison']['first_divergence']['generated_index_zero_based']}"
                )
                break

        monitor.set_phase("lm_head_lockstep_post_replay")
        head_during = delivery.tensor_identity(output_head.weight)
        embedding_during = delivery.tensor_identity(token_embedding.weight)
        memory_before_cleanup = memory_snapshot(torch, device)
        del mirror
        mirror = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    finally:
        if mirror is not None:
            del mirror
            torch.cuda.empty_cache()
        monitor.set_phase("lm_head_lockstep_cleanup")
        monitor.stop()

    structure_after = shared.model_structure_snapshot(model)
    structure = shared.compare_model_structure(structure_before, structure_after)
    head_after = delivery.tensor_identity(output_head.weight)
    embedding_after = delivery.tensor_identity(token_embedding.weight)
    original_identity_unchanged = (
        head_before == head_during == head_after
        and embedding_before == embedding_during == embedding_after
        and output_head.weight.data_ptr() == token_embedding.weight.data_ptr()
    )
    memory_after_cleanup = memory_snapshot(torch, device)
    telemetry = shared.telemetry_report(monitor, telemetry_path)
    telemetry_summary = telemetry["summary"]
    memory_clock = telemetry_summary["numeric"]["memory_mhz"]
    memory_clock_p50 = float(memory_clock["p50"]) if memory_clock else 0.0
    memory_clock_ratio = memory_clock_p50 / float(
        config["measurement"]["memory_clock_reference_mhz"]
    )

    executed_tokens = sum(int(record["executed_tokens"]) for record in records)
    full_requests = sum(bool(record["full_budget"]) for record in records)
    first_divergence = next(
        (
            {
                "request_id": record["request_id"],
                **record["token_comparison"]["first_divergence"],
            }
            for record in records
            if record["token_comparison"]["first_divergence"] is not None
        ),
        None,
    )
    expected_tokens = int(config["replay"]["expected_completion_tokens"])
    expected_requests = int(config["replay"]["expected_requests"])
    fully_exact = bool(
        first_divergence is None
        and executed_tokens == expected_tokens
        and full_requests == expected_requests
        and len(records) == expected_requests
        and all(record["token_comparison"]["identical"] for record in records)
    )
    dispatch_a = sum(record["head_dispatch"]["A_cpu_calls"] for record in records)
    dispatch_b = sum(record["head_dispatch"]["B_cuda_calls"] for record in records)
    cache_pass_count = sum(record["final_cache"]["pass"] for record in records)
    runtime_a = aggregate_counterfactual(records, "A_cpu_head")
    runtime_b = aggregate_counterfactual(records, "B_cuda_head")
    a_tpot = runtime_a["aggregate_incremental_tpot_ms"]
    b_tpot = runtime_b["aggregate_incremental_tpot_ms"]
    speedup = (
        float(a_tpot) / float(b_tpot)
        if fully_exact and a_tpot is not None and b_tpot not in (None, 0)
        else None
    )
    minimum_speedup = float(config["measurement"]["minimum_end_to_end_speedup"])
    performance_pass = bool(speedup is not None and speedup >= minimum_speedup)
    memory_budget = int(float(config["carrier"]["gpu_budget_gib"]) * 1024**3)
    peak_allocated = max(
        int(memory_before_cleanup["peak_allocated_bytes"]),
        runtime_a["peak_cuda_allocated_bytes"],
        runtime_b["peak_cuda_allocated_bytes"],
    )

    validity_gates = {
        "owner_no_sha_direction_honored": "PASS",
        "upstream_existing_path_authorization": "PASS",
        "custom_kernel_remains_unauthorized": "PASS",
        "formal_c02_nf4_carrier_343_modules": "PASS"
        if carrier["quantization"]["active_language_linear4bit_count"] == 343
        else "FAIL",
        "formal_population_or_conclusive_fail_fast": "PASS"
        if fully_exact or first_divergence is not None
        else "FAIL",
        "paired_head_dispatch_reconciled": "PASS"
        if dispatch_a == dispatch_b == executed_tokens and executed_tokens > 0
        else "FAIL",
        "cache_structure_all_executed_requests": "PASS"
        if cache_pass_count == len(records)
        else "FAIL",
        "original_tied_cpu_weight_and_model_structure_unchanged": "PASS"
        if original_identity_unchanged and structure["pass"]
        else "FAIL",
        "gpu_memory_below_7_5_gib": "PASS"
        if peak_allocated < memory_budget
        else "FAIL",
        "telemetry_present": "PASS"
        if int(telemetry["raw_jsonl"]["samples"]) > 0
        else "FAIL",
        "memory_clock_gate": "PASS"
        if memory_clock_ratio
        >= float(config["measurement"]["minimum_memory_clock_ratio"])
        else "FAIL",
        "thermal_hardware_throttle_absent": "PASS"
        if not telemetry_summary["thermal_or_hardware_throttle"]
        else "FAIL",
    }
    validity_pass = all(value == "PASS" for value in validity_gates.values())
    equivalence_gate = "PASS" if fully_exact else "FAIL"
    performance_gate = (
        "PASS" if performance_pass else "FAIL" if fully_exact else "NOT_ELIGIBLE"
    )
    if not validity_pass:
        status = config["decision"]["on_invalid"]
    elif first_divergence is not None:
        status = config["decision"]["on_token_fail"]
    elif not fully_exact:
        status = config["decision"]["on_invalid"]
    elif not performance_pass:
        status = config["decision"]["on_performance_fail"]
    else:
        status = config["decision"]["on_pass"]
    deployment_authorized = status == config["decision"]["on_pass"]
    return {
        "schema_version": 1,
        "stage": config["stage"],
        "target": config["target"],
        "status": status,
        "validity_pass": validity_pass,
        "hard_gates_pass": deployment_authorized,
        "deployment_authorized": deployment_authorized,
        "custom_candidate_implemented_or_timed": False,
        "started_at": started_at,
        "finished_at": now_iso(),
        "wall_seconds": time.perf_counter() - wall_started,
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "environment": shared.environment_identity(),
        "protocol": {
            "path": str(config_path),
            "preregistration": str(prereg_path),
            "method": config["replay"]["method"],
            "equivalence_induction": config["replay"]["equivalence_induction"],
            "replication_disclosure": config["authority"]["replication_disclosure"],
        },
        "validated_inputs": validated,
        "carrier": carrier,
        "mirror": {
            "policy": config["carrier"]["mirror_policy"],
            "identity": mirror_identity,
            "memory_after_allocation": memory_after_mirror,
            "warmup": {
                "cpu_calls": int(config["measurement"]["cpu_head_warmup"]),
                "cuda_calls": int(config["measurement"]["cuda_head_warmup"]),
            },
        },
        "records": records,
        "token_correctness_summary": {
            "expected_requests": expected_requests,
            "started_requests": len(records),
            "full_budget_requests": full_requests,
            "expected_token_decisions_per_path": expected_tokens,
            "executed_token_decisions_per_path": executed_tokens,
            "matching_token_decisions": (
                executed_tokens if first_divergence is None else executed_tokens - 1
            ),
            "all_tokens_identical": fully_exact,
            "first_divergence": first_divergence,
            "fail_fast_applied": first_divergence is not None,
        },
        "head_dispatch": {
            "A_cpu_calls": dispatch_a,
            "B_cuda_calls": dispatch_b,
            "paired": dispatch_a == dispatch_b == executed_tokens,
        },
        "cache_summary": {
            "passing_requests": cache_pass_count,
            "executed_requests": len(records),
        },
        "runtime": {
            "measurement_convention": config["measurement"][
                "counterfactual_path_time"
            ],
            "A_cpu": runtime_a,
            "B_cuda": runtime_b,
            "speedup_A_over_B": speedup,
            "minimum_required_speedup": minimum_speedup,
        },
        "memory": {
            "gpu_budget_bytes": memory_budget,
            "peak_allocated_bytes": peak_allocated,
            "before_cleanup": memory_before_cleanup,
            "after_cleanup": memory_after_cleanup,
        },
        "structure": {
            "head_before": head_before,
            "head_during": head_during,
            "head_after": head_after,
            "embedding_before": embedding_before,
            "embedding_during": embedding_during,
            "embedding_after": embedding_after,
            "original_identity_unchanged": original_identity_unchanged,
            "model_structure": structure,
        },
        "telemetry": telemetry,
        "memory_clock_ratio_vs_reference": memory_clock_ratio,
        "validity_gates": validity_gates,
        "equivalence_gate": equivalence_gate,
        "performance_gate": performance_gate,
        "decision": status,
        "non_authorizations": config["decision"]["non_authorizations"],
    }


def build_receipt(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    token = result["token_correctness_summary"]
    runtime = result["runtime"]
    return {
        "schema_version": 1,
        "stage": result["stage"],
        "target": result["target"],
        "timestamp": result["finished_at"],
        "status": result["status"],
        "validity_pass": result["validity_pass"],
        "hard_gates_pass": result["hard_gates_pass"],
        "deployment_authorized": result["deployment_authorized"],
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "custom_candidate_implemented_or_timed": False,
        "result": str(result_path),
        "validity_gates": result["validity_gates"],
        "equivalence_gate": result["equivalence_gate"],
        "performance_gate": result["performance_gate"],
        "summary": {
            "started_requests": token["started_requests"],
            "full_budget_requests": token["full_budget_requests"],
            "expected_token_decisions_per_path": token[
                "expected_token_decisions_per_path"
            ],
            "executed_token_decisions_per_path": token[
                "executed_token_decisions_per_path"
            ],
            "matching_token_decisions": token["matching_token_decisions"],
            "first_divergence": token["first_divergence"],
            "A_counterfactual_incremental_tpot_ms": runtime["A_cpu"][
                "aggregate_incremental_tpot_ms"
            ],
            "B_counterfactual_incremental_tpot_ms": runtime["B_cuda"][
                "aggregate_incremental_tpot_ms"
            ],
            "A_over_B_speedup": runtime["speedup_A_over_B"],
            "peak_cuda_allocated_bytes": result["memory"]["peak_allocated_bytes"],
            "telemetry_samples": result["telemetry"]["raw_jsonl"]["samples"],
            "memory_clock_ratio_vs_reference": result[
                "memory_clock_ratio_vs_reference"
            ],
            "thermal_or_hardware_throttle": result["telemetry"]["summary"][
                "thermal_or_hardware_throttle"
            ],
            "wall_seconds": result["wall_seconds"],
        },
        "replication_disclosure": result["protocol"]["replication_disclosure"],
        "next_action": (
            "Record the selected eager CUDA lm_head mirror as authorized only for the frozen local C02 PyTorch path; keep custom-kernel and extrapolation boundaries closed."
            if result["deployment_authorized"]
            else "Close or remediate only according to the explicit token/performance/validity branch in the result; do not implement a custom lm_head kernel."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = absolute(args.config)
    prereg_path = absolute(args.prereg)
    result_path = absolute(args.result)
    receipt_path = absolute(args.receipt)
    progress = ProgressWriter(absolute(args.progress))
    config = load_json(config_path)
    try:
        result = run_formal(
            config,
            config_path,
            prereg_path,
            progress,
            absolute(args.telemetry),
            args.device,
        )
        atomic_write_json(result_path, result)
        atomic_write_json(receipt_path, build_receipt(result, result_path))
        progress.write("COMPLETE")
        log(f"formal lm_head reconnection finished: {result['status']}")
        return 0 if result["deployment_authorized"] else 2
    except BaseException as exc:
        failed_at = now_iso()
        failure_path = (
            ROOT
            / "results/operator/failures"
            / (
                "op5_lm_head_reconnect_r1_"
                + failed_at.replace(":", "").replace("+", "_")
                + ".json"
            )
        )
        atomic_write_json(
            failure_path,
            {
                "schema_version": 1,
                "stage": "OP5_LM_HEAD_RECONNECT_R1",
                "status": "EXECUTION_FAILURE",
                "timestamp": failed_at,
                "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
                "custom_candidate_implemented_or_timed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "progress": str(progress.path),
            },
        )
        progress.current = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failure_receipt": str(failure_path),
        }
        progress.write("FAILED")
        log(f"formal lm_head reconnection failed: {failure_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
