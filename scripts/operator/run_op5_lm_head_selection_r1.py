#!/usr/bin/env python3
"""Run the frozen lm_head A/B/C/D selection probe without a custom candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any, Callable

import torch
import torch.nn.functional as F

import run_k3_reconnect_r4 as shared
import run_op5_lm_head_deliverability_probe_r1 as delivery
from load_smoke_nf4 import get_submodule, memory_snapshot
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, summarize
from run_op5_selection import run_same_day_bandwidth


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/operator/op5_lm_head_selection_r1.json"
DEFAULT_PREREG = (
    ROOT / "results/operator/pool/op5_lm_head_selection_r1_prereg_receipt_20260916.json"
)
DEFAULT_RESULT = ROOT / "results/operator/pool/op5_lm_head_selection_r1_20260916.json"
DEFAULT_RECEIPT = (
    ROOT / "results/operator/pool/op5_lm_head_selection_r1_receipt_20260916.json"
)
DEFAULT_TELEMETRY = (
    ROOT / "results/operator/pool/op5_lm_head_selection_r1_telemetry_20260916.jsonl"
)


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(
    config: dict[str, Any], prereg_path: Path
) -> dict[str, Any]:
    if config.get("stage") != "OP5_LM_HEAD_SELECTION_R1":
        raise AssertionError("unexpected lm_head selection identity")
    if config.get("authority", {}).get("sha_verification") != "SKIPPED_BY_OWNER_DIRECTION":
        raise AssertionError("selection must retain the owner no-SHA direction")
    if config.get("no_sha_policy", {}).get("calculate_or_compare_sha") is not False:
        raise AssertionError("selection no-SHA policy is not explicit")
    if not config.get("frozen_before_formal_measurement"):
        raise AssertionError("selection was not frozen before measurement")
    if not config.get("frozen_before_custom_candidate"):
        raise AssertionError("selection was not frozen before candidate work")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise AssertionError("selection requires the frozen sm_89 CUDA device")

    prereg = load_json(prereg_path)
    if prereg.get("status") != "FROZEN_BEFORE_FORMAL_SELECTION":
        raise AssertionError("selection preregistration is not frozen")
    if prereg.get("formal_selection_started") is not False:
        raise AssertionError("preregistration was not captured before selection")
    if prereg.get("custom_candidate_implemented_or_timed") is not False:
        raise AssertionError("custom candidate existed before selection")

    paths = {name: absolute(path) for name, path in config["inputs"].items()}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing input {name}: {path}")
    inventory = shared.validate_model_inventory_without_sha(paths["model"])

    delivery_receipt = load_json(paths["deliverability_receipt"])
    expected_delivery = "LM_HEAD_DELIVERABILITY_HOLD_LIFTED_SELECTION_AUTHORIZED"
    if delivery_receipt.get("status") != expected_delivery:
        raise AssertionError("upstream deliverability probe did not authorize selection")
    delivery_result = load_json(paths["deliverability_result"])
    if not delivery_result.get("hard_gates_pass"):
        raise AssertionError("upstream deliverability hard gates did not pass")

    backlog = load_json(paths["backlog_selection"])
    lm_head_rows = [
        row for row in backlog["backlog"] if row["item"].startswith("lm_head ")
    ]
    if len(lm_head_rows) != 1 or lm_head_rows[0]["disposition"] != "HOLD_CURRENT_CARRIER_DELIVERABILITY":
        raise AssertionError("inherited lm_head backlog HOLD is not intact")

    census = load_json(paths["shape_census"])
    lm_head = census["op_families"]["lm_head"]
    projected = sum(int(row["invocations"]) for row in lm_head["shapes"])
    if census.get("status") != "PASS" or projected != int(config["authenticity"]["projected_calls"]):
        raise AssertionError("lm_head census population drift")
    if int(lm_head["calls_per_output_token"]) != 1:
        raise AssertionError("lm_head call frequency drift")
    if float(lm_head["final_logit_softcap"]) != float(config["authenticity"]["softcap"]):
        raise AssertionError("lm_head softcap drift")

    tapes = shared.load_tapes_without_sha(paths["tapes_manifest"])
    archived = shared.load_qualified_baseline(paths["archived_baseline_metadata"])
    return {
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "preregistration": {"path": str(prereg_path), "status": prereg["status"]},
        "model_inventory": inventory,
        "backlog": lm_head_rows[0],
        "deliverability": {
            "result": str(paths["deliverability_result"]),
            "receipt": str(paths["deliverability_receipt"]),
            "status": delivery_receipt["status"],
            "live_peak_allocated_bytes": delivery_receipt["summary"][
                "live_peak_allocated_bytes"
            ],
        },
        "census": {
            "path": str(paths["shape_census"]),
            "status": census["status"],
            "projected_calls": projected,
            "shapes": lm_head["shapes"],
        },
        "paths": paths,
        "tapes": tapes,
        "archived": archived,
        "census_full": census,
    }


def cpu_reference(
    output_head: Any, hidden_cpu: torch.Tensor, softcap: float
) -> tuple[int, torch.Tensor]:
    with torch.inference_mode():
        logits = output_head(hidden_cpu)
        capped = torch.tanh(logits / softcap) * softcap
        token = int(capped[0, -1].argmax().item())
    return token, capped


def capture_real_states(
    model: Any,
    rendered: list[dict[str, Any]],
    seed: int,
    device: int,
    softcap: float,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    text_model = get_submodule(model, "model.language_model")
    output_head = get_submodule(model, "lm_head")
    token_embedding = text_model.embed_tokens
    per_layer_embedding = text_model.embed_tokens_per_layer
    num_layers = int(text_model.config.num_hidden_layers)
    per_layer_width = int(text_model.config.hidden_size_per_layer_input)
    cuda_device = torch.device(f"cuda:{device}")
    records: list[dict[str, Any]] = []
    states: list[torch.Tensor] = []
    for request in rendered:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prompt_ids = request["rendered_token_ids"]
        current_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
        attention_mask_cpu = torch.ones_like(current_ids, device="cpu")
        cache = None
        for phase in ("prefill", "first_decode"):
            inputs_embeds_cpu = token_embedding(current_ids)
            per_layer_inputs_cpu = per_layer_embedding(current_ids).reshape(
                *current_ids.shape, num_layers, per_layer_width
            )
            inputs_embeds = inputs_embeds_cpu.to(cuda_device)
            per_layer_inputs = per_layer_inputs_cpu.to(cuda_device)
            attention_mask = attention_mask_cpu.to(cuda_device)
            del inputs_embeds_cpu, per_layer_inputs_cpu
            with torch.inference_mode():
                output = text_model(
                    inputs_embeds=inputs_embeds,
                    per_layer_inputs=per_layer_inputs,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=False,
                )
            cache = output.past_key_values
            # The last-token view inherits prompt-dependent leading strides.
            # Size-one dimensions can still report contiguous with those strides,
            # so force a contiguous-format clone rather than calling contiguous().
            hidden_cpu = output.last_hidden_state[:, -1:, :].to("cpu").clone(
                memory_format=torch.contiguous_format
            )
            if not bool(torch.isfinite(hidden_cpu).all().item()):
                raise FloatingPointError(f"non-finite hidden state: {request['request_id']}/{phase}")
            token, capped = cpu_reference(output_head, hidden_cpu, softcap)
            if not bool(torch.isfinite(capped).all().item()):
                raise FloatingPointError(f"non-finite CPU logits: {request['request_id']}/{phase}")
            states.append(hidden_cpu)
            records.append(
                {
                    "request_id": request["request_id"],
                    "request_number": request["request_number"],
                    "phase": phase,
                    "rendered_prompt_tokens": request["rendered_prompt_tokens"],
                    "cache_sequence_length": int(cache.get_seq_length()),
                    "cpu_reference_token_id": token,
                }
            )
            del capped, output, inputs_embeds, per_layer_inputs, attention_mask
            if phase == "prefill":
                current_ids = torch.tensor([[token]], dtype=torch.long, device="cpu")
                attention_mask_cpu = torch.cat(
                    (attention_mask_cpu, torch.ones((1, 1), dtype=attention_mask_cpu.dtype)),
                    dim=1,
                )
        del cache, current_ids, attention_mask_cpu
        torch.cuda.empty_cache()
        log(f"lm_head selection: captured {request['request_id']} prefill+decode")
    if len(records) != 22 or len(states) != 22:
        raise AssertionError("real-state capture did not produce 22 states")
    return records, states


def eager_softcap_token(
    hidden: torch.Tensor, weight: torch.Tensor, softcap: float
) -> torch.Tensor:
    logits = F.linear(hidden, weight)
    return (torch.tanh(logits / softcap) * softcap).argmax(dim=-1)


def eager_raw_token(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden, weight).argmax(dim=-1)


def benchmark_cpu(
    output_head: Any,
    states: list[torch.Tensor],
    softcap: float,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    for index in range(warmup):
        cpu_reference(output_head, states[index % len(states)], softcap)
    samples: list[float] = []
    tokens: list[int] = []
    for index in range(repetitions):
        started = time.perf_counter_ns()
        token, capped = cpu_reference(output_head, states[index % len(states)], softcap)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        tokens.append(token)
        del capped
    return {
        "warmup": warmup,
        "repetitions": repetitions,
        "latency_ms": summarize(samples),
        "distinct_token_ids": len(set(tokens)),
    }


def benchmark_gpu(
    operations: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    states: list[torch.Tensor],
    weight: torch.Tensor,
    warmup: int,
    repetitions: int,
    monitor: NvidiaSmiMonitor,
    device: int,
) -> dict[str, Any]:
    names = list(operations)
    monitor.set_phase("lm_head_gpu_measurement")
    monitor.sample_now()
    with torch.inference_mode():
        for repetition in range(warmup):
            hidden = states[repetition % len(states)]
            rotated = names[repetition % len(names) :] + names[: repetition % len(names)]
            for name in rotated:
                operations[name](hidden, weight)
        torch.cuda.synchronize(device)
        samples: dict[str, list[float]] = {name: [] for name in names}
        for repetition in range(repetitions):
            hidden = states[repetition % len(states)]
            rotated = names[repetition % len(names) :] + names[: repetition % len(names)]
            pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
            for name in rotated:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                operations[name](hidden, weight)
                end.record()
                pending.append((name, start, end))
            torch.cuda.synchronize(device)
            for name, start, end in pending:
                samples[name].append(float(start.elapsed_time(end)))
    monitor.sample_now()
    return {name: {"latency_ms": summarize(values)} for name, values in samples.items()}


def run_selection(
    config: dict[str, Any],
    config_path: Path,
    prereg_path: Path,
    telemetry_path: Path,
    device: int,
) -> dict[str, Any]:
    started_at = now_iso()
    wall_started = time.perf_counter()
    validated = validate_inputs(config, prereg_path)
    paths = validated.pop("paths")
    tapes = validated.pop("tapes")
    archived = validated.pop("archived")
    census = validated.pop("census_full")
    monitor = NvidiaSmiMonitor(
        interval_seconds=float(config["measurement"]["telemetry_interval_seconds"])
    )
    monitor.start()
    model = None
    mirror = None
    try:
        bandwidth = run_same_day_bandwidth(config, monitor, device)
        monitor.set_phase("load_formal_carrier")
        model, tokenizer, carrier = shared.load_carrier_without_sha(config, device)
        structure_before = shared.model_structure_snapshot(model)
        output_head = get_submodule(model, "lm_head")
        token_embedding = get_submodule(model, "model.language_model.embed_tokens")
        if output_head.weight.data_ptr() != token_embedding.weight.data_ptr():
            raise AssertionError("lm_head is not tied to the token embedding")
        if output_head.weight.device.type != "cpu":
            raise AssertionError("lm_head is not CPU-resident")
        head_before = delivery.tensor_identity(output_head.weight)
        embedding_before = delivery.tensor_identity(token_embedding.weight)
        rendered = shared.render_requests(tokenizer, tapes, census, archived)
        monitor.set_phase("capture_22_real_hidden_states")
        captures, states_cpu = capture_real_states(
            model,
            rendered,
            int(config["measurement"]["seed"]),
            device,
            float(config["authenticity"]["softcap"]),
        )

        monitor.set_phase("mirror_and_compile")
        mirror = output_head.weight.detach().to(
            device=torch.device(f"cuda:{device}"), dtype=torch.bfloat16, copy=True
        ).contiguous()
        states_gpu = [
            state.to(device).clone(memory_format=torch.contiguous_format)
            for state in states_cpu
        ]
        softcap = float(config["authenticity"]["softcap"])

        def softcap_operation(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return eager_softcap_token(hidden, weight, softcap)

        def raw_operation(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return eager_raw_token(hidden, weight)

        compiled_softcap = torch.compile(
            softcap_operation, fullgraph=True, dynamic=False, mode="reduce-overhead"
        )
        compiled_raw = torch.compile(
            raw_operation, fullgraph=True, dynamic=False, mode="reduce-overhead"
        )
        compiled_softcap(states_gpu[0], mirror).clone()
        compiled_raw(states_gpu[0], mirror).clone()
        torch.cuda.synchronize(device)
        operations: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
            "cuda_eager_softcap_argmax": softcap_operation,
            "cuda_compile_softcap_argmax": compiled_softcap,
            "cuda_eager_raw_argmax": raw_operation,
            "cuda_compile_raw_argmax": compiled_raw,
        }

        correctness_rows: list[dict[str, Any]] = []
        exact_counts = {name: 0 for name in operations}
        logit_allclose_count = 0
        max_abs = 0.0
        with torch.inference_mode():
            for capture, hidden_cpu, hidden_gpu in zip(
                captures, states_cpu, states_gpu, strict=True
            ):
                reference_token = int(capture["cpu_reference_token_id"])
                tokens = {
                    name: int(operation(hidden_gpu, mirror).clone().item())
                    for name, operation in operations.items()
                }
                for name, token in tokens.items():
                    exact_counts[name] += int(token == reference_token)
                _, cpu_capped = cpu_reference(output_head, hidden_cpu, softcap)
                gpu_logits = F.linear(hidden_gpu, mirror)
                gpu_capped = (torch.tanh(gpu_logits / softcap) * softcap).to("cpu")
                cpu_float = cpu_capped.float()
                gpu_float = gpu_capped.float()
                row_max_abs = float((cpu_float - gpu_float).abs().max().item())
                row_allclose = bool(
                    torch.allclose(
                        cpu_float,
                        gpu_float,
                        atol=float(config["correctness"]["logit_diagnostic_tolerance"]["atol"]),
                        rtol=float(config["correctness"]["logit_diagnostic_tolerance"]["rtol"]),
                    )
                )
                max_abs = max(max_abs, row_max_abs)
                logit_allclose_count += int(row_allclose)
                correctness_rows.append(
                    {
                        **capture,
                        "tokens": {"cpu_reference": reference_token, **tokens},
                        "exact_vs_cpu": {
                            name: token == reference_token for name, token in tokens.items()
                        },
                        "cuda_eager_softcapped_logits": {
                            "max_abs_vs_cpu": row_max_abs,
                            "allclose": row_allclose,
                        },
                    }
                )
                del cpu_capped, gpu_logits, gpu_capped, cpu_float, gpu_float
        eligible = {
            "cuda_eager_softcap_argmax": exact_counts[
                "cuda_eager_softcap_argmax"
            ]
            == 22,
            "cuda_compile_softcap_argmax": exact_counts[
                "cuda_compile_softcap_argmax"
            ]
            == 22,
        }

        monitor.set_phase("lm_head_cpu_measurement")
        cpu_performance = benchmark_cpu(
            output_head,
            states_cpu,
            softcap,
            int(config["measurement"]["cpu_warmup"]),
            int(config["measurement"]["cpu_repetitions"]),
        )
        gpu_performance = benchmark_gpu(
            operations,
            states_gpu,
            mirror,
            int(config["measurement"]["gpu_warmup_per_implementation"]),
            int(config["measurement"]["gpu_repetitions_per_implementation"]),
            monitor,
            device,
        )
        eligible_names = [name for name, value in eligible.items() if value]
        if not eligible_names:
            best_name = None
            best_ms = None
            best_gbps = None
            relative_score = None
            native_speedup = None
        else:
            best_name = min(
                eligible_names,
                key=lambda name: float(gpu_performance[name]["latency_ms"]["p50"]),
            )
            best_ms = float(gpu_performance[best_name]["latency_ms"]["p50"])
            byte_count = int(config["analytic_bytes"]["bytes_per_call"])
            best_gbps = byte_count / (best_ms * 1_000_000.0)
            relative_score = best_gbps / float(bandwidth["ceiling_gbps_decimal"])
            native_speedup = float(cpu_performance["latency_ms"]["p50"]) / best_ms

        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        gpu_budget = int(float(config["carrier"]["gpu_budget_gib"]) * 1024**3)
        head_during = delivery.tensor_identity(output_head.weight)
        embedding_during = delivery.tensor_identity(token_embedding.weight)
        mirror_identity = delivery.tensor_identity(mirror)
        memory_during_measurement = memory_snapshot(torch, device)
        del states_gpu
        del mirror
        mirror = None
        del compiled_softcap, compiled_raw, operations
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
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
    finally:
        if mirror is not None:
            del mirror
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        monitor.set_phase("selection_cleanup")
        monitor.stop()

    telemetry = shared.telemetry_report(monitor, telemetry_path)
    measurement_telemetry = monitor.summary_for_phase("lm_head_gpu_measurement")
    memory_clock = measurement_telemetry["numeric"]["memory_mhz"]
    memory_clock_p50 = float(memory_clock["p50"]) if memory_clock else 0.0
    memory_clock_ratio = memory_clock_p50 / float(
        config["measurement"]["op0_memory_clock_mhz_p50"]
    )
    correctness_pass = all(eligible.values())
    bottleneck_threshold = float(
        config["selection_gate"]["bottleneck_max_relative_score"]
    )
    bottleneck_pass = bool(relative_score is not None and relative_score <= bottleneck_threshold)
    validity_gates = {
        "owner_no_sha_direction_honored": "PASS",
        "upstream_deliverability": "PASS",
        "authenticity": "PASS"
        if validated["census"]["projected_calls"] == 21705
        else "FAIL",
        "formal_c02_nf4_carrier_343_modules": "PASS"
        if carrier["quantization"]["active_language_linear4bit_count"] == 343
        else "FAIL",
        "all_22_real_states_captured": "PASS"
        if len(captures) == 22
        else "FAIL",
        "softcap_cuda_baselines_exact_22_of_22": "PASS"
        if correctness_pass
        else "FAIL",
        "same_day_bandwidth": "PASS"
        if bandwidth["correctness"]["pass"]
        else "FAIL",
        "gpu_memory_below_7_5_gib": "PASS"
        if peak_allocated < gpu_budget
        else "FAIL",
        "original_tied_cpu_weight_and_model_structure_unchanged": "PASS"
        if original_identity_unchanged and structure["pass"]
        else "FAIL",
        "telemetry_present": "PASS"
        if telemetry["raw_jsonl"]["samples"] > 0
        else "FAIL",
        "memory_clock_gate": "PASS"
        if memory_clock_ratio
        >= float(config["measurement"]["minimum_memory_clock_ratio_vs_op0"])
        else "FAIL",
        "thermal_hardware_throttle_absent": "PASS"
        if not measurement_telemetry["thermal_or_hardware_throttle"]
        else "FAIL",
    }
    probe_valid = all(value == "PASS" for value in validity_gates.values())
    native_eligible = bool(
        probe_valid
        and best_name is not None
        and native_speedup is not None
        and native_speedup >= 1.50
    )
    if not probe_valid:
        status = config["decision"]["invalid"]
    elif native_eligible:
        status = config["decision"]["existing_path"]
    elif bottleneck_pass:
        status = config["decision"]["custom_path"]
    else:
        status = config["decision"]["negative"]
    return {
        "schema_version": 1,
        "stage": config["stage"],
        "target": config["target"],
        "status": status,
        "probe_valid": probe_valid,
        "custom_candidate_implemented_or_timed": False,
        "started_at": started_at,
        "finished_at": now_iso(),
        "wall_seconds": time.perf_counter() - wall_started,
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "environment": shared.environment_identity(),
        "protocol": {"path": str(config_path), "preregistration": str(prereg_path)},
        "validated_inputs": validated,
        "carrier": carrier,
        "same_day_bandwidth": bandwidth,
        "real_state_capture": {
            "count": len(captures),
            "records": captures,
        },
        "correctness": {
            "reference": config["correctness"]["reference"],
            "exact_counts_of_22": exact_counts,
            "eligible_softcap_baselines": eligible,
            "raw_argmax_is_diagnostic_only": True,
            "cuda_eager_softcapped_logit_allclose_count": logit_allclose_count,
            "cuda_eager_softcapped_logit_max_abs": max_abs,
            "rows": correctness_rows,
        },
        "performance": {
            "cpu": cpu_performance,
            "gpu": gpu_performance,
            "best_correctness_eligible_cuda_baseline": best_name,
            "best_p50_ms": best_ms,
            "analytic_bytes_per_call": int(config["analytic_bytes"]["bytes_per_call"]),
            "best_measured_gbps_decimal": best_gbps,
            "best_tier_a_relative_score": relative_score,
            "bottleneck_threshold_max_relative_score": bottleneck_threshold,
            "bottleneck_gate": "PASS" if bottleneck_pass else "FAIL",
            "best_cuda_vs_cpu_p50_speedup": native_speedup,
        },
        "memory": {
            "gpu_budget_bytes": gpu_budget,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "mirror": mirror_identity,
            "during_measurement": memory_during_measurement,
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
        "measurement_telemetry": measurement_telemetry,
        "memory_clock_ratio_vs_op0": memory_clock_ratio,
        "validity_gates": validity_gates,
        "bottleneck_gate": "PASS" if bottleneck_pass else "FAIL",
        "existing_path_eligible": native_eligible,
        "decision": status,
        "non_authorizations": config["decision"]["non_authorizations"],
    }


def build_receipt(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    performance = result["performance"]
    return {
        "schema_version": 1,
        "stage": result["stage"],
        "target": result["target"],
        "timestamp": result["finished_at"],
        "status": result["status"],
        "probe_valid": result["probe_valid"],
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "custom_candidate_implemented_or_timed": False,
        "result": str(result_path),
        "validity_gates": result["validity_gates"],
        "bottleneck_gate": result["bottleneck_gate"],
        "existing_path_eligible": result["existing_path_eligible"],
        "summary": {
            "real_states": result["real_state_capture"]["count"],
            "softcap_eager_exact": result["correctness"]["exact_counts_of_22"][
                "cuda_eager_softcap_argmax"
            ],
            "softcap_compile_exact": result["correctness"]["exact_counts_of_22"][
                "cuda_compile_softcap_argmax"
            ],
            "best_cuda_baseline": performance["best_correctness_eligible_cuda_baseline"],
            "cpu_p50_ms": performance["cpu"]["latency_ms"]["p50"],
            "best_cuda_p50_ms": performance["best_p50_ms"],
            "best_cuda_vs_cpu_speedup": performance["best_cuda_vs_cpu_p50_speedup"],
            "same_day_ceiling_gbps": result["same_day_bandwidth"][
                "ceiling_gbps_decimal"
            ],
            "best_tier_a_relative_score": performance["best_tier_a_relative_score"],
            "peak_cuda_allocated_bytes": result["memory"]["peak_allocated_bytes"],
            "memory_clock_ratio_vs_op0": result["memory_clock_ratio_vs_op0"],
            "thermal_or_hardware_throttle": result["measurement_telemetry"][
                "thermal_or_hardware_throttle"
            ],
        },
        "next_action": (
            "Freeze a same-load full C02 A/B reconnection protocol for the existing CUDA mirror path; do not implement a custom lm_head kernel."
            if result["existing_path_eligible"]
            else "Follow the selected custom/negative/invalid branch in the full result."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = absolute(args.config)
    prereg_path = absolute(args.prereg)
    result_path = absolute(args.result)
    receipt_path = absolute(args.receipt)
    telemetry_path = absolute(args.telemetry)
    config = load_json(config_path)
    try:
        result = run_selection(
            config, config_path, prereg_path, telemetry_path, args.device
        )
        atomic_write_json(result_path, result)
        atomic_write_json(receipt_path, build_receipt(result, result_path))
        log(f"lm_head selection finished: {result['status']}")
        return 0 if result["probe_valid"] else 2
    except BaseException as exc:
        failed_at = now_iso()
        failure_path = (
            ROOT
            / "results/operator/failures"
            / (
                "op5_lm_head_selection_r1_"
                + failed_at.replace(":", "").replace("+", "_")
                + ".json"
            )
        )
        atomic_write_json(
            failure_path,
            {
                "schema_version": 1,
                "stage": "OP5_LM_HEAD_SELECTION_R1",
                "status": "EXECUTION_FAILURE",
                "timestamp": failed_at,
                "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
                "custom_candidate_implemented_or_timed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        log(f"lm_head selection failed: {failure_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
