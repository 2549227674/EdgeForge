#!/usr/bin/env python3
"""Run the bounded post-K3 lm_head carrier-deliverability probe."""

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
from load_smoke_nf4 import get_submodule, memory_snapshot
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, summarize


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/operator/op5_lm_head_deliverability_probe_r1.json"
DEFAULT_PREREG = (
    ROOT
    / "results/operator/pool/op5_lm_head_deliverability_probe_r1_prereg_receipt_20260916.json"
)
DEFAULT_RESULT = (
    ROOT / "results/operator/pool/op5_lm_head_deliverability_probe_r1_20260916.json"
)
DEFAULT_RECEIPT = (
    ROOT
    / "results/operator/pool/op5_lm_head_deliverability_probe_r1_receipt_20260916.json"
)
DEFAULT_TELEMETRY = (
    ROOT
    / "results/operator/pool/op5_lm_head_deliverability_probe_r1_telemetry_20260916.jsonl"
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
    if config.get("stage") != "OP5_LM_HEAD_DELIVERABILITY_PROBE_R1":
        raise AssertionError("unexpected probe identity")
    if config.get("authority", {}).get("sha_verification") != "SKIPPED_BY_OWNER_DIRECTION":
        raise AssertionError("probe must retain the owner no-SHA direction")
    if config.get("no_sha_policy", {}).get("calculate_or_compare_sha") is not False:
        raise AssertionError("probe no-SHA policy is not explicit")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise AssertionError("probe requires the frozen sm_89 CUDA device")

    prereg = load_json(prereg_path)
    if prereg.get("status") != "FROZEN_BEFORE_FORMAL_PROBE":
        raise AssertionError("probe preregistration is not frozen")
    if prereg.get("formal_probe_started") is not False:
        raise AssertionError("preregistration was not captured before the probe")

    paths = {name: absolute(path) for name, path in config["inputs"].items()}
    for name, path in paths.items():
        if name == "runner":
            continue
        if not path.exists():
            raise FileNotFoundError(f"missing required input {name}: {path}")

    inventory = shared.validate_model_inventory_without_sha(paths["model"])
    backlog = load_json(paths["backlog_selection"])
    lm_head_rows = [
        row for row in backlog["backlog"] if row["item"].startswith("lm_head ")
    ]
    if len(lm_head_rows) != 1:
        raise AssertionError("backlog has no unique lm_head entry")
    if lm_head_rows[0]["disposition"] != "HOLD_CURRENT_CARRIER_DELIVERABILITY":
        raise AssertionError("lm_head is not at the expected deliverability HOLD")

    census = load_json(paths["shape_census"])
    if census.get("status") != "PASS":
        raise AssertionError("shape census is not PASS")
    lm_head = census["op_families"]["lm_head"]
    projected_calls = sum(int(row["invocations"]) for row in lm_head["shapes"])
    if projected_calls != int(config["population"]["projected_calls"]):
        raise AssertionError("lm_head projected-call population drift")
    if float(lm_head["final_logit_softcap"]) != float(config["population"]["softcap"]):
        raise AssertionError("lm_head softcap drift")

    k3_result = load_json(paths["k3_r5_result"])
    if k3_result.get("status") != "K3_C02_RECONNECTION_FAIL_CLOSE_DEPLOYMENT_PATH":
        raise AssertionError("upstream K3 result is not terminal fail-close")
    k3_adjudication = load_json(paths["k3_r5_adjudication"])
    if not k3_adjudication.get("decision_unchanged"):
        raise AssertionError("K3 post-run adjudication is incomplete")

    tapes = shared.load_tapes_without_sha(paths["tapes_manifest"])
    archived = shared.load_qualified_baseline(paths["archived_baseline_metadata"])
    return {
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "preregistration": {"path": str(prereg_path), "status": prereg["status"]},
        "model_inventory": inventory,
        "backlog": lm_head_rows[0],
        "census": {
            "path": str(paths["shape_census"]),
            "status": census["status"],
            "projected_calls": projected_calls,
            "final_logit_softcap": lm_head["final_logit_softcap"],
            "shapes": lm_head["shapes"],
        },
        "k3_r5": {
            "result": str(paths["k3_r5_result"]),
            "status": k3_result["status"],
            "peak_cuda_allocated_bytes": int(k3_result["peak_cuda_allocated_bytes"]),
            "adjudication": str(paths["k3_r5_adjudication"]),
        },
        "paths": paths,
        "tapes": tapes,
        "archived": archived,
        "census_full": census,
    }


def tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "data_ptr": tensor.data_ptr(),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def capture_longest_prompt_hidden(
    model: Any,
    tokenizer: Any,
    rendered: list[dict[str, Any]],
    seed: int,
    device: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, Any]:
    request = max(rendered, key=lambda row: int(row["rendered_prompt_tokens"]))
    text_model = get_submodule(model, "model.language_model")
    token_embedding = text_model.embed_tokens
    per_layer_embedding = text_model.embed_tokens_per_layer
    num_layers = int(text_model.config.num_hidden_layers)
    per_layer_width = int(text_model.config.hidden_size_per_layer_input)
    prompt_ids = request["rendered_token_ids"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
    attention_mask_cpu = torch.ones_like(input_ids, device="cpu")
    inputs_embeds_cpu = token_embedding(input_ids)
    per_layer_inputs_cpu = per_layer_embedding(input_ids).reshape(
        *input_ids.shape, num_layers, per_layer_width
    )
    cuda_device = torch.device(f"cuda:{device}")
    inputs_embeds = inputs_embeds_cpu.to(cuda_device)
    per_layer_inputs = per_layer_inputs_cpu.to(cuda_device)
    attention_mask = attention_mask_cpu.to(cuda_device)
    del inputs_embeds_cpu, per_layer_inputs_cpu
    with torch.inference_mode():
        output = text_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
        )
    torch.cuda.synchronize(device)
    hidden_gpu = output.last_hidden_state[:, -1:, :].detach()
    hidden_cpu = hidden_gpu.to("cpu")
    cache = output.past_key_values
    cache_length = int(cache.get_seq_length())
    if cache_length != len(prompt_ids):
        raise AssertionError(f"prefill cache length {cache_length} != {len(prompt_ids)}")
    if not bool(torch.isfinite(hidden_gpu).all().item()):
        raise FloatingPointError("captured real hidden state is non-finite")
    record = {
        "request_id": request["request_id"],
        "request_number": request["request_number"],
        "rendered_prompt_tokens": request["rendered_prompt_tokens"],
        "all_rendered_prompt_lengths": sorted(
            int(row["rendered_prompt_tokens"]) for row in rendered
        ),
        "selection": "maximum rendered_prompt_tokens across all 11 frozen requests",
        "hidden": tensor_identity(hidden_gpu),
        "cache_sequence_length": cache_length,
    }
    del output, inputs_embeds, per_layer_inputs, attention_mask
    del input_ids, attention_mask_cpu
    return record, hidden_cpu, hidden_gpu, cache


def cpu_path(
    output_head: Any, hidden_cpu: torch.Tensor, softcap: float
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    with torch.inference_mode():
        raw = output_head(hidden_cpu)
        capped = torch.tanh(raw / softcap) * softcap
        raw_token = int(raw[0, -1].argmax().item())
        capped_token = int(capped[0, -1].argmax().item())
    return raw, capped, raw_token, capped_token


def gpu_path(
    weight: torch.Tensor, hidden_gpu: torch.Tensor, softcap: float
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    with torch.inference_mode():
        raw = F.linear(hidden_gpu, weight)
        capped = torch.tanh(raw / softcap) * softcap
        raw_token = int(raw[0, -1].argmax().item())
        capped_token = int(capped[0, -1].argmax().item())
    return raw, capped, raw_token, capped_token


def benchmark(
    function: Callable[[], int], warmup: int, repetitions: int, device: int | None
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    if device is not None:
        torch.cuda.synchronize(device)
    samples_ms: list[float] = []
    tokens: list[int] = []
    for _ in range(repetitions):
        if device is not None:
            torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        tokens.append(function())
        if device is not None:
            torch.cuda.synchronize(device)
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    if len(set(tokens)) != 1:
        raise AssertionError("benchmark path produced unstable greedy tokens")
    return {
        "warmup": warmup,
        "repetitions": repetitions,
        "token_id": tokens[0],
        "latency_ms": summarize(samples_ms),
    }


def run_probe(
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
    monitor.set_phase("load_formal_carrier")
    monitor.start()
    model = None
    mirror = None
    cache = None
    hidden_gpu = None
    hidden_cpu = None
    try:
        model, tokenizer, carrier = shared.load_carrier_without_sha(config, device)
        structure_before = shared.model_structure_snapshot(model)
        output_head = get_submodule(model, "lm_head")
        token_embedding = get_submodule(model, "model.language_model.embed_tokens")
        if output_head.weight.data_ptr() != token_embedding.weight.data_ptr():
            raise AssertionError("lm_head is not tied to the CPU token embedding")
        if output_head.weight.device.type != "cpu":
            raise AssertionError("formal lm_head is not CPU-resident")
        head_before = tensor_identity(output_head.weight)
        embedding_before = tensor_identity(token_embedding.weight)

        rendered = shared.render_requests(tokenizer, tapes, census, archived)
        monitor.set_phase("longest_prompt_prefill")
        capture, hidden_cpu, hidden_gpu, cache = capture_longest_prompt_hidden(
            model,
            tokenizer,
            rendered,
            int(config["measurement"]["seed"]),
            device,
        )
        softcap = float(config["population"]["softcap"])
        cpu_raw, cpu_capped, cpu_raw_token, cpu_capped_token = cpu_path(
            output_head, hidden_cpu, softcap
        )
        cpu_finite = bool(torch.isfinite(cpu_capped).all().item())
        top2 = torch.topk(cpu_raw[0, -1].float(), k=2)
        cpu_top2_margin = float((top2.values[0] - top2.values[1]).item())

        mirror_bytes = int(output_head.weight.numel() * output_head.weight.element_size())
        gpu_budget_bytes = int(float(config["carrier"]["gpu_budget_gib"]) * 1024**3)
        upstream_peak = int(validated["k3_r5"]["peak_cuda_allocated_bytes"])
        projected_peak = upstream_peak + mirror_bytes
        before_mirror = memory_snapshot(torch, device)
        torch.cuda.reset_peak_memory_stats(device)
        monitor.set_phase("mirror_allocation_and_correctness")
        mirror = output_head.weight.detach().to(
            device=torch.device(f"cuda:{device}"),
            dtype=torch.bfloat16,
            copy=True,
        ).contiguous()
        torch.cuda.synchronize(device)
        after_mirror = memory_snapshot(torch, device)
        gpu_raw, gpu_capped, gpu_raw_token, gpu_capped_token = gpu_path(
            mirror, hidden_gpu, softcap
        )
        torch.cuda.synchronize(device)
        with torch.inference_mode():
            cpu_capped_float = cpu_capped.float()
            gpu_capped_cpu = gpu_capped.to("cpu").float()
            max_abs = float((cpu_capped_float - gpu_capped_cpu).abs().max().item())
            mean_abs = float((cpu_capped_float - gpu_capped_cpu).abs().mean().item())
            allclose = bool(
                torch.allclose(
                    cpu_capped_float,
                    gpu_capped_cpu,
                    atol=0.5,
                    rtol=0.02,
                )
            )
        live_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        live_peak_reserved = int(torch.cuda.max_memory_reserved(device))

        warmup = int(config["measurement"]["warmup"])
        repetitions = int(config["measurement"]["repetitions"])

        def run_cpu() -> int:
            raw, capped, _, token = cpu_path(output_head, hidden_cpu, softcap)
            del raw, capped
            return token

        def run_gpu() -> int:
            raw, capped, _, token = gpu_path(mirror, hidden_gpu, softcap)
            del raw, capped
            return token

        monitor.set_phase("cpu_lm_head_diagnostic")
        cpu_benchmark = benchmark(run_cpu, warmup, repetitions, None)
        monitor.set_phase("cuda_mirror_diagnostic")
        gpu_benchmark = benchmark(run_gpu, warmup, repetitions, device)
        speedup = (
            float(cpu_benchmark["latency_ms"]["p50"])
            / float(gpu_benchmark["latency_ms"]["p50"])
        )

        head_during = tensor_identity(output_head.weight)
        embedding_during = tensor_identity(token_embedding.weight)
        mirror_identity = tensor_identity(mirror)
        del gpu_raw, gpu_capped, gpu_capped_cpu
        del cpu_raw, cpu_capped, cpu_capped_float
        del mirror
        mirror = None
        del cache
        cache = None
        del hidden_gpu, hidden_cpu
        hidden_gpu = None
        hidden_cpu = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        after_cleanup = memory_snapshot(torch, device)
        structure_after = shared.model_structure_snapshot(model)
        structure = shared.compare_model_structure(structure_before, structure_after)
        head_after = tensor_identity(output_head.weight)
        embedding_after = tensor_identity(token_embedding.weight)
        tied_after = output_head.weight.data_ptr() == token_embedding.weight.data_ptr()
        original_identity_unchanged = (
            head_before == head_during == head_after
            and embedding_before == embedding_during == embedding_after
            and tied_after
        )
    finally:
        if mirror is not None:
            del mirror
        if cache is not None:
            del cache
        if hidden_gpu is not None:
            del hidden_gpu
        if hidden_cpu is not None:
            del hidden_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        monitor.set_phase("probe_cleanup")
        monitor.stop()

    telemetry = shared.telemetry_report(monitor, telemetry_path)
    telemetry_summary = telemetry["summary"]
    gates = {
        "owner_no_sha_direction_honored": "PASS",
        "formal_c02_nf4_carrier_343_modules": "PASS"
        if carrier["quantization"]["active_language_linear4bit_count"] == 343
        else "FAIL",
        "longest_of_11_frozen_prompts_real_hidden_finite": "PASS"
        if len(capture["all_rendered_prompt_lengths"]) == 11
        and capture["rendered_prompt_tokens"]
        == max(capture["all_rendered_prompt_lengths"])
        and cpu_finite
        else "FAIL",
        "static_r5_peak_plus_exact_mirror_below_7_5_gib": "PASS"
        if projected_peak < gpu_budget_bytes
        else "FAIL",
        "live_longest_prompt_cache_plus_mirror_below_7_5_gib": "PASS"
        if live_peak_allocated < gpu_budget_bytes
        else "FAIL",
        "same_load_cpu_cuda_softcapped_greedy_token_exact": "PASS"
        if cpu_capped_token == gpu_capped_token
        else "FAIL",
        "softcap_elision_argmax_exact_cpu_and_cuda": "PASS"
        if cpu_raw_token == cpu_capped_token and gpu_raw_token == gpu_capped_token
        else "FAIL",
        "original_tied_cpu_weight_and_model_structure_unchanged": "PASS"
        if original_identity_unchanged and structure["pass"]
        else "FAIL",
        "telemetry_present": "PASS"
        if int(telemetry["raw_jsonl"]["samples"]) > 0
        else "FAIL",
        "thermal_hardware_throttle_absent": "PASS"
        if not telemetry_summary["thermal_or_hardware_throttle"]
        else "FAIL",
    }
    hard_pass = all(value == "PASS" for value in gates.values())
    status = config["decision"]["on_pass"] if hard_pass else config["decision"]["on_fail"]
    return {
        "schema_version": 1,
        "stage": config["stage"],
        "target": config["target"],
        "status": status,
        "hard_gates_pass": hard_pass,
        "started_at": started_at,
        "finished_at": now_iso(),
        "wall_seconds": time.perf_counter() - wall_started,
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "environment": shared.environment_identity(),
        "protocol": {"path": str(config_path), "preregistration": str(prereg_path)},
        "validated_inputs": validated,
        "carrier": carrier,
        "capture": capture,
        "mirror": {
            "identity": mirror_identity,
            "bytes": mirror_bytes,
            "unregistered_tensor": True,
            "original_parameter_moved": False,
        },
        "correctness": {
            "cpu_raw_token_id": cpu_raw_token,
            "cpu_softcapped_token_id": cpu_capped_token,
            "cuda_raw_token_id": gpu_raw_token,
            "cuda_softcapped_token_id": gpu_capped_token,
            "cpu_top2_raw_logit_margin": cpu_top2_margin,
            "softcapped_logits_max_abs_cpu_vs_cuda": max_abs,
            "softcapped_logits_mean_abs_cpu_vs_cuda": mean_abs,
            "softcapped_logits_allclose_atol_0_5_rtol_0_02": allclose,
            "token_exact": cpu_capped_token == gpu_capped_token,
            "softcap_elision_exact": cpu_raw_token == cpu_capped_token
            and gpu_raw_token == gpu_capped_token,
        },
        "memory": {
            "gpu_budget_bytes": gpu_budget_bytes,
            "upstream_r5_peak_allocated_bytes": upstream_peak,
            "projected_peak_plus_mirror_bytes": projected_peak,
            "projected_margin_bytes": gpu_budget_bytes - projected_peak,
            "before_mirror": before_mirror,
            "after_mirror": after_mirror,
            "live_peak_allocated_bytes": live_peak_allocated,
            "live_peak_reserved_bytes": live_peak_reserved,
            "live_margin_bytes": gpu_budget_bytes - live_peak_allocated,
            "after_cleanup": after_cleanup,
        },
        "diagnostic_performance": {
            "hard_gate": False,
            "cpu_tied_lm_head_softcap_argmax": cpu_benchmark,
            "cuda_bf16_mirror_linear_softcap_argmax": gpu_benchmark,
            "cuda_vs_cpu_p50_speedup": speedup,
        },
        "structure": {
            "original_head_before": head_before,
            "original_head_during_mirror": head_during,
            "original_head_after_cleanup": head_after,
            "embedding_before": embedding_before,
            "embedding_during_mirror": embedding_during,
            "embedding_after_cleanup": embedding_after,
            "tied_after_cleanup": tied_after,
            "original_identity_unchanged": original_identity_unchanged,
            "model_structure": structure,
        },
        "telemetry": telemetry,
        "gates": gates,
        "decision_boundary": config["decision"]["pass_boundary"],
        "non_authorizations": config["decision"]["non_authorizations"],
    }


def build_receipt(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": result["stage"],
        "target": result["target"],
        "timestamp": result["finished_at"],
        "status": result["status"],
        "hard_gates_pass": result["hard_gates_pass"],
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "result": str(result_path),
        "gates": result["gates"],
        "summary": {
            "request_id": result["capture"]["request_id"],
            "rendered_prompt_tokens": result["capture"]["rendered_prompt_tokens"],
            "mirror_bytes": result["mirror"]["bytes"],
            "projected_peak_plus_mirror_bytes": result["memory"][
                "projected_peak_plus_mirror_bytes"
            ],
            "live_peak_allocated_bytes": result["memory"]["live_peak_allocated_bytes"],
            "gpu_budget_bytes": result["memory"]["gpu_budget_bytes"],
            "token_exact": result["correctness"]["token_exact"],
            "softcap_elision_exact": result["correctness"]["softcap_elision_exact"],
            "cpu_p50_ms": result["diagnostic_performance"][
                "cpu_tied_lm_head_softcap_argmax"
            ]["latency_ms"]["p50"],
            "cuda_p50_ms": result["diagnostic_performance"][
                "cuda_bf16_mirror_linear_softcap_argmax"
            ]["latency_ms"]["p50"],
            "cuda_vs_cpu_p50_speedup": result["diagnostic_performance"][
                "cuda_vs_cpu_p50_speedup"
            ],
            "telemetry_samples": result["telemetry"]["raw_jsonl"]["samples"],
            "thermal_or_hardware_throttle": result["telemetry"]["summary"][
                "thermal_or_hardware_throttle"
            ],
        },
        "authorization_boundary": result["decision_boundary"],
        "next_action": (
            "Freeze and run a separate lm_head A/B/C/D selection probe."
            if result["hard_gates_pass"]
            else "Close the current OP5 backlog with no remaining executable kernel on this carrier."
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
        result = run_probe(config, config_path, prereg_path, telemetry_path, args.device)
        atomic_write_json(result_path, result)
        atomic_write_json(receipt_path, build_receipt(result, result_path))
        log(f"lm_head deliverability probe finished: {result['status']}")
        return 0 if result["hard_gates_pass"] else 2
    except BaseException as exc:
        failed_at = now_iso()
        failure_path = (
            ROOT
            / "results/operator/failures"
            / (
                "op5_lm_head_deliverability_probe_r1_"
                + failed_at.replace(":", "").replace("+", "_")
                + ".json"
            )
        )
        atomic_write_json(
            failure_path,
            {
                "schema_version": 1,
                "stage": "OP5_LM_HEAD_DELIVERABILITY_PROBE_R1",
                "status": "EXECUTION_FAILURE",
                "timestamp": failed_at,
                "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        log(f"lm_head deliverability probe failed: {failure_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
