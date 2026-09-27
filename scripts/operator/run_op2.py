#!/usr/bin/env python3
"""Execute the frozen OP2 K1 Gemma4 RMSNorm microbenchmark protocol."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

from safetensors import safe_open
import torch
import triton

from k1_rmsnorm import (
    SCALE_FREE_SEMANTICS,
    SCALED_SEMANTICS,
    candidate_rmsnorm,
    eager_scale_free,
    eager_scaled,
    make_compiled_baselines,
    reference_fp32,
)
from k0_harness import cuda_event_measure
from op0_common import (
    NvidiaSmiMonitor,
    atomic_write_json,
    now_iso,
    sha256,
    summarize,
    text_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/operator/k1_rmsnorm_prereg.json"
DEFAULT_RESULT = ROOT / "results/operator/k1/k1_rmsnorm_op2.json"
DEFAULT_RECEIPT = ROOT / "results/operator/k1/op2_receipt.json"
MODEL_MANIFEST_NAME = "artifact_manifest.json"
MODEL_INDEX_NAME = "model.safetensors.index.json"


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _assert_close(value: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(value, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"value {value} does not match frozen value {expected}")


def verify_frozen_inputs(
    config: dict[str, Any], prereg_path: Path, prereg_sha: str
) -> dict[str, Any]:
    if config.get("stage") != "OP2" or config.get("kernel") != "k1_gemma4_rmsnorm":
        raise AssertionError("unexpected K1 pre-registration identity")
    if config.get("frozen_before_any_k1_measurement") is not True:
        raise AssertionError("K1 pre-registration is not marked frozen")

    verified: dict[str, Any] = {
        "preregistration": {
            "path": str(prereg_path),
            "sha256": prereg_sha,
            "frozen_at": config["frozen_at"],
        },
        "sources": {},
    }
    for name, identity in config["source_identity"].items():
        if name == "git_sha":
            continue
        path = _absolute(identity["path"])
        observed = sha256(path)
        if observed != identity["sha256"]:
            raise AssertionError(
                f"frozen source hash mismatch for {name}: {observed} != {identity['sha256']}"
            )
        verified["sources"][name] = {
            "path": str(path),
            "sha256": observed,
            "status": "PASS",
        }

    current_git_sha = _git_sha()
    if current_git_sha != config["source_identity"]["git_sha"]:
        raise AssertionError("Git HEAD differs from the frozen OP2 source identity")
    verified["git_sha"] = current_git_sha

    op0 = json.loads(
        _absolute(config["source_identity"]["op0_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    op1 = json.loads(
        _absolute(config["source_identity"]["op1_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    census = json.loads(
        _absolute(config["source_identity"]["shape_census"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    bandwidth = json.loads(
        _absolute(config["source_identity"]["bandwidth_ceiling"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    env_identity = json.loads(
        _absolute(config["source_identity"]["environment_identity"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if op0.get("status") != "PASS" or op1.get("status") != "PASS":
        raise AssertionError("OP0 and OP1 must both be PASS before OP2")
    if census.get("status") != "PASS" or bandwidth["receipt"]["status"] != "PASS":
        raise AssertionError("shape census and bandwidth ceiling must be valid")
    observed_ceiling = float(
        bandwidth["ceiling"]["p50_across_stable_sizes_gbps_decimal"]
    )
    _assert_close(observed_ceiling, float(config["analytic_bytes"]["ceiling_gbps_decimal"]))
    rmsnorm = census["op_families"]["rmsnorm"]
    if rmsnorm["op_semantics_sha"] != config["semantics"]["census_op_semantics_sha256"]:
        raise AssertionError("RMSNorm census semantics hash mismatch")
    semantics = rmsnorm["implementation_semantics"]
    if "no implicit +1" not in semantics["with_scale"]:
        raise AssertionError("census does not prove direct-scale semantics")
    correction = census["op2_preregistration_inputs"]["mandatory_semantics_correction"]
    if not correction["source_uses_direct_weight_scale"] or correction[
        "source_uses_one_plus_weight"
    ]:
        raise AssertionError("OP1 semantic correction is not closed")

    device = env_identity["cuda"]["device"]
    if device["compute_capability"] != "sm_89":
        raise AssertionError("frozen environment is not sm_89")
    if not torch.cuda.is_available():
        raise RuntimeError("OP2 requires CUDA")
    if torch.cuda.get_device_capability(0) != (8, 9):
        raise AssertionError("live CUDA device is not sm_89")

    p0_nf4_path = _absolute(census["source_identity"]["p0_nf4_receipt"])
    if sha256(p0_nf4_path) != census["source_identity"]["p0_nf4_receipt_sha256"]:
        raise AssertionError("P0 NF4 receipt hash drift")
    p0_nf4 = json.loads(p0_nf4_path.read_text(encoding="utf-8"))
    if p0_nf4.get("status") != "PASS" or not p0_nf4.get(
        "formal_evidence_eligible", False
    ):
        raise AssertionError("formal C02 NF4 carrier receipt is not eligible")

    verified["upstream_states"] = {
        "op0": op0["status"],
        "op1": op1["status"],
        "shape_census": census["status"],
        "bandwidth": bandwidth["receipt"]["status"],
        "p0_nf4": p0_nf4["status"],
        "ncu_op4_soft_gate": op0["gates"]["ncu_op4_soft_gate"],
    }
    verified["live_device"] = {
        "name": torch.cuda.get_device_name(0),
        "compute_capability": "sm_89",
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    return {
        "evidence": verified,
        "census": census,
        "bandwidth": bandwidth,
        "p0_nf4": p0_nf4,
    }


def expand_formal_cases(
    census: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    source_rows = census["op_families"]["rmsnorm"]["shapes"]
    expected_rows = int(config["target_population"]["source_census_rows"])
    if len(source_rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} RMSNorm census rows")

    split_variants = set(
        config["target_population"]["semantic_expansion"]["split_variants"]
    )
    cases: list[dict[str, Any]] = []
    for source_index, row in enumerate(source_rows):
        original_variant = row["variant"]
        expansions: list[tuple[str, bool, float]]
        if original_variant in split_variants:
            suffix = original_variant.removeprefix("kv_norm_")
            expansions = [
                (f"k_norm_{suffix}", True, 0.5),
                (f"v_norm_{suffix}", False, 0.5),
            ]
        else:
            expansions = [(original_variant, True, 1.0)]

        m_rows, n_cols = (int(value) for value in row["dims"])
        for semantic_variant, with_scale, fraction in expansions:
            invocations_float = int(row["invocations"]) * fraction
            calls_float = int(row["calls_per_forward"]) * fraction
            analytic_elements_float = int(row["analytic_elements"]) * fraction
            for label, value in (
                ("invocations", invocations_float),
                ("calls_per_forward", calls_float),
                ("analytic_elements", analytic_elements_float),
            ):
                if not float(value).is_integer():
                    raise AssertionError(f"semantic split made {label} non-integral")

            outer_multiplier = {
                "decoder_hidden_norms": 1,
                "q_norm_sliding": 8,
                "q_norm_full": 8,
                "kv_norm_sliding_nonshared": 2,
                "kv_norm_full_nonshared": 2,
                "ple_projection_norm": 42,
            }[original_variant]
            token_count = m_rows // outer_multiplier if row["phase"] == "prefill" else 1
            if m_rows != token_count * outer_multiplier:
                raise AssertionError("RMSNorm flattened-row geometry is inconsistent")
            cases.append(
                {
                    "case_id": f"k1_{source_index:03d}_{'scaled' if with_scale else 'scale_free'}",
                    "source_census_row": source_index,
                    "source_variant": original_variant,
                    "variant": semantic_variant,
                    "with_scale": with_scale,
                    "dims": [m_rows, n_cols],
                    "native_outer_shape_note": row["native_outer_shape_note"],
                    "dtype": row["dtype"],
                    "phase": row["phase"],
                    "token_count": token_count,
                    "source_request_id": row.get("source_request_id"),
                    "source_request_ids": row.get("source_request_ids"),
                    "calls_per_forward": int(calls_float),
                    "forward_count": int(row["forward_count"]),
                    "invocations": int(invocations_float),
                    "analytic_elements": int(analytic_elements_float),
                    "weight": float(row["weight"]) * fraction,
                    "work_weight": float(row["work_weight"]) * fraction,
                }
            )

    expected_cases = int(
        config["target_population"]["semantic_expansion"]["expected_formal_cases"]
    )
    if len(cases) != expected_cases:
        raise AssertionError(f"expanded {len(cases)} cases, expected {expected_cases}")
    if sum(case["invocations"] for case in cases) != int(
        config["four_questions"]["authenticity"]["projected_workload_invocations"]
    ):
        raise AssertionError("expanded invocation total does not match pre-registration")
    _assert_close(sum(case["weight"] for case in cases), 1.0)
    _assert_close(sum(case["work_weight"] for case in cases), 1.0)
    return cases


def verify_model_and_load_weights(
    census: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model_path = _absolute(census["source_identity"]["model"])
    manifest_path = model_path / MODEL_MANIFEST_NAME
    observed_manifest_sha = sha256(manifest_path)
    expected_manifest_sha = census["source_identity"]["model_manifest_sha256"]
    if observed_manifest_sha != expected_manifest_sha:
        raise AssertionError("formal C02 model manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_records = {record["path"]: record for record in manifest["files_excluding_manifest"]}
    index_path = model_path / MODEL_INDEX_NAME
    if sha256(index_path) != file_records[MODEL_INDEX_NAME]["sha256"]:
        raise AssertionError("formal C02 tensor index hash mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    named_keys = config["target_population"]["timing_weight_tensors"]
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for variant, tensor_key in named_keys.items():
        shard = index["weight_map"].get(tensor_key)
        if shard is None:
            raise AssertionError(f"timing tensor absent from C02 index: {tensor_key}")
        by_shard[shard].append((variant, tensor_key))

    weights: dict[str, torch.Tensor] = {}
    tensor_evidence: dict[str, Any] = {}
    shard_evidence: dict[str, Any] = {}
    for shard, entries in sorted(by_shard.items()):
        shard_path = model_path / shard
        expected = file_records[shard]
        observed_bytes = shard_path.stat().st_size
        if observed_bytes != int(expected["bytes"]):
            raise AssertionError(f"selected C02 shard size mismatch: {shard}")
        shard_evidence[shard] = {
            "path": str(shard_path),
            "bytes": observed_bytes,
            "expected_sha256_from_upstream_full_hash_receipt": expected["sha256"],
            "current_check": "size plus indexed tensor read; full hash inherited from bound OP1/P0 receipts",
        }
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for variant, tensor_key in entries:
                tensor = handle.get_tensor(tensor_key)
                if tensor.dtype != torch.bfloat16 or tensor.ndim != 1:
                    raise AssertionError(f"unexpected timing scale tensor: {tensor_key}")
                tensor_float = tensor.float()
                tensor_evidence[variant] = {
                    "tensor_key": tensor_key,
                    "shard": shard,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": float(tensor_float.min().item()),
                    "max": float(tensor_float.max().item()),
                    "mean": float(tensor_float.mean().item()),
                }
                weights[variant] = tensor.contiguous().to(device="cuda")
    torch.cuda.synchronize()
    return weights, {
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": observed_manifest_sha,
        "artifact_set_sha256": manifest["artifact_set_sha256_excluding_manifest"],
        "index_sha256": file_records[MODEL_INDEX_NAME]["sha256"],
        "selected_shards": shard_evidence,
        "timing_scale_tensors": tensor_evidence,
        "status": "PASS",
    }


def _make_random_input(
    dims: list[int], dtype: torch.dtype, distribution: str, generator: torch.Generator
) -> torch.Tensor:
    values = torch.empty(tuple(dims), device="cuda", dtype=torch.float32)
    if distribution == "unit_normal":
        values.normal_(mean=0.0, std=1.0, generator=generator)
    elif distribution == "wide_normal":
        values.normal_(mean=0.0, std=32.0, generator=generator)
    elif distribution == "near_zero_uniform":
        values.uniform_(-0.001, 0.001, generator=generator)
    else:
        raise ValueError(f"unknown correctness distribution: {distribution}")
    return values.to(dtype=dtype)


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    delta = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(torch.finfo(torch.float32).tiny)
    relative = delta / denominator
    return {
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "max_rel": float(relative.max().item()) if relative.numel() else 0.0,
    }


def _check_output(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    errors = _error_metrics(actual, expected)
    checks = {
        "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "shape": tuple(actual.shape) == tuple(expected.shape),
        "dtype": actual.dtype == expected.dtype,
        "contiguous": actual.is_contiguous(),
        "finite": bool(torch.isfinite(actual).all().item()),
    }
    return {**errors, "checks": checks, "pass": all(checks.values())}


def run_correctness(
    cases: list[dict[str, Any]],
    config: dict[str, Any],
    compiled_scaled: Callable[..., torch.Tensor],
    compiled_scale_free: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    protocol = config["correctness_protocol"]
    prefill_tokens = set(int(value) for value in protocol["prefill_prompt_tokens"])
    selected = [
        case
        for case in cases
        if case["phase"] == "decode" or case["token_count"] in prefill_tokens
    ]
    expected_selected = int(protocol["expected_formal_cases_per_dtype_distribution_seed"])
    if len(selected) != expected_selected:
        raise AssertionError(
            f"correctness selected {len(selected)} cases, expected {expected_selected}"
        )
    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    epsilon = float(config["semantics"]["epsilon"])
    trials: list[dict[str, Any]] = []
    started = time.monotonic()
    total_expected = (
        len(selected)
        * len(protocol["seeds"])
        * len(protocol["input_distributions"])
        * len(config["semantics"]["supported_correctness_dtypes"])
    )
    completed = 0
    torch.cuda.reset_peak_memory_stats()
    for dtype_name in config["semantics"]["supported_correctness_dtypes"]:
        dtype = dtype_map[dtype_name]
        tolerance = protocol["tolerances"][dtype_name]
        atol, rtol = float(tolerance["atol"]), float(tolerance["rtol"])
        for distribution_record in protocol["input_distributions"]:
            distribution = distribution_record["name"]
            for seed in protocol["seeds"]:
                for case in selected:
                    generator = torch.Generator(device="cuda")
                    generator.manual_seed(int(seed))
                    hidden_states = _make_random_input(
                        case["dims"], dtype, distribution, generator
                    )
                    input_before = hidden_states.clone()
                    weight = None
                    if case["with_scale"]:
                        weight_fp32 = torch.empty(
                            case["dims"][1], device="cuda", dtype=torch.float32
                        )
                        weight_fp32.uniform_(-0.5, 96.0, generator=generator)
                        weight = weight_fp32.to(dtype=dtype)

                    oracle_fp32 = reference_fp32(hidden_states, weight, epsilon)
                    expected = oracle_fp32.to(dtype=dtype)
                    if case["with_scale"]:
                        eager_output = eager_scaled(hidden_states, weight, epsilon)
                        compile_output = compiled_scaled(hidden_states, weight, epsilon)
                    else:
                        eager_output = eager_scale_free(hidden_states, epsilon)
                        compile_output = compiled_scale_free(hidden_states, epsilon)
                    candidate_output = candidate_rmsnorm(hidden_states, weight, epsilon)
                    torch.cuda.synchronize()

                    implementations = {
                        "eager": _check_output(eager_output, expected, atol, rtol),
                        "compile": _check_output(compile_output, expected, atol, rtol),
                        "candidate": _check_output(candidate_output, expected, atol, rtol),
                    }
                    input_unchanged = bool(torch.equal(hidden_states, input_before))
                    reference_finite = bool(torch.isfinite(oracle_fp32).all().item())
                    scale_contract = (weight is None) == (not case["with_scale"])
                    passed = (
                        all(record["pass"] for record in implementations.values())
                        and input_unchanged
                        and reference_finite
                        and scale_contract
                    )
                    trials.append(
                        {
                            "case_id": case["case_id"],
                            "variant": case["variant"],
                            "phase": case["phase"],
                            "dims": case["dims"],
                            "with_scale": case["with_scale"],
                            "dtype": dtype_name,
                            "distribution": distribution,
                            "seed": int(seed),
                            "tolerance": {"atol": atol, "rtol": rtol},
                            "implementations": implementations,
                            "input_unchanged": input_unchanged,
                            "reference_finite": reference_finite,
                            "scale_contract": scale_contract,
                            "pass": passed,
                        }
                    )
                    completed += 1
                    del (
                        hidden_states,
                        input_before,
                        weight,
                        oracle_fp32,
                        expected,
                        eager_output,
                        compile_output,
                        candidate_output,
                    )
                print(
                    f"correctness {completed}/{total_expected}: "
                    f"dtype={dtype_name} distribution={distribution} seed={seed}",
                    flush=True,
                )

    by_dtype_implementation: dict[str, Any] = {}
    for dtype_name in dtype_map:
        by_dtype_implementation[dtype_name] = {}
        dtype_trials = [trial for trial in trials if trial["dtype"] == dtype_name]
        for implementation in ("eager", "compile", "candidate"):
            records = [
                trial["implementations"][implementation] for trial in dtype_trials
            ]
            by_dtype_implementation[dtype_name][implementation] = {
                "max_abs": max(record["max_abs"] for record in records),
                "max_rel": max(record["max_rel"] for record in records),
                "failed_trials": sum(not record["pass"] for record in records),
                "pass": all(record["pass"] for record in records),
            }
    failed_trials = [trial for trial in trials if not trial["pass"]]
    return {
        "reference": "explicit PyTorch fp32 semantic result, cast to input dtype for the carrier contract",
        "selection_case_ids": [case["case_id"] for case in selected],
        "selected_formal_case_count": len(selected),
        "dtypes": list(config["semantics"]["supported_correctness_dtypes"]),
        "distributions": [item["name"] for item in protocol["input_distributions"]],
        "seeds": [int(value) for value in protocol["seeds"]],
        "tolerance_policy": protocol["tolerances"],
        "n_trials": len(trials),
        "expected_trials": total_expected,
        "failed_trials": len(failed_trials),
        "by_dtype_implementation": by_dtype_implementation,
        "trials": trials,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "pass": not failed_trials and len(trials) == total_expected,
    }


def _compact_clocks(telemetry: dict[str, Any]) -> dict[str, Any]:
    numeric = telemetry["numeric"]
    return {
        "sm_mhz_p50": numeric["sm_mhz"]["p50"] if numeric["sm_mhz"] else None,
        "sm_mhz_min": numeric["sm_mhz"]["min"] if numeric["sm_mhz"] else None,
        "memory_mhz_p50": (
            numeric["memory_mhz"]["p50"] if numeric["memory_mhz"] else None
        ),
        "temp_c_max": (
            numeric["temperature_c"]["max"] if numeric["temperature_c"] else None
        ),
        "power_w_p50": numeric["power_w"]["p50"] if numeric["power_w"] else None,
        "pstates": telemetry["pstates"],
        "throttle_flags": telemetry["throttle_flags"],
        "thermal_or_hardware_throttle": telemetry["thermal_or_hardware_throttle"],
        "sample_count": telemetry["sample_count"],
        "monitor_errors": telemetry["errors"],
    }


def _analytic_bytes(case: dict[str, Any], element_bytes: int) -> int:
    m_rows, n_cols = case["dims"]
    input_and_output = 2 * m_rows * n_cols * element_bytes
    scale = n_cols * element_bytes if case["with_scale"] else 0
    return input_and_output + scale


def _actual_weight_for_case(
    case: dict[str, Any], weights: dict[str, torch.Tensor]
) -> torch.Tensor | None:
    if not case["with_scale"]:
        return None
    return weights[case["source_variant"]]


def run_performance(
    cases: list[dict[str, Any]],
    config: dict[str, Any],
    weights: dict[str, torch.Tensor],
    compiled_scaled: Callable[..., torch.Tensor],
    compiled_scale_free: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    protocol = config["measurement_protocol"]
    epsilon = float(config["semantics"]["epsilon"])
    warmup = int(protocol["warmup_repetitions_per_implementation_per_case"])
    repetitions = int(protocol["measured_repetitions_per_implementation_per_case"])
    ceiling_gbps = float(config["analytic_bytes"]["ceiling_gbps_decimal"])
    element_bytes = int(config["analytic_bytes"]["formal_element_bytes"])
    monitor = NvidiaSmiMonitor(float(protocol["gpu_telemetry_interval_seconds"]))
    monitor.set_phase("op2_stabilization")
    monitor.start()
    started = time.monotonic()
    measurements: list[dict[str, Any]] = []
    telemetry: dict[str, Any] | None = None
    try:
        largest = max(cases, key=lambda item: item["dims"][0] * item["dims"][1])
        stabilizer_input = torch.randn(
            tuple(largest["dims"]), device="cuda", dtype=torch.bfloat16
        )
        stabilizer_weight = _actual_weight_for_case(largest, weights)
        stabilization_started = time.monotonic()
        stabilization_launches = 0
        while (
            time.monotonic() - stabilization_started
            < float(protocol["steady_state_warmup_min_seconds"])
        ):
            for _ in range(16):
                candidate_rmsnorm(stabilizer_input, stabilizer_weight, epsilon)
                stabilization_launches += 1
            torch.cuda.synchronize()
        stabilization_seconds = time.monotonic() - stabilization_started
        del stabilizer_input, stabilizer_weight

        torch.cuda.reset_peak_memory_stats()
        ordered_cases = sorted(
            cases,
            key=lambda item: item["dims"][0] * item["dims"][1],
            reverse=True,
        )
        for measured_index, case in enumerate(ordered_cases, start=1):
            phase_name = f"op2_timing_{case['case_id']}"
            monitor.set_phase(phase_name)
            generator = torch.Generator(device="cuda")
            generator.manual_seed(20260915 + int(case["source_census_row"]))
            hidden_states = torch.empty(
                tuple(case["dims"]), device="cuda", dtype=torch.bfloat16
            )
            hidden_states.normal_(mean=0.0, std=1.0, generator=generator)
            weight = _actual_weight_for_case(case, weights)
            if case["with_scale"]:
                operations: dict[str, Callable[[], torch.Tensor]] = {
                    "eager": lambda: eager_scaled(hidden_states, weight, epsilon),
                    "compile": lambda: compiled_scaled(hidden_states, weight, epsilon),
                    "candidate": lambda: candidate_rmsnorm(hidden_states, weight, epsilon),
                }
            else:
                operations = {
                    "eager": lambda: eager_scale_free(hidden_states, epsilon),
                    "compile": lambda: compiled_scale_free(hidden_states, epsilon),
                    "candidate": lambda: candidate_rmsnorm(hidden_states, None, epsilon),
                }
            samples = cuda_event_measure(
                operations,
                list(protocol["timed_implementations"]),
                warmup,
                repetitions,
                before_synchronize=monitor.sample_now,
            )
            case_telemetry = monitor._summarize_samples(  # noqa: SLF001
                [sample for sample in monitor.samples if sample["phase"] == phase_name]
            )
            analytic_bytes = _analytic_bytes(case, element_bytes)
            baselines: dict[str, Any] = {}
            for name, values in samples.items():
                distribution = summarize(values)
                if distribution["p50"] <= 0:
                    raise AssertionError(f"non-positive event time for {case['case_id']} {name}")
                baselines[name] = {
                    "p50_us": distribution["p50"],
                    "iqr_us": distribution["iqr"],
                    "reps": distribution["count"],
                    "distribution_us": distribution,
                }
            baselines["reference"] = {
                "p50_us": None,
                "iqr_us": None,
                "reps": 0,
                "timed": False,
                "reason": config["baselines"]["reference"]["reason"],
            }
            candidate_seconds = baselines["candidate"]["p50_us"] / 1_000_000.0
            compile_seconds = baselines["compile"]["p50_us"] / 1_000_000.0
            candidate_gbps = analytic_bytes / candidate_seconds / 1_000_000_000.0
            compile_gbps = analytic_bytes / compile_seconds / 1_000_000_000.0
            record = {
                **case,
                "formal_dtype": "bfloat16",
                "analytic_bytes": analytic_bytes,
                "baselines": baselines,
                "speedup": {
                    "candidate_vs_eager": round(
                        baselines["eager"]["p50_us"]
                        / baselines["candidate"]["p50_us"],
                        6,
                    ),
                    "candidate_vs_compile": round(
                        baselines["compile"]["p50_us"]
                        / baselines["candidate"]["p50_us"],
                        6,
                    ),
                },
                "roofline": {
                    "tier": "Tier-A analytic",
                    "analytic_bytes": analytic_bytes,
                    "candidate_measured_gbps": round(candidate_gbps, 6),
                    "candidate_relative_score": round(candidate_gbps / ceiling_gbps, 6),
                    "compile_measured_gbps": round(compile_gbps, 6),
                    "compile_relative_score": round(compile_gbps / ceiling_gbps, 6),
                    "ceiling_gbps": ceiling_gbps,
                },
                "clocks": _compact_clocks(case_telemetry),
            }
            measurements.append(record)
            del hidden_states, weight, operations, samples
            print(
                f"timing {measured_index}/{len(ordered_cases)}: {case['case_id']} "
                f"{case['variant']} {case['dims']} candidate={baselines['candidate']['p50_us']:.3f} us",
                flush=True,
            )
    finally:
        telemetry = monitor.stop()

    measurements.sort(key=lambda item: item["case_id"])
    timing_telemetry = monitor._summarize_samples(  # noqa: SLF001
        [sample for sample in monitor.samples if sample["phase"].startswith("op2_timing_")]
    )
    return {
        "formal_case_count": len(measurements),
        "measurements": measurements,
        "steady_state_warmup": {
            "minimum_seconds": float(protocol["steady_state_warmup_min_seconds"]),
            "observed_seconds": round(stabilization_seconds, 6),
            "untimed_candidate_launches": stabilization_launches,
            "shape": largest["dims"],
            "case_id": largest["case_id"],
        },
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "telemetry": telemetry,
        "timing_telemetry": timing_telemetry,
        "clocks": _compact_clocks(timing_telemetry),
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def aggregate_performance(
    performance: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    measurements = performance["measurements"]
    implementations = list(config["measurement_protocol"]["timed_implementations"])
    ceiling = float(config["analytic_bytes"]["ceiling_gbps_decimal"])
    aggregates: dict[str, Any] = {}
    for name in implementations:
        total_us = sum(
            item["baselines"][name]["p50_us"] * item["invocations"]
            for item in measurements
        )
        weighted_p50 = sum(
            item["baselines"][name]["p50_us"] * item["weight"]
            for item in measurements
        )
        weighted_iqr = sum(
            item["baselines"][name]["iqr_us"] * item["weight"]
            for item in measurements
        )
        aggregates[name] = {
            "aggregation": "projected invocation-weighted per-case p50/IQR over the frozen workload",
            "p50_us": round(weighted_p50, 6),
            "iqr_us": round(weighted_iqr, 6),
            "reps": int(config["measurement_protocol"]["measured_repetitions_per_implementation_per_case"]),
            "formal_cases": len(measurements),
            "projected_workload_gpu_seconds": round(total_us / 1_000_000.0, 6),
        }
    aggregates["reference"] = {
        "implementation": config["baselines"]["reference"]["definition"],
        "p50_us": None,
        "iqr_us": None,
        "reps": 0,
        "timed": False,
        "reason": config["baselines"]["reference"]["reason"],
    }

    total_projected_bytes = sum(
        item["analytic_bytes"] * item["invocations"] for item in measurements
    )
    phase: dict[str, Any] = {}
    for phase_name in ("prefill", "decode"):
        phase_items = [item for item in measurements if item["phase"] == phase_name]
        phase_bytes = sum(item["analytic_bytes"] * item["invocations"] for item in phase_items)
        phase_result: dict[str, Any] = {
            "formal_cases": len(phase_items),
            "projected_invocations": sum(item["invocations"] for item in phase_items),
            "projected_analytic_bytes": phase_bytes,
        }
        for name in implementations:
            phase_us = sum(
                item["baselines"][name]["p50_us"] * item["invocations"]
                for item in phase_items
            )
            phase_result[f"{name}_projected_gpu_seconds"] = round(
                phase_us / 1_000_000.0, 6
            )
        phase_result["candidate_speedup_vs_eager"] = round(
            phase_result["eager_projected_gpu_seconds"]
            / phase_result["candidate_projected_gpu_seconds"],
            6,
        )
        phase_result["candidate_speedup_vs_compile"] = round(
            phase_result["compile_projected_gpu_seconds"]
            / phase_result["candidate_projected_gpu_seconds"],
            6,
        )
        phase[phase_name] = phase_result

    candidate_seconds = aggregates["candidate"]["projected_workload_gpu_seconds"]
    compile_seconds = aggregates["compile"]["projected_workload_gpu_seconds"]
    candidate_gbps = total_projected_bytes / candidate_seconds / 1_000_000_000.0
    compile_gbps = total_projected_bytes / compile_seconds / 1_000_000_000.0
    speedup_compile = compile_seconds / candidate_seconds
    thresholds = config["success_and_kill_criteria"]["aggregate_classification"]
    if speedup_compile >= 1.05:
        classification = "POSITIVE"
    elif speedup_compile >= 0.95:
        classification = "PARITY"
    else:
        classification = "NEGATIVE"
    no_headroom_threshold = 1.0 - float(
        config["four_questions"]["bottleneck"]["secondary_compile_roofline_gap_to_report"]
    )
    slower_than_compile_over_30pct = speedup_compile < (1.0 / 1.3)
    no_bandwidth_headroom = (candidate_gbps / ceiling) >= no_headroom_threshold
    return {
        "baselines": aggregates,
        "phase_projection": phase,
        "speedup": {
            "candidate_vs_eager": round(
                aggregates["eager"]["projected_workload_gpu_seconds"]
                / candidate_seconds,
                6,
            ),
            "candidate_vs_compile": round(speedup_compile, 6),
        },
        "performance_classification": {
            "value": classification,
            "frozen_thresholds": thresholds,
            "is_op2_pass_gate": False,
        },
        "roofline": {
            "tier": "Tier-A analytic",
            "analytic_bytes": total_projected_bytes,
            "measured_gbps": round(candidate_gbps, 6),
            "compile_measured_gbps": round(compile_gbps, 6),
            "ceiling_gbps": ceiling,
            "relative_score": round(candidate_gbps / ceiling, 6),
            "compile_relative_score": round(compile_gbps / ceiling, 6),
            "byte_formula_scaled": config["analytic_bytes"]["scaled_formula"],
            "byte_formula_scale_free": config["analytic_bytes"]["scale_free_formula"],
            "tier_b_status": "BLOCKED_WINDOWS_ERR_NVGPUCTRPERM_DEFERRED_TO_OP4",
        },
        "kill_criterion_2_observation": {
            "candidate_slower_than_compile_over_30_percent": slower_than_compile_over_30pct,
            "no_bandwidth_headroom_threshold_relative_score": no_headroom_threshold,
            "candidate_has_no_bandwidth_headroom": no_bandwidth_headroom,
            "triggered": slower_than_compile_over_30pct and no_bandwidth_headroom,
            "effect_if_triggered": "optimization-negative conclusion; methodology completion remains valid",
        },
    }


def _environment() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = args.config.resolve()
    prereg_sha_start = sha256(prereg_path)
    config = json.loads(prereg_path.read_text(encoding="utf-8"))
    verified = verify_frozen_inputs(config, prereg_path, prereg_sha_start)
    cases = expand_formal_cases(verified["census"], config)
    weights, model_evidence = verify_model_and_load_weights(
        verified["census"], config
    )
    compiled_scaled, compiled_scale_free = make_compiled_baselines()

    print(
        f"OP2 prereg={prereg_sha_start} formal_cases={len(cases)}; starting correctness",
        flush=True,
    )
    with torch.inference_mode():
        correctness = run_correctness(
            cases, config, compiled_scaled, compiled_scale_free
        )
        if not correctness["pass"]:
            raise AssertionError(
                f"K1 randomized correctness failed in {correctness['failed_trials']} trials"
            )
        print("K1 correctness PASS; starting 96-case timing", flush=True)
        performance = run_performance(
            cases,
            config,
            weights,
            compiled_scaled,
            compiled_scale_free,
        )
    aggregate = aggregate_performance(performance, config)

    prereg_sha_end = sha256(prereg_path)
    prereg_immutable = prereg_sha_start == prereg_sha_end
    clocks = performance["clocks"]
    memory_clock = clocks["memory_mhz_p50"]
    op0_memory_clock = float(config["measurement_protocol"]["op0_memory_clock_mhz_p50"])
    memory_clock_ratio = (
        float(memory_clock) / op0_memory_clock if memory_clock is not None else None
    )
    clock_pass = (
        memory_clock_ratio is not None
        and memory_clock_ratio
        >= float(config["measurement_protocol"]["minimum_memory_clock_ratio_vs_op0"])
    )
    cases_pass = (
        performance["formal_case_count"]
        == int(config["target_population"]["semantic_expansion"]["expected_formal_cases"])
        and all(
            item["baselines"][name]["reps"]
            == int(config["measurement_protocol"]["measured_repetitions_per_implementation_per_case"])
            for item in performance["measurements"]
            for name in config["measurement_protocol"]["timed_implementations"]
        )
    )
    roofline_pass = all(
        item["roofline"]["tier"] == "Tier-A analytic"
        and item["roofline"]["analytic_bytes"] > 0
        for item in performance["measurements"]
    )
    thermal_pass = not clocks["thermal_or_hardware_throttle"]
    gates = {
        "preregistration_frozen_before_measurement": "PASS"
        if config["frozen_before_any_k1_measurement"]
        else "FAIL",
        "preregistration_hash_immutable": "PASS" if prereg_immutable else "FAIL",
        "upstream_identity": "PASS",
        "direct_scale_semantics": "PASS",
        "formal_shape_expansion": "PASS" if len(cases) == 96 else "FAIL",
        "randomized_correctness": "PASS" if correctness["pass"] else "FAIL",
        "three_timed_implementations": "PASS" if cases_pass else "FAIL",
        "tier_a_roofline": "PASS" if roofline_pass else "FAIL",
        "memory_clock_comparability": "PASS" if clock_pass else "FAIL",
        "thermal_hardware_throttle": "PASS" if thermal_pass else "FAIL",
        "formal_reconnection": "DEFERRED_TO_OP3",
        "ncu_tier_b": "BLOCKED_WINDOWS_ERR_NVGPUCTRPERM_DEFERRED_TO_OP4",
    }
    blocking_gate_values = [
        value
        for key, value in gates.items()
        if key not in ("formal_reconnection", "ncu_tier_b")
    ]
    status = "PASS" if all(value == "PASS" for value in blocking_gate_values) else "INVALID"

    result = {
        "schema_version": 1,
        "stage": "OP2",
        "kernel": "k1_gemma4_rmsnorm",
        "op_semantics_sha": text_sha256(SCALED_SEMANTICS + "\n" + SCALE_FREE_SEMANTICS),
        "carrier": "C02-S.O BF16 RMSNorm tensors within the formal NF4 PyTorch carrier geometry",
        "timestamp": now_iso(),
        "preregistration": {
            "path": str(prereg_path),
            "sha256_at_start": prereg_sha_start,
            "sha256_at_end": prereg_sha_end,
            "immutable_during_measurement": prereg_immutable,
            "frozen_at": config["frozen_at"],
        },
        "source_verification": verified["evidence"],
        "model_weight_evidence": model_evidence,
        "four_questions": config["four_questions"],
        "semantics": config["semantics"],
        "shapes": cases,
        "shape_accounting": {
            "source_census_rows": config["target_population"]["source_census_rows"],
            "formal_cases_after_k_v_semantic_split": len(cases),
            "projected_invocations": sum(item["invocations"] for item in cases),
            "invocation_weight_sum": sum(item["weight"] for item in cases),
            "work_weight_sum": sum(item["work_weight"] for item in cases),
            "phase_frequency": config["target_population"]["phase_frequency"],
        },
        "baselines": aggregate["baselines"],
        "correctness": correctness,
        "formal_case_measurements": performance["measurements"],
        "phase_projection": aggregate["phase_projection"],
        "speedup": aggregate["speedup"],
        "performance_classification": aggregate["performance_classification"],
        "roofline": aggregate["roofline"],
        "kill_criterion_2_observation": aggregate["kill_criterion_2_observation"],
        "clocks": clocks,
        "telemetry": performance["telemetry"],
        "timing_phase_telemetry": performance["timing_telemetry"],
        "memory": {
            "correctness": {
                "peak_allocated_bytes": correctness["peak_allocated_bytes"],
                "peak_reserved_bytes": correctness["peak_reserved_bytes"],
            },
            "timing": performance["memory"],
        },
        "measurement_receipt": {
            "warmup_repetitions_per_implementation_per_case": int(
                config["measurement_protocol"][
                    "warmup_repetitions_per_implementation_per_case"
                ]
            ),
            "measured_repetitions_per_implementation_per_case": int(
                config["measurement_protocol"][
                    "measured_repetitions_per_implementation_per_case"
                ]
            ),
            "interleave_order": config["measurement_protocol"]["interleave_order"],
            "event_timer": config["measurement_protocol"]["event_timer"],
            "steady_state_warmup": performance["steady_state_warmup"],
            "elapsed_seconds": performance["elapsed_seconds"],
            "memory_clock_comparability": {
                "op0_memory_mhz_p50": op0_memory_clock,
                "op2_timing_memory_mhz_p50": memory_clock,
                "observed_ratio": round(memory_clock_ratio, 6)
                if memory_clock_ratio is not None
                else None,
                "minimum_ratio": float(
                    config["measurement_protocol"][
                        "minimum_memory_clock_ratio_vs_op0"
                    ]
                ),
                "pass": clock_pass,
            },
        },
        "backlink": {
            "carrier": "C02-S.O + bitsandbytes NF4 PyTorch path",
            "tape_ids": sorted(
                {
                    request_id.split(":", 1)[0]
                    for case in cases
                    for request_id in (
                        ([case["source_request_id"]] if case["source_request_id"] else [])
                        + (case["source_request_ids"] or [])
                    )
                }
            ),
            "tokens_per_s_delta": None,
            "ttft_delta": None,
            "side_effects": [
                "OP2 is an isolated formal-shape microbenchmark; C02 module replacement and token-level effects are deferred to OP3",
                "NF4 and Q4_K_M figures are not compared",
            ],
        },
        "env_identity_ref": config["source_identity"]["environment_identity"]["path"],
        "environment": _environment(),
        "implementation": {
            "kernel_source": str((ROOT / "scripts/operator/k1_rmsnorm.py").resolve()),
            "kernel_source_sha256": sha256(ROOT / "scripts/operator/k1_rmsnorm.py"),
            "runner_source": str(Path(__file__).resolve()),
            "runner_source_sha256": sha256(Path(__file__).resolve()),
        },
        "git_sha": _git_sha(),
        "gates": gates,
        "status": status,
    }
    atomic_write_json(args.result.resolve(), result)

    receipt = {
        "schema_version": 1,
        "stage": "OP2",
        "kernel": "k1_gemma4_rmsnorm",
        "timestamp": result["timestamp"],
        "artifacts": {
            "preregistration": {
                "path": str(prereg_path),
                "sha256": prereg_sha_end,
            },
            "implementation": {
                "path": result["implementation"]["kernel_source"],
                "sha256": result["implementation"]["kernel_source_sha256"],
            },
            "result": {
                "path": str(args.result.resolve()),
                "sha256": sha256(args.result.resolve()),
            },
        },
        "gates": gates,
        "summary": {
            "source_census_rows": config["target_population"]["source_census_rows"],
            "formal_cases": len(cases),
            "correctness_trials": correctness["n_trials"],
            "correctness_max_abs_by_dtype_implementation": {
                dtype_name: {
                    name: values["max_abs"]
                    for name, values in implementations.items()
                }
                for dtype_name, implementations in correctness[
                    "by_dtype_implementation"
                ].items()
            },
            "candidate_weighted_p50_us": aggregate["baselines"]["candidate"][
                "p50_us"
            ],
            "compile_weighted_p50_us": aggregate["baselines"]["compile"]["p50_us"],
            "candidate_speedup_vs_eager": aggregate["speedup"][
                "candidate_vs_eager"
            ],
            "candidate_speedup_vs_compile": aggregate["speedup"][
                "candidate_vs_compile"
            ],
            "performance_classification": aggregate["performance_classification"][
                "value"
            ],
            "tier_a_relative_score": aggregate["roofline"]["relative_score"],
            "memory_clock_ratio_vs_op0": round(memory_clock_ratio, 6)
            if memory_clock_ratio is not None
            else None,
            "thermal_or_hardware_throttle": clocks[
                "thermal_or_hardware_throttle"
            ],
        },
        "scope": {
            "op2_complete": status == "PASS",
            "op3_formal_reconnection_pending": True,
            "op4_ncu_tier_b_soft_blocked": True,
            "performance_gain_required": False,
        },
        "git_sha": _git_sha(),
        "status": status,
    }
    atomic_write_json(args.receipt.resolve(), receipt)
    return receipt


def _write_failure(
    args: argparse.Namespace,
    started_at: str,
    prereg_sha: str | None,
    error: BaseException,
) -> Path:
    failure_dir = ROOT / "results/operator/failures"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = failure_dir / f"k1_op2_failure_{stamp}.json"
    payload = {
        "schema_version": 1,
        "stage": "OP2",
        "kernel": "k1_gemma4_rmsnorm",
        "started_at": started_at,
        "failed_at": now_iso(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "config": str(args.config.resolve()),
        "config_sha256_at_failure": prereg_sha,
        "environment": _environment(),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "status": "FAIL",
    }
    atomic_write_json(path, payload)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = now_iso()
    prereg_sha: str | None = None
    try:
        prereg_sha = sha256(args.config.resolve())
        receipt = run(args)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0 if receipt["status"] == "PASS" else 2
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        failure = _write_failure(args, started_at, prereg_sha, error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        print(f"failure receipt: {failure}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
