#!/usr/bin/env python3
"""Run the frozen OP5 GQA decode-attention selection probe."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import torch

from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, sha256, summarize
from run_op5_selection import environment_identity, load_json, run_same_day_bandwidth
from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward


ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = datetime.now().astimezone().strftime("%Y%m%d")
DEFAULT_PROTOCOL = ROOT / "configs/operator/op5_gqa_decode_selection_probe.json"
DEFAULT_RESULT = ROOT / f"results/operator/pool/op5_gqa_selection_{DATE_TAG}.json"
DEFAULT_RECEIPT = ROOT / f"results/operator/pool/op5_gqa_selection_receipt_{DATE_TAG}.json"
PREREG_RECEIPT = ROOT / f"results/operator/pool/op5_gqa_prereg_receipt_{DATE_TAG}.json"
PREREG_RECEIPT_SHA256 = "6fd79a4fcaa47bd652b3e7f8b2174827ff348851e22e027d4ebd13f955c74c8f"
PRIOR_HARNESS_FAILURE = (
    ROOT / "results/operator/failures/op5_gqa_selection_failure_20260916T170832.json"
)
PRIOR_HARNESS_FAILURE_SHA256 = (
    "c3b542ecfa12e414e20d05afacf13d8ccb2656283f2a8499dbd112bb40809fe9"
)
SECOND_HARNESS_FAILURE = (
    ROOT / "results/operator/failures/op5_gqa_selection_failure_20260916T170952.json"
)
SECOND_HARNESS_FAILURE_SHA256 = (
    "091548bdb95f95186dc492beebe166e44fbb65abb8201e41021c08dc2d0c05a4"
)


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_prior_harness_failures(protocol_sha: str) -> dict[str, Any]:
    if sha256(PRIOR_HARNESS_FAILURE) != PRIOR_HARNESS_FAILURE_SHA256:
        raise AssertionError("GQA prior harness failure identity drift")
    failure = load_json(PRIOR_HARNESS_FAILURE)
    expected = {
        "stage": "OP5_GQA_DECODE_SELECTION",
        "protocol_sha256": protocol_sha,
        "error_type": "RuntimeError",
        "error": "Inference tensors do not track version counter.",
        "custom_candidate_implemented_or_timed": False,
        "status": "FAIL",
    }
    for field, value in expected.items():
        if failure.get(field) != value:
            raise AssertionError(f"GQA prior harness failure field drift: {field}")
    if "versions = [tensor._version for tensor in inputs]" not in str(
        failure.get("traceback", "")
    ):
        raise AssertionError("GQA prior failure is not the frozen version-counter defect")
    first = {
        "path": str(PRIOR_HARNESS_FAILURE),
        "sha256": PRIOR_HARNESS_FAILURE_SHA256,
        "classification": "HARNESS_ONLY_INFERENCE_TENSOR_VERSION_COUNTER",
        "formal_attention_baseline_executed_before_failure": False,
        "custom_candidate_implemented_or_timed": False,
        "protocol_changed_after_failure": False,
        "repair_scope": "create audited input tensors outside torch.inference_mode",
    }
    if sha256(SECOND_HARNESS_FAILURE) != SECOND_HARNESS_FAILURE_SHA256:
        raise AssertionError("GQA second harness failure identity drift")
    second_failure = load_json(SECOND_HARNESS_FAILURE)
    second_expected = {
        "stage": "OP5_GQA_DECODE_SELECTION",
        "protocol_sha256": protocol_sha,
        "error_type": "RuntimeError",
        "custom_candidate_implemented_or_timed": False,
        "status": "FAIL",
    }
    for field, value in second_expected.items():
        if second_failure.get(field) != value:
            raise AssertionError(f"GQA second harness failure field drift: {field}")
    if "CUDAGraphs that has been overwritten" not in str(second_failure.get("error", "")):
        raise AssertionError("GQA second failure is not the frozen CUDAGraph output defect")
    second = {
        "path": str(SECOND_HARNESS_FAILURE),
        "sha256": SECOND_HARNESS_FAILURE_SHA256,
        "classification": "HARNESS_ONLY_CUDAGRAPH_OUTPUT_LIFETIME",
        "correctness_operations_started": True,
        "formal_performance_measurement_started": False,
        "custom_candidate_implemented_or_timed": False,
        "protocol_changed_after_failure": False,
        "repair_scope": "clone correctness outputs immediately outside compiled graphs; timed path unchanged",
    }
    return {"count": 2, "failures": [first, second]}


class AttentionProxy(torch.nn.Module):
    num_key_value_groups = 4
    head_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.eval()


ATTENTION_PROXY = AttentionProxy()


def eager_operation(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    output, _ = eager_attention_forward(
        ATTENTION_PROXY,
        query,
        key,
        value,
        mask,
        dropout=0.0,
        scaling=1.0,
    )
    return output


def sdpa_operation(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0,
        enable_gqa=True,
    )
    return output.transpose(1, 2).contiguous()


def verify_protocol(protocol_path: Path, protocol_sha_start: str) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    if protocol.get("stage") != "OP5_GQA_DECODE_SELECTION":
        raise AssertionError("unexpected GQA selection protocol identity")
    if protocol.get("frozen_before_formal_probe_measurement") is not True:
        raise AssertionError("GQA protocol was not frozen before measurement")
    if protocol.get("frozen_before_gqa_candidate_implementation") is not True:
        raise AssertionError("GQA candidate existed before selection freeze")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("formal GQA selection requires the local sm_89 CUDA device")

    sources: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for name, identity in protocol["source_identity"].items():
        if name == "git_sha":
            continue
        path = absolute(identity["path"])
        observed = sha256(path)
        if observed != identity["sha256"]:
            raise AssertionError(
                f"GQA frozen source hash mismatch for {name}: {observed} != {identity['sha256']}"
            )
        sources[name] = {"path": str(path), "sha256": observed, "status": "PASS"}
        if path.suffix == ".json":
            loaded[name] = load_json(path)
    if git_sha() != protocol["source_identity"]["git_sha"]:
        raise AssertionError("Git HEAD differs from frozen GQA selection identity")
    if loaded["op1_receipt"].get("status") != "PASS":
        raise AssertionError("OP1 prerequisite is not PASS")
    if loaded["op1_channel_b_profiler"].get("status") != "PASS":
        raise AssertionError("OP1 channel-B prerequisite is not PASS")
    if loaded["k2_r2_closure_receipt"].get("status") != "K2_R2_CLOSED_DAY1_NOT_POSITIVE":
        raise AssertionError("K2-R2 closure does not authorize the next backlog decision")
    if loaded["op0_receipt"].get("status") != "PASS":
        raise AssertionError("OP0 prerequisite is not PASS")
    if sha256(PREREG_RECEIPT) != PREREG_RECEIPT_SHA256:
        raise AssertionError("GQA preregistration receipt identity drift")
    prereg_receipt = load_json(PREREG_RECEIPT)
    if prereg_receipt.get("status") != "PASS":
        raise AssertionError("GQA preregistration receipt is not PASS")
    if prereg_receipt.get("protocol", {}).get("sha256") != protocol_sha_start:
        raise AssertionError("GQA preregistration receipt protocol hash drift")

    census = loaded["shape_census"]
    attention = census["op_families"]["attention"]
    if attention["implementation"] != protocol["authenticity_contract"]["implementation"]:
        raise AssertionError("GQA implementation identity drift")
    if int(attention["calls_per_model_forward"]) != 42:
        raise AssertionError("GQA calls-per-forward drift")
    decode_rows = [row for row in attention["shapes"] if row["phase"] == "decode"]
    sliding_rows = [row for row in decode_rows if row["variant"] == "sliding_attention_eager"]
    full_rows = [row for row in decode_rows if row["variant"] == "full_attention_eager"]
    if len(sliding_rows) != 1 or len(full_rows) != 11:
        raise AssertionError("GQA decode census row coverage drift")
    sliding = sliding_rows[0]
    auth = protocol["authenticity_contract"]
    if int(sliding["dims"][3]) != int(auth["sliding"]["kv_tokens"]):
        raise AssertionError("GQA sliding K drift")
    if int(sliding["invocations"]) != int(auth["sliding"]["projected_invocations"]):
        raise AssertionError("GQA sliding invocation count drift")

    lengths: list[int] = []
    for row in full_rows:
        low, high = (int(value) for value in row["dimension_range"]["kv_tokens_inclusive"])
        if high - low + 1 != int(row["forward_count"]):
            raise AssertionError("GQA full-attention inclusive K range drift")
        lengths.extend(range(low, high + 1))
    lengths.sort()
    if len(lengths) != int(auth["full"]["projected_forwards"]):
        raise AssertionError("GQA full-attention forward count drift")
    expected_bins: list[dict[str, int]] = []
    bin_count = int(protocol["full_length_stratification"]["bin_count"])
    for index in range(bin_count):
        start = round(index * len(lengths) / bin_count)
        end = round((index + 1) * len(lengths) / bin_count)
        values = lengths[start:end]
        expected_bins.append(
            {
                "bin": index,
                "min_k": values[0],
                "max_k": values[-1],
                "representative_k": values[(len(values) - 1) // 2],
                "forward_count": len(values),
                "projected_invocations": len(values) * int(auth["full"]["calls_per_forward"]),
            }
        )
    if expected_bins != protocol["full_length_stratification"]["bins"]:
        raise AssertionError("GQA frozen full-length stratification drift")
    full_invocations = sum(row["projected_invocations"] for row in expected_bins)
    if full_invocations != int(auth["full"]["projected_invocations"]):
        raise AssertionError("GQA stratified full invocation count drift")
    if full_invocations + int(auth["sliding"]["projected_invocations"]) != int(
        auth["total_projected_decode_invocations"]
    ):
        raise AssertionError("GQA total decode invocation count drift")
    if sha256(protocol_path) != protocol_sha_start:
        raise AssertionError("GQA protocol changed during verification")
    return {
        "protocol": protocol,
        "sources": sources,
        "loaded": loaded,
        "decode_rows": decode_rows,
        "expected_bins": expected_bins,
        "git_sha": git_sha(),
        "prereg_receipt": {
            "path": str(PREREG_RECEIPT),
            "sha256": PREREG_RECEIPT_SHA256,
        },
    }


def build_cases(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    auth = protocol["authenticity_contract"]
    cases = [
        {
            "case": "sliding_k512_d256",
            "family": "sliding",
            "k_tokens": int(auth["sliding"]["kv_tokens"]),
            "head_dim": int(auth["sliding"]["head_dim"]),
            "projected_invocations": int(auth["sliding"]["projected_invocations"]),
            "source_bin": None,
        }
    ]
    for row in protocol["full_length_stratification"]["bins"]:
        cases.append(
            {
                "case": f"full_bin{row['bin']:02d}_k{row['representative_k']}_d512",
                "family": "full",
                "k_tokens": int(row["representative_k"]),
                "head_dim": int(auth["full"]["head_dim"]),
                "projected_invocations": int(row["projected_invocations"]),
                "source_bin": row,
            }
        )
    return cases


def make_inputs(case: dict[str, Any], seed: int, device: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=f"cuda:{device}")
    generator.manual_seed(seed)
    k_tokens = int(case["k_tokens"])
    head_dim = int(case["head_dim"])
    query = torch.randn(
        (1, 8, 1, head_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    key = torch.randn(
        (1, 2, k_tokens, head_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    value = torch.randn(
        (1, 2, k_tokens, head_dim),
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).to(torch.bfloat16)
    mask = torch.zeros((1, 1, 1, k_tokens), dtype=torch.bfloat16, device=device)
    return query, key, value, mask


def comparison(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    delta = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(torch.finfo(torch.float32).tiny)
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    exact = actual == expected
    return {
        "allclose": bool(close.all().item()),
        "max_abs": float(delta.max().item()),
        "max_rel": float((delta / denominator).max().item()),
        "exact_elements": int(exact.sum().item()),
        "allclose_elements": int(close.sum().item()),
        "non_allclose_elements": int((~close).sum().item()),
        "output_elements": actual.numel(),
    }


def analytic_bytes(case: dict[str, Any]) -> int:
    k_tokens = int(case["k_tokens"])
    head_dim = int(case["head_dim"])
    return 2 * (
        8 * head_dim
        + 2 * k_tokens * head_dim
        + 2 * k_tokens * head_dim
        + k_tokens
        + 8 * head_dim
    )


def run_correctness(
    protocol: dict[str, Any],
    cases: list[dict[str, Any]],
    operations: dict[str, Callable[..., torch.Tensor]],
    device: int,
) -> dict[str, Any]:
    settings = protocol["correctness_protocol"]
    atol = float(settings["tolerance"]["atol"])
    rtol = float(settings["tolerance"]["rtol"])
    rows: list[dict[str, Any]] = []
    allclose_by_implementation = {name: True for name in operations}
    log("GQA selection: correctness over 13 cases x 3 seeds x 4 implementations")
    for case in cases:
        for seed in (int(value) for value in settings["seeds"]):
            inputs = make_inputs(case, seed, device)
            snapshots = [tensor.clone() for tensor in inputs]
            pointers = [tensor.data_ptr() for tensor in inputs]
            versions = [tensor._version for tensor in inputs]
            with torch.inference_mode():
                outputs: dict[str, torch.Tensor] = {}
                for name, operation in operations.items():
                    outputs[name] = operation(*inputs).clone()
                torch.cuda.synchronize(device)
            reference = outputs["transformers_eager"]
            expected_shape = (1, 1, 8, int(case["head_dim"]))
            comparisons: dict[str, Any] = {}
            for name, output in outputs.items():
                contract = bool(
                    tuple(output.shape) == expected_shape
                    and output.dtype == torch.bfloat16
                    and output.device.type == "cuda"
                    and output.is_contiguous()
                    and torch.isfinite(output).all().item()
                )
                if not contract:
                    raise AssertionError(
                        f"GQA output contract failure for {case['case']}/{seed}/{name}"
                    )
                metrics = comparison(output, reference, atol, rtol)
                metrics["output_contract"] = contract
                comparisons[name] = metrics
                allclose_by_implementation[name] &= bool(metrics["allclose"])
            unmodified = all(
                torch.equal(tensor, snapshot)
                and tensor.data_ptr() == pointer
                and tensor._version == version
                for tensor, snapshot, pointer, version in zip(
                    inputs, snapshots, pointers, versions
                )
            )
            if not unmodified:
                raise AssertionError(f"GQA source tensor mutation for {case['case']}/{seed}")
            if not comparisons["compile_eager"]["allclose"]:
                raise AssertionError(
                    f"mandatory compiled eager mismatch for {case['case']}/{seed}"
                )
            rows.append(
                {
                    "case": case["case"],
                    "family": case["family"],
                    "k_tokens": case["k_tokens"],
                    "head_dim": case["head_dim"],
                    "seed": seed,
                    "comparisons_vs_transformers_eager": comparisons,
                    "source_tensors_unmodified": unmodified,
                }
            )
            del outputs, snapshots, inputs
        torch.cuda.empty_cache()
        log(f"GQA selection: correctness completed {case['case']}")
    eligibility = {
        "transformers_eager": True,
        "compile_eager": allclose_by_implementation["compile_eager"],
        "torch_sdpa_gqa": allclose_by_implementation["torch_sdpa_gqa"],
        "compile_sdpa_gqa": allclose_by_implementation["compile_sdpa_gqa"],
    }
    if not eligibility["compile_eager"]:
        raise AssertionError("compiled eager is not correctness-eligible")
    return {
        "tolerance": {"atol": atol, "rtol": rtol},
        "cases": len(cases),
        "seeds": settings["seeds"],
        "trials": len(rows),
        "rows": rows,
        "best_baseline_eligibility": eligibility,
        "status": "PASS",
    }


def benchmark(
    protocol: dict[str, Any],
    cases: list[dict[str, Any]],
    operations: dict[str, Callable[..., torch.Tensor]],
    eligibility: dict[str, bool],
    monitor: NvidiaSmiMonitor,
    ceiling_gbps: float,
    device: int,
) -> dict[str, Any]:
    settings = protocol["measurement_protocol"]
    warmups = int(settings["warmup_repetitions_per_case_per_implementation"])
    repetitions = int(settings["measured_repetitions_per_case_per_implementation"])
    implementation_names = list(operations)
    rows: list[dict[str, Any]] = []
    monitor.set_phase("gqa_probe_measurement")
    monitor.sample_now()
    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            inputs = make_inputs(case, int(protocol["correctness_protocol"]["seeds"][0]), device)
            for repetition in range(warmups):
                rotated = (
                    implementation_names[repetition % len(implementation_names) :]
                    + implementation_names[: repetition % len(implementation_names)]
                )
                for name in rotated:
                    operations[name](*inputs)
            torch.cuda.synchronize(device)

            samples: dict[str, list[float]] = {name: [] for name in implementation_names}
            for repetition in range(repetitions):
                rotated = (
                    implementation_names[repetition % len(implementation_names) :]
                    + implementation_names[: repetition % len(implementation_names)]
                )
                pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
                for name in rotated:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    operations[name](*inputs)
                    end.record()
                    pending.append((name, start, end))
                torch.cuda.synchronize(device)
                for name, start, end in pending:
                    samples[name].append(start.elapsed_time(end) * 1000.0)
            summaries = {name: summarize(values) for name, values in samples.items()}
            eligible_names = [name for name in implementation_names if eligibility[name]]
            best_name = min(eligible_names, key=lambda name: float(summaries[name]["p50"]))
            byte_count = analytic_bytes(case)
            best_us = float(summaries[best_name]["p50"])
            rows.append(
                {
                    **case,
                    "analytic_bytes": byte_count,
                    "implementations": summaries,
                    "best_correctness_eligible_baseline": best_name,
                    "best_baseline_p50_us": best_us,
                    "best_baseline_measured_gbps_decimal": byte_count / (best_us * 1000.0),
                    "best_baseline_tier_a_relative_score": byte_count
                    / (best_us * 1000.0)
                    / ceiling_gbps,
                }
            )
            del inputs
            torch.cuda.empty_cache()
            monitor.sample_now()
            log(f"GQA selection: measured {case_index + 1}/{len(cases)} cases")

    total_invocations = sum(int(row["projected_invocations"]) for row in rows)

    def weighted_latency(name: str) -> float:
        return sum(
            float(row["implementations"][name]["p50"])
            * int(row["projected_invocations"])
            for row in rows
        ) / total_invocations

    weighted_bytes = sum(
        int(row["analytic_bytes"]) * int(row["projected_invocations"])
        for row in rows
    ) / total_invocations
    weighted_by_implementation = {
        name: weighted_latency(name) for name in implementation_names
    }
    weighted_best_us = sum(
        float(row["best_baseline_p50_us"]) * int(row["projected_invocations"])
        for row in rows
    ) / total_invocations
    best_gbps = weighted_bytes / (weighted_best_us * 1000.0)
    sdpa_names = [
        name
        for name in ("torch_sdpa_gqa", "compile_sdpa_gqa")
        if eligibility[name]
    ]
    weighted_best_sdpa_us = (
        min(weighted_by_implementation[name] for name in sdpa_names)
        if sdpa_names
        else None
    )
    eager_us = weighted_by_implementation["transformers_eager"]
    sdpa_speedup = eager_us / weighted_best_sdpa_us if weighted_best_sdpa_us else None
    family_aggregates: dict[str, Any] = {}
    for family in ("sliding", "full"):
        family_rows = [row for row in rows if row["family"] == family]
        family_invocations = sum(int(row["projected_invocations"]) for row in family_rows)
        family_bytes = sum(
            int(row["analytic_bytes"]) * int(row["projected_invocations"])
            for row in family_rows
        ) / family_invocations
        family_best_us = sum(
            float(row["best_baseline_p50_us"]) * int(row["projected_invocations"])
            for row in family_rows
        ) / family_invocations
        family_gbps = family_bytes / (family_best_us * 1000.0)
        family_aggregates[family] = {
            "projected_invocations": family_invocations,
            "invocation_weighted_analytic_bytes": family_bytes,
            "invocation_weighted_best_baseline_p50_us": family_best_us,
            "best_baseline_measured_gbps_decimal": family_gbps,
            "best_baseline_tier_a_relative_score": family_gbps / ceiling_gbps,
        }
    return {
        "protocol": {
            "warmups_per_case_per_implementation": warmups,
            "measured_repetitions_per_case_per_implementation": repetitions,
            "interleave": settings["interleave"],
        },
        "rows": rows,
        "aggregate": {
            "projected_decode_invocations": total_invocations,
            "invocation_weighted_analytic_bytes": weighted_bytes,
            "invocation_weighted_p50_us_by_implementation": weighted_by_implementation,
            "invocation_weighted_best_baseline_p50_us": weighted_best_us,
            "best_baseline_measured_gbps_decimal": best_gbps,
            "same_day_ceiling_gbps_decimal": ceiling_gbps,
            "best_baseline_tier_a_relative_score": best_gbps / ceiling_gbps,
            "best_baseline_headroom_fraction": 1.0 - best_gbps / ceiling_gbps,
            "best_correctness_eligible_sdpa_p50_us": weighted_best_sdpa_us,
            "best_sdpa_speedup_vs_current_eager": sdpa_speedup,
        },
        "family_aggregates": family_aggregates,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol_sha_start = sha256(protocol_path)
    prior_harness_failures = verify_prior_harness_failures(protocol_sha_start)
    verified = verify_protocol(protocol_path, protocol_sha_start)
    protocol = verified["protocol"]
    torch.cuda.set_device(args.device)
    torch.manual_seed(int(protocol["correctness_protocol"]["seeds"][0]))
    torch.cuda.manual_seed_all(int(protocol["correctness_protocol"]["seeds"][0]))
    torch._dynamo.config.recompile_limit = int(
        protocol["measurement_protocol"]["torch_dynamo_recompile_limit"]
    )
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        torch._dynamo.config.accumulated_recompile_limit = 128
    cases = build_cases(protocol)
    compile_started = time.perf_counter()
    compiled_eager = torch.compile(
        eager_operation,
        fullgraph=True,
        mode="reduce-overhead",
        dynamic=False,
    )
    compiled_sdpa = torch.compile(
        sdpa_operation,
        fullgraph=True,
        mode="reduce-overhead",
        dynamic=False,
    )
    wrapper_creation_seconds = time.perf_counter() - compile_started
    operations: dict[str, Callable[..., torch.Tensor]] = {
        "transformers_eager": eager_operation,
        "compile_eager": compiled_eager,
        "torch_sdpa_gqa": sdpa_operation,
        "compile_sdpa_gqa": compiled_sdpa,
    }

    environment = environment_identity()
    monitor = NvidiaSmiMonitor(
        interval_seconds=float(protocol["measurement_protocol"]["gpu_telemetry_interval_seconds"])
    )
    monitor.start()
    telemetry: dict[str, Any] | None = None
    try:
        bandwidth_config = verified["loaded"]["same_day_bandwidth_config"]
        log("GQA selection: measuring frozen same-day bandwidth ceiling")
        bandwidth = run_same_day_bandwidth(bandwidth_config, monitor, args.device)
        materialization_started = time.perf_counter()
        correctness = run_correctness(protocol, cases, operations, args.device)
        compile_materialization_seconds = time.perf_counter() - materialization_started
        benchmark_result = benchmark(
            protocol,
            cases,
            operations,
            correctness["best_baseline_eligibility"],
            monitor,
            float(bandwidth["ceiling_gbps_decimal"]),
            args.device,
        )
        telemetry = monitor.stop()
        measurement_telemetry = monitor.summary_for_phase("gqa_probe_measurement")
        bandwidth_telemetry = monitor.summary_for_phase("op5_same_day_bandwidth")
        memory_summary = measurement_telemetry["numeric"]["memory_mhz"]
        if memory_summary is None:
            raise AssertionError("GQA selection has no memory-clock witness")
        memory_clock_ratio = float(memory_summary["p50"]) / float(
            protocol["measurement_protocol"]["op0_memory_clock_mhz_p50"]
        )
        thermal_valid = not telemetry["thermal_or_hardware_throttle"]
        memory_clock_valid = memory_clock_ratio >= float(
            protocol["measurement_protocol"]["minimum_memory_clock_ratio_vs_op0"]
        )
        day_valid = thermal_valid and memory_clock_valid
        relative_score = float(
            benchmark_result["aggregate"]["best_baseline_tier_a_relative_score"]
        )
        bottleneck_pass = day_valid and relative_score <= 0.80
        sdpa_speedup = benchmark_result["aggregate"]["best_sdpa_speedup_vs_current_eager"]
        sdpa_precedence = bool(
            day_valid and sdpa_speedup is not None and float(sdpa_speedup) >= 1.05
        )
        if not day_valid:
            decision = "INCONCLUSIVE_INVALID_DAY"
            next_action = "Repeat the unchanged GQA selection probe; do not authorize a candidate."
            status = "INVALID_TELEMETRY"
        elif not bottleneck_pass:
            decision = "NOT_SELECTED_NO_BOTTLENECK_HEADROOM"
            next_action = "Do not implement a custom GQA candidate; retain this valid probe and move the OP5 backlog decision onward."
            status = "PASS_NOT_SELECTED"
        elif sdpa_precedence:
            decision = "ROUTE_NATIVE_SDPA_RECONNECTION_BEFORE_CUSTOM"
            next_action = "Freeze and run a configuration-only C02 SDPA reconnection/equivalence probe before considering any custom GQA kernel."
            status = "PASS_NATIVE_SDPA_FIRST"
        else:
            decision = "CUSTOM_GQA_DESIGN_PREREG_AUTHORIZED"
            next_action = "Complete C/D design, then freeze a custom GQA candidate protocol before implementation."
            status = "PASS_CUSTOM_DESIGN_AUTHORIZED"

        protocol_sha_end = sha256(protocol_path)
        if protocol_sha_end != protocol_sha_start:
            raise AssertionError("GQA selection protocol changed during formal probe")
        result = {
            "schema_version": 1,
            "stage": "OP5_GQA_DECODE_SELECTION",
            "timestamp": now_iso(),
            "evidence_scope": protocol["host_scope"],
            "protocol": {
                "path": str(protocol_path),
                "sha256_start": protocol_sha_start,
                "sha256_end": protocol_sha_end,
            },
            "preregistration_receipt": verified["prereg_receipt"],
            "harness_execution_attempt": 3,
            "prior_harness_failures": prior_harness_failures,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "verified_sources": verified["sources"],
            "git_sha": verified["git_sha"],
            "environment": environment,
            "compile": {
                "wrapper_creation_seconds": wrapper_creation_seconds,
                "materialization_and_correctness_seconds": compile_materialization_seconds,
                "fullgraph": True,
                "mode": "reduce-overhead",
                "dynamic": False,
                "recompile_limit": protocol["measurement_protocol"]["torch_dynamo_recompile_limit"],
            },
            "authenticity": {
                "status": "PASS",
                "decode_rows": len(verified["decode_rows"]),
                "cases": cases,
            },
            "correctness": correctness,
            "same_day_bandwidth": bandwidth,
            "benchmark": benchmark_result,
            "telemetry": telemetry,
            "bandwidth_telemetry": bandwidth_telemetry,
            "measurement_telemetry": measurement_telemetry,
            "telemetry_gates": {
                "thermal_or_hardware_throttle_absent": thermal_valid,
                "memory_clock_ratio_vs_op0": memory_clock_ratio,
                "minimum_memory_clock_ratio": protocol["measurement_protocol"]["minimum_memory_clock_ratio_vs_op0"],
                "memory_clock_gate": memory_clock_valid,
                "day_valid": day_valid,
            },
            "selection": {
                "authenticity": "PASS",
                "bottleneck": "PASS" if bottleneck_pass else "FAIL",
                "bottleneck_relative_score_threshold": 0.80,
                "observed_best_baseline_relative_score": relative_score,
                "native_sdpa_precedence": sdpa_precedence,
                "decision": decision,
            },
            "custom_candidate_implemented_or_timed": False,
            "next_authorized_action": next_action,
            "status": status,
        }
        result_path = args.result.resolve()
        atomic_write_json(result_path, result)
        receipt = {
            "schema_version": 1,
            "stage": "OP5_GQA_DECODE_SELECTION",
            "timestamp": now_iso(),
            "protocol": {"path": str(protocol_path), "sha256": protocol_sha_start},
            "preregistration_receipt": verified["prereg_receipt"],
            "harness_execution_attempt": 3,
            "prior_harness_failures": prior_harness_failures,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "result": {"path": str(result_path), "sha256": sha256(result_path)},
            "gates": {
                "frozen_protocol_immutable": "PASS",
                "prior_harness_failures_preserved_and_classified": "PASS",
                "source_and_git_identity": "PASS",
                "k2_r2_terminal_transition": "PASS",
                "authenticity": "PASS",
                "correctness": correctness["status"],
                "all_13_cases_x_3_seeds": "PASS",
                "four_existing_baselines_measured": "PASS",
                "same_day_bandwidth": "PASS",
                "thermal_or_hardware_throttle_absent": "PASS" if thermal_valid else "FAIL",
                "memory_clock_gate": "PASS" if memory_clock_valid else "FAIL",
                "bottleneck": "PASS" if bottleneck_pass else "FAIL",
            },
            "best_baseline_eligibility": correctness["best_baseline_eligibility"],
            "best_baseline_tier_a_relative_score": relative_score,
            "best_sdpa_speedup_vs_current_eager": sdpa_speedup,
            "decision": decision,
            "custom_candidate_implemented_or_timed": False,
            "next_authorized_action": next_action,
            "status": status,
        }
        receipt_path = args.receipt.resolve()
        atomic_write_json(receipt_path, receipt)
        if sha256(protocol_path) != protocol_sha_start:
            raise AssertionError("GQA protocol changed through receipt write")
        return receipt
    finally:
        if telemetry is None:
            monitor.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = now_iso()
    try:
        receipt = execute(args)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        failure_path = ROOT / (
            "results/operator/failures/"
            f"op5_gqa_selection_failure_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}.json"
        )
        protocol_path = args.protocol.resolve()
        failure = {
            "schema_version": 1,
            "stage": "OP5_GQA_DECODE_SELECTION",
            "started_at": started,
            "failed_at": now_iso(),
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path) if protocol_path.is_file() else None,
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "git_sha": git_sha(),
            "environment": environment_identity(),
            "custom_candidate_implemented_or_timed": False,
            "status": "FAIL",
        }
        atomic_write_json(failure_path, failure)
        print(
            json.dumps({**failure, "failure_path": str(failure_path)}, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
