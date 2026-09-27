#!/usr/bin/env python3
"""Run the frozen bounded K3 GQA decode-attention Day-1 performance gate."""

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

from k3_gqa_attention import autotune_choices, candidate_gqa_attention
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, sha256, summarize
from run_op5_gqa_selection import (
    analytic_bytes,
    eager_operation,
    environment_identity,
    load_json,
    make_inputs,
)
from run_op5_selection import run_same_day_bandwidth


ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = datetime.now().astimezone().strftime("%Y%m%d")
DEFAULT_PROTOCOL = ROOT / "configs/operator/k3_gqa_decode_attention_prereg.json"
DEFAULT_CORRECTNESS_RECEIPT = (
    ROOT / f"results/operator/pool/k3_gqa_correctness_receipt_{DATE_TAG}.json"
)
DEFAULT_RESULT = ROOT / f"results/operator/pool/k3_gqa_day1_{DATE_TAG}.json"
DEFAULT_RECEIPT = ROOT / f"results/operator/pool/k3_gqa_day1_receipt_{DATE_TAG}.json"
CANDIDATE = ROOT / "scripts/operator/k3_gqa_attention.py"
CANDIDATE_SHA256 = "3d3d57ac20e390e9f117ac763af7d5ffcfabe084fce861461e8e62e183ac76f0"
CORRECTNESS_RECEIPT_SHA256 = "ac8e9834ccfd32d731ef6374c962989486b3779caa9fa90e848a1392b8f7cf32"
CORRECTNESS_RESULT_SHA256 = "8ba0bca4ee44af54d7da368b72a61833adaba0a0ebcfbe8fa700d376129df514"


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


def verify_preconditions(
    protocol_path: Path,
    protocol_sha: str,
    correctness_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    if protocol.get("stage") != "OP5_K3":
        raise AssertionError("unexpected K3 Day-1 protocol identity")
    if sha256(CANDIDATE) != CANDIDATE_SHA256:
        raise AssertionError("locked K3 candidate changed before Day-1")
    if sha256(correctness_receipt_path) != CORRECTNESS_RECEIPT_SHA256:
        raise AssertionError("K3 correctness receipt identity drift")
    correctness_receipt = load_json(correctness_receipt_path)
    if (
        correctness_receipt.get("stage")
        != "OP5_K3_GQA_DECODE_ATTENTION_CORRECTNESS"
        or correctness_receipt.get("status") != "PASS"
    ):
        raise AssertionError("K3 correctness receipt is not PASS")
    if correctness_receipt.get("protocol") != {
        "path": str(protocol_path),
        "sha256": protocol_sha,
    }:
        raise AssertionError("K3 correctness receipt protocol identity drift")
    if correctness_receipt.get("locked_candidate") != {
        "path": str(CANDIDATE),
        "sha256": CANDIDATE_SHA256,
    }:
        raise AssertionError("K3 correctness receipt candidate identity drift")
    required_gates = (
        "protocol_immutable",
        "candidate_source_locked",
        "source_and_git_identity",
        "all_length_cases",
        "fresh_representative_matrix",
        "mask_matrix",
        "gqa_head_mapping_microcase",
        "source_tensors_unmodified",
        "fallback_rejection_matrix",
        "frozen_autotune_space",
    )
    if any(
        correctness_receipt.get("gates", {}).get(name) != "PASS"
        for name in required_gates
    ):
        raise AssertionError("K3 correctness receipt contains a non-PASS gate")
    if correctness_receipt.get("counts", {}).get("candidate_trials") != 75:
        raise AssertionError("K3 formal correctness trial-count drift")
    result_pointer = correctness_receipt.get("result", {})
    result_path = absolute(str(result_pointer.get("path", "")))
    if (
        not result_path.is_file()
        or sha256(result_path) != CORRECTNESS_RESULT_SHA256
        or result_pointer.get("sha256") != CORRECTNESS_RESULT_SHA256
    ):
        raise AssertionError("K3 correctness result identity drift")
    correctness_result = load_json(result_path)
    if correctness_result.get("status") != "PASS":
        raise AssertionError("K3 correctness result is not PASS")
    if correctness_result.get("formal_candidate_performance_started") is not False:
        raise AssertionError("K3 correctness result unexpectedly includes performance")

    verified_sources: dict[str, Any] = {}
    selection_config: dict[str, Any] | None = None
    for name, identity in protocol["source_identity"].items():
        if name == "git_sha":
            continue
        path = absolute(identity["path"])
        observed = sha256(path)
        if observed != identity["sha256"]:
            raise AssertionError(f"K3 frozen source hash mismatch: {name}")
        verified_sources[name] = {
            "path": str(path),
            "sha256": observed,
            "status": "PASS",
        }
        if name == "gqa_selection_protocol":
            selection_config = load_json(path)
    if selection_config is None:
        raise AssertionError("K3 selection protocol source is missing")
    if git_sha() != protocol["source_identity"]["git_sha"]:
        raise AssertionError("Git HEAD differs from frozen K3 identity")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("formal K3 Day-1 requires the local sm_89 CUDA device")

    declared = protocol["analytic_bytes"]["same_day_ceiling_protocol"]
    inherited = selection_config["analytic_bytes"]["same_day_ceiling"]
    for field in (
        "operation",
        "array_size_mib",
        "warmup_repetitions",
        "measured_repetitions",
        "block_size",
    ):
        if declared[field] != inherited[field]:
            raise AssertionError(f"K3 same-day bandwidth protocol drift: {field}")
    if int(inherited["analytic_bytes_per_repetition"]) != 3 * int(
        declared["array_size_mib"]
    ) * 1024 * 1024:
        raise AssertionError("K3 same-day bandwidth byte declaration drift")
    if sha256(protocol_path) != protocol_sha:
        raise AssertionError("K3 protocol changed during Day-1 precondition checks")
    correctness = {
        "receipt": {
            "path": str(correctness_receipt_path),
            "sha256": CORRECTNESS_RECEIPT_SHA256,
        },
        "result": {
            "path": str(result_path),
            "sha256": CORRECTNESS_RESULT_SHA256,
        },
        "status": "PASS",
    }
    return protocol, selection_config, verified_sources, correctness


def build_cases(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(protocol["target_population"]["performance_cases"]):
        key_tokens = int(row["K"])
        head_dim = int(row["D"])
        cases.append(
            {
                "case": (
                    f"sliding_k{key_tokens}_d{head_dim}"
                    if row["family"] == "sliding"
                    else f"full_bin{index - 1:02d}_k{key_tokens}_d{head_dim}"
                ),
                "family": row["family"],
                "k_tokens": key_tokens,
                "head_dim": head_dim,
                "projected_invocations": int(row["projected_invocations"]),
            }
        )
    if len(cases) != 13:
        raise AssertionError("K3 frozen performance case count drift")
    if sum(row["projected_invocations"] for row in cases) != int(
        protocol["target_population"]["total_projected_invocations"]
    ):
        raise AssertionError("K3 projected invocation total drift")
    return cases


def benchmark(
    protocol: dict[str, Any],
    cases: list[dict[str, Any]],
    operations: dict[str, Callable[..., torch.Tensor]],
    monitor: NvidiaSmiMonitor,
    ceiling_gbps: float,
    device: int,
) -> dict[str, Any]:
    settings = protocol["measurement_protocol"]
    warmups = int(settings["warmup_repetitions_per_case_per_implementation"])
    repetitions = int(settings["measured_repetitions_per_case_per_implementation"])
    implementation_names = list(operations)
    if implementation_names != ["current_eager", "compile_eager", "candidate"]:
        raise AssertionError("K3 timed implementation order drift")
    seed = int(protocol["formal_correctness_protocol"]["fresh_seeds"][0])
    rows: list[dict[str, Any]] = []
    monitor.set_phase("k3_day1_measurement")
    monitor.sample_now()
    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            inputs = make_inputs(case, seed, device)
            for repetition in range(warmups):
                rotated = (
                    implementation_names[repetition % len(implementation_names) :]
                    + implementation_names[: repetition % len(implementation_names)]
                )
                for name in rotated:
                    operations[name](*inputs)
            torch.cuda.synchronize(device)

            samples: dict[str, list[float]] = {
                name: [] for name in implementation_names
            }
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
            summaries = {
                name: summarize(values) for name, values in samples.items()
            }
            best_baseline = min(
                ("current_eager", "compile_eager"),
                key=lambda name: float(summaries[name]["p50"]),
            )
            best_us = float(summaries[best_baseline]["p50"])
            candidate_us = float(summaries["candidate"]["p50"])
            byte_count = analytic_bytes(case)
            rows.append(
                {
                    **case,
                    "input_seed": seed,
                    "analytic_bytes": byte_count,
                    "implementations": summaries,
                    "best_correctness_eligible_baseline": best_baseline,
                    "best_baseline_p50_us": best_us,
                    "candidate_p50_us": candidate_us,
                    "candidate_speedup_vs_best_baseline": best_us / candidate_us,
                    "candidate_measured_gbps_decimal": byte_count
                    / (candidate_us * 1000.0),
                    "candidate_tier_a_relative_score": byte_count
                    / (candidate_us * 1000.0)
                    / ceiling_gbps,
                }
            )
            del inputs
            torch.cuda.empty_cache()
            monitor.sample_now()
            log(f"K3 Day-1: measured {case_index + 1}/{len(cases)} cases")

    total_invocations = sum(row["projected_invocations"] for row in rows)

    def weighted(field: Callable[[dict[str, Any]], float], selected: list[dict[str, Any]]) -> float:
        denominator = sum(row["projected_invocations"] for row in selected)
        return sum(field(row) * row["projected_invocations"] for row in selected) / denominator

    weighted_by_implementation = {
        name: weighted(
            lambda row, implementation=name: float(
                row["implementations"][implementation]["p50"]
            ),
            rows,
        )
        for name in implementation_names
    }
    weighted_best = weighted(lambda row: float(row["best_baseline_p50_us"]), rows)
    weighted_candidate = weighted(lambda row: float(row["candidate_p50_us"]), rows)
    family_aggregates: dict[str, Any] = {}
    for family in ("sliding", "full"):
        family_rows = [row for row in rows if row["family"] == family]
        family_best = weighted(
            lambda row: float(row["best_baseline_p50_us"]), family_rows
        )
        family_candidate = weighted(
            lambda row: float(row["candidate_p50_us"]), family_rows
        )
        family_aggregates[family] = {
            "projected_invocations": sum(
                row["projected_invocations"] for row in family_rows
            ),
            "invocation_weighted_best_baseline_p50_us": family_best,
            "invocation_weighted_candidate_p50_us": family_candidate,
            "candidate_speedup_vs_best_baseline": family_best / family_candidate,
        }
    aggregate_speedup = weighted_best / weighted_candidate
    family_minimum = min(
        float(row["candidate_speedup_vs_best_baseline"])
        for row in family_aggregates.values()
    )
    positive_by_performance = bool(aggregate_speedup >= 1.05 and family_minimum >= 0.95)
    return {
        "protocol": {
            "input_seed": seed,
            "warmups_per_case_per_implementation": warmups,
            "measured_repetitions_per_case_per_implementation": repetitions,
            "interleave": settings["interleave"],
        },
        "rows": rows,
        "aggregate": {
            "projected_decode_invocations": total_invocations,
            "invocation_weighted_p50_us_by_implementation": weighted_by_implementation,
            "invocation_weighted_best_baseline_p50_us": weighted_best,
            "invocation_weighted_candidate_p50_us": weighted_candidate,
            "candidate_speedup_vs_best_baseline": aggregate_speedup,
            "minimum_family_speedup_vs_best_baseline": family_minimum,
            "performance_gate_positive": positive_by_performance,
            "classification": "POSITIVE" if positive_by_performance else "NEGATIVE",
        },
        "family_aggregates": family_aggregates,
    }


def validate_autotune(
    protocol: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    choices = autotune_choices()
    design = protocol["candidate_design"]
    frozen = {
        "score": design["score_search_space"],
        "softmax": design["softmax_search_space"],
        "pv": design["pv_search_space"],
    }
    fields = {
        "score": ("BLOCK_K", "BLOCK_D", "num_warps", "num_stages"),
        "softmax": ("num_warps", "num_stages"),
        "pv": ("BLOCK_D", "BLOCK_K", "num_warps", "num_stages"),
    }
    for stage, rows in choices.items():
        allowed = {
            tuple(config[field] for field in fields[stage])
            for config in frozen[stage]
        }
        for row in rows:
            selected = tuple(row.get(field) for field in fields[stage])
            if selected not in allowed:
                raise AssertionError(f"non-frozen K3 {stage} autotune choice: {row}")
    expected_geometries = {(row["k_tokens"], row["head_dim"]) for row in cases}
    expected_lengths = {row["k_tokens"] for row in cases}
    score_geometries = {(int(row["key"][0]), int(row["key"][1])) for row in choices["score"]}
    pv_geometries = {(int(row["key"][0]), int(row["key"][1])) for row in choices["pv"]}
    softmax_lengths = {int(row["key"][0]) for row in choices["softmax"]}
    if score_geometries != expected_geometries or pv_geometries != expected_geometries:
        raise AssertionError("K3 Day-1 score/PV autotune coverage drift")
    if softmax_lengths != expected_lengths:
        raise AssertionError("K3 Day-1 softmax autotune coverage drift")
    return {"choices": choices, "status": "PASS"}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol_sha_start = sha256(protocol_path)
    candidate_sha_start = sha256(CANDIDATE)
    protocol, selection_config, verified_sources, correctness = verify_preconditions(
        protocol_path,
        protocol_sha_start,
        args.correctness_receipt.resolve(),
    )
    torch.cuda.set_device(args.device)
    torch._dynamo.config.recompile_limit = int(
        selection_config["measurement_protocol"]["torch_dynamo_recompile_limit"]
    )
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        torch._dynamo.config.accumulated_recompile_limit = 128
    compile_started = time.perf_counter()
    compiled_eager = torch.compile(
        eager_operation,
        fullgraph=True,
        mode="reduce-overhead",
        dynamic=False,
    )
    wrapper_creation_seconds = time.perf_counter() - compile_started
    operations: dict[str, Callable[..., torch.Tensor]] = {
        "current_eager": eager_operation,
        "compile_eager": compiled_eager,
        "candidate": candidate_gqa_attention,
    }
    cases = build_cases(protocol)
    environment = environment_identity()
    monitor = NvidiaSmiMonitor(
        interval_seconds=float(
            protocol["measurement_protocol"]["gpu_telemetry_interval_seconds"]
        )
    )
    monitor.start()
    telemetry: dict[str, Any] | None = None
    performance_started = False
    try:
        log("K3 Day-1: measuring the frozen same-day bandwidth ceiling")
        bandwidth = run_same_day_bandwidth(selection_config, monitor, args.device)
        performance_started = True
        benchmark_result = benchmark(
            protocol,
            cases,
            operations,
            monitor,
            float(bandwidth["ceiling_gbps_decimal"]),
            args.device,
        )
        tuning = validate_autotune(protocol, cases)
        telemetry = monitor.stop()
        measurement_telemetry = monitor.summary_for_phase("k3_day1_measurement")
        bandwidth_telemetry = monitor.summary_for_phase("op5_same_day_bandwidth")
        memory_summary = measurement_telemetry["numeric"]["memory_mhz"]
        if memory_summary is None:
            raise AssertionError("K3 Day-1 has no memory-clock witness")
        memory_clock_ratio = float(memory_summary["p50"]) / float(
            protocol["measurement_protocol"]["op0_memory_clock_mhz_p50"]
        )
        thermal_valid = not telemetry["thermal_or_hardware_throttle"]
        memory_clock_valid = memory_clock_ratio >= float(
            protocol["measurement_protocol"]["minimum_memory_clock_ratio_vs_op0"]
        )
        day_valid = thermal_valid and memory_clock_valid
        performance_positive = bool(
            benchmark_result["aggregate"]["performance_gate_positive"]
        )
        day1_positive = day_valid and performance_positive
        if not day_valid:
            status = "K3_DAY1_INVALID_TELEMETRY"
            conclusion = "NO_VALID_DAY1_CONCLUSION"
            next_action = "Repeat the unchanged K3 Day-1 measurement under valid telemetry; do not run Day-2, ncu, or reconnection."
            downstream = "NOT_AUTHORIZED_INVALID_DAY1"
        elif day1_positive:
            status = "K3_PROVISIONAL_DAY1_POSITIVE"
            conclusion = "PENDING_SECOND_VALID_CALENDAR_DAY"
            next_action = "Run the unchanged K3 measurement on a later calendar day; ncu and reconnection remain unauthorized until the two-day consistency gate passes."
            downstream = "PENDING_SECOND_VALID_CALENDAR_DAY"
        else:
            status = "K3_CLOSED_DAY1_NOT_POSITIVE"
            conclusion = "K3_CLOSED_BY_FROZEN_DAY1_GATE"
            next_action = "Do not run K3 Day-2, ncu, or reconnection; retain the evidence and advance the rolling backlog decision."
            downstream = "SKIPPED_K3_CLOSED"

        protocol_sha_end = sha256(protocol_path)
        candidate_sha_end = sha256(CANDIDATE)
        if protocol_sha_end != protocol_sha_start:
            raise AssertionError("K3 protocol changed during Day-1")
        if candidate_sha_end != candidate_sha_start:
            raise AssertionError("locked K3 candidate changed during Day-1")
        result = {
            "schema_version": 1,
            "stage": "OP5_K3_GQA_DECODE_ATTENTION_ISOLATED_DAY1",
            "calendar_day": datetime.now().astimezone().date().isoformat(),
            "timestamp": now_iso(),
            "evidence_scope": protocol["host_scope"],
            "protocol": {
                "path": str(protocol_path),
                "sha256_start": protocol_sha_start,
                "sha256_end": protocol_sha_end,
            },
            "locked_candidate": {
                "path": str(CANDIDATE),
                "sha256_start": candidate_sha_start,
                "sha256_end": candidate_sha_end,
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "correctness_precondition": correctness,
            "verified_sources": verified_sources,
            "git_sha": git_sha(),
            "environment": environment,
            "compile": {
                "wrapper_creation_seconds": wrapper_creation_seconds,
                "fullgraph": True,
                "mode": "reduce-overhead",
                "dynamic": False,
                "recompile_limit": selection_config["measurement_protocol"]["torch_dynamo_recompile_limit"],
            },
            "same_day_bandwidth": bandwidth,
            "benchmark": benchmark_result,
            "autotune": tuning,
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
            "bounded_continuation_gate": {
                "aggregate_speedup_threshold": 1.05,
                "minimum_family_speedup_threshold": 0.95,
                "performance_positive": performance_positive,
                "day1_positive": day1_positive,
                "conclusion": conclusion,
            },
            "cross_day_replication": downstream,
            "ncu_tier_b": downstream,
            "reconnection": downstream,
            "next_authorized_action": next_action,
            "status": status,
        }
        result_path = args.result.resolve()
        atomic_write_json(result_path, result)
        receipt = {
            "schema_version": 1,
            "stage": "OP5_K3_GQA_DECODE_ATTENTION_ISOLATED_DAY1",
            "calendar_day": result["calendar_day"],
            "timestamp": now_iso(),
            "protocol": {"path": str(protocol_path), "sha256": protocol_sha_start},
            "locked_candidate": {"path": str(CANDIDATE), "sha256": candidate_sha_end},
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "correctness_precondition": correctness,
            "result": {"path": str(result_path), "sha256": sha256(result_path)},
            "gates": {
                "correctness_precondition": "PASS",
                "protocol_immutable": "PASS",
                "candidate_source_locked": "PASS",
                "same_day_bandwidth": "PASS",
                "three_way_13_case_50_warmup_200_repetition_measurement": "PASS",
                "frozen_autotune_space": "PASS",
                "thermal_or_hardware_throttle_absent": "PASS" if thermal_valid else "FAIL",
                "memory_clock_gate": "PASS" if memory_clock_valid else "FAIL",
                "day1_positive_continuation_gate": "PASS" if day1_positive else "FAIL",
            },
            "performance_outcome": benchmark_result["aggregate"]["classification"],
            "candidate_speedup_vs_best_baseline": benchmark_result["aggregate"]["candidate_speedup_vs_best_baseline"],
            "family_speedups_vs_best_baseline": {
                family: values["candidate_speedup_vs_best_baseline"]
                for family, values in benchmark_result["family_aggregates"].items()
            },
            "bounded_conclusion": conclusion,
            "next_authorized_action": next_action,
            "status": status,
        }
        receipt_path = args.receipt.resolve()
        atomic_write_json(receipt_path, receipt)
        if sha256(protocol_path) != protocol_sha_start:
            raise AssertionError("K3 protocol changed through Day-1 receipt write")
        if sha256(CANDIDATE) != candidate_sha_start:
            raise AssertionError("K3 candidate changed through Day-1 receipt write")
        return receipt
    except BaseException as exc:
        failure_path = ROOT / (
            "results/operator/failures/"
            f"k3_gqa_day1_failure_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}.json"
        )
        failure = {
            "schema_version": 1,
            "stage": "OP5_K3_GQA_DECODE_ATTENTION_ISOLATED_DAY1",
            "failed_at": now_iso(),
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "locked_candidate": str(CANDIDATE),
            "locked_candidate_sha256": sha256(CANDIDATE),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "correctness_precondition": correctness,
            "formal_candidate_performance_started": performance_started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "environment": environment,
            "status": "FAIL",
        }
        atomic_write_json(failure_path, failure)
        raise RuntimeError(f"K3 Day-1 failed; retained {failure_path}") from exc
    finally:
        if telemetry is None:
            monitor.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--correctness-receipt", type=Path, default=DEFAULT_CORRECTNESS_RECEIPT
    )
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = execute(args)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
