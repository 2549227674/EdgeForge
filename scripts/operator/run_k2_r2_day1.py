#!/usr/bin/env python3
"""Run the bounded K2-R2 Day-1 isolated performance gate."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import torch

from k2_nf4_gemv import autotune_choices
from load_smoke_nf4 import memory_snapshot
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, sha256
from run_k2_day import benchmark_day
from run_k2_r2_correctness import LOCKED_CANDIDATE, validate_choices, verify_protocol
from run_op5_selection import (
    environment_identity,
    load_formal_carrier,
    load_json,
    prepare_modules,
    run_same_day_bandwidth,
)


ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = datetime.now().astimezone().strftime("%Y%m%d")
DEFAULT_PROTOCOL = ROOT / "configs/operator/k2_nf4_decode_gemv_prereg_r2.json"
DEFAULT_CORRECTNESS_RECEIPT = (
    ROOT / f"results/operator/pool/k2_r2_correctness_receipt_{DATE_TAG}.json"
)
DEFAULT_RESULT = ROOT / f"results/operator/pool/k2_r2_day1_{DATE_TAG}.json"
DEFAULT_RECEIPT = ROOT / f"results/operator/pool/k2_r2_day1_receipt_{DATE_TAG}.json"


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def verify_correctness_receipt(
    receipt_path: Path,
    protocol_path: Path,
    protocol_sha: str,
    candidate_sha: str,
) -> dict[str, Any]:
    if not receipt_path.is_file():
        raise AssertionError("K2-R2 correctness receipt is missing")
    receipt = load_json(receipt_path)
    if receipt.get("stage") != "OP5_K2_R2_CORRECTNESS" or receipt.get("status") != "PASS":
        raise AssertionError("K2-R2 correctness receipt is not PASS")
    if receipt.get("protocol") != {"path": str(protocol_path), "sha256": protocol_sha}:
        raise AssertionError("K2-R2 correctness receipt protocol identity drift")
    locked = receipt.get("locked_candidate", {})
    if locked.get("path") != str(LOCKED_CANDIDATE) or locked.get("sha256") != candidate_sha:
        raise AssertionError("K2-R2 correctness receipt candidate identity drift")
    required_gates = (
        "protocol_immutable",
        "candidate_source_locked",
        "source_and_git_identity",
        "semantic_microcase",
        "all_343_actual_weights_vs_single_primary_oracle",
        "fresh_14_x_4_x_3_holdout_vs_single_primary_oracle",
        "source_tensors_unmodified",
        "fallback_rejection_matrix",
        "frozen_autotune_space",
    )
    if any(receipt.get("gates", {}).get(name) != "PASS" for name in required_gates):
        raise AssertionError("K2-R2 correctness receipt has a non-PASS prerequisite gate")

    result_pointer = receipt.get("result", {})
    result_path = absolute(str(result_pointer.get("path", "")))
    if not result_path.is_file() or sha256(result_path) != result_pointer.get("sha256"):
        raise AssertionError("K2-R2 correctness result identity drift")
    result = load_json(result_path)
    if result.get("stage") != "OP5_K2_R2_CORRECTNESS" or result.get("status") != "PASS":
        raise AssertionError("K2-R2 correctness result is not PASS")
    if result.get("formal_candidate_performance_started") is not False:
        raise AssertionError("K2-R2 correctness result unexpectedly includes performance")
    summary = result["correctness"]["gating_candidate_vs_primary_reference"]
    actual = summary["actual_weights"]
    holdout = summary["representative_holdout"]
    if (actual["cases"], actual["allclose_cases"]) != (343, 343):
        raise AssertionError("K2-R2 343-weight correctness coverage drift")
    if (holdout["cases"], holdout["allclose_cases"]) != (168, 168):
        raise AssertionError("K2-R2 holdout correctness coverage drift")
    return {
        "path": str(receipt_path),
        "sha256": sha256(receipt_path),
        "result": {"path": str(result_path), "sha256": sha256(result_path)},
        "status": "PASS",
    }


def verify_bandwidth_protocol(protocol: dict[str, Any], selection: dict[str, Any]) -> None:
    declared = protocol["analytic_bytes"]["same_day_ceiling_protocol"]
    inherited = selection["analytic_bytes"]["same_day_ceiling"]
    for field in (
        "operation",
        "array_size_mib",
        "warmup_repetitions",
        "measured_repetitions",
        "block_size",
    ):
        if declared[field] != inherited[field]:
            raise AssertionError(f"K2-R2 same-day bandwidth protocol drift: {field}")
    expected_bytes = 3 * int(declared["array_size_mib"]) * 1024 * 1024
    if int(inherited["analytic_bytes_per_repetition"]) != expected_bytes:
        raise AssertionError("same-day bandwidth analytic byte declaration drift")


def benchmark_protocol_adapter(protocol: dict[str, Any]) -> dict[str, Any]:
    """Adapt only the legacy helper's baseline display schema."""
    adapted = copy.deepcopy(protocol)
    for name in ("eager", "compile", "candidate"):
        definition = adapted["baselines"][name]
        if isinstance(definition, str):
            adapted["baselines"][name] = {"definition": definition}
    return adapted


def execute(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol_sha_start = sha256(protocol_path)
    candidate_sha_start = sha256(LOCKED_CANDIDATE)
    verified = verify_protocol(protocol_path, protocol_sha_start)
    protocol = verified["protocol"]
    candidate_sha_frozen = protocol["source_identity"]["candidate_source"]["sha256"]
    if candidate_sha_start != candidate_sha_frozen:
        raise AssertionError("locked K2-R2 candidate changed before Day-1")
    correctness = verify_correctness_receipt(
        args.correctness_receipt.resolve(),
        protocol_path,
        protocol_sha_start,
        candidate_sha_start,
    )
    verify_bandwidth_protocol(protocol, verified["selection"])

    environment = environment_identity()
    monitor = NvidiaSmiMonitor(
        interval_seconds=float(protocol["measurement_protocol"]["gpu_telemetry_interval_seconds"])
    )
    monitor.start()
    telemetry: dict[str, Any] | None = None
    performance_started = False
    try:
        log("K2-R2 Day-1: measuring the frozen same-day bandwidth ceiling")
        bandwidth = run_same_day_bandwidth(verified["selection"], monitor, args.device)
        log("K2-R2 Day-1: loading the formal C02 carrier")
        model, model_load = load_formal_carrier(verified["selection"], args.device)
        prepared, layout = prepare_modules(
            model,
            verified["selection"],
            verified["decode_rows"],
            args.device,
        )
        performance_started = True
        benchmark = benchmark_day(
            benchmark_protocol_adapter(protocol),
            prepared,
            verified["decode_rows"],
            monitor,
            float(bandwidth["ceiling_gbps_decimal"]),
            args.device,
        )
        choices = validate_choices(protocol, prepared)
        if choices != autotune_choices():
            raise AssertionError("K2-R2 autotune choices changed during Day-1 validation")
        telemetry = monitor.stop()
        measurement_telemetry = monitor.summary_for_phase("k2_day_measurement")
        bandwidth_telemetry = monitor.summary_for_phase("op5_same_day_bandwidth")
        memory_summary = measurement_telemetry["numeric"]["memory_mhz"]
        if memory_summary is None:
            raise AssertionError("K2-R2 Day-1 has no memory-clock witness")
        memory_clock_ratio = float(memory_summary["p50"]) / float(
            protocol["measurement_protocol"]["op0_memory_clock_mhz_p50"]
        )
        thermal_valid = not telemetry["thermal_or_hardware_throttle"]
        memory_clock_valid = memory_clock_ratio >= float(
            protocol["measurement_protocol"]["minimum_memory_clock_ratio_vs_op0"]
        )
        day_valid = thermal_valid and memory_clock_valid
        classification = benchmark["aggregate"]["classification"]
        day1_positive = day_valid and classification == "POSITIVE"

        if not day_valid:
            status = "DAY1_INVALID_TELEMETRY"
            conclusion = "NO_VALID_DAY1_CONCLUSION"
            next_action = "Repeat an unchanged R2 Day-1 measurement under valid telemetry; do not run Day-2, ncu, or reconnection."
            downstream = "NOT_AUTHORIZED_INVALID_DAY1"
        elif day1_positive:
            status = "PROVISIONAL_DAY1_POSITIVE"
            conclusion = "PENDING_SECOND_VALID_CALENDAR_DAY"
            next_action = "Run the unchanged second valid measurement on a later calendar day; ncu and reconnection remain unauthorized until the two-day gate passes."
            downstream = "PENDING_SECOND_VALID_CALENDAR_DAY"
        else:
            status = "K2_R2_CLOSED_DAY1_NOT_POSITIVE"
            conclusion = "K2_R2_CLOSED_BY_FROZEN_DAY1_GATE"
            next_action = "Do not run K2 Day-2, ncu, or reconnection; move the OP5 backlog decision to GQA decode attention."
            downstream = "SKIPPED_K2_R2_CLOSED"

        protocol_sha_end = sha256(protocol_path)
        candidate_sha_end = sha256(LOCKED_CANDIDATE)
        if protocol_sha_end != protocol_sha_start:
            raise AssertionError("K2-R2 protocol changed during Day-1")
        if candidate_sha_end != candidate_sha_start:
            raise AssertionError("locked K2-R2 candidate changed during Day-1")

        result = {
            "schema_version": 1,
            "stage": "OP5_K2_R2_ISOLATED_DAY1",
            "calendar_day": datetime.now().astimezone().date().isoformat(),
            "timestamp": now_iso(),
            "evidence_scope": protocol["host_scope"],
            "adaptive_revision_disclosure": protocol["revision_disclosure"],
            "protocol": {
                "path": str(protocol_path),
                "sha256_start": protocol_sha_start,
                "sha256_end": protocol_sha_end,
            },
            "locked_candidate": {
                "path": str(LOCKED_CANDIDATE),
                "sha256_start": candidate_sha_start,
                "sha256_end": candidate_sha_end,
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "correctness_precondition": correctness,
            "verified_sources": verified["sources"],
            "git_sha": verified["git_sha"],
            "environment": environment,
            "same_day_bandwidth": bandwidth,
            "model_load": model_load,
            "layout": layout,
            "benchmark": benchmark,
            "autotune_choices": choices,
            "telemetry": telemetry,
            "bandwidth_telemetry": bandwidth_telemetry,
            "measurement_telemetry": measurement_telemetry,
            "telemetry_gates": {
                "thermal_or_hardware_throttle_absent": thermal_valid,
                "memory_clock_ratio_vs_op0": memory_clock_ratio,
                "minimum_memory_clock_ratio": protocol["measurement_protocol"]["minimum_memory_clock_ratio_vs_op0"],
                "memory_clock_gate": memory_clock_valid,
            },
            "memory_final": memory_snapshot(torch, args.device),
            "bounded_continuation_gate": {
                "day_valid": day_valid,
                "performance_classification": classification,
                "aggregate_speedup_threshold": 1.05,
                "mlp_minimum_speedup_threshold": 0.95,
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
            "stage": "OP5_K2_R2_ISOLATED_DAY1",
            "calendar_day": result["calendar_day"],
            "timestamp": now_iso(),
            "protocol": {"path": str(protocol_path), "sha256": protocol_sha_start},
            "locked_candidate": {"path": str(LOCKED_CANDIDATE), "sha256": candidate_sha_end},
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "correctness_precondition": correctness,
            "result": {"path": str(result_path), "sha256": sha256(result_path)},
            "gates": {
                "r2_correctness_precondition": "PASS",
                "protocol_immutable": "PASS",
                "candidate_source_locked": "PASS",
                "formal_c02_full_hash": model_load["integrity"]["status"],
                "all_343_weights_and_14_variants": "PASS",
                "same_day_bandwidth": "PASS",
                "three_way_200_full_sequence_measurement": "PASS",
                "frozen_autotune_space": "PASS",
                "thermal_or_hardware_throttle_absent": "PASS" if thermal_valid else "FAIL",
                "memory_clock_gate": "PASS" if memory_clock_valid else "FAIL",
                "day1_positive_continuation_gate": "PASS" if day1_positive else "FAIL",
            },
            "performance_outcome": classification,
            "candidate_speedup_vs_best_baseline": benchmark["aggregate"]["candidate_speedup_vs_best_baseline"],
            "mlp_speedups_vs_best_baseline": benchmark["aggregate"]["mlp_speedups_vs_best_baseline"],
            "bounded_conclusion": conclusion,
            "next_authorized_action": next_action,
            "status": status,
        }
        receipt_path = args.receipt.resolve()
        atomic_write_json(receipt_path, receipt)
        if sha256(protocol_path) != protocol_sha_start:
            raise AssertionError("K2-R2 protocol changed through Day-1 receipt write")
        if sha256(LOCKED_CANDIDATE) != candidate_sha_start:
            raise AssertionError("K2-R2 candidate changed through Day-1 receipt write")
        return receipt
    except BaseException as exc:
        failure_path = ROOT / (
            "results/operator/failures/"
            f"k2_r2_day1_failure_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}.json"
        )
        failure = {
            "schema_version": 1,
            "stage": "OP5_K2_R2_ISOLATED_DAY1",
            "failed_at": now_iso(),
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "locked_candidate": str(LOCKED_CANDIDATE),
            "locked_candidate_sha256": sha256(LOCKED_CANDIDATE),
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
        raise RuntimeError(f"K2-R2 Day-1 failed; retained {failure_path}") from exc
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
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
