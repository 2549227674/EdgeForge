#!/usr/bin/env python3
"""Run owner-authorized K3 three-case Nsight Compute attribution."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import torch

from k3_gqa_attention import candidate_gqa_attention
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso
from run_op5_gqa_selection import analytic_bytes, environment_identity, load_json, make_inputs


ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = datetime.now().astimezone().strftime("%Y%m%d")
DEFAULT_CONFIG = ROOT / "configs/operator/k3_gqa_ncu_override_r3.json"
DEFAULT_RESULT = ROOT / f"results/operator/pool/k3_gqa_ncu_r3_{DATE_TAG}.json"
DEFAULT_RECEIPT = ROOT / f"results/operator/pool/k3_gqa_ncu_r3_receipt_{DATE_TAG}.json"
DEFAULT_TARGET_RECEIPT = ROOT / f"results/operator/pool/k3_gqa_ncu_r3_target_receipt_{DATE_TAG}.json"
DEFAULT_RAW = ROOT / f"results/operator/pool/k3_gqa_ncu_r3_raw_{DATE_TAG}.csv"


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"required file is missing: {path}")
    return load_json(path)


def absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def verify_status_preconditions(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("stage") != "OP5_K3_NCU_R3":
        raise AssertionError("unexpected K3 ncu R3 config identity")
    if config.get("identity_policy", {}).get("sha256_comparison") != "SKIPPED_BY_OWNER_DIRECTION":
        raise AssertionError("K3 ncu R3 SHA policy does not match owner direction")
    sources = config["sources"]
    paths = {name: absolute(value) for name, value in sources.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise AssertionError(f"missing K3 ncu source {name}: {path}")
    correctness = required_json(paths["correctness_receipt_path"])
    day1_result = required_json(paths["day1_result_path"])
    day1_receipt = required_json(paths["day1_receipt_path"])
    r2_receipt = required_json(paths["r2_replication_receipt_path"])
    if correctness.get("status") != "PASS":
        raise AssertionError("K3 correctness prerequisite is not PASS")
    if day1_result.get("status") != "K3_PROVISIONAL_DAY1_POSITIVE":
        raise AssertionError("K3 Day-1 result prerequisite is not positive")
    if day1_receipt.get("status") != "K3_PROVISIONAL_DAY1_POSITIVE":
        raise AssertionError("K3 Day-1 receipt prerequisite is not positive")
    if r2_receipt.get("status") != "SUPERSEDED_UNEXECUTED_BY_OWNER":
        raise AssertionError("K3 R2 replication was not cleanly superseded")
    if r2_receipt.get("formal_replication_started") is not False:
        raise AssertionError("K3 R2 replication unexpectedly started")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("K3 ncu R3 requires the local sm_89 CUDA device")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "day1_result": day1_result,
    }


def build_cases(config: dict[str, Any], day1: dict[str, Any]) -> list[dict[str, Any]]:
    day1_rows = {row["case"]: row for row in day1["benchmark"]["rows"]}
    cases: list[dict[str, Any]] = []
    for frozen in config["formal_cases"]:
        case_name = frozen["case"]
        day1_name = (
            case_name
            if case_name.startswith("sliding_")
            else next(
                name
                for name in day1_rows
                if name.endswith(f"_k{int(frozen['K'])}_d{int(frozen['D'])}")
            )
        )
        row = day1_rows[day1_name]
        cases.append(
            {
                "case": case_name,
                "day1_case": day1_name,
                "family": frozen["family"],
                "k_tokens": int(frozen["K"]),
                "head_dim": int(frozen["D"]),
                "projected_invocations": int(row["projected_invocations"]),
                "day1_candidate_p50_us": float(row["candidate_p50_us"]),
                "analytic_bytes": int(row["analytic_bytes"]),
            }
        )
    if len(cases) != 3:
        raise AssertionError("K3 ncu frozen case count drift")
    return cases


def target_run(args: argparse.Namespace) -> int:
    config = required_json(args.config.resolve())
    verified = verify_status_preconditions(config)
    cases = build_cases(config, verified["day1_result"])
    warmups = int(config["attribution_protocol"]["warmup_candidate_calls_per_case"])
    seed = int(config["attribution_protocol"]["input_seed"])
    inputs: list[tuple[torch.Tensor, ...]] = []
    with torch.inference_mode():
        for index, case in enumerate(cases, 1):
            case_inputs = make_inputs(case, seed, args.device)
            for _ in range(warmups):
                candidate_gqa_attention(*case_inputs)
            torch.cuda.synchronize(args.device)
            inputs.append(case_inputs)
            print(f"K3_NCU_TARGET_WARMUP {index}/{len(cases)}", flush=True)

        outputs: list[torch.Tensor] = []
        start_status = torch.cuda.cudart().cudaProfilerStart()
        try:
            for case_inputs in inputs:
                outputs.append(candidate_gqa_attention(*case_inputs))
            torch.cuda.synchronize(args.device)
        finally:
            stop_status = torch.cuda.cudart().cudaProfilerStop()

    expected_launches = int(config["attribution_protocol"]["expected_profiled_kernel_launches"])
    target_receipt = {
        "schema_version": 1,
        "stage": "OP5_K3_NCU_R3_TARGET",
        "timestamp": now_iso(),
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "protocol": {
            "warmup_candidate_calls_per_case": warmups,
            "profiled_candidate_calls_per_case": 1,
            "expected_profiled_kernel_launches": expected_launches,
            "input_seed": seed,
            "cuda_profiler_start_status": str(start_status),
            "cuda_profiler_stop_status": str(stop_status),
        },
        "cases": cases,
        "output_witnesses": [float(output.float().flatten()[0].item()) for output in outputs],
        "status": "PASS"
        if (
            len(outputs) == len(cases)
            and str(start_status) == "cudaError.success"
            and str(stop_status) == "cudaError.success"
        )
        else "FAIL",
    }
    atomic_write_json(args.target_receipt.resolve(), target_receipt)
    print(f"K3_NCU_TARGET_{target_receipt['status']}", flush=True)
    return 0 if target_receipt["status"] == "PASS" else 2


def parse_ncu_csv(raw_output: str, metrics: list[str]) -> list[dict[str, Any]]:
    lines = raw_output.splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith('"ID",')
        )
    except StopIteration as error:
        raise AssertionError("ncu raw CSV header was not found") from error
    parsed = list(csv.reader(lines[header_index:]))
    if len(parsed) < 3:
        raise AssertionError("ncu raw CSV is incomplete")
    header = parsed[0]
    missing = sorted(set(metrics) - set(header))
    if missing:
        raise AssertionError(f"ncu raw CSV is missing metrics: {missing}")
    rows: list[dict[str, Any]] = []
    for values in parsed[2:]:
        if len(values) != len(header) or not values[0].isdigit():
            continue
        raw = dict(zip(header, values, strict=True))
        metric_values = {name: float(raw[name]) for name in metrics}
        rows.append(
            {
                "ncu_id": int(raw["ID"]),
                "kernel_name": raw["Kernel Name"],
                "grid_size": raw["Grid Size"],
                "block_size": raw["Block Size"],
                "device": raw["Device"],
                "compute_capability": raw["CC"],
                "replayer_passes": int(float(raw["profiler__replayer_passes"])),
                "metrics": metric_values,
            }
        )
    rows.sort(key=lambda row: row["ncu_id"])
    return rows


def attribute(
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    day1: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_per_case = int(
        config["attribution_protocol"]["expected_kernels_per_candidate_call"]
    )
    expected_order = config["attribution_protocol"]["kernel_order_per_case"]
    if len(rows) != len(cases) * expected_per_case:
        raise AssertionError(
            f"ncu returned {len(rows)} launches; expected {len(cases) * expected_per_case}"
        )
    tolerance = float(config["attribution_protocol"]["analytic_tolerance_fraction"])
    ceiling = float(day1["same_day_bandwidth"]["ceiling_gbps_decimal"])
    measurements: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        kernel_rows = rows[
            case_index * expected_per_case : (case_index + 1) * expected_per_case
        ]
        for expected_name, row in zip(expected_order, kernel_rows, strict=True):
            if expected_name not in row["kernel_name"]:
                raise AssertionError(
                    f"K3 ncu launch order mismatch for {case['case']}: {row['kernel_name']}"
                )
            if row["metrics"]["dram__bytes.sum"] <= 0:
                raise AssertionError(f"non-positive K3 DRAM bytes: {case['case']}")
            if row["metrics"]["gpu__time_duration.sum"] <= 0:
                raise AssertionError(f"non-positive K3 kernel duration: {case['case']}")
        measured_bytes = int(
            round(sum(row["metrics"]["dram__bytes.sum"] for row in kernel_rows))
        )
        duration_ns = sum(
            row["metrics"]["gpu__time_duration.sum"] for row in kernel_rows
        )

        def weighted_percent(metric: str) -> float:
            return sum(
                row["metrics"][metric]
                * row["metrics"]["gpu__time_duration.sum"]
                for row in kernel_rows
            ) / duration_ns

        analytic = int(case["analytic_bytes"])
        relative_delta = measured_bytes / analytic - 1.0
        within = abs(relative_delta) <= tolerance
        accounting_bytes = analytic if within else measured_bytes
        seconds = float(case["day1_candidate_p50_us"]) / 1_000_000.0
        measurements.append(
            {
                **case,
                "kernel_rows": kernel_rows,
                "measured_dram_bytes": measured_bytes,
                "analytic_bytes": analytic,
                "measured_to_analytic_ratio": measured_bytes / analytic,
                "relative_delta": relative_delta,
                "within_analytic_tolerance": within,
                "accounting_bytes": accounting_bytes,
                "accounting_source": (
                    "Tier-A analytic (ncu-backed within ±15%)"
                    if within
                    else "Tier-B ncu measured (supersedes analytic)"
                ),
                "ncu_total_duration_ns_diagnostic": duration_ns,
                "duration_weighted_dram_throughput_pct": weighted_percent(
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
                ),
                "duration_weighted_sm_throughput_pct": weighted_percent(
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
                ),
                "day1_candidate_p50_us": case["day1_candidate_p50_us"],
                "corrected_candidate_gbps_decimal": accounting_bytes / seconds / 1e9,
                "corrected_tier_relative_score": accounting_bytes / seconds / 1e9 / ceiling,
            }
        )
    return measurements, {
        "cases": len(measurements),
        "analytic_backed_cases": sum(row["within_analytic_tolerance"] for row in measurements),
        "ncu_superseded_cases": sum(not row["within_analytic_tolerance"] for row in measurements),
        "all_cases_closed": len(measurements) == len(cases),
        "max_absolute_relative_delta": max(abs(row["relative_delta"]) for row in measurements),
        "ncu_duration_use": "diagnostic only; Day-1 CUDA-event p50 remains the performance timing",
    }


def compact_telemetry(summary: dict[str, Any]) -> dict[str, Any]:
    numeric = summary["numeric"]
    return {
        "sample_count": summary["sample_count"],
        "errors": summary["errors"],
        "sm_mhz_p50": numeric["sm_mhz"]["p50"] if numeric["sm_mhz"] else None,
        "memory_mhz_p50": numeric["memory_mhz"]["p50"] if numeric["memory_mhz"] else None,
        "temperature_c_max": numeric["temperature_c"]["max"] if numeric["temperature_c"] else None,
        "power_w_p50": numeric["power_w"]["p50"] if numeric["power_w"] else None,
        "throttle_flags": summary["throttle_flags"],
        "thermal_or_hardware_throttle": summary["thermal_or_hardware_throttle"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = required_json(args.config.resolve())
    verified = verify_status_preconditions(config)
    day1 = verified["day1_result"]
    cases = build_cases(config, day1)
    metrics = list(config["attribution_protocol"]["metrics"])
    expected_launches = int(config["attribution_protocol"]["expected_profiled_kernel_launches"])
    ncu_path = config["toolchain"]["ncu_path"]
    version = subprocess.run(
        [ncu_path, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if config["toolchain"]["expected_version"] not in version:
        raise AssertionError("unexpected ncu version")

    command = [
        ncu_path,
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(metrics),
        "--disable-extra-suffixes",
        "--cache-control",
        config["attribution_protocol"]["cache_control"],
        "--replay-mode",
        config["attribution_protocol"]["replay_mode"],
        "--profile-from-start",
        "off",
        "--launch-count",
        str(expected_launches),
        "--target-processes",
        "application-only",
        sys.executable,
        str(Path(__file__).resolve()),
        "--target",
        "--config",
        str(args.config.resolve()),
        "--target-receipt",
        str(args.target_receipt.resolve()),
        "--device",
        str(args.device),
    ]
    monitor = NvidiaSmiMonitor(float(config["telemetry_gate"]["interval_seconds"]))
    monitor.set_phase("k3_ncu_r3")
    monitor.start()
    monitor.sample_now()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    finally:
        monitor.sample_now()
        telemetry = monitor.stop()
    elapsed = time.monotonic() - started
    raw_output = completed.stdout
    if completed.stderr:
        raw_output += "\n# --- STDERR ---\n" + completed.stderr
    write_text_atomic(args.raw.resolve(), raw_output)
    if completed.returncode != 0:
        raise RuntimeError(f"ncu exited {completed.returncode}; see {args.raw.resolve()}")
    target_receipt = required_json(args.target_receipt.resolve())
    if target_receipt.get("status") != "PASS":
        raise AssertionError("K3 ncu target receipt is not PASS")
    rows = parse_ncu_csv(completed.stdout, metrics)
    measurements, aggregate = attribute(config, cases, rows, day1)
    compact = compact_telemetry(telemetry)
    samples_pass = compact["sample_count"] >= int(config["telemetry_gate"]["minimum_samples"])
    telemetry_pass = samples_pass and not compact["errors"]
    thermal_pass = not compact["thermal_or_hardware_throttle"]
    memory_ratio = (
        float(compact["memory_mhz_p50"])
        / float(config["telemetry_gate"]["op0_memory_clock_mhz_p50"])
        if compact["memory_mhz_p50"] is not None
        else None
    )
    memory_pass = bool(
        memory_ratio is not None
        and memory_ratio
        >= float(config["telemetry_gate"]["minimum_memory_clock_ratio_vs_op0"])
    )
    row_count_pass = len(rows) == expected_launches
    metrics_pass = all(
        all(metric in row["metrics"] for metric in metrics) for row in rows
    )
    status = (
        config["success_policy"]["pass_status"]
        if (
            row_count_pass
            and metrics_pass
            and aggregate["all_cases_closed"]
            and telemetry_pass
            and thermal_pass
            and memory_pass
        )
        else config["success_policy"]["invalid_status"]
    )
    result = {
        "schema_version": 1,
        "stage": "OP5_K3_NCU_R3",
        "timestamp": now_iso(),
        "status": status,
        "revision_disclosure": config["disclosure"],
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "config_path": str(args.config.resolve()),
        "runner_path": str(Path(__file__).resolve()),
        "source_paths": verified["paths"],
        "toolchain": {"version_output": version, "command": command},
        "environment": environment_identity(),
        "target_receipt": target_receipt,
        "raw_ncu_csv_path": str(args.raw.resolve()),
        "elapsed_seconds": elapsed,
        "measurements": measurements,
        "aggregate": aggregate,
        "telemetry": telemetry,
        "compact_telemetry": compact,
        "gates": {
            "correctness_precondition": "PASS",
            "positive_day1_precondition": "PASS",
            "replication_waiver_recorded": "PASS",
            "exactly_nine_profiled_kernels": "PASS" if row_count_pass else "FAIL",
            "required_metrics": "PASS" if metrics_pass else "FAIL",
            "three_case_byte_accounts_closed": "PASS" if aggregate["all_cases_closed"] else "FAIL",
            "telemetry_samples": "PASS" if telemetry_pass else "FAIL",
            "memory_clock": "PASS" if memory_pass else "FAIL",
            "thermal_hardware_throttle_absent": "PASS" if thermal_pass else "FAIL",
        },
        "memory_clock_ratio_vs_op0": memory_ratio,
        "reconnection": (
            "AUTHORIZED_AS_NEXT_STAGE_BY_R3_OWNER_OVERRIDE"
            if status == config["success_policy"]["pass_status"]
            else "NOT_AUTHORIZED"
        ),
        "next_authorized_action": (
            config["success_policy"]["on_pass"]
            if status == config["success_policy"]["pass_status"]
            else "Do not reconnect; resolve the invalid ncu gate."
        ),
    }
    atomic_write_json(args.result.resolve(), result)
    receipt = {
        "schema_version": 1,
        "stage": "OP5_K3_NCU_R3",
        "timestamp": now_iso(),
        "status": status,
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "config_path": str(args.config.resolve()),
        "runner_path": str(Path(__file__).resolve()),
        "result_path": str(args.result.resolve()),
        "raw_ncu_csv_path": str(args.raw.resolve()),
        "gates": result["gates"],
        "aggregate": aggregate,
        "case_accounts": [
            {
                "case": row["case"],
                "analytic_bytes": row["analytic_bytes"],
                "measured_dram_bytes": row["measured_dram_bytes"],
                "measured_to_analytic_ratio": row["measured_to_analytic_ratio"],
                "accounting_source": row["accounting_source"],
            }
            for row in measurements
        ],
        "reconnection": result["reconnection"],
        "next_authorized_action": result["next_authorized_action"],
    }
    atomic_write_json(args.receipt.resolve(), receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--target-receipt", type=Path, default=DEFAULT_TARGET_RECEIPT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--target", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.target:
            return target_run(args)
        receipt = run(args)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        failure_path = ROOT / (
            "results/operator/failures/"
            f"k3_gqa_ncu_r3_failure_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}.json"
        )
        failure = {
            "schema_version": 1,
            "stage": "OP5_K3_NCU_R3",
            "failed_at": now_iso(),
            "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
            "config_path": str(args.config.resolve()),
            "runner_path": str(Path(__file__).resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "status": "FAIL",
        }
        atomic_write_json(failure_path, failure)
        traceback.print_exc()
        print(f"failure retained: {failure_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
