#!/usr/bin/env python3
"""Validate and summarize the single formal OPE5-05 A55 FP16 GEMV segment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SCOPE_SHA256 = "8f3a2c6949980d04096a1e995ff2aafae653aed031fdf0960b1261c86059d1f4"
PREREG_SHA256 = "98a3fe20f7982f7fb58fca01f1fe23fa483f4ca11f387d24767805f1e7fb43ce"
SOURCE_LOCK_SHA256 = "40fe76d2ce6f411247f88be4efc2494ed08055ef1c7beb0689a03842c9185483"
A76_RECEIPT_SHA256 = "221e26f63cfde04362ae6a718f9c85ea40331b4cca084bae385374ef8213421a"
A76_STATUS = "VALID_OPE5_02_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_4XA76_FP16_GEMV"
OPE0_GBPS = 23.1951846685
ANALYTIC_BYTES = 52_459_520
OPS = 52_428_800
A76_MAX_KHZ = 2_352_000
A55_MAX_KHZ = 1_800_000
VARIANTS = ("scalar_fp32", "autovec_o3", "neon_fp16_f32acc")
DISTRIBUTIONS = (
    "quarter_grid_exact",
    "unit_normal_clipped_3",
    "wide_normal_std4_clipped_8",
)
SEEDS = (20260924, 20260925, 20260926)
EVENTS = ("cycles:u", "instructions:u", "l2d_cache_refill:u", "bus_access:u")
OWNER_INSTRUCTION = "不需要复测，取消同日复测，只测一次"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, value = line.split("=", 1)
            values[key] = value
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_perf_stat(path: Path) -> dict[str, int]:
    observed: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#") or len(row) < 3:
                continue
            event = row[2].strip()
            if event not in EVENTS:
                continue
            value = row[0].strip()
            if value in {"", "<not counted>", "<not supported>"}:
                raise AssertionError((path, event, value))
            observed[event] = int(value)
    if set(observed) != set(EVENTS):
        raise AssertionError((path, observed))
    return observed


def close(left: float, right: float, rel: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=1e-9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    segment = (repo / args.segment).resolve()
    output = (repo / args.output).resolve()
    scope_path = repo / "configs/operator/board/ope5_a55_fp16_gemv_scope.json"
    prereg_path = repo / "configs/operator/board/ope5_a55_fp16_gemv_prereg.json"
    source_lock_path = (
        repo
        / "results/operator/board/ope5/ope5_a55_fp16_gemv_source_lock_20260921.json"
    )
    prelock_path = (
        repo
        / "results/operator/board/ope5/ope5_a55_fp16_gemv_prelock_receipt_20260921.json"
    )
    a76_receipt_path = (
        repo / "results/operator/board/ope5/ope5_fp16_gemv_receipt_20260921.json"
    )
    scope = read_json(scope_path)
    prereg = read_json(prereg_path)
    source_lock = read_json(source_lock_path)
    prelock = read_json(prelock_path)
    a76 = read_json(a76_receipt_path)
    controller = read_json(segment / "controller_meta.json")
    run_meta = read_env(segment / "run_meta.env")
    checks: dict[str, bool] = {}

    checks["scope_frozen_identity"] = (
        sha256(scope_path) == SCOPE_SHA256
        and scope["status"] == "PASS_OPE5_05_FOUR_QUESTIONS_PREREG_AUTHORIZED"
        and scope["board_execution_started"] is False
    )
    checks["prereg_frozen_identity"] = (
        sha256(prereg_path) == PREREG_SHA256
        and prereg["status"] == "PREREGISTERED_FROZEN_NOT_RUN"
        and prereg["formal_execution_started"] is False
        and prereg["selection"]["scope_sha256"] == SCOPE_SHA256
    )
    checks["prereg_single_segment_owner_waiver"] = (
        prereg["measurement_protocol"]["formal_segments"] == 1
        and prereg["measurement_protocol"]["owner_instruction"]
        == OWNER_INSTRUCTION
        and prereg["measurement_protocol"]["replication_status"]
        == "WAIVED_BY_OWNER_NOT_SATISFIED"
        and "no automatic retry" in prereg["measurement_protocol"]["stability_gate"]
        and "No separate performance smoke" in prereg["measurement_protocol"]["smoke_policy"]
    )
    checks["prelock_passed_without_benchmark"] = (
        prelock["status"] == "PASS_A55_PRELOCK_NO_BENCHMARK_EXECUTION"
        and prelock["check_summary"] == {"failed": [], "passed": 13, "total": 13}
        and prelock["benchmark_execution_count"] == 0
        and prelock["formal_segment_count"] == 0
        and sha256(prelock_path)
        == source_lock["reference_files"]["prelock_audit_receipt"]["sha256"]
    )
    checks["source_lock_precedes_formal_execution"] = (
        sha256(source_lock_path) == SOURCE_LOCK_SHA256
        and source_lock["status"] == "LOCKED_BEFORE_FORMAL_EXECUTION"
        and source_lock["formal_execution_started"] is False
        and source_lock["board_benchmark_process_started"] is False
        and source_lock["recorded_at"] < controller["controller_before"]
    )
    checks["source_lock_frozen_identity"] = (
        source_lock["scope"]["sha256"] == SCOPE_SHA256
        and source_lock["preregistration"]["sha256"] == PREREG_SHA256
        and run_meta["prereg_sha256"] == PREREG_SHA256
        and run_meta["source_lock_sha256"] == SOURCE_LOCK_SHA256
    )

    base_source = repo / "scripts/operator/board/ope5_fp16_gemv_bench.cpp"
    candidate_source = repo / "scripts/operator/board/ope5_a55_fp16_gemv_bench.cpp"
    expected_candidate = (
        base_source.read_text(encoding="utf-8")
        .replace("constexpr int kFirstCpu = 4;", "constexpr int kFirstCpu = 0;")
        .replace(
            "failed to pin main worker to CPU4",
            "failed to pin main worker to CPU0",
        )
        .replace("OPE5_FP16_GEMV", "OPE5_A55_FP16_GEMV")
    )
    checks["source_exact_allowlist_derivative"] = (
        candidate_source.read_text(encoding="utf-8") == expected_candidate
        and source_lock["source_derivation"]["exact_allowlist_match"] is True
    )

    locked_files: dict[str, dict[str, str]] = {}
    for section in ("implementation_files", "runtime_dependencies", "reference_files"):
        for key, item in source_lock[section].items():
            path = repo / item["path"]
            observed = sha256(path)
            locked_files[key] = {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "observed_sha256": observed,
            }
    checks["all_local_locked_files_match"] = all(
        item["expected_sha256"] == item["observed_sha256"]
        for item in locked_files.values()
    )
    board_hashes = read_tsv(segment / "implementation_hash_observed.tsv")
    checks["all_board_locked_files_match"] = (
        len(board_hashes) == len(locked_files)
        and all(row["match"] == "yes" for row in board_hashes)
        and {row["path"] for row in board_hashes}
        == {item["path"] for item in locked_files.values()}
    )
    checks["immutable_a76_anchor_identity"] = (
        sha256(a76_receipt_path) == A76_RECEIPT_SHA256
        and a76["status"] == A76_STATUS
        and a76["measurement_valid"] is True
        and a76["evidence_strength"]["formal_segments"] == 1
        and a76["evidence_strength"]["replication_status"]
        == "WAIVED_BY_OWNER_NOT_SATISFIED"
    )
    checks["formal_process_exit_and_restore_zero"] = (
        controller["ssh_exit_code"] == 0
        and run_meta["run_exit_code"] == "0"
        and run_meta["governor_restore_exit_code"] == "0"
    )
    segment_dirs = [
        path
        for path in segment.parent.glob("ope5_a55_fp16_gemv_segment*")
        if path.is_dir()
    ]
    checks["exactly_one_formal_segment_no_retest"] = (
        controller["formal_segment_ordinal"] == 1
        and controller["additional_segment_scheduled"] is False
        and controller["same_day_retest_scheduled"] is False
        and controller["instability_retry_authorized"] is False
        and run_meta["segment_label"] == "segment1"
        and run_meta["formal_segments"] == "1"
        and source_lock["measurement_identity"]["same_day_retest_scheduled"]
        is False
        and len(segment_dirs) == 1
        and segment_dirs[0] == segment
    )
    checks["no_a76_rerun_or_performance_smoke"] = (
        controller["a76_rerun_performed"] is False
        and controller["performance_smoke_performed"] is False
        and source_lock["measurement_identity"]["a76_rerun_scheduled"] is False
        and source_lock["measurement_identity"]["performance_smoke_scheduled"]
        is False
        and not list(segment.glob("*smoke*"))
    )
    checks["four_a55_fixed_topology"] = (
        run_meta["primary_cpus"] == "0,1,2,3"
        and run_meta["threads"] == "4"
        and run_meta["policy0_expected_max_khz"] == "1800000"
        and prereg["formal_shape"]["cpu_cores"] == [0, 1, 2, 3]
        and prereg["formal_shape"]["threads"] == 4
    )
    checks["clock_domains_recorded"] = (
        controller["controller_before"] < controller["controller_after"]
        and controller["controller_elapsed_seconds"] > 0
        and int(run_meta["board_end_monotonic_raw_ns"])
        > int(run_meta["board_start_monotonic_raw_ns"])
        and "not subtracted" in controller["clock_domain_note"]
    )

    correctness_rows = read_csv(segment / "correctness.csv")
    observed_comparisons = sum(int(row["elements"]) for row in correctness_rows)
    correctness_by_variant: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        rows = [row for row in correctness_rows if row["variant"] == variant]
        correctness_by_variant[variant] = {
            "cases": len(rows),
            "comparisons": sum(int(row["elements"]) for row in rows),
            "passing_cases": sum(row["pass"] == "1" for row in rows),
            "mismatch_elements": sum(int(row["mismatch_count"]) for row in rows),
            "max_abs_error": max(float(row["max_abs_error"]) for row in rows),
            "max_rel_error": max(float(row["max_rel_error"]) for row in rows),
        }
    checks["correctness_69120_of_69120"] = (
        len(correctness_rows) == 27
        and observed_comparisons == 69_120
        and all(row["pass"] == "1" for row in correctness_rows)
        and {row["variant"] for row in correctness_rows} == set(VARIANTS)
        and {row["distribution"] for row in correctness_rows}
        == set(DISTRIBUTIONS)
        and {int(row["seed"]) for row in correctness_rows} == set(SEEDS)
        and (segment / "correctness_gate.txt").read_text(encoding="utf-8").strip()
        == "PASS_69120_OF_69120"
    )
    checks["correctness_structural_assertions"] = all(
        row["all_finite"] == "1"
        and row["input_unchanged"] == "1"
        and row["weight_unchanged"] == "1"
        and row["output_shape_pass"] == "1"
        and row["partition_coverage_pass"] == "1"
        and row["affinity_pass"] == "1"
        for row in correctness_rows
    )
    checks["correctness_frozen_tolerances"] = all(
        (
            row["distribution"] == "quarter_grid_exact"
            and float(row["atol"]) == 0.0
            and float(row["rtol"]) == 0.0
            and float(row["max_abs_error"]) == 0.0
            and float(row["max_rel_error"]) == 0.0
        )
        or (
            row["distribution"] != "quarter_grid_exact"
            and float(row["atol"]) == 0.05
            and float(row["rtol"]) == 0.001
        )
        for row in correctness_rows
    )

    scalar_disassembly = (segment / "disassembly_scalar_fp32.txt").read_text(
        encoding="utf-8"
    ).lower()
    candidate_disassembly = (
        segment / "disassembly_neon_fp16_f32acc.txt"
    ).read_text(encoding="utf-8").lower()
    codegen = {
        "scalar_vector_fmla_matches": len(
            re.findall(r"\bfmla\s+v", scalar_disassembly)
        ),
        "candidate_fcvtl_matches": len(
            re.findall(r"\bfcvtl2?\s+v", candidate_disassembly)
        ),
        "candidate_vector_fmla_matches": len(
            re.findall(r"\bfmla\s+v", candidate_disassembly)
        ),
    }
    checks["codegen_gates"] = (
        codegen["scalar_vector_fmla_matches"] == 0
        and codegen["candidate_fcvtl_matches"] > 0
        and codegen["candidate_vector_fmla_matches"] > 0
        and (segment / "disassembly_autovec_o3.txt").stat().st_size > 0
        and (segment / "vectorization.log").is_file()
    )

    timing_rows = read_csv(segment / "timings.csv")
    checks["timing_population_complete"] = (
        len(timing_rows) == 3
        and {row["variant"] for row in timing_rows} == set(VARIANTS)
        and all(row["scope"] == "formal_invocation" for row in timing_rows)
        and all(
            (int(row["m"]), int(row["k"]), int(row["n"]))
            == (1, 10240, 2560)
            for row in timing_rows
        )
        and all(int(row["threads"]) == 4 for row in timing_rows)
        and all(int(row["repetitions"]) >= 200 for row in timing_rows)
        and all(int(row["analytic_bytes"]) == ANALYTIC_BYTES for row in timing_rows)
        and all(int(row["ops"]) == OPS for row in timing_rows)
        and all(int(row["permutations_used"]) == 6 for row in timing_rows)
        and all(
            int(row["working_set_weight_bytes"]) == 52_428_800
            for row in timing_rows
        )
        and all(
            row["affinity_pass"] == "1"
            and row["partition_coverage_pass"] == "1"
            for row in timing_rows
        )
        and all(
            int(row["repetitions"]) * float(row["p50_us"]) / 1000.0
            >= 3000.0
            for row in timing_rows
        )
    )
    checks["timing_stability_gate"] = (
        all(float(row["iqr_over_p50"]) <= 0.30 for row in timing_rows)
        and (segment / "timing_gate.txt").read_text(encoding="utf-8").strip()
        == "PASS_ALL_THREE_VARIANTS_STABLE"
    )
    timing: dict[str, dict[str, Any]] = {}
    arithmetic_pass = True
    for row in timing_rows:
        variant = row["variant"]
        p50_us = float(row["p50_us"])
        expected_gops = OPS / (p50_us * 1e-6) / 1e9
        expected_gbps = ANALYTIC_BYTES / (p50_us * 1e-6) / 1e9
        arithmetic_pass &= close(float(row["measured_gops"]), expected_gops)
        arithmetic_pass &= close(float(row["measured_gbps"]), expected_gbps)
        arithmetic_pass &= close(
            float(row["memory_relative_to_ope0"]), expected_gbps / OPE0_GBPS
        )
        timing[variant] = {
            "repetitions": int(row["repetitions"]),
            "p25_us": float(row["p25_us"]),
            "p50_us": p50_us,
            "p75_us": float(row["p75_us"]),
            "iqr_us": float(row["iqr_us"]),
            "iqr_over_p50": float(row["iqr_over_p50"]),
            "measured_gops": float(row["measured_gops"]),
            "analytic_gbps": float(row["measured_gbps"]),
            "shared_ope0_relative_context": float(
                row["memory_relative_to_ope0"]
            ),
        }
    checks["timing_arithmetic_reproduces"] = arithmetic_pass
    speedup_vs_autovec = (
        timing["autovec_o3"]["p50_us"]
        / timing["neon_fp16_f32acc"]["p50_us"]
    )
    speedup_vs_scalar = (
        timing["scalar_fp32"]["p50_us"]
        / timing["neon_fp16_f32acc"]["p50_us"]
    )
    candidate_classification = (
        "POSITIVE"
        if speedup_vs_autovec >= 1.05
        else "PARITY"
        if speedup_vs_autovec >= 0.95
        else "NEGATIVE"
    )

    governor_rows = read_tsv(segment / "governor_state.tsv")
    governor_by_event = {row["event"]: row for row in governor_rows}
    before = governor_by_event["before"]
    after_set = governor_by_event["after_set"]
    after_restore = governor_by_event["after_restore"]
    checks["policy0_performance_and_restored"] = (
        after_set["policy0_governor"] == "performance"
        and after_restore["policy0_governor"] == before["policy0_governor"]
    )
    checks["comparison_and_npu_governors_unchanged"] = all(
        before[key] == after_set[key] == after_restore[key]
        for key in (
            "policy4_governor",
            "policy6_governor",
            "npu_governor",
        )
    )

    telemetry_rows = read_csv(segment / "telemetry_1hz.csv")
    timed_rows = [
        row
        for row in telemetry_rows
        if row["phase"] == "formal_interleaved_timing"
        or row["phase"].startswith("tier_b_")
    ]
    frequency_ratios = [
        int(row["policy0_khz"]) / int(row["policy0_max_khz"])
        for row in timed_rows
    ]
    checks["a55_policy0_frequency_gate"] = bool(timed_rows) and (
        min(frequency_ratios) >= 0.95
    )
    checks["timed_window_governor_gate"] = all(
        row["policy0_governor"] == "performance" for row in timed_rows
    )
    checks["telemetry_read_only_governors_unchanged"] = all(
        row["policy4_governor"] == before["policy4_governor"]
        and row["policy6_governor"] == before["policy6_governor"]
        and row["npu_governor"] == before["npu_governor"]
        for row in telemetry_rows
    )
    uptime_values = [float(row["uptime_s"]) for row in telemetry_rows]
    telemetry_gaps = [
        right - left for left, right in zip(uptime_values, uptime_values[1:])
    ]
    checks["telemetry_approximately_1hz"] = (
        len(telemetry_rows) >= 2 and max(telemetry_gaps) <= 1.25
    )

    trip_rows = read_tsv(segment / "thermal_trip_points.tsv")
    lowest_trip_by_zone: dict[str, int] = {}
    zone_name: dict[str, str] = {}
    for row in trip_rows:
        zone = row["zone"]
        temperature = int(row["trip_temp_mC"])
        zone_name[zone] = row["type"]
        lowest_trip_by_zone[zone] = min(
            lowest_trip_by_zone.get(zone, temperature), temperature
        )
    max_temperature_by_zone: dict[str, int] = {}
    for zone in lowest_trip_by_zone:
        prefix = f"thermal_zone{zone}_"
        columns = [column for column in timed_rows[0] if column.startswith(prefix)]
        if len(columns) != 1:
            raise AssertionError((zone, columns))
        max_temperature_by_zone[zone] = max(
            int(row[columns[0]]) for row in timed_rows
        )
    thermal_headroom = {
        zone: lowest_trip_by_zone[zone] - max_temperature_by_zone[zone]
        for zone in lowest_trip_by_zone
    }
    checks["thermal_gate"] = all(value > 0 for value in thermal_headroom.values())

    attempt_rows = read_tsv(segment / "perf_event_attempts.tsv")
    perf: dict[str, dict[str, Any]] = {}
    perf_contract_pass = (
        len(attempt_rows) == 3
        and {row["variant"] for row in attempt_rows} == set(VARIANTS)
        and all(row["events"].split(",") == list(EVENTS) for row in attempt_rows)
        and all(row["exit_code"] == "0" for row in attempt_rows)
    )
    for variant in VARIANTS:
        counts = parse_perf_stat(segment / f"perf_{variant}_invocation.stat.csv")
        bench_rows = read_csv(segment / f"perf_{variant}_invocation.bench.csv")
        if len(bench_rows) != 1:
            raise AssertionError((variant, bench_rows))
        row = bench_rows[0]
        invocations = int(row["invocations"])
        per_invocation = {event: counts[event] / invocations for event in EVENTS}
        perf[variant] = {
            "invocations": invocations,
            "elapsed_ms": float(row["elapsed_ms"]),
            "counts": counts,
            "per_invocation": per_invocation,
            "ipc": counts["instructions:u"] / counts["cycles:u"],
        }
        perf_contract_pass &= (
            row["scope"] == "perf_invocation"
            and row["variant"] == variant
            and invocations >= 200
            and float(row["elapsed_ms"]) >= 3000.0
            and int(row["analytic_bytes_per_invocation"]) == ANALYTIC_BYTES
            and int(row["ops_per_invocation"]) == OPS
            and int(row["threads"]) == 4
            and int(row["working_set_weight_bytes"]) == 52_428_800
            and row["affinity_pass"] == "1"
            and row["partition_coverage_pass"] == "1"
            and (segment / f"perf_{variant}_invocation.exit_code.txt")
            .read_text(encoding="utf-8")
            .strip()
            == "0"
        )
    checks["tier_b_all_events_numeric"] = (
        perf_contract_pass
        and (segment / "tier_b_gate.txt").read_text(encoding="utf-8").strip()
        == "PASS_ALL_FOUR_EVENTS_ALL_THREE_VARIANTS"
    )
    checks["a55_bus_access_has_no_byte_multiplier"] = (
        "Do not multiply A55 bus_access by 8 or 64"
        in prereg["tier_b"]["bus_access_policy"]
        and all(
            "bytes" not in key and "ratio" not in key
            for variant_data in perf.values()
            for key in variant_data
        )
    )

    a76_timing = a76["timing"]["variants"]
    a76_candidate_perf = a76["tier_b"]["variants"]["neon_fp16_f32acc"]
    candidate_latency_ratio = (
        timing["neon_fp16_f32acc"]["p50_us"]
        / a76_timing["neon_fp16_f32acc"]["p50_us"]
    )
    candidate_bandwidth_ratio = (
        timing["neon_fp16_f32acc"]["analytic_gbps"]
        / a76_timing["neon_fp16_f32acc"]["analytic_gbps"]
    )
    candidate_cycles_ratio = (
        perf["neon_fp16_f32acc"]["per_invocation"]["cycles:u"]
        / a76_candidate_perf["per_invocation"]["cycles:u"]
    )
    cluster_classification = (
        "A55_FASTER"
        if candidate_latency_ratio < 0.95
        else "PARITY"
        if candidate_latency_ratio <= 1.05
        else "A55_SLOWER"
    )
    checks["cluster_comparison_arithmetic"] = (
        close(A76_MAX_KHZ / A55_MAX_KHZ, 1.3066666666666666)
        and candidate_latency_ratio > 0
        and candidate_bandwidth_ratio > 0
        and candidate_cycles_ratio > 0
    )

    measurement_valid = all(checks.values())
    status = (
        "VALID_OPE5_05_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_4XA55_FP16_GEMV_COMPARATOR"
        if measurement_valid
        else "INVALID_OPE5_05_A55_FP16_GEMV_MEASUREMENT"
    )
    thermal = {
        zone: {
            "name": zone_name[zone],
            "maximum_timed_temperature_mC": max_temperature_by_zone[zone],
            "lowest_enabled_trip_mC": lowest_trip_by_zone[zone],
            "headroom_mC": thermal_headroom[zone],
        }
        for zone in sorted(lowest_trip_by_zone, key=int)
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "OPE5-05-A55-FP16-GEMV",
        "status": status,
        "measurement_valid": measurement_valid,
        "recorded_at": controller["controller_after"],
        "check_summary": {
            "passed": sum(checks.values()),
            "total": len(checks),
            "failed": [name for name, passed in checks.items() if not passed],
        },
        "checks": checks,
        "identity": {
            "segment": str(segment.relative_to(repo)),
            "scope": {"path": str(scope_path.relative_to(repo)), "sha256": sha256(scope_path)},
            "preregistration": {"path": str(prereg_path.relative_to(repo)), "sha256": sha256(prereg_path)},
            "prelock_receipt": {"path": str(prelock_path.relative_to(repo)), "sha256": sha256(prelock_path)},
            "source_lock": {"path": str(source_lock_path.relative_to(repo)), "sha256": sha256(source_lock_path)},
            "locked_files": locked_files,
            "controller_meta": controller,
            "board_run_meta": run_meta,
        },
        "correctness": {
            "oracle": "independent FP64 scalar reference",
            "expected_element_comparisons": 69_120,
            "observed_element_comparisons": observed_comparisons,
            "case_rows": len(correctness_rows),
            "passing_case_rows": sum(row["pass"] == "1" for row in correctness_rows),
            "mismatch_elements": sum(int(row["mismatch_count"]) for row in correctness_rows),
            "by_variant": correctness_by_variant,
        },
        "codegen": codegen,
        "timing": {
            "clock": "CLOCK_MONOTONIC_RAW",
            "shape_mkn": [1, 10240, 2560],
            "cpu_cores": [0, 1, 2, 3],
            "threads": 4,
            "analytic_bytes_per_invocation": ANALYTIC_BYTES,
            "ops_per_invocation": OPS,
            "variants": timing,
            "candidate_speedup_vs_autovec": speedup_vs_autovec,
            "candidate_speedup_vs_scalar": speedup_vs_scalar,
            "candidate_classification_vs_autovec": candidate_classification,
        },
        "immutable_a76_anchor": {
            "receipt_path": str(a76_receipt_path.relative_to(repo)),
            "receipt_sha256": sha256(a76_receipt_path),
            "status": a76["status"],
            "candidate_p50_us": a76_timing["neon_fp16_f32acc"]["p50_us"],
            "candidate_analytic_gbps": a76_timing["neon_fp16_f32acc"]["analytic_gbps"],
            "candidate_cycles_per_invocation": a76_candidate_perf["per_invocation"]["cycles:u"],
            "candidate_instructions_per_invocation": a76_candidate_perf["per_invocation"]["instructions:u"],
            "candidate_ipc": a76_candidate_perf["ipc"],
            "recorded_max_frequency_khz": A76_MAX_KHZ,
            "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
        },
        "cluster_comparison": {
            "a55_candidate_p50_us": timing["neon_fp16_f32acc"]["p50_us"],
            "a76_candidate_p50_us": a76_timing["neon_fp16_f32acc"]["p50_us"],
            "a55_to_a76_candidate_latency_ratio": candidate_latency_ratio,
            "a55_candidate_analytic_gbps": timing["neon_fp16_f32acc"]["analytic_gbps"],
            "a76_candidate_analytic_gbps": a76_timing["neon_fp16_f32acc"]["analytic_gbps"],
            "a55_to_a76_candidate_bandwidth_ratio": candidate_bandwidth_ratio,
            "a55_candidate_cycles_per_invocation": perf["neon_fp16_f32acc"]["per_invocation"]["cycles:u"],
            "a76_candidate_cycles_per_invocation": a76_candidate_perf["per_invocation"]["cycles:u"],
            "a55_to_a76_candidate_cycles_ratio": candidate_cycles_ratio,
            "a55_candidate_instructions_per_invocation": perf["neon_fp16_f32acc"]["per_invocation"]["instructions:u"],
            "a55_candidate_ipc": perf["neon_fp16_f32acc"]["ipc"],
            "a76_candidate_instructions_per_invocation": a76_candidate_perf["per_invocation"]["instructions:u"],
            "a76_candidate_ipc": a76_candidate_perf["ipc"],
            "a76_to_a55_max_frequency_ratio": A76_MAX_KHZ / A55_MAX_KHZ,
            "classification": cluster_classification,
            "guard": "System-level nonconcurrent cluster comparison; not a frequency-normalized single-variable microarchitecture causal score.",
        },
        "tier_b": {
            "events": list(EVENTS),
            "variants": perf,
            "primary_cross_cluster_metric": "candidate cycles per invocation ratio",
            "bus_access_policy": "Raw count and count per invocation only. No x8/x64 byte conversion and no direct A55/A76 L2D or bus ratio.",
        },
        "telemetry": {
            "samples": len(telemetry_rows),
            "timed_window_samples": len(timed_rows),
            "max_interval_seconds": max(telemetry_gaps),
            "minimum_policy0_frequency_ratio": min(frequency_ratios),
            "thermal_by_zone": thermal,
            "minimum_thermal_headroom_mC": min(thermal_headroom.values()),
        },
        "governors": {
            "before": before,
            "after_set": after_set,
            "after_restore": after_restore,
            "policy0_write_performed": True,
            "policy4_policy6_npu_write_performed": False,
        },
        "evidence_strength": {
            "formal_segments": 1,
            "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
            "same_day_retest_performed": False,
            "same_day_retest_scheduled": False,
            "automatic_retry_performed": False,
            "a76_rerun_performed": False,
            "performance_smoke_performed": False,
        },
        "claim_limits": [
            "A55 placement is a synthetic isolated comparator and does not prove closed RKLLM placement on A55 or CPU.",
            "The A55/A76 ratios combine clock, microarchitecture, cache, memory service, compiler target, and scheduling effects.",
            "The A76 anchor is one owner-waived nonconcurrent segment; this is not paired replication.",
            "OPE0 is a shared A76-derived platform ceiling, not an A55-specific roofline.",
            "No A55 bus_access byte conversion, cross-cluster L2D/bus ratio, OPE4 attribution, deployment-speed, quality, energy, or cost claim is made.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "checks": f"{sum(checks.values())}/{len(checks)}",
                "correctness": f"{observed_comparisons}/69120",
                "a55_candidate_p50_us": timing["neon_fp16_f32acc"]["p50_us"],
                "speedup_vs_a55_autovec": speedup_vs_autovec,
                "a55_to_a76_latency_ratio": candidate_latency_ratio,
                "a55_to_a76_cycles_ratio": candidate_cycles_ratio,
            },
            ensure_ascii=False,
        )
    )
    return 0 if measurement_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
