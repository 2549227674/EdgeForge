#!/usr/bin/env python3
"""Validate and summarize the single formal OPE5-02 FP16 GEMV segment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SCOPE_SHA256 = "51c6d942cbd7fcd494ea076a48b434cbebb870365897ab7da965978e13c07068"
PREREG_SHA256 = "b89dc6e03bcae62ab817a0814f48541ff6c8526c4b23f60108bbd438434fb514"
SOURCE_LOCK_SHA256 = "cee44b4f4734655d44cf38bdadc106ec9ac0d956e02f3c5d8ff92f710638ad5b"
OPE0_GBPS = 23.1951846685
OPE3_INT8_P50_US = 944.7080005
OPE3_INT8_BYTES = 26_234_880
OPE4_TPOT_MS = 446.1262611013072
ANALYTIC_BYTES = 52_459_520
OPS = 52_428_800
CALLS_PER_FORWARD = 42
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


def close(left: float, right: float, rel: float = 1e-9, abs_: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=abs_)


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
    scope_path = repo / "configs/operator/board/ope5_fp16_gemv_scope.json"
    prereg_path = repo / "configs/operator/board/ope5_fp16_gemv_prereg.json"
    source_lock_path = (
        repo / "results/operator/board/ope5/ope5_fp16_gemv_source_lock_20260921.json"
    )
    scope = read_json(scope_path)
    prereg = read_json(prereg_path)
    source_lock = read_json(source_lock_path)
    controller = read_json(segment / "controller_meta.json")
    run_meta = read_env(segment / "run_meta.env")
    checks: dict[str, bool] = {}

    checks["scope_frozen_identity"] = (
        sha256(scope_path) == SCOPE_SHA256
        and scope["status"] == "PASS_OPE5_02_FOUR_QUESTIONS_PREREG_AUTHORIZED"
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
        and prereg["measurement_protocol"]["owner_instruction"] == OWNER_INSTRUCTION
        and prereg["measurement_protocol"]["replication_status"]
        == "WAIVED_BY_OWNER_NOT_SATISFIED"
        and "no automatic retry" in prereg["measurement_protocol"]["stability_gate"]
    )
    checks["source_lock_precedes_formal_execution"] = (
        sha256(source_lock_path) == SOURCE_LOCK_SHA256
        and source_lock["status"] == "LOCKED_BEFORE_FORMAL_EXECUTION"
        and source_lock["formal_execution_started"] is False
        and source_lock["board_process_started"] is False
        and source_lock["recorded_at"] < controller["controller_before"]
    )
    checks["source_lock_frozen_identity"] = (
        source_lock["scope"]["sha256"] == SCOPE_SHA256
        and source_lock["preregistration"]["sha256"] == PREREG_SHA256
        and run_meta["prereg_sha256"] == PREREG_SHA256
        and run_meta["source_lock_sha256"] == SOURCE_LOCK_SHA256
    )

    locked_files: dict[str, dict[str, str]] = {}
    for section in ("implementation_files", "runtime_dependencies"):
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
    observed_hashes = read_tsv(segment / "implementation_hash_observed.tsv")
    checks["all_board_locked_files_match"] = (
        len(observed_hashes) == 4
        and all(row["match"] == "yes" for row in observed_hashes)
        and {row["path"] for row in observed_hashes}
        == {item["path"] for item in locked_files.values()}
    )
    checks["formal_process_exit_and_restore_zero"] = (
        controller["ssh_exit_code"] == 0
        and run_meta["run_exit_code"] == "0"
        and run_meta["governor_restore_exit_code"] == "0"
    )
    segment_dirs = [
        path
        for path in segment.parent.glob("ope5_fp16_gemv_segment*")
        if path.is_dir()
    ]
    checks["exactly_one_formal_segment_no_retest"] = (
        controller["formal_segment_ordinal"] == 1
        and controller["additional_segment_scheduled"] is False
        and controller["instability_retry_authorized"] is False
        and run_meta["segment_label"] == "segment1"
        and run_meta["formal_segments"] == "1"
        and source_lock["measurement_identity"]["same_day_retest_scheduled"] is False
        and source_lock["measurement_identity"]["instability_retry_authorized"] is False
        and len(segment_dirs) == 1
        and segment_dirs[0] == segment
    )
    checks["four_a76_fixed_topology"] = (
        run_meta["primary_cpus"] == "4,5,6,7"
        and run_meta["threads"] == "4"
        and prereg["formal_shape"]["cpu_cores"] == [4, 5, 6, 7]
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
        and {row["distribution"] for row in correctness_rows} == set(DISTRIBUTIONS)
        and {int(row["seed"]) for row in correctness_rows} == set(SEEDS)
        and all(int(row["elements"]) == 2560 for row in correctness_rows)
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
    codegen_counts = {
        "scalar_vector_fmla": len(re.findall(r"\bfmla\s+v", scalar_disassembly)),
        "candidate_fcvtl": len(re.findall(r"\bfcvtl2?\s+v", candidate_disassembly)),
        "candidate_vector_fmla": len(
            re.findall(r"\bfmla\s+v", candidate_disassembly)
        ),
    }
    checks["codegen_gates"] = (
        codegen_counts["scalar_vector_fmla"] == 0
        and codegen_counts["candidate_fcvtl"] > 0
        and codegen_counts["candidate_vector_fmla"] > 0
        and (segment / "disassembly_autovec_o3.txt").stat().st_size > 0
        and (segment / "vectorization.log").exists()
    )

    timing_rows = read_csv(segment / "timings.csv")
    checks["timing_population_complete"] = (
        len(timing_rows) == 3
        and {row["variant"] for row in timing_rows} == set(VARIANTS)
        and all(row["scope"] == "formal_invocation" for row in timing_rows)
        and all((int(row["m"]), int(row["k"]), int(row["n"])) == (1, 10240, 2560) for row in timing_rows)
        and all(int(row["threads"]) == 4 for row in timing_rows)
        and all(int(row["repetitions"]) >= 200 for row in timing_rows)
        and all(int(row["analytic_bytes"]) == ANALYTIC_BYTES for row in timing_rows)
        and all(int(row["ops"]) == OPS for row in timing_rows)
        and all(int(row["permutations_used"]) == 6 for row in timing_rows)
        and all(int(row["working_set_weight_bytes"]) == 52_428_800 for row in timing_rows)
        and all(row["affinity_pass"] == "1" and row["partition_coverage_pass"] == "1" for row in timing_rows)
        and all(int(row["repetitions"]) * float(row["p50_us"]) / 1000.0 >= 3000.0 for row in timing_rows)
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
        expected_relative = expected_gbps / OPE0_GBPS
        arithmetic_pass &= close(float(row["measured_gops"]), expected_gops, rel=2e-10)
        arithmetic_pass &= close(float(row["measured_gbps"]), expected_gbps, rel=2e-10)
        arithmetic_pass &= close(
            float(row["memory_relative_to_ope0"]), expected_relative, rel=2e-10
        )
        if variant == "neon_fp16_f32acc":
            expected_projection_ms = CALLS_PER_FORWARD * p50_us / 1000.0
            expected_projection_fraction = expected_projection_ms / OPE4_TPOT_MS
            arithmetic_pass &= close(
                float(row["projection_42_calls_ms"]), expected_projection_ms, rel=2e-10
            )
            arithmetic_pass &= close(
                float(row["projection_fraction_of_ope4"]),
                expected_projection_fraction,
                rel=2e-10,
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
            "memory_relative_to_ope0": float(row["memory_relative_to_ope0"]),
        }
    checks["timing_arithmetic_reproduces"] = arithmetic_pass
    candidate = next(row for row in timing_rows if row["variant"] == "neon_fp16_f32acc")
    candidate_projection_ms = float(candidate["projection_42_calls_ms"])
    candidate_projection_fraction = float(candidate["projection_fraction_of_ope4"])
    speedup_vs_autovec = timing["autovec_o3"]["p50_us"] / timing["neon_fp16_f32acc"]["p50_us"]
    speedup_vs_scalar = timing["scalar_fp32"]["p50_us"] / timing["neon_fp16_f32acc"]["p50_us"]
    candidate_classification = (
        "POSITIVE"
        if speedup_vs_autovec >= 1.05
        else "PARITY"
        if speedup_vs_autovec >= 0.95
        else "NEGATIVE"
    )
    projection_classification = (
        "SMALL"
        if candidate_projection_fraction < 0.10
        else "MATERIAL"
        if candidate_projection_fraction <= 0.30
        else "LARGE"
    )

    governor_rows = read_tsv(segment / "governor_state.tsv")
    governor_by_event = {row["event"]: row for row in governor_rows}
    before = governor_by_event["before"]
    after_set = governor_by_event["after_set"]
    after_restore = governor_by_event["after_restore"]
    checks["cpu_governors_performance_and_restored"] = (
        after_set["policy4_governor"] == "performance"
        and after_set["policy6_governor"] == "performance"
        and after_restore["policy4_governor"] == before["policy4_governor"]
        and after_restore["policy6_governor"] == before["policy6_governor"]
    )
    checks["npu_governor_unchanged"] = (
        before["npu_governor"]
        == after_set["npu_governor"]
        == after_restore["npu_governor"]
    )

    telemetry_rows = read_csv(segment / "telemetry_1hz.csv")
    timed_rows = [
        row
        for row in telemetry_rows
        if row["phase"] == "formal_interleaved_timing"
        or row["phase"].startswith("tier_b_")
    ]
    frequency_ratios = {
        "policy4": [int(row["policy4_khz"]) / int(row["policy4_max_khz"]) for row in timed_rows],
        "policy6": [int(row["policy6_khz"]) / int(row["policy6_max_khz"]) for row in timed_rows],
    }
    checks["both_a76_policy_frequency_gates"] = bool(timed_rows) and all(
        min(values) >= 0.95 for values in frequency_ratios.values()
    )
    checks["timed_window_governor_gate"] = all(
        row["policy4_governor"] == "performance"
        and row["policy6_governor"] == "performance"
        for row in timed_rows
    )
    checks["telemetry_npu_governor_unchanged"] = all(
        row["npu_governor"] == before["npu_governor"] for row in telemetry_rows
    )
    uptime_values = [float(row["uptime_s"]) for row in telemetry_rows]
    telemetry_gaps = [right - left for left, right in zip(uptime_values, uptime_values[1:])]
    checks["telemetry_approximately_1hz"] = (
        len(telemetry_rows) >= 2 and max(telemetry_gaps) <= 1.20
    )
    checks["no_performance_smoke_phase"] = (
        all("smoke" not in row["phase"] for row in telemetry_rows)
        and not list(segment.glob("*smoke*"))
    )

    trip_rows = read_tsv(segment / "thermal_trip_points.tsv")
    lowest_trip_by_zone: dict[str, int] = {}
    zone_name: dict[str, str] = {}
    for row in trip_rows:
        zone = row["zone"]
        temperature = int(row["trip_temp_mC"])
        zone_name[zone] = row["type"]
        lowest_trip_by_zone[zone] = min(lowest_trip_by_zone.get(zone, temperature), temperature)
    max_temperature_by_zone: dict[str, int] = {}
    for zone in lowest_trip_by_zone:
        prefix = f"thermal_zone{zone}_"
        columns = [column for column in timed_rows[0] if column.startswith(prefix)]
        if len(columns) != 1:
            raise AssertionError((zone, columns))
        max_temperature_by_zone[zone] = max(int(row[columns[0]]) for row in timed_rows)
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
        beat8_bytes_per_invocation = per_invocation["bus_access:u"] * 8.0
        perf[variant] = {
            "invocations": invocations,
            "elapsed_ms": float(row["elapsed_ms"]),
            "counts": counts,
            "per_invocation": per_invocation,
            "ipc": counts["instructions:u"] / counts["cycles:u"],
            "bus_access_x8_hypothesis_bytes_per_invocation": beat8_bytes_per_invocation,
            "bus_access_x8_to_analytic_bytes_ratio": beat8_bytes_per_invocation / ANALYTIC_BYTES,
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
    checks["old_bus_access_x64_gate_not_reintroduced"] = (
        "count*64" in prereg["tier_b"]["bus_access_policy"]
        and "not an analytic-byte acceptance gate" in prereg["tier_b"]["bus_access_policy"]
    )

    measurement_valid = all(checks.values())
    status = (
        "VALID_OPE5_02_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_4XA76_FP16_GEMV"
        if measurement_valid
        else "INVALID_OPE5_02_MEASUREMENT"
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
        "stage": "OPE5-02-FP16-GEMV",
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
        "codegen": codegen_counts,
        "timing": {
            "clock": "CLOCK_MONOTONIC_RAW",
            "shape_mkn": [1, 10240, 2560],
            "threads": 4,
            "analytic_bytes_per_invocation": ANALYTIC_BYTES,
            "ops_per_invocation": OPS,
            "variants": timing,
            "candidate_speedup_vs_autovec": speedup_vs_autovec,
            "candidate_speedup_vs_scalar": speedup_vs_scalar,
            "candidate_classification_vs_autovec": candidate_classification,
        },
        "ope3_int8_same_geometry_context": {
            "candidate_fp16_p50_us": timing["neon_fp16_f32acc"]["p50_us"],
            "ope3_int8_neon_sdot_p50_us": OPE3_INT8_P50_US,
            "fp16_to_int8_latency_ratio": timing["neon_fp16_f32acc"]["p50_us"] / OPE3_INT8_P50_US,
            "fp16_to_int8_analytic_byte_ratio": ANALYTIC_BYTES / OPE3_INT8_BYTES,
            "guard": "Same geometry, row-major layout, and four-A76 topology only; this ratio is not a pure dtype causal effect or a hardware leaderboard.",
        },
        "ope4_counterfactual": {
            "calls": CALLS_PER_FORWARD,
            "projection_ms": candidate_projection_ms,
            "ope4_primary_decode_ms_per_token": OPE4_TPOT_MS,
            "fraction_of_ope4_tpot": candidate_projection_fraction,
            "percent_of_ope4_tpot": candidate_projection_fraction * 100.0,
            "classification": projection_classification,
            "guard": "The 42-call value is p50 multiplication, not a measured population, RKLLM component attribution, or subtractable TPOT decomposition.",
        },
        "tier_b": {
            "events": list(EVENTS),
            "variants": perf,
            "bus_access_policy": "count*8 is a transferred-beat hypothesis diagnostic only; count*64 remains rejected and no bus counter ratio is an acceptance gate.",
        },
        "telemetry": {
            "samples": len(telemetry_rows),
            "timed_window_samples": len(timed_rows),
            "max_interval_seconds": max(telemetry_gaps),
            "minimum_frequency_ratio": {key: min(values) for key, values in frequency_ratios.items()},
            "thermal_by_zone": thermal,
            "minimum_thermal_headroom_mC": min(thermal_headroom.values()),
        },
        "governors": {
            "before": before,
            "after_set": after_set,
            "after_restore": after_restore,
            "npu_write_performed": False,
        },
        "evidence_strength": {
            "formal_segments": 1,
            "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
            "same_day_retest_performed": False,
            "same_day_retest_scheduled": False,
            "automatic_retry_performed": False,
        },
        "claim_limits": [
            "Only M/K/N and call frequency are deployment-authentic; CPU FP16 dtype, row-major layout, and execution are isolated comparator choices.",
            "The 42-call result is a derived counterfactual, not a measured full linear population and not RKLLM component time.",
            "OPE2 NPU FP16, OPE3 A76 int8, OPE4 model-stage, and PC context remain lineage-separated and are not a hardware leaderboard.",
            "The result has one owner-waived formal segment; replication is explicitly unsatisfied.",
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
                "candidate_p50_us": timing["neon_fp16_f32acc"]["p50_us"],
                "speedup_vs_autovec": speedup_vs_autovec,
                "projection_percent_of_ope4": candidate_projection_fraction * 100.0,
            },
            ensure_ascii=False,
        )
    )
    return 0 if measurement_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
