#!/usr/bin/env python3
"""Validate and summarize the single formal OPE5 RMSNorm board segment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PREREG_SHA256 = "971f97200383d341c0751d82b6c6b598804e95e4e80f5bdc00201381ea4cb224"
OPE0_GBPS = 23.1951846685
OPE4_TPOT_MS = 446.1262611013072
ANALYTIC_BYTES = 3_840_000
CALLS_PER_FORWARD = 302
VARIANTS = ("scalar_ref", "autovec_o3", "neon_fp16")
EVENTS = ("cycles:u", "instructions:u", "l2d_cache_refill:u", "bus_access:u")


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    prereg_path = repo / "configs/operator/board/ope5_rmsnorm_prereg.json"
    source_lock_path = (
        repo
        / "results/operator/board/ope5/ope5_rmsnorm_source_lock_20260921.json"
    )
    prereg = read_json(prereg_path)
    source_lock = read_json(source_lock_path)
    controller = read_json(segment / "controller_meta.json")
    run_meta = read_env(segment / "run_meta.env")
    checks: dict[str, bool] = {}

    checks["prereg_sha256"] = sha256(prereg_path) == PREREG_SHA256
    checks["prereg_single_segment_owner_waiver"] = (
        prereg["measurement_protocol"]["formal_segments"] == 1
        and prereg["measurement_protocol"]["owner_instruction"]
        == "不需要复测，取消同日复测，只测一次"
        and prereg["measurement_protocol"]["replication_status"]
        == "WAIVED_BY_OWNER_NOT_SATISFIED"
    )
    checks["source_lock_precedes_formal_execution"] = (
        source_lock["status"] == "LOCKED_BEFORE_FORMAL_EXECUTION"
        and source_lock["formal_execution_started"] is False
        and datetime.fromisoformat(source_lock["recorded_at"])
        < datetime.fromisoformat(controller["controller_before"])
    )
    checks["source_lock_prereg_identity"] = (
        source_lock["preregistration"]["sha256"] == PREREG_SHA256
        and run_meta["prereg_sha256"] == PREREG_SHA256
        and run_meta["source_lock_sha256"] == sha256(source_lock_path)
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
    checks["all_locked_files_match"] = all(
        item["expected_sha256"] == item["observed_sha256"]
        for item in locked_files.values()
    )
    observed_hashes = read_tsv(segment / "implementation_hash_observed.tsv")
    checks["board_locked_files_match"] = (
        len(observed_hashes) == 4
        and all(row["match"] == "yes" for row in observed_hashes)
    )
    checks["formal_process_exit_zero"] = (
        controller["ssh_exit_code"] == 0
        and run_meta["run_exit_code"] == "0"
        and run_meta["governor_restore_exit_code"] == "0"
    )
    checks["single_segment_identity"] = (
        controller["formal_segment_ordinal"] == 1
        and controller["additional_segment_scheduled"] is False
        and run_meta["segment_label"] == "segment1"
        and len(list(segment.parent.glob("ope5_rmsnorm_segment*"))) == 1
    )
    checks["board_uses_a76_cpu7_single_thread"] = (
        run_meta["primary_cpu"] == "7" and run_meta["threads"] == "1"
    )
    checks["clock_domains_recorded"] = (
        controller["controller_before"] < controller["controller_after"]
        and int(run_meta["board_end_monotonic_raw_ns"])
        > int(run_meta["board_start_monotonic_raw_ns"])
        and "not subtracted" in controller["clock_domain_note"]
    )

    correctness_rows = read_csv(segment / "correctness.csv")
    correctness_by_variant: dict[str, dict[str, Any]] = {}
    for variant in ("autovec_o3", "neon_fp16"):
        rows = [row for row in correctness_rows if row["variant"] == variant]
        correctness_by_variant[variant] = {
            "comparisons": len(rows),
            "passes": sum(row["pass"] == "1" for row in rows),
            "mismatch_elements": sum(int(row["mismatch_count"]) for row in rows),
            "max_abs_error": max(float(row["max_abs_error"]) for row in rows),
            "max_rel_error": max(float(row["max_rel_error"]) for row in rows),
        }
    checks["correctness_144_of_144"] = (
        len(correctness_rows) == 144
        and all(row["pass"] == "1" for row in correctness_rows)
        and len({row["case_id"] for row in correctness_rows}) == 8
        and {row["distribution"] for row in correctness_rows}
        == {
            "unit_normal",
            "wide_normal_std32",
            "near_zero_uniform_plus_minus_0.001",
        }
        and {int(row["seed"]) for row in correctness_rows}
        == {20260921, 20260922, 20260923}
        and (segment / "correctness_gate.txt").read_text().strip()
        == "PASS_144_OF_144"
    )
    checks["correctness_structural_assertions"] = all(
        row["all_finite"] == "1"
        and row["input_unchanged"] == "1"
        and row["scale_unchanged"] == "1"
        and row["scale_pointer_contract"] == "1"
        for row in correctness_rows
    )

    timing_rows = read_csv(segment / "timings.csv")
    per_case_rows = [row for row in timing_rows if row["scope"] == "per_case"]
    population_rows = [row for row in timing_rows if row["scope"] == "population"]
    checks["timing_population_complete"] = (
        len(per_case_rows) == 24
        and len(population_rows) == 3
        and {row["variant"] for row in population_rows} == set(VARIANTS)
        and all(int(row["repetitions"]) == 200 for row in per_case_rows)
        and all(int(row["repetitions"]) >= 200 for row in population_rows)
        and all(int(row["calls_per_forward"]) == CALLS_PER_FORWARD for row in population_rows)
        and all(int(row["analytic_bytes"]) == ANALYTIC_BYTES for row in population_rows)
        and all(int(row["permutations_used"]) == 6 for row in timing_rows)
        and all(int(row["ring_frames"]) == 16 for row in timing_rows)
    )
    checks["primary_population_stable"] = (
        all(float(row["iqr_over_p50"]) <= 0.30 for row in population_rows)
        and (segment / "timing_gate.txt").read_text().strip()
        == "PASS_PRIMARY_POPULATION_STABLE"
    )
    population: dict[str, dict[str, Any]] = {}
    for row in population_rows:
        variant = row["variant"]
        population[variant] = {
            "repetitions": int(row["repetitions"]),
            "p25_us": float(row["p25_us"]),
            "p50_us": float(row["p50_us"]),
            "p75_us": float(row["p75_us"]),
            "iqr_us": float(row["iqr_us"]),
            "iqr_over_p50": float(row["iqr_over_p50"]),
            "analytic_gbps": float(row["measured_gbps"]),
            "memory_relative_to_ope0": float(row["memory_relative_to_ope0"]),
            "fraction_of_ope4_tpot": float(row["fraction_of_ope4_tpot"]),
            "percent_of_ope4_tpot": float(row["fraction_of_ope4_tpot"]) * 100.0,
        }
    speedup_vs_autovec = (
        population["autovec_o3"]["p50_us"] / population["neon_fp16"]["p50_us"]
    )
    speedup_vs_scalar = (
        population["scalar_ref"]["p50_us"] / population["neon_fp16"]["p50_us"]
    )
    if speedup_vs_autovec >= 1.05:
        candidate_classification = "POSITIVE"
    elif speedup_vs_autovec >= 0.95:
        candidate_classification = "PARITY"
    else:
        candidate_classification = "NEGATIVE"
    neon_percent = population["neon_fp16"]["percent_of_ope4_tpot"]
    if neon_percent < 1.0:
        anchor_classification = "SUPPORTS_MINOR_CPU_RMSNORM_INSTRUMENT_BOUND"
    elif neon_percent <= 5.0:
        anchor_classification = "SMALL_BUT_NONZERO_ISOLATED_COMPONENT"
    else:
        anchor_classification = (
            "NONTRIVIAL_ISOLATED_COMPONENT_REQUIRES_NEW_RUNTIME_LINKAGE_BEFORE_ATTRIBUTION"
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
    frequency_ratios = [
        int(row["policy6_khz"]) / int(row["policy6_max_khz"])
        for row in timed_rows
    ]
    checks["a76_frequency_gate"] = bool(timed_rows) and min(frequency_ratios) >= 0.95
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

    trip_rows = read_tsv(segment / "thermal_trip_points.tsv")
    lowest_trip_by_zone: dict[str, int] = {}
    for row in trip_rows:
        zone = row["zone"]
        temperature = int(row["trip_temp_mC"])
        lowest_trip_by_zone[zone] = min(lowest_trip_by_zone.get(zone, temperature), temperature)
    max_temperature_by_zone: dict[str, int] = {}
    for zone in lowest_trip_by_zone:
        prefix = f"thermal_zone{zone}_"
        columns = [column for column in timed_rows[0] if column.startswith(prefix)]
        if len(columns) != 1:
            raise AssertionError((zone, columns))
        column = columns[0]
        max_temperature_by_zone[zone] = max(int(row[column]) for row in timed_rows)
    thermal_headroom = {
        zone: lowest_trip_by_zone[zone] - max_temperature_by_zone[zone]
        for zone in lowest_trip_by_zone
    }
    checks["thermal_gate"] = all(value > 0 for value in thermal_headroom.values())

    perf: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        counts = parse_perf_stat(segment / f"perf_{variant}_population.stat.csv")
        bench_rows = read_csv(segment / f"perf_{variant}_population.bench.csv")
        if len(bench_rows) != 1:
            raise AssertionError((variant, bench_rows))
        row = bench_rows[0]
        sequences = int(row["population_sequences"])
        per_sequence = {event: counts[event] / sequences for event in EVENTS}
        beat8_bytes_per_sequence = per_sequence["bus_access:u"] * 8.0
        perf[variant] = {
            "population_sequences": sequences,
            "elapsed_ms": float(row["elapsed_ms"]),
            "counts": counts,
            "per_sequence": per_sequence,
            "ipc": counts["instructions:u"] / counts["cycles:u"],
            "bus_access_x8_hypothesis_bytes_per_sequence": beat8_bytes_per_sequence,
            "bus_access_x8_to_analytic_bytes_ratio": beat8_bytes_per_sequence
            / ANALYTIC_BYTES,
        }
    checks["tier_b_all_events_numeric"] = (
        all(
            item["population_sequences"] >= 200 and item["elapsed_ms"] >= 3000.0
            for item in perf.values()
        )
        and (segment / "tier_b_gate.txt").read_text().strip()
        == "PASS_ALL_FOUR_EVENTS_ALL_THREE_VARIANTS"
        and all(
            (segment / f"perf_{variant}_population.exit_code.txt").read_text().strip()
            == "0"
            for variant in VARIANTS
        )
    )
    checks["old_bus_access_x64_gate_not_reintroduced"] = (
        "count*64" in prereg["tier_b"]["bus_access_policy"]
        and "not an analytic-backing acceptance gate"
        in prereg["tier_b"]["bus_access_policy"]
    )

    measurement_valid = all(checks.values())
    status = (
        "VALID_OPE5_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_A76_RMSNORM"
        if measurement_valid
        else "INVALID_OPE5_MEASUREMENT"
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "OPE5-03-RMSNorm",
        "status": status,
        "measurement_valid": measurement_valid,
        "recorded_at": controller["controller_after"],
        "checks": checks,
        "identity": {
            "segment": str(segment.relative_to(repo)),
            "segment_sha_scope": "Raw segment files are hashed by the OPE5 evidence manifest.",
            "preregistration": {
                "path": str(prereg_path.relative_to(repo)),
                "sha256": sha256(prereg_path),
            },
            "source_lock": {
                "path": str(source_lock_path.relative_to(repo)),
                "sha256": sha256(source_lock_path),
            },
            "locked_files": locked_files,
            "controller_meta": controller,
            "board_run_meta": run_meta,
        },
        "correctness": {
            "expected_comparisons": 144,
            "observed_comparisons": len(correctness_rows),
            "passes": sum(row["pass"] == "1" for row in correctness_rows),
            "by_variant": correctness_by_variant,
        },
        "timing": {
            "clock": "CLOCK_MONOTONIC_RAW",
            "formal_cases": 8,
            "per_case_rows": len(per_case_rows),
            "population_calls_per_forward": CALLS_PER_FORWARD,
            "population_analytic_bytes": ANALYTIC_BYTES,
            "streaming_ring_frames": 16,
            "streaming_ring_analytic_bytes": 61_440_000,
            "population": population,
            "candidate_speedup_vs_autovec": speedup_vs_autovec,
            "candidate_speedup_vs_scalar": speedup_vs_scalar,
            "candidate_classification_vs_autovec": candidate_classification,
        },
        "ope4_anchor": {
            "ope4_primary_decode_ms_per_token": OPE4_TPOT_MS,
            "neon_isolated_population_ms": population["neon_fp16"]["p50_us"] / 1000.0,
            "neon_fraction_of_ope4_tpot": population["neon_fp16"]["fraction_of_ope4_tpot"],
            "neon_percent_of_ope4_tpot": neon_percent,
            "classification": anchor_classification,
            "guard": "This is an isolated A76 instrument ratio, not RKLLM component attribution and not a subtractable TPOT decomposition.",
        },
        "tier_b": {
            "events": list(EVENTS),
            "variants": perf,
            "bus_access_policy": "count*8 is retained only as the preregistered beat-size hypothesis diagnostic; count*64 remains rejected and neither ratio is an acceptance gate.",
        },
        "telemetry": {
            "samples": len(telemetry_rows),
            "timed_window_samples": len(timed_rows),
            "max_interval_seconds": max(telemetry_gaps),
            "minimum_cpu7_frequency_ratio": min(frequency_ratios),
            "max_temperature_mC": max(max_temperature_by_zone.values()),
            "lowest_trip_by_zone_mC": lowest_trip_by_zone,
            "max_temperature_by_zone_mC": max_temperature_by_zone,
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
        },
        "claim_limits": [
            "The result is an isolated single-thread A76 FP16 RMSNorm instrument, not proof that RKLLM executes the same population on CPU.",
            "The OPE4 ratio must not be subtracted from TPOT or represented as a measured RKLLM component breakdown.",
            "The PC K1 BF16 GPU lineage is methodology context only and is not directly performance-ranked against this FP16 A76 result.",
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
                "correctness": f"{receipt['correctness']['passes']}/144",
                "neon_p50_us": population["neon_fp16"]["p50_us"],
                "speedup_vs_autovec": speedup_vs_autovec,
                "percent_of_ope4_tpot": neon_percent,
            },
            ensure_ascii=False,
        )
    )
    return 0 if measurement_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
