#!/usr/bin/env python3
"""Validate and summarize the sole OPE5-01 attention-sequence segment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


STATUS = "VALID_PRIMARY_OPE5_01_SINGLE_SEGMENT_OWNER_WAIVER_PARTIAL_TIER_B_PROTOCOL_CASE_ID_DEFECT"
VARIANTS = (
    "cpu_scalar_sequence",
    "cpu_neon_sequence",
    "hybrid_rknpu_sequence",
)
EVENTS = (
    "cycles:u",
    "instructions:u",
    "task-clock",
    "context-switches",
    "cpu-migrations",
)
RUN_REL = Path(
    "results/operator/board/ope5/"
    "ope5_attention_sequence_segment1_20260921T182337+0800"
)
PREREG_REL = Path("configs/operator/board/ope5_attention_sequence_prereg.json")
LOCK_REL = Path(
    "results/operator/board/ope5/ope5_attention_sequence_source_lock_20260921.json"
)
ADJUDICATION_REL = Path(
    "configs/operator/board/ope5_attention_sequence_tier_b_adjudication_20260921.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        result[key] = value
    return result


def parse_perf(path: Path) -> dict[str, float]:
    observed: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#") or len(row) < 3:
                continue
            event = row[2].strip()
            if event in EVENTS:
                value = row[0].strip()
                if value in {"", "<not counted>", "<not supported>"}:
                    raise ValueError((path, event, value))
                observed[event] = float(value)
    if set(observed) != set(EVENTS):
        raise ValueError((path, observed))
    return observed


def weighted(rows: list[dict[str, str]]) -> float:
    denominator = sum(int(row["projected_invocations"]) for row in rows)
    return sum(
        int(row["projected_invocations"]) * float(row["p50_us"])
        for row in rows
    ) / denominator


def variant_summary(rows: list[dict[str, str]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    sliding = [row for row in selected if row["family"] == "sliding"]
    full = [row for row in selected if row["family"] == "full"]
    all_us = weighted(selected)
    full_us = weighted(full)
    forward_ms = 42.0 * all_us / 1000.0
    return {
        "case_count": len(selected),
        "sliding_p50_us": float(sliding[0]["p50_us"]),
        "full_weighted_case_p50_us": full_us,
        "all_invocation_weighted_case_p50_us": all_us,
        "model_forward_equivalent_ms": forward_ms,
        "counterfactual_fraction_of_ope4": forward_ms / 446.1262611013072,
        "minimum_repetitions": min(int(row["repetitions"]) for row in selected),
        "maximum_repetitions": max(int(row["repetitions"]) for row in selected),
        "maximum_iqr_over_p50": max(float(row["iqr_over_p50"]) for row in selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/operator/board/ope5/"
            "ope5_attention_sequence_receipt_20260921.json"
        ),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    run = repo / RUN_REL
    prereg = load_json(repo / PREREG_REL)
    lock = load_json(repo / LOCK_REL)
    adjudication = load_json(repo / ADJUDICATION_REL)

    checks: dict[str, bool] = {}
    checks["prereg_sha_locked"] = (
        sha256(repo / PREREG_REL)
        == lock["preregistration"]["sha256"]
        == "a0cc5d0c516f1b9e12028debec03743c386980041e4a5c6e2502e3d110cff464"
    )
    checks["source_lock_identity"] = (
        sha256(repo / LOCK_REL)
        == adjudication["source_lock"]["sha256"]
        == "fc2d1bd1c4d3a701fa081a80460c1dafb7738ed99afc835ac978edd4a043ae8b"
    )
    checks["adjudication_narrow_no_rerun"] = (
        adjudication["status"] == "OWNER_WAIVER_ACCEPT_PRIMARY_DROP_FULL_TIER_B"
        and adjudication["adjudication"]["additional_board_execution_authorized"] is False
        and adjudication["adjudication"]["primary_sequence_timing_accepted"] is True
        and adjudication["adjudication"]["full_tier_b_accepted"] is False
    )

    capability = load_csv(run / "capability.csv")
    checks["capability_26_of_26"] = (
        len(capability) == 26
        and {row["stage"] for row in capability} == {"qk", "pv"}
        and all(row["pass"] == "1" for row in capability)
        and all(row["core_mask"] == "core0" for row in capability)
    )

    correctness = load_csv(run / "correctness.csv")
    correctness_by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [row for row in correctness if row["variant"] == variant]
        correctness_by_variant[variant] = {
            "trials": len(rows),
            "elements": sum(int(row["elements"]) for row in rows),
            "mismatches": sum(int(row["mismatches"]) for row in rows),
            "max_abs": max(float(row["max_abs"]) for row in rows),
            "max_rel": max(float(row["max_rel"]) for row in rows),
            "max_pre_round_row_sum_error": max(
                float(row["pre_round_row_sum_error"]) for row in rows
            ),
            "max_fp16_row_sum_error": max(
                float(row["fp16_row_sum_error"]) for row in rows
            ),
        }
    checks["correctness_270336_of_270336"] = (
        len(correctness) == 72
        and set(correctness_by_variant) == set(VARIANTS)
        and sum(item["elements"] for item in correctness_by_variant.values()) == 270336
        and all(item["mismatches"] == 0 for item in correctness_by_variant.values())
        and all(row["pass"] == "1" for row in correctness)
    )
    state_gates = load_csv(run / "state_gates.csv")
    checks["state_gates_8_of_8"] = (
        len(state_gates) == 8 and all(row["pass"] == "1" for row in state_gates)
    )

    timings = load_csv(run / "timings.csv")
    checks["primary_timing_39_of_39"] = (
        len(timings) == 39
        and len({row["case_id"] for row in timings}) == 13
        and {row["variant"] for row in timings} == set(VARIANTS)
        and all(50 <= int(row["repetitions"]) <= 5000 for row in timings)
        and all(float(row["elapsed_ms"]) >= 500.0 for row in timings)
        and all(float(row["p50_us"]) > 0.0 for row in timings)
        and all(row["affinity_pass"] == "1" for row in timings)
    )
    timing = {variant: variant_summary(timings, variant) for variant in VARIANTS}
    scalar = timing["cpu_scalar_sequence"]
    neon = timing["cpu_neon_sequence"]
    hybrid = timing["hybrid_rknpu_sequence"]
    comparisons = {
        "hybrid_latency_ratio_vs_neon": hybrid["all_invocation_weighted_case_p50_us"]
        / neon["all_invocation_weighted_case_p50_us"],
        "hybrid_speedup_vs_neon": neon["all_invocation_weighted_case_p50_us"]
        / hybrid["all_invocation_weighted_case_p50_us"],
        "hybrid_speedup_vs_scalar": scalar["all_invocation_weighted_case_p50_us"]
        / hybrid["all_invocation_weighted_case_p50_us"],
        "neon_speedup_vs_scalar": scalar["all_invocation_weighted_case_p50_us"]
        / neon["all_invocation_weighted_case_p50_us"],
        "sliding_hybrid_latency_ratio_vs_neon": hybrid["sliding_p50_us"]
        / neon["sliding_p50_us"],
        "full_hybrid_latency_ratio_vs_neon": hybrid["full_weighted_case_p50_us"]
        / neon["full_weighted_case_p50_us"],
        "classification": "NEGATIVE_HYBRID_SLOWER_THAN_NEON",
    }

    tier_b: dict[str, Any] = {}
    for variant in VARIANTS:
        prefix = run / f"perf_sliding_k512_d256_{variant}"
        bench_rows = load_csv(prefix.with_suffix(".bench.csv"))
        assert len(bench_rows) == 1
        tier_b[variant] = {
            "case_id": "sliding_k512_d256",
            "benchmark": {
                "warmup": int(bench_rows[0]["warmup"]),
                "repetitions": int(bench_rows[0]["repetitions"]),
                "elapsed_ms": float(bench_rows[0]["elapsed_ms"]),
            },
            "process_block_events": parse_perf(prefix.with_suffix(".stat.csv")),
        }
    checks["swa_tier_b_3_of_3_numeric"] = len(tier_b) == 3
    attempts = [
        line.split("\t")
        for line in (run / "perf_event_attempts.tsv").read_text(encoding="utf-8").splitlines()[1:]
        if line
    ]
    failed_stderr = (run / "perf_full_k3123_d512_cpu_scalar_sequence.stderr.log").read_text(
        encoding="utf-8"
    ).strip()
    checks["full_tier_b_protocol_case_id_defect_exact"] = (
        len(attempts) == 4
        and attempts[-1][0] == "full_k3123_d512"
        and attempts[-1][1] == "cpu_scalar_sequence"
        and attempts[-1][-1] == "90"
        and failed_stderr
        == "OPE5_ATTENTION_ERROR=unknown performance case: full_k3123_d512"
        and (run / "perf_full_k3123_d512_cpu_scalar_sequence.bench.csv").stat().st_size == 0
    )

    meta = load_env(run / "run_meta.env")
    checks["run_exit_and_restore_disclosed"] = (
        meta["run_exit_code"] == "90"
        and meta["governor_restore_exit_code"] == "0"
        and meta["bench_binary_sha256"] == lock["board_binary"]["sha256"]
    )
    governors = load_tsv(run / "governor_state.tsv")
    before = governors[0]
    after_set = governors[1]
    after_restore = governors[-1]
    checks["governors_exactly_restored"] = (
        before["policy4_governor"] == after_set["policy4_governor"] == after_restore["policy4_governor"]
        and after_set["policy6_governor"] == "performance"
        and after_set["npu_governor"] == "performance"
        and after_restore["policy6_governor"] == before["policy6_governor"]
        and after_restore["npu_governor"] == before["npu_governor"]
    )

    telemetry = load_csv(run / "telemetry_1hz.csv")
    checks["telemetry_frequency_and_governor_gate"] = (
        len(telemetry) >= 180
        and all(row["policy6_governor"] == "performance" for row in telemetry)
        and all(row["npu_governor"] == "performance" for row in telemetry)
        and all(row["policy4_governor"] == before["policy4_governor"] for row in telemetry)
        and {int(row["policy6_khz"]) for row in telemetry} == {2352000}
        and {int(row["policy6_max_khz"]) for row in telemetry} == {2352000}
        and {int(row["npu_hz"]) for row in telemetry} == {1000000000}
        and {int(row["npu_max_hz"]) for row in telemetry} == {1000000000}
    )
    thermal_columns = [key for key in telemetry[0] if key.startswith("thermal_zone")]
    thermal_max = {key: max(int(row[key]) for row in telemetry) for key in thermal_columns}
    trip_rows = load_tsv(run / "thermal_trip_points.tsv")
    first_trip = {}
    for row in trip_rows:
        zone = row["zone"]
        first_trip[zone] = min(
            int(row["trip_temp_mC"]), first_trip.get(zone, int(row["trip_temp_mC"]))
        )
    headrooms = []
    for column, maximum in thermal_max.items():
        zone = column.split("_")[1].removeprefix("zone")
        headrooms.append(first_trip[zone] - maximum)
    checks["thermal_headroom_positive"] = min(headrooms) > 0

    npu_rows = load_csv(run / "npu_load_10hz.csv")
    npu_load = {}
    for core in range(3):
        key = f"npu_load_core{core}_pct"
        values = [float(row[key]) for row in npu_rows if row[key] not in {"", "NA"}]
        npu_load[f"core{core}"] = {
            "samples": len(values),
            "mean_pct": statistics.fmean(values),
            "p50_pct": statistics.median(values),
            "max_pct": max(values),
        }
    checks["npu_core0_only"] = (
        npu_load["core0"]["max_pct"] > 0
        and npu_load["core1"]["max_pct"] == 0
        and npu_load["core2"]["max_pct"] == 0
    )

    passed = sum(checks.values())
    if passed != len(checks):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"receipt checks failed: {failed}")

    receipt = {
        "schema_version": 1,
        "stage": "OPE5-01-DECODE-ATTENTION-SEQUENCE-FINAL",
        "status": STATUS,
        "recorded_at": "2026-09-21T18:27:57+08:00",
        "primary_sequence_measurement_valid": True,
        "strict_original_prereg_complete": False,
        "owner_waiver": adjudication["status"],
        "check_summary": {"passed": passed, "total": len(checks)},
        "checks": checks,
        "sources": {
            "preregistration": {"path": PREREG_REL.as_posix(), "sha256": sha256(repo / PREREG_REL)},
            "source_lock": {"path": LOCK_REL.as_posix(), "sha256": sha256(repo / LOCK_REL)},
            "postrun_adjudication": {
                "path": ADJUDICATION_REL.as_posix(),
                "sha256": sha256(repo / ADJUDICATION_REL),
            },
            "run_directory": RUN_REL.as_posix(),
        },
        "controller_window": {
            "before": "2026-09-21T18:23:37.095589070+08:00",
            "after": "2026-09-21T18:27:57.278515145+08:00",
        },
        "board_window": {
            "start_monotonic_raw_ns": int(meta["board_start_monotonic_raw_ns"]),
            "end_monotonic_raw_ns": int(meta["board_end_monotonic_raw_ns"]),
            "duration_seconds": (
                int(meta["board_end_monotonic_raw_ns"])
                - int(meta["board_start_monotonic_raw_ns"])
            )
            / 1e9,
            "runner_exit_code": 90,
            "governor_restore_exit_code": 0,
            "wall_clock_advisory_only": True,
        },
        "correctness": {
            "capability_cells": 26,
            "logical_trials": 24,
            "implementation_trials": 72,
            "output_element_comparisons": 270336,
            "mismatches": 0,
            "state_gates": "8/8",
            "by_variant": correctness_by_variant,
        },
        "primary_timing": {
            "method": "sum(projected_invocations * case_p50_us) / 911148; forward equivalent multiplies by 42",
            "variants": timing,
            "comparisons": comparisons,
            "case_rows": [
                {
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "K": int(row["K"]),
                    "D": int(row["D"]),
                    "projected_invocations": int(row["projected_invocations"]),
                    "variant": row["variant"],
                    "repetitions": int(row["repetitions"]),
                    "p50_us": float(row["p50_us"]),
                    "iqr_over_p50": float(row["iqr_over_p50"]),
                }
                for row in timings
            ],
        },
        "tier_b": {
            "status": "PARTIAL_SWA_ONLY_OWNER_WAIVER",
            "swa": tier_b,
            "full": {
                "status": "NOT_MEASURED_PROTOCOL_CASE_ID_DEFECT",
                "requested_case_id": "full_k3123_d512",
                "runner_exit_code": 90,
                "benchmark_work_started": False,
            },
            "claim_limit": "SWA process-block CPU counters include process setup and warmup and do not measure NPU operations or bytes; no full-context Tier-B claim is available.",
        },
        "telemetry": {
            "system_samples": len(telemetry),
            "npu_load_10hz": npu_load,
            "policy6_khz": 2352000,
            "npu_hz": 1000000000,
            "maximum_temperature_mC": max(thermal_max.values()),
            "minimum_thermal_headroom_mC": min(headrooms),
            "governors_restored": True,
            "policy4_unchanged": True,
        },
        "evidence_strength": {
            "formal_segments": 1,
            "performance_smoke_performed": False,
            "automatic_retry_performed": False,
            "same_day_retest_performed": False,
            "cross_day_retest_performed": False,
            "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
            "strict_prereg_tier_b_complete": False,
        },
        "bounded_conclusion": (
            "The complete CPU NEON sequence is faster than the host-orchestrated core0-RKNPU hybrid "
            "for both SWA and full families in this synthetic FP16 carrier. This supports a negative "
            "hybrid result, not an RKLLM attention placement or deployment conclusion."
        ),
        "claim_limits": adjudication["claim_limits"],
    }
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": STATUS,
                "checks": receipt["check_summary"],
                "neon_weighted_us": neon["all_invocation_weighted_case_p50_us"],
                "hybrid_weighted_us": hybrid["all_invocation_weighted_case_p50_us"],
                "hybrid_latency_ratio_vs_neon": comparisons["hybrid_latency_ratio_vs_neon"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
