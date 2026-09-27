#!/usr/bin/env python3
"""Validate and summarize the two formal OPE3 K-E1 board segments."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VARIANTS = ("scalar", "autovec", "neon_sdot")
PERF_EVENTS = ("cycles:u", "instructions:u", "l2d_cache_refill:u", "bus_access:u")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"malformed CSV: {path}")
    return rows


def read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    ordered = sorted(values)
    if len(ordered) == 1:
        p25 = p50 = p75 = ordered[0]
    else:
        p25, p50, p75 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "min": min(ordered),
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "max": max(ordered),
        "iqr": p75 - p25,
        "iqr_over_p50": (p75 - p25) / p50 if p50 else float("inf"),
    }


def numeric(rows: Iterable[dict[str, str]], key: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in {None, "", "NA"}:
            continue
        try:
            result.append(float(value))
        except ValueError:
            continue
    return result


def analyze_governors(path: Path) -> dict[str, Any]:
    rows = read_csv(path, delimiter="\t")
    events = {row["event"]: row for row in rows}
    fields = ("policy4_governor", "policy6_governor", "npu_governor")
    expected = {"before", "after_set", "after_restore"}
    shape_pass = set(events) == expected
    set_pass = shape_pass and all(
        events["after_set"][field] == "performance" for field in fields
    )
    restore_pass = shape_pass and all(
        events["after_restore"][field] == events["before"][field]
        for field in fields
    )
    return {
        "events": events,
        "event_shape_pass": shape_pass,
        "performance_set_pass": set_pass,
        "exact_restore_pass": restore_pass,
    }


def analyze_correctness(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    expected = {
        (20260919, "small_symmetric_int8", "autovec"),
        (20260919, "small_symmetric_int8", "neon_sdot"),
        (20260920, "full_symmetric_int8", "autovec"),
        (20260920, "full_symmetric_int8", "neon_sdot"),
        (20260921, "alternating_extrema_with_random_sign", "autovec"),
        (20260921, "alternating_extrema_with_random_sign", "neon_sdot"),
    }
    observed = {
        (int(row["seed"]), row["distribution"], row["variant"]) for row in rows
    }
    row_passes = [
        int(row["checked"]) == 2560
        and int(row["mismatches"]) == 0
        and int(row["max_abs_diff"]) == 0
        and row["reference_checksum"] == row["actual_checksum"]
        for row in rows
    ]
    return {
        "rows": rows,
        "expected_trials": len(expected),
        "observed_trials": len(rows),
        "trial_identity_pass": observed == expected and len(rows) == len(expected),
        "exact_comparisons": sum(int(row["checked"]) for row in rows),
        "pass": observed == expected and len(rows) == len(expected) and all(row_passes),
    }


def read_perf_stat(path: Path) -> dict[str, float]:
    counters: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for fields in csv.reader(handle):
            if len(fields) < 3:
                continue
            raw_value = fields[0].strip()
            event = fields[2].strip()
            if event not in PERF_EVENTS or raw_value.startswith("<"):
                continue
            try:
                counters[event] = float(raw_value)
            except ValueError:
                continue
    return counters


def perf_for_variant(path: Path, variant: str) -> dict[str, Any]:
    group_exit_path = path / f"perf_{variant}_group.exit_code.txt"
    group_exit = int(group_exit_path.read_text().strip())
    source_by_event: dict[str, str] = {}
    counters: dict[str, float] = {}
    if group_exit == 0:
        counters = read_perf_stat(path / f"perf_{variant}_group.stat.csv")
        source_by_event = {event: "group" for event in counters}
    else:
        for event in PERF_EVENTS:
            event_slug = event.replace(":", "_")
            exit_path = path / f"perf_{variant}_{event_slug}.exit_code.txt"
            if not exit_path.is_file() or int(exit_path.read_text().strip()) != 0:
                continue
            isolated = read_perf_stat(path / f"perf_{variant}_{event_slug}.stat.csv")
            if event in isolated:
                counters[event] = isolated[event]
                source_by_event[event] = "isolated"
    missing = sorted(set(PERF_EVENTS) - set(counters))
    return {
        "group_exit_code": group_exit,
        "counters": counters,
        "source_by_event": source_by_event,
        "missing_events": missing,
        "all_events_pass": not missing,
    }


def read_thermal_trips(path: Path) -> dict[int, float]:
    rows = read_csv(path, delimiter="\t")
    by_zone: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        try:
            by_zone[int(row["zone"])].append(float(row["trip_temp_mC"]))
        except (KeyError, TypeError, ValueError):
            continue
    return {zone: min(values) for zone, values in by_zone.items() if values}


def analyze_segment(segment: dict[str, Any], prereg: dict[str, Any]) -> dict[str, Any]:
    path = Path(segment["path"])
    meta = read_env(path / "run_meta.env")
    controller_elapsed_s = (
        dt.datetime.fromisoformat(segment["controller_after"])
        - dt.datetime.fromisoformat(segment["controller_before"])
    ).total_seconds()
    board_monotonic_elapsed_s = (
        float(meta["board_end_uptime_s"]) - float(meta["board_start_uptime_s"])
    )
    dual_timebase_pass = controller_elapsed_s > 0 and board_monotonic_elapsed_s > 0
    timing_rows = read_csv(path / "timings.csv")
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in timing_rows:
        by_variant[row["variant"]].append(row)

    expected_reps = prereg["measurement_protocol"]["timed_repetitions_per_variant"]
    timing: dict[str, Any] = {}
    count_pass = True
    for variant in VARIANTS:
        rows = by_variant[variant]
        count_pass &= len(rows) == expected_reps
        summary = quantiles([float(row["elapsed_us"]) for row in rows])
        timing[variant] = {
            "samples": len(rows),
            "latency_us": summary,
            "checksum_values": sorted({row["checksum"] for row in rows}),
            "stability_pass": summary["iqr_over_p50"]
            <= 0.30,
        }
    checksum_values = {
        checksum
        for variant in VARIANTS
        for checksum in timing[variant]["checksum_values"]
    }
    checksum_pass = len(checksum_values) == 1

    telemetry = read_csv(path / "telemetry_1hz.csv")
    formal_telemetry = [
        row
        for row in telemetry
        if row["phase"] == "formal_interleaved_three_baselines"
    ]
    governor_values = {
        (row["policy4_governor"], row["policy6_governor"], row["npu_governor"])
        for row in formal_telemetry
    }
    governor_telemetry_pass = governor_values == {
        ("performance", "performance", "performance")
    }
    frequency: dict[str, Any] = {}
    for policy in ("policy4", "policy6"):
        ratios = [
            float(row[f"{policy}_khz"]) / float(row[f"{policy}_max_khz"])
            for row in formal_telemetry
            if row[f"{policy}_khz"] not in {"", "NA"}
            and row[f"{policy}_max_khz"] not in {"", "NA", "0"}
        ]
        frequency[policy] = {
            "samples": len(ratios),
            "min_fraction_of_max": min(ratios) if ratios else None,
            "p50_fraction_of_max": statistics.median(ratios) if ratios else None,
            "pass": bool(ratios) and min(ratios) >= 0.95,
        }

    trips = read_thermal_trips(path / "thermal_trip_points.tsv")
    thermal: dict[str, Any] = {}
    thermal_pass = True
    thermal_columns = [key for key in telemetry[0] if key.startswith("thermal_zone")]
    for column in thermal_columns:
        zone_text = column.split("_", 2)[1]
        zone = int(zone_text.removeprefix("zone"))
        values = numeric(formal_telemetry, column)
        maximum = max(values) if values else None
        trip = trips.get(zone)
        zone_pass = maximum is not None and trip is not None and maximum < trip
        thermal_pass &= zone_pass
        thermal[column.removesuffix("_mC")] = {
            "max_c": maximum / 1000.0 if maximum is not None else None,
            "lowest_trip_c": trip / 1000.0 if trip is not None else None,
            "pass": zone_pass,
        }

    governor = analyze_governors(path / "governor_state.tsv")
    codegen = {
        variant: {
            "static_sdot_instruction_count": len(
                re.findall(
                    r"\bsdot\b",
                    (path / f"disassembly_{variant}.txt").read_text(
                        encoding="utf-8", errors="replace"
                    ),
                )
            )
        }
        for variant in VARIANTS
    }
    codegen_pass = (
        codegen["scalar"]["static_sdot_instruction_count"] == 0
        and codegen["neon_sdot"]["static_sdot_instruction_count"] > 0
    )
    segment_pass = all(
        [
            count_pass,
            checksum_pass,
            bool(formal_telemetry),
            governor_telemetry_pass,
            thermal_pass,
            governor["performance_set_pass"],
            governor["exact_restore_pass"],
            codegen_pass,
            dual_timebase_pass,
            meta.get("run_exit_code") == "0",
            *(timing[variant]["stability_pass"] for variant in VARIANTS),
            *(frequency[policy]["pass"] for policy in frequency),
        ]
    )

    result: dict[str, Any] = {
        "label": segment["label"],
        "path": str(path),
        "controller_before": segment["controller_before"],
        "controller_after": segment["controller_after"],
        "run_meta": meta,
        "timebase": {
            "controller_elapsed_s": controller_elapsed_s,
            "board_monotonic_elapsed_s": board_monotonic_elapsed_s,
            "pass": dual_timebase_pass,
            "board_wall_clock_disposition": (
                "RECORDED_BUT_NON_AUTHORITATIVE_CONTROLLER_AND_MONOTONIC_USED"
            ),
        },
        "timing": timing,
        "timing_count_pass": count_pass,
        "cross_variant_checksum_pass": checksum_pass,
        "telemetry": {
            "samples": len(telemetry),
            "formal_window_samples": len(formal_telemetry),
            "governor_values": [list(value) for value in sorted(governor_values)],
            "governor_pass": governor_telemetry_pass,
            "frequency": frequency,
            "thermal": thermal,
            "thermal_pass": thermal_pass,
        },
        "governor_guard": governor,
        "codegen": {
            "variants": codegen,
            "scalar_has_no_sdot_and_candidate_has_sdot_pass": codegen_pass,
        },
        "segment_pass": segment_pass,
    }
    if (path.joinpath("correctness.csv").is_file()):
        result["correctness"] = analyze_correctness(path / "correctness.csv")
    if path.joinpath("perf_event_attempts.tsv").is_file():
        cache_line_bytes = int(meta["cache_line_bytes"])
        perf = {variant: perf_for_variant(path, variant) for variant in VARIANTS}
        expected_bus_bytes = (
            prereg["formal_shape"]["analytics"]["total_unique_bytes"]
            * prereg["tier_b"]["stat_repetitions_per_variant"]
        )
        for variant in VARIANTS:
            bus_access = perf[variant]["counters"].get("bus_access:u")
            cycles = perf[variant]["counters"].get("cycles:u")
            instructions = perf[variant]["counters"].get("instructions:u")
            repetitions = prereg["tier_b"]["stat_repetitions_per_variant"]
            perf[variant]["derived_counts"] = {
                "ipc": (
                    instructions / cycles
                    if instructions is not None and cycles not in {None, 0}
                    else None
                ),
                "cycles_per_requested_repetition": (
                    cycles / repetitions if cycles is not None else None
                ),
                "instructions_per_requested_repetition": (
                    instructions / repetitions if instructions is not None else None
                ),
                "scope_limit": (
                    "perf stat covers process initialization plus the requested "
                    "kernel repetitions; values are not region-instrumented."
                ),
            }
            proxy_bytes = bus_access * cache_line_bytes if bus_access is not None else None
            ratio = proxy_bytes / expected_bus_bytes if proxy_bytes is not None else None
            perf[variant]["bus_access_proxy"] = {
                "status": "PREREGISTERED_GATE_DO_NOT_REINTERPRET",
                "cache_line_bytes": cache_line_bytes,
                "proxy_bytes": proxy_bytes,
                "expected_analytic_bytes": expected_bus_bytes,
                "ratio": ratio,
                "within_20_percent": ratio is not None and 0.80 <= ratio <= 1.20,
            }
            # This does not replace the frozen cache-line proxy above.  It records
            # the observed scale needed to explain the analytic byte model so the
            # miss can be investigated without silently normalizing it away.
            effective_bytes_per_count = (
                expected_bus_bytes / bus_access
                if bus_access not in {None, 0}
                else None
            )
            eight_byte_ratio = (
                bus_access * 8 / expected_bus_bytes
                if bus_access is not None
                else None
            )
            perf[variant]["posthoc_bus_access_scale_diagnostic"] = {
                "status": "POSTHOC_DIAGNOSTIC_NOT_AN_ACCEPTANCE_GATE",
                "effective_bytes_per_count_to_match_analytic": effective_bytes_per_count,
                "eight_bytes_per_count_ratio_to_analytic": eight_byte_ratio,
                "interpretation_limit": (
                    "A ratio near one is an empirical scale observation only; it "
                    "does not establish the PMU event's architectural unit."
                ),
            }
        record_exit = int((path / "perf_record_exit_code.txt").read_text().strip())
        report_exit_path = path / "perf_report_exit_code.txt"
        report_exit = (
            int(report_exit_path.read_text().strip())
            if report_exit_path.is_file()
            else None
        )
        result["tier_b"] = {
            "variants": perf,
            "all_stat_events_pass": all(
                perf[variant]["all_events_pass"] for variant in VARIANTS
            ),
            "perf_record_exit_code": record_exit,
            "perf_report_exit_code": report_exit,
            "perf_record_and_report_pass": record_exit == 0 and report_exit == 0,
            "preregistered_analytic_backing_pass": all(
                perf[variant]["bus_access_proxy"]["within_20_percent"]
                for variant in VARIANTS
            ),
            "proxy_investigation": (
                "The frozen bus_access*cache-line proxy is evaluated as written. "
                "Any miss is retained; the separate 8-byte-scale diagnostic is "
                "post-hoc and cannot convert that miss into a pass."
            ),
            "authoritative_event_semantics": {
                "source": (
                    "https://raw.githubusercontent.com/ARM-software/data/"
                    "0806afb1f3479432c30a9488872bd7040a1fe874/"
                    "pmu/cortex-a76.json"
                ),
                "source_owner": "Arm",
                "source_revision": "0806afb1f3479432c30a9488872bd7040a1fe874",
                "source_cpuid": "0x41d0b",
                "finding": (
                    "Cortex-A76 BUS_ACCESS counts transferred data beats between "
                    "the core and SCU, not cache-line transactions."
                ),
                "adjudication": (
                    "Multiplying every event by the 64-byte cache-line size is not "
                    "architecturally justified, so the preregistered proxy is "
                    "rejected when it misses; no replacement scale is promoted."
                ),
            },
        }
    return result


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def main() -> int:
    args = parse_args()
    prereg = read_json(args.prereg)
    manifest = read_json(args.run_manifest)
    if prereg["stage"] != "OPE3-K-E1":
        raise ValueError("wrong preregistration stage")
    if len(manifest["formal_segments"]) != 2:
        raise ValueError("OPE3 requires exactly two formal segments")

    segments = [analyze_segment(item, prereg) for item in manifest["formal_segments"]]
    correctness_segments = [item for item in segments if "correctness" in item]
    correctness_pass = (
        len(correctness_segments) == 1 and correctness_segments[0]["correctness"]["pass"]
    )
    tier_b_segments = [item for item in segments if "tier_b" in item]
    tier_b_present_once = len(tier_b_segments) == 1
    tier_b = tier_b_segments[0]["tier_b"] if tier_b_present_once else None
    tier_b_collection_pass = bool(
        tier_b
        and tier_b["all_stat_events_pass"]
        and tier_b["perf_record_and_report_pass"]
    )
    tier_b_analytic_backing_pass = bool(
        tier_b and tier_b["preregistered_analytic_backing_pass"]
    )

    first_after = parse_time(segments[0]["controller_after"])
    second_before = parse_time(segments[1]["controller_before"])
    gap_seconds = (second_before - first_after).total_seconds()
    gap_pass = gap_seconds >= prereg["measurement_protocol"][
        "minimum_same_day_segment_separation_seconds"
    ]
    same_day = first_after.date() == second_before.date()
    boot_ids = {segment["run_meta"].get("board_start_boot_id") for segment in segments}

    aggregate: dict[str, Any] = {}
    for variant in VARIANTS:
        p50s = [segment["timing"][variant]["latency_us"]["p50"] for segment in segments]
        aggregate_p50 = statistics.median(p50s)
        aggregate[variant] = {
            "segment_p50_us": p50s,
            "p50_us": aggregate_p50,
            "measured_gops": prereg["formal_shape"]["analytics"]["ops"]
            / aggregate_p50
            / 1000.0,
            "measured_gbps": prereg["formal_shape"]["analytics"]["total_unique_bytes"]
            / aggregate_p50
            / 1000.0,
        }
        aggregate[variant]["tier_a_relative_to_ope0"] = (
            aggregate[variant]["measured_gbps"]
            / prereg["sources"]["ope0_bandwidth_ceiling"]["ceiling_gbps_decimal"]
        )

    candidate_p50s = aggregate["neon_sdot"]["segment_p50_us"]
    candidate_relative_delta = abs(candidate_p50s[1] - candidate_p50s[0]) / statistics.median(
        candidate_p50s
    )
    directions = [
        "neon_faster"
        if segment["timing"]["neon_sdot"]["latency_us"]["p50"]
        < segment["timing"]["autovec"]["latency_us"]["p50"]
        else "neon_not_faster"
        for segment in segments
    ]
    replication_pass = (
        same_day
        and gap_pass
        and len(set(directions)) == 1
        and candidate_relative_delta <= 0.10
    )

    candidate_speedup = aggregate["autovec"]["p50_us"] / aggregate["neon_sdot"]["p50_us"]
    autovec_emits_sdot_in_both_segments = all(
        segment["codegen"]["variants"]["autovec"][
            "static_sdot_instruction_count"
        ]
        > 0
        for segment in segments
    )
    optimization_class = (
        "NEGATIVE_GT_30_PERCENT_SLOWER_THAN_AUTOVEC"
        if candidate_speedup < (1.0 / 1.30)
        else ("NEON_FASTER" if candidate_speedup > 1.0 else "NEON_NOT_FASTER_WITHIN_30_PERCENT")
    )

    prereg_sha = sha256(args.prereg)
    expected_prereg_sha = manifest.get("prereg_sha256")
    expected_source_sha = prereg.get("candidate", {}).get("source_sha256")
    source_hashes = {segment["run_meta"].get("candidate_source_sha256") for segment in segments}
    prereg_hashes = {segment["run_meta"].get("prereg_sha256") for segment in segments}
    identity_pass = (
        expected_prereg_sha == prereg_sha
        and prereg_hashes == {prereg_sha}
        and expected_source_sha is not None
        and source_hashes == {expected_source_sha}
    )

    execution_validity = {
        "identity_pass": identity_pass,
        "correctness_pass": correctness_pass,
        "both_segments_pass": all(segment["segment_pass"] for segment in segments),
        "same_day_pass": same_day,
        "minimum_gap_pass": gap_pass,
        "replication_pass": replication_pass,
        "tier_b_present_once": tier_b_present_once,
        "tier_b_collection_pass": tier_b_collection_pass,
    }
    execution_pass = all(execution_validity.values())
    method_gates = {
        "tier_b_preregistered_analytic_backing_pass": tier_b_analytic_backing_pass,
        "ope4_model_stage_anchor_present": False,
    }
    isolated_measurement_pass = execution_pass
    full_ke1_four_gate_closure_pass = execution_pass and all(method_gates.values())
    status = (
        "PASS_ISOLATED_OPE3_OPE4_ANCHOR_PENDING"
        if execution_pass and tier_b_analytic_backing_pass
        else (
            "VALID_OPE3_TIER_B_CACHELINE_PROXY_REJECTED_OPE4_ANCHOR_PENDING"
            if execution_pass
            else "FAIL_OPE3_VALIDITY_GATE"
        )
    )

    npu_anchor = prereg["cross_isa_anchors"]["npu_primitive"]
    pc_anchor = prereg["cross_isa_anchors"]["pc_sm89_reference"]
    table = {
        "schema_version": 1,
        "stage": "OPE3-K-E1-three-horizontal",
        "shape_mkn": prereg["semantic_contract"]["shape_mkn"],
        "rows": [
            {
                "carrier": "a76-neon",
                "lineage": "OPE3 int8 row-major [N,K], four A76 cores, neon_sdot",
                **aggregate["neon_sdot"],
            },
            {
                "carrier": "rknpu-matmul",
                "lineage": npu_anchor["lineage"],
                "p50_us": npu_anchor["p50_us"],
                "measured_gops": npu_anchor["measured_gops"],
                "measured_gbps": npu_anchor["measured_gbps"],
                "tier_a_relative_to_ope0": npu_anchor["memory_relative_to_ope0"],
            },
            {
                "carrier": "pc-sm89-reference",
                "lineage": pc_anchor["lineage"],
                "p50_us": pc_anchor["p50_us"],
                "analytic_bytes": pc_anchor["analytic_bytes"],
                "comparison_limit": pc_anchor["use"],
            },
        ],
        "ranking_prohibited": True,
        "ope4_anchor_status": "PENDING_OPE4",
    }

    receipt = {
        "schema_version": 1,
        "stage": "OPE3-K-E1",
        "status": status,
        "prereg_ref": str(args.prereg),
        "prereg_sha256": prereg_sha,
        "run_manifest_ref": str(args.run_manifest),
        "segments": segments,
        "correctness": correctness_segments[0]["correctness"] if correctness_segments else None,
        "aggregate": aggregate,
        "candidate_vs_autovec": {
            "speedup": candidate_speedup,
            "speedup_vs_scalar": (
                aggregate["scalar"]["p50_us"] / aggregate["neon_sdot"]["p50_us"]
            ),
            "segment_directions": directions,
            "classification": optimization_class,
            "candidate_p50_relative_delta_between_segments": candidate_relative_delta,
            "autovec_emits_sdot_in_both_segments": (
                autovec_emits_sdot_in_both_segments
            ),
            "interpretation": (
                "When autovec_emits_sdot_in_both_segments is true, this is a "
                "hand-unrolled intrinsic schedule versus GCC's sdot loop, not "
                "a scalar-instruction versus dot-product comparison."
            ),
        },
        "replication": {
            "same_day": same_day,
            "gap_seconds": gap_seconds,
            "minimum_gap_seconds": prereg["measurement_protocol"][
                "minimum_same_day_segment_separation_seconds"
            ],
            "pass": replication_pass,
        },
        "time_authority": {
            "controller_wall_clock": "AUTHORITATIVE_FOR_SEGMENT_ORDER_AND_GAP",
            "board_monotonic_uptime": "AUTHORITATIVE_FOR_BOARD_RUN_DURATION",
            "board_wall_clock": "NON_AUTHORITATIVE_RECORDED_ONLY",
            "same_board_boot_between_segments": len(boot_ids) == 1,
            "board_boot_ids": sorted(value for value in boot_ids if value is not None),
        },
        "tier_b": tier_b,
        "posthoc_analysis_adjudication": {
            "recorded_at": "2026-09-19T22:05:07+08:00",
            "trigger": (
                "The first segment's preregistered bus_access*64-byte proxy was "
                "about 8.0 times the analytic byte model."
            ),
            "resolution": (
                "Arm's Cortex-A76 PMU description establishes that BUS_ACCESS "
                "counts data beats rather than cache-line transactions. The "
                "frozen proxy therefore remains a failed gate. An observed "
                "8-byte scale is reported only as a post-hoc diagnostic."
            ),
            "protocol_effect": (
                "None: preregistration, candidate source, inputs, repetitions, "
                "timing, correctness, telemetry, Tier-B events, and the second "
                "segment protocol were unchanged."
            ),
            "analysis_code_effect": (
                "The summarizer was amended after segment 1 to separate valid "
                "data collection from the rejected proxy and to prevent the "
                "post-hoc scale from being promoted to an acceptance gate."
            ),
            "detail_ref": (
                "results/operator/board/ope3/"
                "ope3_bus_access_semantics_adjudication_20260919.json"
            ),
        },
        "three_horizontal_table_ref": str(args.table_output),
        "ope4_anchor": prereg["cross_isa_anchors"]["ope4_anchor"],
        "execution_validity_gates": execution_validity,
        "method_gates": method_gates,
        "isolated_measurement_pass": isolated_measurement_pass,
        "full_ke1_four_gate_closure_pass": full_ke1_four_gate_closure_pass,
        "claim_limits": prereg["claim_limits"],
    }

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.table_output.write_text(
        json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "isolated_measurement_pass": isolated_measurement_pass,
                "full_ke1_four_gate_closure_pass": full_ke1_four_gate_closure_pass,
                "candidate_speedup_vs_autovec": candidate_speedup,
                "replication_pass": replication_pass,
                "tier_b_collection_pass": tier_b_collection_pass,
                "tier_b_preregistered_analytic_backing_pass": (
                    tier_b_analytic_backing_pass
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if execution_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
