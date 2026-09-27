#!/usr/bin/env python3
"""Summarize an OPE2 board matrix into a compact receipt and cell table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--board-run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cells-output", type=Path, required=True)
    parser.add_argument("--correctness-adjudication", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"malformed CSV: {path}")
    return rows


def read_csvs(paths: list[Path]) -> list[dict[str, str]]:
    if not paths:
        raise ValueError("no CSV inputs found")
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv(path))
    return rows


def read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def numeric(rows: Iterable[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in {None, "", "NA"}:
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
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
    }


def analyze_governor(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    events = {row["event"]: row for row in rows}
    fields = ["policy4_governor", "policy6_governor", "npu_governor"]
    before = events.get("before")
    after_set = events.get("after_set")
    after_restore = events.get("after_restore")
    return {
        "source": str(path),
        "events": events,
        "performance_set_pass": bool(after_set)
        and all(after_set[field] == "performance" for field in fields),
        "exact_restore_pass": bool(before and after_restore)
        and all(after_restore[field] == before[field] for field in fields),
    }


def analyze_governor_sessions(paths: list[Path]) -> dict[str, Any]:
    sessions = [analyze_governor(path) for path in paths]
    sessions.sort(
        key=lambda session: session["events"].get("before", {}).get("board_wall_time", "")
    )
    fields = ["policy4_governor", "policy6_governor", "npu_governor"]
    recovered_interrupted: list[str] = []
    unrecovered_interrupted: list[str] = []
    for index, session in enumerate(sessions):
        if session["exact_restore_pass"]:
            continue
        before = session["events"].get("before")
        next_before = (
            sessions[index + 1]["events"].get("before")
            if index + 1 < len(sessions)
            else None
        )
        if before and next_before and all(next_before[field] == before[field] for field in fields):
            recovered_interrupted.append(session["source"])
        else:
            unrecovered_interrupted.append(session["source"])
    final_exact_restore = bool(sessions) and sessions[-1]["exact_restore_pass"]
    return {
        "sessions": sessions,
        "performance_set_pass": bool(sessions)
        and all(session["performance_set_pass"] for session in sessions),
        "final_exact_restore_pass": final_exact_restore,
        "interrupted_sessions_recovered_before_resume": recovered_interrupted,
        "unrecovered_interrupted_sessions": unrecovered_interrupted,
        "closure_pass": final_exact_restore and not unrecovered_interrupted,
    }


def final_result(terminal: dict[str, Any]) -> dict[str, Any] | None:
    attempts = terminal.get("attempts", [])
    if not attempts:
        return None
    return attempts[-1].get("result")


def command_arg(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def sampling_from_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    command = attempt.get("command", [])
    return {
        "warmup_repetitions": int(command_arg(command, "--warmup") or -1),
        "timed_repetitions_minimum": int(command_arg(command, "--min-reps") or -1),
        "timed_window_minimum_ms": float(command_arg(command, "--min-timed-ms") or -1),
    }


def group_status(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["terminal_status"] for row in rows).items()))


def phase_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[row["phase"]].append(row)
    return result


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> int:
    args = parse_args()
    prereg = read_json(args.prereg)
    matrix_dir = args.board_run / "matrix"
    terminals = read_jsonl(matrix_dir / "matrix_terminals.jsonl")
    capability = read_json(matrix_dir / "capability_probe.json")
    runner_summary = read_json(matrix_dir / "runner_summary.json")
    telemetry_paths = sorted(args.board_run.glob("telemetry_1hz*.csv"))
    npu_load_paths = sorted(args.board_run.glob("npu_load_10hz*.csv"))
    telemetry = read_csvs(telemetry_paths)
    npu_load = read_csvs(npu_load_paths)
    telemetry_by_phase = phase_rows(telemetry)
    load_by_phase = phase_rows(npu_load)
    governor = analyze_governor_sessions(sorted(args.board_run.glob("governor_state*.tsv")))
    run_meta = read_env(args.board_run / "run_meta.env")
    amendment = prereg.get("protocol_amendment")
    correctness_adjudication = (
        read_json(args.correctness_adjudication)
        if args.correctness_adjudication
        else None
    )

    expected_count = prereg["expected_terminal_counts"]["all_matrix_terminals"]
    duplicate_ids = [item for item, count in Counter(row["cell_id"] for row in terminals).items() if count != 1]
    scopes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for terminal in terminals:
        scopes[terminal["scope"]].append(terminal)

    thermal_columns = [key for key in telemetry[0] if key.startswith("thermal_")]
    thermal_max_c = {
        key.removesuffix("_mC"): max(numeric(telemetry, key)) / 1000.0
        for key in thermal_columns
        if numeric(telemetry, key)
    }
    timed_telemetry = [row for row in telemetry if row["phase"] in {t["cell_id"] for t in terminals}]
    governor_values = {
        (row["policy4_governor"], row["policy6_governor"], row["npu_governor"])
        for row in timed_telemetry
    }
    npu_ratios = [
        float(row["npu_hz"]) / float(row["npu_max_hz"])
        for row in timed_telemetry
        if row["npu_hz"] not in {"", "NA"}
        and row["npu_max_hz"] not in {"", "NA", "0"}
    ]
    policy6_ratios = [
        float(row["policy6_khz"]) / float(row["policy6_max_khz"])
        for row in timed_telemetry
        if row["policy6_khz"] not in {"", "NA"}
        and row["policy6_max_khz"] not in {"", "NA", "0"}
    ]

    cell_rows: list[dict[str, Any]] = []
    missing_telemetry: list[str] = []
    iova_matches: list[dict[str, Any]] = []
    for terminal in terminals:
        result = final_result(terminal)
        last_attempt = terminal.get("attempts", [])[-1] if terminal.get("attempts") else None
        actual_sampling = sampling_from_attempt(last_attempt) if last_attempt else None
        shape = terminal.get("shape", {})
        mkn = shape.get("logical_mkn", [None, None, None])
        telemetry_rows = telemetry_by_phase.get(terminal["cell_id"], [])
        load_rows = load_by_phase.get(terminal["cell_id"], [])
        if terminal["scope"] in {"formal", "synthetic_m_sweep"} and terminal["terminal_status"] == "PASS":
            if not telemetry_rows or not load_rows:
                missing_telemetry.append(terminal["cell_id"])
        for attempt in terminal.get("attempts", []):
            failure = attempt.get("failure_dmesg", {})
            for line in failure.get("iova_failure_matches", []):
                iova_matches.append({"cell_id": terminal["cell_id"], "line": line})

        cell_rows.append(
            {
                "cell_id": terminal["cell_id"],
                "scope": terminal["scope"],
                "terminal_status": terminal["terminal_status"],
                "terminal_reason": terminal.get("terminal_reason"),
                "adjudicated_disposition": (
                    correctness_adjudication["terminal_disposition"]
                    if correctness_adjudication
                    and terminal["cell_id"]
                    in set(correctness_adjudication["affected_formal_cell_ids"])
                    else None
                ),
                "shape_id": shape.get("shape_id"),
                "phase": shape.get("phase"),
                "M": mkn[0],
                "K": mkn[1],
                "N": mkn[2],
                "type": terminal.get("axes", {}).get("type"),
                "core_mask": terminal.get("axes", {}).get("core_mask"),
                "b_layout": terminal.get("axes", {}).get("b_layout"),
                "attempt_count": len(terminal.get("attempts", [])),
                "warmup_repetitions": (
                    actual_sampling["warmup_repetitions"] if actual_sampling else None
                ),
                "timed_repetitions_minimum": (
                    actual_sampling["timed_repetitions_minimum"] if actual_sampling else None
                ),
                "timed_window_minimum_ms": (
                    actual_sampling["timed_window_minimum_ms"] if actual_sampling else None
                ),
                "timed_reps": result.get("timed_reps") if result else None,
                "p50_us": result.get("run_only_us", {}).get("p50") if result else None,
                "iqr_us": result.get("run_only_us", {}).get("iqr") if result else None,
                "iqr_over_p50": result.get("run_only_us", {}).get("iqr_over_p50") if result else None,
                "measured_gops": result.get("throughput", {}).get("measured_gops") if result else None,
                "compute_relative_to_2tops_core0": (
                    result.get("throughput", {}).get("measured_gops") / 2000.0
                    if result and terminal.get("axes", {}).get("type") == "int8xint8_to_int32"
                    else None
                ),
                "measured_gbps": result.get("throughput", {}).get("measured_gbps") if result else None,
                "memory_relative_to_ope0": result.get("throughput", {}).get("memory_relative_to_ope0") if result else None,
                "layout_conv_us": result.get("layout_conv_us") if result else None,
                "correctness_pass": result.get("correctness", {}).get("pass") if result else None,
                "telemetry_1hz_samples": len(telemetry_rows),
                "npu_load_10hz_samples": len(load_rows),
                "npu_load_core0_max_pct": max(numeric(load_rows, "npu_load_core0_pct"), default=None),
                "npu_load_core1_max_pct": max(numeric(load_rows, "npu_load_core1_pct"), default=None),
                "npu_load_core2_max_pct": max(numeric(load_rows, "npu_load_core2_pct"), default=None),
            }
        )

    fieldnames = list(cell_rows[0])
    args.cells_output.parent.mkdir(parents=True, exist_ok=True)
    with args.cells_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cell_rows)

    preflight = scopes["randomized_preflight"]
    formal = scopes["formal"]
    attention = scopes["attention_standby"]
    sweep_rows = scopes["synthetic_m_sweep"]
    formal_pass = [row for row in formal if row["terminal_status"] == "PASS"]
    formal_non_pass_non_na = [row for row in formal if row["terminal_status"] not in {"PASS", "N/A"}]
    correctness_failures = [
        row["cell_id"]
        for row in terminals
        if row["terminal_status"] == "FAIL_CORRECTNESS"
    ]
    adjudication_pass = False
    adjudicated_ids: set[str] = set()
    adjudication_checks: dict[str, Any] = {}
    if correctness_adjudication:
        adjudicated_ids = set(correctness_adjudication["affected_formal_cell_ids"])
        failed_by_id = {
            row["cell_id"]: row
            for row in formal
            if row["terminal_status"] == "FAIL_CORRECTNESS"
        }
        expected_axis = correctness_adjudication["axis"]
        threshold = int(correctness_adjudication["boundary"]["last_verified_correct_k"])
        affected_exact = adjudicated_ids == set(failed_by_id)
        axes_match = affected_exact and all(
            failed_by_id[cell_id]["axes"] == expected_axis for cell_id in adjudicated_ids
        )
        boundary_match = affected_exact and all(
            int(failed_by_id[cell_id]["shape"]["logical_mkn"][1]) > threshold
            for cell_id in adjudicated_ids
        )
        probe_map = {
            (int(row["K"]), row["pattern"]): row["status"]
            for row in correctness_adjudication["diagnostic_probes"]
        }
        adjacent_probe_pass = (
            probe_map.get((threshold, "constant")) == "PASS"
            and probe_map.get((threshold, "random")) == "PASS"
            and probe_map.get((threshold + 32, "constant")) == "CORRECTNESS_FAIL"
            and probe_map.get((threshold + 32, "random")) == "CORRECTNESS_FAIL"
        )
        evidence_paths_exist = all(
            Path(path).is_file()
            for path in correctness_adjudication["evidence"].values()
        )
        performance_excluded = all(
            row["cell_id"] not in {item["cell_id"] for item in formal_pass}
            for row in failed_by_id.values()
        )
        adjudication_checks = {
            "affected_cells_exactly_match_correctness_failures": affected_exact,
            "axes_match": axes_match,
            "boundary_predicate_match": boundary_match,
            "adjacent_aligned_k_constant_and_random_probes": adjacent_probe_pass,
            "evidence_paths_exist": evidence_paths_exist,
            "failed_performance_excluded": performance_excluded,
        }
        adjudication_pass = all(adjudication_checks.values())

    # Per-type/layout extrema and distribution over executable formal cells.
    performance_by_axis: dict[str, Any] = {}
    for type_id in [item["id"] for item in prereg["axes"]["types"]]:
        for layout_id in [item["id"] for item in prereg["axes"]["b_layouts"]]:
            rows = [
                final_result(row)
                for row in formal_pass
                if row["axes"]["type"] == type_id and row["axes"]["b_layout"] == layout_id
            ]
            results = [row for row in rows if row]
            if not results:
                continue
            key = f"{type_id}__{layout_id}"
            performance_by_axis[key] = {
                "cells": len(results),
                "p50_us": quantiles([float(row["run_only_us"]["p50"]) for row in results]),
                "measured_gops": quantiles([float(row["throughput"]["measured_gops"]) for row in results]),
                "measured_gbps": quantiles([float(row["throughput"]["measured_gbps"]) for row in results]),
                "memory_relative_to_ope0": quantiles(
                    [float(row["throughput"]["memory_relative_to_ope0"]) for row in results]
                ),
                "layout_conv_us": quantiles([float(row["layout_conv_us"]) for row in results]),
            }
            if type_id == "int8xint8_to_int32":
                performance_by_axis[key]["compute_relative_to_2tops_core0"] = quantiles(
                    [float(row["throughput"]["measured_gops"]) / 2000.0 for row in results]
                )

    # Census-weighted decode primitive estimate for all 343 non-lm-head linears.
    decode_estimates: dict[str, Any] = {}
    for layout_id in ("normal", "native"):
        selected = [
            row
            for row in formal_pass
            if row["axes"] == {
                "type": "int8xint8_to_int32",
                "core_mask": "core0",
                "b_layout": layout_id,
            }
            and row["shape"].get("phase") == "decode"
            and row["shape"].get("scope") in {"main_projection", "ple_supplemental"}
        ]
        core_ms = 0.0
        ple_ms = 0.0
        call_count = 0
        for row in selected:
            result = final_result(row)
            assert result is not None
            calls = int(row["shape"]["frequency"]["calls_per_model_forward"])
            contribution_ms = float(result["run_only_us"]["p50"]) * calls / 1000.0
            call_count += calls
            if row["shape"]["scope"] == "main_projection":
                core_ms += contribution_ms
            else:
                ple_ms += contribution_ms
        decode_estimates[layout_id] = {
            "shape_rows": len(selected),
            "linear_calls_per_forward": call_count,
            "core_projection_258_calls_ms": core_ms,
            "ple_85_calls_ms": ple_ms,
            "all_343_non_lm_head_linear_ms": core_ms + ple_ms,
            "ope1_bandwidth_lower_bound_ms": 171.27659799989493,
            "lower_bound_fraction_of_measured": (
                171.27659799989493 / (core_ms + ple_ms) if core_ms + ple_ms else None
            ),
            "claim_limit": "Sum of isolated primitive medians weighted by census calls; excludes runtime orchestration, lm_head, attention, norms, sampling, and all RKLLM graph effects.",
        }

    # Boundary probe terminals are reported explicitly rather than hidden in N/A.
    boundary_rows = [row for row in formal if row["shape"].get("scope") == "boundary_probe"]
    boundary = [
        {
            "axes": row["axes"],
            "terminal_status": row["terminal_status"],
            "terminal_reason": row.get("terminal_reason"),
            "result": final_result(row),
        }
        for row in boundary_rows
    ]

    # M-sweep turnover. Only core0 is executable; rejected multi-core rows remain N/A.
    sweep_pass = [row for row in sweep_rows if row["terminal_status"] == "PASS"]
    sweep_points = []
    for row in sorted(sweep_pass, key=lambda item: item["shape"]["logical_mkn"][0]):
        result = final_result(row)
        assert result is not None
        m, k, n = row["shape"]["logical_mkn"]
        ops = float(result["analytic_ops"])
        byte_count = float(result["normal_analytic_bytes"])
        sweep_points.append(
            {
                "M": m,
                "p50_us": result["run_only_us"]["p50"],
                "measured_gops": result["throughput"]["measured_gops"],
                "measured_gbps": result["throughput"]["measured_gbps"],
                "compute_relative_to_2tops_core0": result["throughput"]["measured_gops"] / 2000.0,
                "memory_relative_to_ope0": result["throughput"]["memory_relative_to_ope0"],
                "arithmetic_intensity_ops_per_byte": ops / byte_count,
            }
        )
    balance_intensity = 2000.0 / float(prereg["source"]["tier_a_gbps"])
    analytic_cross = next(
        (point["M"] for point in sweep_points if point["arithmetic_intensity_ops_per_byte"] >= balance_intensity),
        None,
    )
    max_sweep_gops = max((point["measured_gops"] for point in sweep_points), default=0.0)
    measured_saturation = next(
        (point["M"] for point in sweep_points if point["measured_gops"] >= 0.8 * max_sweep_gops),
        None,
    )

    pass_load_rows = [
        load_by_phase[row["cell_id"]]
        for row in formal_pass + sweep_pass
        if load_by_phase.get(row["cell_id"])
    ]
    flattened_load = [item for group in pass_load_rows for item in group]
    load_max = {
        "core0": max(numeric(flattened_load, "npu_load_core0_pct"), default=None),
        "core1": max(numeric(flattened_load, "npu_load_core1_pct"), default=None),
        "core2": max(numeric(flattened_load, "npu_load_core2_pct"), default=None),
    }

    telemetry_pass = all(
        [
            not missing_telemetry,
            governor_values == {("performance", "performance", "performance")},
            bool(npu_ratios) and min(npu_ratios) >= 0.95,
            bool(policy6_ratios) and min(policy6_ratios) >= 0.95,
            load_max["core0"] is not None and load_max["core0"] > 0,
        ]
    )
    count_pass = len(terminals) == expected_count and not duplicate_ids
    unadjudicated_non_pass = [
        row for row in formal_non_pass_non_na if row["cell_id"] not in adjudicated_ids
    ]
    terminal_pass = not unadjudicated_non_pass and (
        not correctness_failures or adjudication_pass
    )
    preflight_pass = group_status(preflight) == {"N/A": 39, "PASS": 15}
    runner_pass = runner_summary["status"] == "PASS_TERMINAL_MATRIX_COMPLETE"
    sampling_mismatches: list[dict[str, Any]] = []
    sampling_groups: Counter[str] = Counter()
    if amendment:
        cutoff = int(amendment["effective_after_terminal_count"])
        original_sampling = amendment["original_sampling"]
        amended_sampling = amendment["amended_sampling"]
        cutoff_identity_pass = (
            cutoff <= len(terminals)
            and terminals[cutoff - 1]["cell_id"] == amendment["last_terminal_cell_id"]
        )
        for position, terminal in enumerate(terminals, start=1):
            if terminal["scope"] not in {"formal", "synthetic_m_sweep"}:
                continue
            for attempt in terminal.get("attempts", []):
                observed = sampling_from_attempt(attempt)
                expected = original_sampling if position <= cutoff else amended_sampling
                label = "original" if position <= cutoff else "amended"
                sampling_groups[label] += 1
                if observed != expected:
                    sampling_mismatches.append(
                        {
                            "position": position,
                            "cell_id": terminal["cell_id"],
                            "expected": expected,
                            "observed": observed,
                        }
                    )
    else:
        cutoff_identity_pass = True
    protocol_amendment_pass = bool(amendment) and cutoff_identity_pass and not sampling_mismatches
    overall_pass = all(
        [
            count_pass,
            terminal_pass,
            preflight_pass,
            telemetry_pass,
            governor["performance_set_pass"],
            governor["closure_pass"],
            protocol_amendment_pass if amendment else True,
            adjudication_pass if correctness_adjudication else not correctness_failures,
            runner_pass,
            run_meta.get("run_exit_code") == "0",
            not iova_matches,
        ]
    )

    receipt = {
        "schema_version": 1,
        "stage": "OPE2",
        "status": (
            (
                "PASS_RESUMED_PROTOCOL_AMENDED_MATRIX_WITH_CAPABILITY_NA_AND_DISCARDED_BOUNDARY"
                if amendment
                else "PASS_RESUMED_MATRIX_WITH_CAPABILITY_NA"
                if runner_summary.get("resumed")
                else "PASS_SINGLE_SEGMENT_WITH_CAPABILITY_NA"
            )
            if overall_pass
            else "FAIL_REVIEW_REQUIRED"
        ),
        "prereg_ref": str(args.prereg),
        "board_run_ref": str(args.board_run),
        "counts": {
            "expected_terminals": expected_count,
            "observed_terminals": len(terminals),
            "duplicate_cell_ids": duplicate_ids,
            "by_scope": {scope: len(rows) for scope, rows in sorted(scopes.items())},
            "status_by_scope": {scope: group_status(rows) for scope, rows in sorted(scopes.items())},
        },
        "capability_adjudication": capability["decisions"],
        "protocol_amendment": (
            {
                **amendment,
                "cutoff_identity_pass": cutoff_identity_pass,
                "attempt_groups": dict(sorted(sampling_groups.items())),
                "sampling_mismatches": sampling_mismatches,
                "pass": protocol_amendment_pass,
            }
            if amendment
            else None
        ),
        "correctness_boundary_adjudication": (
            {
                **correctness_adjudication,
                "checks": adjudication_checks,
                "pass": adjudication_pass,
            }
            if correctness_adjudication
            else None
        ),
        "randomized_preflight": {
            "status_counts": group_status(preflight),
            "pass": preflight_pass,
        },
        "formal_matrix": {
            "status_counts": group_status(formal),
            "runnable_pass_cells": len(formal_pass),
            "adjudicated_discarded_cells": sorted(adjudicated_ids),
            "unexpected_non_pass_non_na": [row["cell_id"] for row in unadjudicated_non_pass],
            "performance_by_axis": performance_by_axis,
            "decode_census_weighted_primitive_estimate": decode_estimates,
            "lm_head_boundary": boundary,
        },
        "m_sweep": {
            "status_counts": group_status(sweep_rows),
            "points": sweep_points,
            "roofline_balance_intensity_ops_per_byte": balance_intensity,
            "analytic_first_compute_side_M": analytic_cross,
            "measured_max_gops": max_sweep_gops,
            "measured_first_M_at_80pct_max_gops": measured_saturation,
        },
        "six_tops_attribution": {
            "vendor_nominal": "3 NPU cores x 2 TOPS/core INT8 = 6 TOPS",
            "measured_core_masks": ["core0"],
            "unmeasured_core_masks": ["core0_1", "core0_1_2"],
            "reason": "librknnrt 2.3.2 rejects matmul core masks 0x3 and 0x7; accepting its single-core fallback would falsely label one-core work as 2/3-core evidence.",
            "core0_int8_peak_measured_gops": max(
                (
                    final_result(row)["throughput"]["measured_gops"]
                    for row in formal_pass + sweep_pass
                    if row["axes"]["type"] == "int8xint8_to_int32" and final_result(row)
                ),
                default=None,
            ),
            "claim_limit": "The 6-TOPS aggregate remains a nominal hardware label, not a measured OPE2 result on this runtime. OPE2 directly measures only the accepted core0 axis against 2 TOPS.",
        },
        "telemetry": {
            "full_system_sources": [str(path) for path in telemetry_paths],
            "npu_load_sources": [str(path) for path in npu_load_paths],
            "full_system_samples": len(telemetry),
            "npu_load_samples": len(npu_load),
            "missing_pass_cell_windows": missing_telemetry,
            "timed_governor_values": [list(item) for item in sorted(governor_values)],
            "npu_frequency_fraction_of_max": quantiles(npu_ratios),
            "policy6_frequency_fraction_of_max": quantiles(policy6_ratios),
            "thermal_max_c": thermal_max_c,
            "npu_load_max_pct": load_max,
            "pass": telemetry_pass,
        },
        "governor_guard": governor,
        "iova_guard": {
            "failure_matches": iova_matches,
            "pass": not iova_matches,
        },
        "validity_gates": {
            "terminal_count_pass": count_pass,
            "all_formal_cells_pass_na_or_adjudicated_discard": terminal_pass,
            "correctness_boundary_adjudication_pass": (
                adjudication_pass if correctness_adjudication else not correctness_failures
            ),
            "randomized_preflight_pass": preflight_pass,
            "telemetry_pass": telemetry_pass,
            "performance_governors_pass": governor["performance_set_pass"],
            "governor_closure_pass": governor["closure_pass"],
            "final_session_exact_governor_restore_pass": governor["final_exact_restore_pass"],
            "runner_pass": runner_pass,
            "board_process_exit_pass": run_meta.get("run_exit_code") == "0",
            "protocol_amendment_pass": protocol_amendment_pass if amendment else True,
        },
        "evidence_level": {
            "logical_formal_matrices": 1,
            "execution_sessions": len(governor["sessions"]),
            "interruption": (
                "board reboot; durable per-cell terminals retained and remaining cells resumed"
                if runner_summary.get("resumed")
                else None
            ),
            "replication": "WAIVED_BY_OWNER_NOT_SATISFIED",
            "tier_a_denominator": "OPE0 single formal segment under owner waiver",
        },
        "claim_limits": [
            "This is an isolated rknn_matmul primitive benchmark, not an RKLLM model run or deployment claim.",
            "Attention rows are explicit OPE5-standby N/A terminals and were not silently dropped.",
            "B0-reference prefill rows remain reference probes; only M=5142 carries formal C02 rendered-shape lineage.",
            "int4 and fp16 have measured GOPS but no invented nominal-TOPS relative score.",
            "No same-day second matrix was executed; replication is not claimed.",
            "A board reboot interrupted the matrix after durable terminal 583; this receipt merges the original and resumed telemetry sessions without treating the resume as replication.",
            (
                f"Owner-authorized protocol amendment {amendment['id']} applies only after "
                f"durable terminal {amendment['effective_after_terminal_count']}; completed "
                "cells were neither replaced nor rerun."
                if amendment
                else "No sampling-protocol amendment was applied."
            ),
        ],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "observed_terminals": len(terminals),
                "formal_statuses": group_status(formal),
                "m_sweep_statuses": group_status(sweep_rows),
                "telemetry_pass": telemetry_pass,
                "max_sweep_gops": max_sweep_gops,
            },
            ensure_ascii=False,
        )
    )
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
