#!/usr/bin/env python3
"""Summarize two formal OPE0 board segments into machine receipts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any


RESULT_MARKER = re.compile(r"OPE0_RESULT_JSON=(\{.*\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bandwidth-output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    if any(None in row for row in rows):
        raise ValueError(f"malformed CSV column count: {path}")
    return rows


def read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    q25, _, q75 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "min": min(ordered),
        "p25": q25,
        "p50": statistics.median(ordered),
        "p75": q75,
        "max": max(ordered),
        "iqr": q75 - q25,
    }


def numeric(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def parse_time(text: str) -> dt.datetime:
    # Python accepts both ISO decimal separators, but normalize for clarity.
    match = re.match(r"^(.*T\d\d:\d\d:\d\d),(\d+)([+-].*)$", text)
    if match:
        text = f"{match.group(1)}.{match.group(2)}{match.group(3)}"
    return dt.datetime.fromisoformat(text)


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def analyze_governors(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_event = {row["event"]: row for row in rows}
    required = {"before", "after_set", "after_restore"}
    if set(by_event) != required:
        raise ValueError(f"unexpected governor events in {path}: {sorted(by_event)}")
    fields = ["policy4_governor", "policy6_governor", "npu_governor"]
    set_pass = all(by_event["after_set"][field] == "performance" for field in fields)
    restore_pass = all(
        by_event["after_restore"][field] == by_event["before"][field]
        for field in fields
    )
    return {
        "before": {field: by_event["before"][field] for field in fields},
        "after_set": {field: by_event["after_set"][field] for field in fields},
        "after_restore": {
            field: by_event["after_restore"][field] for field in fields
        },
        "performance_set_pass": set_pass,
        "exact_restore_pass": restore_pass,
    }


def analyze_segment(
    segment: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    path = Path(segment["path"])
    bandwidth_rows = read_csv(path / "bandwidth_samples.csv")
    expected_sizes = config["bandwidth_ceiling"]["sizes_mib"]
    expected_reps = config["bandwidth_ceiling"]["measured_repetitions_per_size"]
    stable_sizes = config["bandwidth_ceiling"]["stable_segment_sizes_mib"]

    by_size: dict[int, dict[str, Any]] = {}
    bandwidth_counts_pass = True
    for size in expected_sizes:
        values = [
            float(row["gbps"])
            for row in bandwidth_rows
            if int(row["size_mib"]) == size
        ]
        bandwidth_counts_pass &= len(values) == expected_reps
        by_size[size] = {"reps": len(values), "gbps": quantiles(values)}

    stable_p50 = [by_size[size]["gbps"]["p50"] for size in stable_sizes]
    ceiling_gbps = statistics.median(stable_p50)
    stable_span_fraction = (max(stable_p50) - min(stable_p50)) / ceiling_gbps
    stable_pass = (
        stable_span_fraction
        <= config["bandwidth_ceiling"]["stable_segment_max_span_fraction"]
    )

    matmul_text = (path / "matmul.log").read_text(encoding="utf-8", errors="replace")
    marker = RESULT_MARKER.search(matmul_text)
    if not marker:
        raise ValueError(f"missing matmul result marker: {path / 'matmul.log'}")
    matmul = json.loads(marker.group(1))
    matmul_exit_code = int((path / "matmul_exit_code.txt").read_text().strip())
    matmul_config = config["matmul_selfcal"]
    matmul_pass = (
        matmul_exit_code == 0
        and matmul["shape_mkn"] == matmul_config["shape_mkn"]
        and matmul["warmup_reps"] == matmul_config["warmup_repetitions"]
        and matmul["timed_reps"] == matmul_config["measured_repetitions"]
        and matmul["core_mask_ret"] == 0
        and matmul["correctness"]["pass"] is True
        and matmul["correctness"]["mismatch_count"] == 0
    )

    npu_load_rows = read_csv(path / "npu_load_10hz.csv")
    load_columns = [
        "npu_load_core0_pct",
        "npu_load_core1_pct",
        "npu_load_core2_pct",
    ]
    load_max_by_core = {
        column: int(max(numeric(npu_load_rows, column))) for column in load_columns
    }
    load_max = max(load_max_by_core.values())
    load_pass = load_max > config["validity_gates"]["rknpu_load_max_percent_must_exceed"]

    telemetry_rows = read_csv(path / "telemetry_1hz.csv")
    timed_rows = [
        row for row in telemetry_rows if row["phase"] in {"bandwidth", "matmul"}
    ]
    bandwidth_telemetry = [
        row for row in telemetry_rows if row["phase"] == "bandwidth"
    ]
    matmul_telemetry = [row for row in telemetry_rows if row["phase"] == "matmul"]
    telemetry_present_pass = bool(bandwidth_telemetry and matmul_telemetry)

    frequency_checks: dict[str, dict[str, float | bool]] = {}
    frequency_specs = [
        (
            "bandwidth_policy4",
            bandwidth_telemetry,
            "policy4_khz",
            "policy4_max_khz",
            config["validity_gates"]["a76_frequency_min_fraction_of_policy_max"],
        ),
        (
            "bandwidth_policy6",
            bandwidth_telemetry,
            "policy6_khz",
            "policy6_max_khz",
            config["validity_gates"]["a76_frequency_min_fraction_of_policy_max"],
        ),
        (
            "matmul_npu",
            matmul_telemetry,
            "npu_hz",
            "npu_max_hz",
            config["validity_gates"]["npu_frequency_min_fraction_of_max_during_matmul"],
        ),
    ]
    for name, rows, current_key, max_key, threshold in frequency_specs:
        ratios = [
            float(row[current_key]) / float(row[max_key])
            for row in rows
            if row[current_key] not in {"", "NA"}
            and row[max_key] not in {"", "NA", "0"}
        ]
        frequency_checks[name] = {
            "samples": len(ratios),
            "min_fraction_of_max": min(ratios) if ratios else 0.0,
            "p50_fraction_of_max": statistics.median(ratios) if ratios else 0.0,
            "threshold": threshold,
            "pass": bool(ratios) and min(ratios) >= threshold,
        }

    governor_values = {
        (
            row["policy4_governor"],
            row["policy6_governor"],
            row["npu_governor"],
        )
        for row in timed_rows
    }
    telemetry_governor_pass = governor_values == {
        ("performance", "performance", "performance")
    }

    thermal_columns = [key for key in telemetry_rows[0] if key.startswith("thermal_")]
    thermal_max_c = {
        key.removesuffix("_mC"): max(numeric(telemetry_rows, key)) / 1000.0
        for key in thermal_columns
    }
    uptime_values = numeric(telemetry_rows, "uptime_s")
    cadence = [right - left for left, right in zip(uptime_values, uptime_values[1:])]

    governor = analyze_governors(path / "governor_state.tsv")
    run_meta = read_env(path / "run_meta.env")
    overall_pass = all(
        [
            bandwidth_counts_pass,
            stable_pass,
            matmul_pass,
            load_pass,
            telemetry_present_pass,
            telemetry_governor_pass,
            governor["performance_set_pass"],
            governor["exact_restore_pass"],
            *(item["pass"] for item in frequency_checks.values()),
            run_meta.get("run_exit_code") == "0",
        ]
    )

    return {
        "label": segment["label"],
        "path": segment["path"],
        "controller_before": segment["controller_before"],
        "controller_after": segment["controller_after"],
        "board_monotonic_window_s": {
            "start": float(run_meta["board_start_uptime_s"]),
            "end": float(run_meta["board_end_uptime_s"]),
        },
        "bandwidth": {
            "sizes": {str(key): value for key, value in by_size.items()},
            "sample_count_gate_pass": bandwidth_counts_pass,
            "stable_segment_sizes_mib": stable_sizes,
            "stable_segment_span_fraction": stable_span_fraction,
            "stable_segment_gate_pass": stable_pass,
            "ceiling_gbps": ceiling_gbps,
        },
        "matmul_selfcal": {
            **matmul,
            "process_exit_code": matmul_exit_code,
            "selfcal_gate_pass": matmul_pass,
            "rknpu_load_10hz_samples": len(npu_load_rows),
            "rknpu_load_max_by_core_pct": load_max_by_core,
            "rknpu_load_gate_pass": load_pass,
        },
        "telemetry": {
            "samples": len(telemetry_rows),
            "bandwidth_window_samples": len(bandwidth_telemetry),
            "matmul_window_samples": len(matmul_telemetry),
            "observed_cadence_seconds": quantiles(cadence),
            "timed_window_governor_values": [list(value) for value in governor_values],
            "timed_window_governor_gate_pass": telemetry_governor_pass,
            "frequency_checks": frequency_checks,
            "thermal_max_c": thermal_max_c,
        },
        "governor_guard": governor,
        "run_meta": run_meta,
        "segment_pass": overall_pass,
    }


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    manifest = read_json(args.run_manifest)
    segments = [analyze_segment(item, config) for item in manifest["formal_segments"]]
    owner_waiver = manifest.get("owner_waiver")
    expected_segments = config["bandwidth_ceiling"]["formal_segments"]
    waiver_active = bool(
        owner_waiver
        and owner_waiver.get("skip_second_formal_segment") is True
        and len(segments) == 1
    )
    if len(segments) != expected_segments and not waiver_active:
        raise ValueError(
            "formal segment count does not match preregistration and no applicable "
            "owner waiver is recorded"
        )

    segment_ceilings = [item["bandwidth"]["ceiling_gbps"] for item in segments]
    tier_a_ceiling = statistics.median(segment_ceilings)
    if len(segments) == expected_segments:
        first_after = parse_time(segments[0]["controller_after"])
        second_before = parse_time(segments[1]["controller_before"])
        gap_seconds: float | None = (second_before - first_after).total_seconds()
        gap_pass = gap_seconds >= config["bandwidth_ceiling"]["minimum_gap_seconds"]
        ceiling_delta: float | None = (
            abs(segment_ceilings[1] - segment_ceilings[0]) / tier_a_ceiling
        )
        replication_pass = (
            ceiling_delta
            <= config["bandwidth_ceiling"]["between_segment_max_relative_delta"]
        )
        replication_status = "PASS" if gap_pass and replication_pass else "FAIL"
    else:
        gap_seconds = None
        gap_pass = False
        ceiling_delta = None
        replication_pass = False
        replication_status = "WAIVED_BY_OWNER_NOT_SATISFIED"

    overall_pass = all(item["segment_pass"] for item in segments) and (
        (gap_pass and replication_pass) or waiver_active
    )
    overall_status = (
        "PASS_SINGLE_SEGMENT_OWNER_WAIVER"
        if overall_pass and waiver_active
        else ("PASS" if overall_pass else "FAIL")
    )

    thermal_columns = sorted(
        {
            key
            for segment in segments
            for key in segment["telemetry"]["thermal_max_c"]
        }
    )
    thermal_max_c = {
        key: max(segment["telemetry"]["thermal_max_c"][key] for segment in segments)
        for key in thermal_columns
    }
    matmul_p50_values = [
        item["matmul_selfcal"]["run_only_us"]["p50"] for item in segments
    ]

    receipt = {
        "schema_version": "board-v1",
        "stage": "OPE0",
        "status": overall_status,
        "measurement_config_ref": str(args.config),
        "run_manifest_ref": str(args.run_manifest),
        "environment_identity_ref": config["environment_identity_ref"],
        "clock_authority": config["clock_protocol"],
        "git_sha": git_head(),
        "formal_segments": segments,
        "replication": {
            "status": replication_status,
            "gap_seconds": gap_seconds,
            "minimum_gap_seconds": config["bandwidth_ceiling"]["minimum_gap_seconds"],
            "gap_gate_pass": gap_pass,
            "segment_ceiling_gbps": segment_ceilings,
            "relative_delta": ceiling_delta,
            "maximum_relative_delta": config["bandwidth_ceiling"][
                "between_segment_max_relative_delta"
            ],
            "replication_gate_pass": replication_pass,
            "owner_waiver": owner_waiver,
            "claim": (
                "single formal segment only; same-day replication was not executed "
                "and is not claimed"
                if waiver_active
                else "two formal segments completed"
            ),
        },
        "bandwidth_ceiling": {
            "status": overall_status,
            "tier": "Tier-A",
            "value_gbps": tier_a_ceiling,
            "definition": (
                "single formal segment stable-size p50 after explicit owner waiver "
                "of the preregistered second segment"
                if waiver_active
                else config["bandwidth_ceiling"]["tier_a_denominator"]
            ),
            "kernel": config["bandwidth_ceiling"]["kernel"],
            "analytic_bytes_per_element": config["bandwidth_ceiling"][
                "analytic_bytes_per_element"
            ],
            "stable_segment_sizes_mib": config["bandwidth_ceiling"][
                "stable_segment_sizes_mib"
            ],
            "taskset_cpus": config["bandwidth_ceiling"]["taskset_cpus"],
            "threads": config["bandwidth_ceiling"]["threads"],
        },
        "matmul_selfcal": {
            "status": "PASS"
            if all(item["matmul_selfcal"]["selfcal_gate_pass"] for item in segments)
            else "FAIL",
            "shape_mkn": config["matmul_selfcal"]["shape_mkn"],
            "type": config["matmul_selfcal"]["type"],
            "exact_mismatch_count_per_segment": [
                item["matmul_selfcal"]["correctness"]["mismatch_count"]
                for item in segments
            ],
            "run_only_p50_us_per_segment": matmul_p50_values,
            "run_only_p50_us_median": statistics.median(matmul_p50_values),
            "rknpu_load_max_pct_per_segment": [
                max(item["matmul_selfcal"]["rknpu_load_max_by_core_pct"].values())
                for item in segments
            ],
        },
        "thermal_max_c": thermal_max_c,
        "governor_restore_pass": all(
            item["governor_guard"]["exact_restore_pass"] for item in segments
        ),
        "invalid_harness_attempts": manifest.get("invalid_harness_attempts", []),
        "scope_guards": config["scope_guards"],
        "negative_statement": (
            "No driver, device tree, kernel, RKNN/RKLLM shared library, frozen "
            "model artifact, multi-core RKLLM path, or model-quality evaluation was "
            "modified or executed. The only temporary system changes were the "
            "preregistered CPU/NPU governors, restored exactly after every attempt."
        ),
    }

    bandwidth_receipt = {
        "schema_version": 1,
        "stage": "OPE0-bandwidth-ceiling",
        "status": overall_status,
        "tier": "Tier-A",
        "ceiling_gbps": tier_a_ceiling,
        "segment_ceiling_gbps": segment_ceilings,
        "between_segment_relative_delta": ceiling_delta,
        "stable_segment_sizes_mib": config["bandwidth_ceiling"][
            "stable_segment_sizes_mib"
        ],
        "kernel": config["bandwidth_ceiling"]["kernel"],
        "definition": receipt["bandwidth_ceiling"]["definition"],
        "replication_status": replication_status,
        "ope0_receipt_ref": str(args.output),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.bandwidth_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.bandwidth_output.write_text(
        json.dumps(bandwidth_receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": overall_status,
                "tier_a_ceiling_gbps": tier_a_ceiling,
                "segment_ceiling_gbps": segment_ceilings,
                "replication_relative_delta": ceiling_delta,
                "gap_seconds": gap_seconds,
                "matmul_p50_us": matmul_p50_values,
            },
            ensure_ascii=False,
        )
    )
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
