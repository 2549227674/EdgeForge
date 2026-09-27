#!/usr/bin/env python3
"""Verify K-E1 preregistration lineage before any formal board execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source(path: Path, expected: str) -> dict[str, Any]:
    observed = file_sha256(path)
    return {
        "path": str(path),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "pass": observed == expected,
    }


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def main() -> int:
    args = parse_args()
    prereg = read_json(args.prereg)
    if prereg["stage"] != "OPE3-K-E1":
        raise ValueError("wrong stage")

    source_checks = {
        "ope1_manifest": check_source(
            Path(prereg["sources"]["ope1_manifest"]["path"]),
            prereg["sources"]["ope1_manifest"]["sha256"],
        ),
        "ope0_bandwidth_ceiling": check_source(
            Path(prereg["sources"]["ope0_bandwidth_ceiling"]["path"]),
            prereg["sources"]["ope0_bandwidth_ceiling"]["sha256"],
        ),
        "ope2_receipt": check_source(
            Path(prereg["sources"]["ope2_anchor"]["receipt_path"]),
            prereg["sources"]["ope2_anchor"]["receipt_sha256"],
        ),
        "ope2_cells": check_source(
            Path(prereg["sources"]["ope2_anchor"]["cells_path"]),
            prereg["sources"]["ope2_anchor"]["cells_sha256"],
        ),
        "pc_reference": check_source(
            Path(prereg["sources"]["pc_reference"]["path"]),
            prereg["sources"]["pc_reference"]["sha256"],
        ),
        "candidate": check_source(
            Path(prereg["candidate"]["source_path"]),
            prereg["candidate"]["source_sha256"],
        ),
    }

    ope1 = read_json(Path(prereg["sources"]["ope1_manifest"]["path"]))
    ope1_rows = [
        row
        for row in ope1["linear_shapes"]["formal_main_and_card_prefill_probes"]
        if row["shape_id"] == "decode_mlp_down_proj"
    ]
    if len(ope1_rows) != 1:
        raise ValueError(f"expected one OPE1 down-projection row, got {len(ope1_rows)}")
    ope1_row = ope1_rows[0]
    ope1_int8 = ope1_row["analytics_by_type"]["int8xint8_to_int32"]
    formal = prereg["formal_shape"]
    ope1_lineage_pass = all(
        [
            ope1_row["logical_mkn"] == prereg["semantic_contract"]["shape_mkn"],
            ope1_row["frequency"] == formal["frequency"],
            ope1_int8["logical_ops"] == formal["analytics"]["ops"],
            ope1_int8["logical_io_bytes"]["A_read"]
            == formal["analytics"]["activation_read_bytes"],
            ope1_int8["logical_io_bytes"]["B_read"]
            == formal["analytics"]["weight_read_bytes"],
            ope1_int8["logical_io_bytes"]["C_write"]
            == formal["analytics"]["output_write_bytes"],
            ope1_int8["logical_io_bytes"]["total"]
            == formal["analytics"]["total_unique_bytes"],
            ope1_int8["alignment"]["padding_required"] is False,
        ]
    )

    with Path(prereg["sources"]["ope2_anchor"]["cells_path"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        ope2_rows = [
            row
            for row in csv.DictReader(handle)
            if row["cell_id"]
            == prereg["cross_isa_anchors"]["npu_primitive"]["cell_id"]
        ]
    if len(ope2_rows) != 1:
        raise ValueError(f"expected one OPE2 anchor row, got {len(ope2_rows)}")
    ope2_row = ope2_rows[0]
    npu = prereg["cross_isa_anchors"]["npu_primitive"]
    ope2_anchor_pass = all(
        [
            ope2_row["terminal_status"] == "PASS",
            [int(ope2_row[key]) for key in ("M", "K", "N")]
            == prereg["semantic_contract"]["shape_mkn"],
            ope2_row["type"] == "int8xint8_to_int32",
            ope2_row["core_mask"] == "core0",
            ope2_row["b_layout"] == "normal",
            close(float(ope2_row["p50_us"]), npu["p50_us"]),
            close(float(ope2_row["measured_gops"]), npu["measured_gops"]),
            close(float(ope2_row["measured_gbps"]), npu["measured_gbps"]),
            close(
                float(ope2_row["memory_relative_to_ope0"]),
                npu["memory_relative_to_ope0"],
            ),
        ]
    )

    pc_result = read_json(Path(prereg["sources"]["pc_reference"]["path"]))
    pc_rows = [
        row for row in pc_result["benchmark"]["rows"] if row["variant"] == "mlp_down_proj"
    ]
    if len(pc_rows) != 1:
        raise ValueError(f"expected one PC anchor row, got {len(pc_rows)}")
    pc_row = pc_rows[0]
    pc = prereg["cross_isa_anchors"]["pc_sm89_reference"]
    pc_anchor_pass = all(
        [
            pc_row["dims"] == pc["shape"],
            close(pc_row["eager"]["p50"], pc["p50_us"]),
            pc_row["analytic_bytes"] == pc["analytic_bytes"],
        ]
    )

    bandwidth = prereg["sources"]["ope0_bandwidth_ceiling"][
        "ceiling_gbps_decimal"
    ]
    analytics = formal["analytics"]
    derived_intensity = analytics["ops"] / analytics["total_unique_bytes"]
    derived_lower_us = analytics["total_unique_bytes"] / (bandwidth * 1e9) * 1e6
    analytic_pass = all(
        [
            close(derived_intensity, analytics["arithmetic_intensity_ops_per_byte"]),
            close(derived_lower_us, analytics["ope0_roofline_lower_bound_us"]),
        ]
    )

    prereg_internal_pass = all(
        [
            prereg["status"] == "PREREGISTERED_FROZEN_NOT_RUN",
            prereg["measurement_protocol"]["warmup_repetitions_per_variant"] == 50,
            prereg["measurement_protocol"]["timed_repetitions_per_variant"] == 200,
            prereg["measurement_protocol"]["formal_segments"] == 2,
            prereg["measurement_protocol"][
                "minimum_same_day_segment_separation_seconds"
            ]
            == 1800,
            prereg["correctness_protocol"]["tolerance"] == {"atol": 0, "rtol": 0},
        ]
    )

    checks = {
        "all_source_hashes_pass": all(item["pass"] for item in source_checks.values()),
        "ope1_shape_frequency_analytics_lineage_pass": ope1_lineage_pass,
        "ope2_exact_anchor_pass": ope2_anchor_pass,
        "pc_reference_exact_anchor_pass": pc_anchor_pass,
        "derived_analytics_pass": analytic_pass,
        "prereg_internal_contract_pass": prereg_internal_pass,
    }
    passed = all(checks.values())
    receipt = {
        "schema_version": 1,
        "stage": "OPE3-K-E1-prereg-verification",
        "status": "PASS" if passed else "FAIL",
        "prereg_ref": str(args.prereg),
        "prereg_sha256": file_sha256(args.prereg),
        "source_checks": source_checks,
        "lineage_checks": {
            "ope1": ope1_lineage_pass,
            "ope2": ope2_anchor_pass,
            "pc_reference": pc_anchor_pass,
        },
        "derived": {
            "arithmetic_intensity_ops_per_byte": derived_intensity,
            "ope0_roofline_lower_bound_us": derived_lower_us,
        },
        "checks": checks,
        "pass": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], **checks}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
