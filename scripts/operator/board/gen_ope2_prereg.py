#!/usr/bin/env python3
"""Freeze the OPE2 RKNPU matmul execution matrix from the OPE1 manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


TYPE_AXES = [
    {
        "id": "int8xint8_to_int32",
        "api_type": "RKNN_INT8_MM_INT8_TO_INT32",
        "cli": "int8",
        "nominal_tops_per_core": 2.0,
    },
    {
        "id": "int4xint4_to_int16",
        "api_type": "RKNN_INT4_MM_INT4_TO_INT16",
        "cli": "int4",
        "nominal_tops_per_core": None,
        "claim_limit": "No vendor nominal TOPS denominator is frozen for int4; report measured GOPS only.",
    },
    {
        "id": "fp16xfp16_to_fp32",
        "api_type": "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32",
        "cli": "fp16",
        "nominal_tops_per_core": None,
        "claim_limit": "No vendor nominal TOPS denominator is frozen for fp16; report measured GOPS only.",
    },
]

CORE_AXES = [
    {"id": "core0", "api_mask": "RKNN_NPU_CORE_0", "cli": "core0", "core_count": 1},
    {"id": "core0_1", "api_mask": "RKNN_NPU_CORE_0_1", "cli": "core01", "core_count": 2},
    {"id": "core0_1_2", "api_mask": "RKNN_NPU_CORE_0_1_2", "cli": "core012", "core_count": 3},
]

LAYOUT_AXES = [
    {
        "id": "normal",
        "b_layout": "RKNN_MM_LAYOUT_NORM",
        "cli": "normal",
        "ac_layout": "RKNN_MM_LAYOUT_NORM",
    },
    {
        "id": "native",
        "b_layout": "RKNN_MM_LAYOUT_NATIVE",
        "cli": "native",
        "ac_layout": "RKNN_MM_LAYOUT_NORM",
        "conversion_timing": "rknn_B_normal_layout_to_native_layout wall time, reported separately and excluded from run_only_us",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bandwidth-ceiling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_shape(row: dict[str, Any], scope: str) -> dict[str, Any]:
    lineage = row.get("prefill_anchor")
    return {
        "shape_id": row["shape_id"],
        "scope": scope,
        "category": row["category"],
        "phase": row["phase"],
        "logical_mkn": row["logical_mkn"],
        "census_ptrs": row.get("census_ptrs", [row.get("census_ptr")]),
        "frequency": row.get(
            "frequency",
            {"calls_per_model_forward": row.get("calls_per_output_token")},
        ),
        "prefill_lineage": lineage,
        "analytics_by_type": row["analytics_by_type"],
        "matrix_disposition": "SCHEDULED",
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen preregistration: {args.output}")

    manifest = read_json(args.manifest)
    bandwidth = read_json(args.bandwidth_ceiling)
    linear = manifest["linear_shapes"]

    scheduled_shapes: list[dict[str, Any]] = []
    scheduled_shapes.extend(
        copy_shape(row, "main_projection")
        for row in linear["formal_main_and_card_prefill_probes"]
    )
    scheduled_shapes.extend(
        copy_shape(row, "ple_supplemental") for row in linear["supplemental_ple"]
    )
    scheduled_shapes.append(copy_shape(linear["lm_head_boundary"], "boundary_probe"))

    standby_shapes = []
    for row in manifest["decode_attention_standby"]["shapes"]:
        standby_shapes.append(
            {
                "shape_id": row["shape_id"],
                "scope": "attention_standby",
                "category": row["category"],
                "phase": row["phase"],
                "logical_mkn": row["logical_mkn"],
                "census_ptrs": row.get("census_ptrs", [row.get("census_ptr")]),
                "analytics_by_type": row["analytics_by_type"],
                "matrix_disposition": "N/A_OPE5_STANDBY",
                "reason": "The R2 card explicitly reserves decode-attention primitive replay for OPE5; OPE2 is the linear/lm_head matmul matrix.",
            }
        )

    cross_product_width = len(TYPE_AXES) * len(CORE_AXES) * len(LAYOUT_AXES)
    prereg = {
        "schema_version": 1,
        "stage": "OPE2",
        "status": "PREREGISTERED_FROZEN",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "source": {
            "ope1_manifest": str(args.manifest),
            "ope1_status": manifest["status"],
            "bandwidth_ceiling": str(args.bandwidth_ceiling),
            "tier_a_gbps": bandwidth["ceiling_gbps"],
            "tier_a_evidence_limit": "single formal OPE0 segment; owner waived the second segment",
        },
        "axes": {
            "types": TYPE_AXES,
            "core_masks": CORE_AXES,
            "b_layouts": LAYOUT_AXES,
        },
        "measurement_protocol": {
            "cpu_taskset": "7",
            "governors_during_window": {
                "policy4": "performance",
                "policy6": "performance",
                "npu": "performance",
            },
            "warmup_repetitions": 50,
            "timed_repetitions_minimum": 200,
            "timed_window_minimum_ms": 1250,
            "summary": ["p25_us", "p50_us", "p75_us", "iqr_us"],
            "layout_pairing": "For each shape/type/core mask, normal and native B-layout cells are adjacent; conversion time is never included in run_only_us.",
            "correctness": {
                "preflight": "Three deterministic randomized seeds for every type/core/layout axis on M=4,K=64,N=64; integer paths exact, fp16-to-fp32 atol=1e-3 and rtol=1e-5.",
                "formal_cells": "Constant-input analytic oracle; full output scan through 1,048,576 elements, otherwise 4,096 deterministic samples including both edges.",
            },
            "unstable_cell_rule": "If run-only IQR/p50 exceeds 0.30, rerun that cell once. If the retry also exceeds 0.30, retain both attempts and terminally discard unless telemetry supplies a throttle explanation.",
            "formal_segments": 1,
            "replication_note": "Owner explicitly waived the same-day second segment; do not claim replicated-matrix evidence.",
            "telemetry": {
                "full_system_hz": 1,
                "rknpu_load_hz": 10,
                "missing_window_rule": "A scheduled cell lacking both phase-tagged telemetry streams is invalid.",
            },
        },
        "formal_shapes": scheduled_shapes,
        "attention_standby_shapes": standby_shapes,
        "m_sweep": {
            "status": "SCHEDULED_SYNTHETIC_ISOLATED_ONLY",
            "source_shape_id": "decode_mlp_down_proj",
            "m_values": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            "k": 10240,
            "n": 2560,
            "type_axis": "int8xint8_to_int32",
            "core_masks": [item["id"] for item in CORE_AXES],
            "b_layout": "native",
            "claim_limit": "Synthetic M values are used only to locate the primitive-level turnover and never enter formal census-weighted scores; M=1 reuses the formal down-projection lineage.",
        },
        "expected_terminal_counts": {
            "randomized_preflight_cells": len(TYPE_AXES)
            * len(CORE_AXES)
            * len(LAYOUT_AXES)
            * 3,
            "scheduled_formal_shape_rows": len(scheduled_shapes),
            "scheduled_formal_cells": len(scheduled_shapes) * cross_product_width,
            "attention_standby_shape_rows": len(standby_shapes),
            "attention_standby_na_cells": len(standby_shapes) * cross_product_width,
            "synthetic_m_sweep_cells": 12 * len(CORE_AXES),
            "all_matrix_terminals": (
                len(TYPE_AXES) * len(CORE_AXES) * len(LAYOUT_AXES) * 3
                + len(scheduled_shapes) * cross_product_width
                + len(standby_shapes) * cross_product_width
                + 12 * len(CORE_AXES)
            ),
        },
        "failure_policy": {
            "boundary_probe": "API create/allocation/run rejection is a valid N/A boundary result when captured with the exact stage/return code.",
            "iova_guard": "On any failed multi-core create/run, capture dmesg; if IOVA or dma-map keywords occur, stop that cell immediately, retain evidence, and do not modify the driver, kernel, DT, or runtime.",
            "implementation_faults": "Harness, packing, layout, and orchestration faults must be diagnosed and fixed; they are not grounds for silently dropping an axis.",
        },
        "claim_limits": [
            "rknn_matmul_run is an isolated primitive benchmark and does not load an RKLLM model.",
            "Multi-core primitive measurements do not reopen or alter the frozen multi-core LLM deployment boundary.",
            "Only int8 has the frozen 2 TOPS/core nominal denominator; int4/fp16 measured GOPS are not divided by an invented peak.",
            "All OPE0 memory-relative scores inherit its single-segment owner-waiver evidence level.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prereg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": prereg["status"],
                **prereg["expected_terminal_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
