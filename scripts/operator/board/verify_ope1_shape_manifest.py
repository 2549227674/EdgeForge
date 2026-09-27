#!/usr/bin/env python3
"""Independent arithmetic and lineage verification for the OPE1 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPECS = {
    "int8xint8_to_int32": (8, 8, 32, 32, 32),
    "int4xint4_to_int16": (4, 4, 16, 32, 64),
    "fp16xfp16_to_fp32": (16, 16, 32, 32, 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nbytes(elements: int, bits: int) -> int:
    return (elements * bits + 7) // 8


def up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def expected_analytics(m: int, k: int, n: int, name: str) -> dict[str, Any]:
    a_bits, b_bits, c_bits, k_align, n_align = SPECS[name]
    pk, pn = up(k, k_align), up(n, n_align)
    logical = [nbytes(m * k, a_bits), nbytes(k * n, b_bits), nbytes(m * n, c_bits)]
    padded = [nbytes(m * pk, a_bits), nbytes(pk * pn, b_bits), nbytes(m * pn, c_bits)]
    return {
        "logical_ops": 2 * m * k * n,
        "logical_total": sum(logical),
        "api_mkn": [m, pk, pn],
        "api_ops": 2 * m * pk * pn,
        "api_total": sum(padded),
        "padding": pk != k or pn != n,
        "k_remainder": k % k_align,
        "n_remainder": n % n_align,
    }


def verify_shape(row: dict[str, Any]) -> int:
    m, k, n = row["logical_mkn"]
    checks = 0
    for name, actual in row["analytics_by_type"].items():
        expected = expected_analytics(m, k, n, name)
        assert actual["logical_ops"] == expected["logical_ops"]
        assert actual["logical_io_bytes"]["total"] == expected["logical_total"]
        assert actual["alignment"]["api_mkn"] == expected["api_mkn"]
        assert actual["alignment"]["padding_required"] == expected["padding"]
        assert actual["alignment"]["k_remainder"] == expected["k_remainder"]
        assert actual["alignment"]["n_remainder"] == expected["n_remainder"]
        assert actual["api_shape_ops"] == expected["api_ops"]
        assert actual["api_shape_io_bytes"]["total"] == expected["api_total"]
        checks += 8
    return checks


def main() -> int:
    args = parse_args()
    census = json.loads(args.census.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS_MANIFEST_FROZEN"
    decode = [
        row
        for row in census["op_families"]["nf4_linear"]["shapes"]
        if row["phase"] == "decode"
    ]
    assert len(decode) == 14

    core = [row for row in decode if not row["variant"].startswith("ple_")]
    ple = [row for row in decode if row["variant"].startswith("ple_")]
    core_calls = sum(row["calls_per_forward"] for row in core)
    ple_calls = sum(row["calls_per_forward"] for row in ple)
    core_bytes = sum(
        row["dims"][1] * row["dims"][2] * row["calls_per_forward"] for row in core
    )
    ple_bytes = sum(
        row["dims"][1] * row["dims"][2] * row["calls_per_forward"] for row in ple
    )
    assert core_calls == 258
    assert ple_calls == 85
    assert core_calls + ple_calls == 343
    assert core_bytes == 3_890_216_960
    assert ple_bytes == 82_575_360
    assert core_bytes + ple_bytes == 3_972_792_320

    accounts = manifest["decode_weight_traffic_accounts"]
    assert accounts["core_projection_scope"]["weight_bytes"] == core_bytes
    assert accounts["ple_scope"]["weight_bytes"] == ple_bytes
    assert accounts["all_343_linear_excluding_lm_head"]["weight_bytes"] == core_bytes + ple_bytes
    assert accounts["all_343_linear_excluding_lm_head"]["linear_calls_per_forward"] == 343

    rows = (
        manifest["linear_shapes"]["formal_main_and_card_prefill_probes"]
        + manifest["linear_shapes"]["supplemental_ple"]
        + [manifest["linear_shapes"]["lm_head_boundary"]]
        + manifest["decode_attention_standby"]["shapes"]
    )
    arithmetic_checks = sum(verify_shape(row) for row in rows)

    source_by_variant = {row["variant"]: row for row in decode}
    lineage_checks = 0
    for row in (
        manifest["linear_shapes"]["formal_main_and_card_prefill_probes"]
        + manifest["linear_shapes"]["supplemental_ple"]
    ):
        source_rows = [source_by_variant[name] for name in row["source_variants"]]
        expected_k_n = {(item["dims"][1], item["dims"][2]) for item in source_rows}
        assert len(expected_k_n) == 1
        assert tuple(row["logical_mkn"][1:]) == next(iter(expected_k_n))
        assert row["frequency"]["calls_per_model_forward"] == sum(
            item["calls_per_forward"] for item in source_rows
        )
        if row["phase"] == "decode":
            assert row["frequency"]["observed_invocations"] == sum(
                item["invocations"] for item in source_rows
            )
        lineage_checks += 4

    requests = census["token_census"]["requests"]
    lengths: list[int] = []
    for request in requests:
        lengths.extend(
            range(
                request["c02_rendered_prompt_tokens"] + 1,
                request["c02_rendered_prompt_tokens"]
                + request["incremental_decode_forwards"]
                + 1,
            )
        )
    lengths.sort()

    def oq(q: float) -> int:
        return lengths[int(q * (len(lengths) - 1) + 0.5)]

    population = manifest["decode_attention_standby"]["population"]
    assert len(lengths) == census["token_census"]["incremental_decode_forwards"]
    assert population["min"] == min(lengths) == 807
    assert population["q25"] == oq(0.25)
    assert population["p50"] == oq(0.5)
    assert population["q75"] == oq(0.75)
    assert population["max"] == max(lengths) == 7359

    anchors = manifest["prefill_anchors"]["values"]
    assert [item["M"] for item in anchors] == [813, 1445, 5142, 7082]
    assert [item["formal_c02_shape"] for item in anchors] == [False, False, True, False]

    receipt = {
        "schema_version": 1,
        "stage": "OPE1-independent-verification",
        "status": "PASS",
        "source_census_ref": str(args.census),
        "manifest_ref": str(args.manifest),
        "checks": {
            "decode_variant_count": len(decode),
            "core_projection_calls": core_calls,
            "ple_calls": ple_calls,
            "all_linear_calls_excluding_lm_head": core_calls + ple_calls,
            "core_projection_weight_bytes": core_bytes,
            "ple_weight_bytes": ple_bytes,
            "all_343_linear_weight_bytes": core_bytes + ple_bytes,
            "shape_rows_verified": len(rows),
            "arithmetic_field_checks": arithmetic_checks,
            "lineage_checks": lineage_checks,
            "attention_decode_forwards_reconstructed": len(lengths),
            "attention_context_range": [min(lengths), max(lengths)],
            "prefill_anchor_values": [item["M"] for item in anchors],
        },
        "adjudication": "3,890,216,960 bytes is exact for the 258 q/k/v/o+MLP calls, not for all 343 non-lm-head linear calls. Adding the 85 PLE calls yields 3,972,792,320 bytes.",
        "claim_limit": "This verification checks lineage and analytic arithmetic only; it is not a board performance run. The 3,972,792,320-byte result happens to equal an M0 IOVA map size, but equality does not establish allocation identity.",
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", **receipt["checks"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
