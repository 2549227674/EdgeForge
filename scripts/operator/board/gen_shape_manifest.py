#!/usr/bin/env python3
"""Generate the OPE1 board matmul shape manifest from the frozen OP1 census."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from pathlib import Path
from typing import Any


MAIN_GROUPS = [
    ("q_proj_sliding", ["q_proj_sliding"]),
    ("q_proj_full", ["q_proj_full"]),
    ("kv_proj_sliding_nonshared", ["k_proj_sliding_nonshared", "v_proj_sliding_nonshared"]),
    ("kv_proj_full_nonshared", ["k_proj_full_nonshared", "v_proj_full_nonshared"]),
    ("o_proj_sliding", ["o_proj_sliding"]),
    ("o_proj_full", ["o_proj_full"]),
    ("mlp_gate_up_proj", ["mlp_gate_proj", "mlp_up_proj"]),
    ("mlp_down_proj", ["mlp_down_proj"]),
]

PLE_GROUPS = [
    ("ple_context_projection", ["ple_context_projection"]),
    ("ple_input_gate", ["ple_input_gate"]),
    ("ple_residual_projection", ["ple_residual_projection"]),
]

TYPE_SPECS = {
    "int8xint8_to_int32": {
        "api_type": "RKNN_INT8_MM_INT8_TO_INT32",
        "a_bits": 8,
        "b_bits": 8,
        "c_bits": 32,
        "k_alignment": 32,
        "n_alignment": 32,
    },
    "int4xint4_to_int16": {
        "api_type": "RKNN_INT4_MM_INT4_TO_INT16",
        "a_bits": 4,
        "b_bits": 4,
        "c_bits": 16,
        "k_alignment": 32,
        "n_alignment": 64,
    },
    "fp16xfp16_to_fp32": {
        "api_type": "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32",
        "a_bits": 16,
        "b_bits": 16,
        "c_bits": 32,
        "k_alignment": 32,
        "n_alignment": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--bandwidth-ceiling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def packed_bytes(elements: int, bits: int) -> int:
    return (elements * bits + 7) // 8


def round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def analytics(m: int, k: int, n: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, spec in TYPE_SPECS.items():
        padded_k = round_up(k, spec["k_alignment"])
        padded_n = round_up(n, spec["n_alignment"])
        logical_a = packed_bytes(m * k, spec["a_bits"])
        logical_b = packed_bytes(k * n, spec["b_bits"])
        logical_c = packed_bytes(m * n, spec["c_bits"])
        padded_a = packed_bytes(m * padded_k, spec["a_bits"])
        padded_b = packed_bytes(padded_k * padded_n, spec["b_bits"])
        padded_c = packed_bytes(m * padded_n, spec["c_bits"])
        result[name] = {
            "api_type": spec["api_type"],
            "logical_ops": 2 * m * k * n,
            "logical_io_bytes": {
                "A_read": logical_a,
                "B_read": logical_b,
                "C_write": logical_c,
                "total": logical_a + logical_b + logical_c,
            },
            "alignment": {
                "k_multiple": spec["k_alignment"],
                "n_multiple": spec["n_alignment"],
                "k_remainder": k % spec["k_alignment"],
                "n_remainder": n % spec["n_alignment"],
                "padding_required": padded_k != k or padded_n != n,
                "api_mkn": [m, padded_k, padded_n],
            },
            "api_shape_ops": 2 * m * padded_k * padded_n,
            "api_shape_io_bytes": {
                "A_read": padded_a,
                "B_read": padded_b,
                "C_write": padded_c,
                "total": padded_a + padded_b + padded_c,
            },
        }
    return result


def decode_rows(census: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = census["op_families"]["nf4_linear"]["shapes"]
    selected = {row["variant"]: row for row in rows if row["phase"] == "decode"}
    if len(selected) != 14:
        raise ValueError(f"expected 14 decode linear variants, found {len(selected)}")
    return selected


def group_facts(
    row_by_variant: dict[str, dict[str, Any]], variants: list[str]
) -> dict[str, Any]:
    source_rows = [row_by_variant[variant] for variant in variants]
    dims = {(row["dims"][1], row["dims"][2]) for row in source_rows}
    if len(dims) != 1:
        raise ValueError(f"group variants do not share K/N: {variants}: {dims}")
    k, n = next(iter(dims))
    return {
        "K": k,
        "N": n,
        "calls_per_model_forward": sum(row["calls_per_forward"] for row in source_rows),
        "observed_decode_invocations": sum(row["invocations"] for row in source_rows),
        "source_variants": variants,
        "census_ptrs": [
            f"$.op_families.nf4_linear.shapes[phase=decode,variant={variant}]"
            for variant in variants
        ],
    }


def make_linear_shape(
    shape_id: str,
    category: str,
    phase: str,
    m: int,
    facts: dict[str, Any],
    forward_count: int | None,
    anchor: dict[str, Any] | None,
) -> dict[str, Any]:
    frequency: dict[str, Any] = {
        "calls_per_model_forward": facts["calls_per_model_forward"]
    }
    if forward_count is not None:
        frequency.update(
            {
                "observed_model_forwards": forward_count,
                "observed_invocations": facts["calls_per_model_forward"] * forward_count,
            }
        )
    return {
        "shape_id": shape_id,
        "category": category,
        "phase": phase,
        "logical_mkn": [m, facts["K"], facts["N"]],
        "source_variants": facts["source_variants"],
        "census_ptrs": facts["census_ptrs"],
        "frequency": frequency,
        "prefill_anchor": anchor,
        "analytics_by_type": analytics(m, facts["K"], facts["N"]),
    }


def prompt_anchors(census: dict[str, Any]) -> list[dict[str, Any]]:
    requests = census["token_census"]["requests"]

    def request_with(field: str, value: int) -> dict[str, Any]:
        matches = [row for row in requests if row[field] == value]
        if len(matches) != 1:
            raise ValueError(f"expected one request with {field}={value}, got {len(matches)}")
        return matches[0]

    b0_min = int(census["token_census"]["b0_reference_prompt_tokens"]["min"])
    b0_p50 = int(census["token_census"]["b0_reference_prompt_tokens"]["p50"])
    b0_max = int(census["token_census"]["b0_reference_prompt_tokens"]["max"])
    c02_max = int(census["token_census"]["c02_rendered_prompt_tokens"]["max"])
    definitions = [
        (b0_min, "b0_reference_min", "b0_reference_prompt_tokens"),
        (b0_p50, "b0_reference_p50", "b0_reference_prompt_tokens"),
        (c02_max, "c02_rendered_max", "c02_rendered_prompt_tokens"),
        (b0_max, "b0_reference_max", "b0_reference_prompt_tokens"),
    ]
    result = []
    for value, role, field in definitions:
        row = request_with(field, value)
        result.append(
            {
                "M": value,
                "role": role,
                "authority": (
                    "formal_c02_rendered"
                    if field == "c02_rendered_prompt_tokens"
                    else "parallel_b0_systems_reference"
                ),
                "request_id": row["request_id"],
                "c02_rendered_prompt_tokens": row["c02_rendered_prompt_tokens"],
                "b0_reference_prompt_tokens": row["b0_reference_prompt_tokens"],
                "formal_c02_shape": field == "c02_rendered_prompt_tokens",
                "claim_limit": (
                    None
                    if field == "c02_rendered_prompt_tokens"
                    else "Reference anchor from B0; do not relabel as an observed C02 rendered prefill shape."
                ),
            }
        )
    return result


def attention_shapes(census: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests = census["token_census"]["requests"]
    context_lengths: list[int] = []
    for request in requests:
        prompt = request["c02_rendered_prompt_tokens"]
        forwards = request["incremental_decode_forwards"]
        context_lengths.extend(range(prompt + 1, prompt + forwards + 1))
    context_lengths.sort()
    total_forwards = census["token_census"]["incremental_decode_forwards"]
    if len(context_lengths) != total_forwards:
        raise ValueError("reconstructed full-attention context population mismatch")

    def observed_quantile(q: float) -> int:
        index = int(q * (len(context_lengths) - 1) + 0.5)
        return context_lengths[index]

    quantile_specs = [
        ("min", 0.0),
        ("q25", 0.25),
        ("p50", 0.5),
        ("q75", 0.75),
        ("max", 1.0),
    ]
    full_layers = census["topology"]["full_layers_all"]
    sliding_layers = census["topology"]["sliding_layers_all"]
    heads = census["topology"]["num_attention_heads"]
    result = [
        {
            "shape_id": "decode_attention_qk_sliding",
            "category": "ope5_attention_standby",
            "phase": "decode",
            "logical_mkn": [1, census["topology"]["sliding_head_dim"], census["topology"]["sliding_window"]],
            "population_role": "all_sliding_contexts_are_capped_at_512",
            "frequency": {
                "layer_calls_per_model_forward": sliding_layers,
                "observed_layer_invocations": sliding_layers * total_forwards,
                "heads_per_layer_call": heads,
                "equivalent_head_matmul_invocations": sliding_layers * total_forwards * heads,
            },
            "census_ptr": "$.op_families.attention.shapes[phase=decode,variant=sliding_attention_eager]",
            "analytics_by_type": analytics(
                1,
                census["topology"]["sliding_head_dim"],
                census["topology"]["sliding_window"],
            ),
        }
    ]
    for role, q in quantile_specs:
        n = observed_quantile(q)
        result.append(
            {
                "shape_id": f"decode_attention_qk_full_{role}",
                "category": "ope5_attention_standby",
                "phase": "decode",
                "logical_mkn": [1, census["topology"]["full_head_dim"], n],
                "population_role": role,
                "population_quantile": q,
                "frequency": {
                    "layer_calls_per_model_forward": full_layers,
                    "observed_layer_invocations_total_population": full_layers * total_forwards,
                    "heads_per_layer_call": heads,
                    "equivalent_head_matmul_invocations_total_population": full_layers
                    * total_forwards
                    * heads,
                },
                "census_ptr": "$.token_census.requests + $.op_families.attention.shapes[phase=decode,variant=full_attention_eager]",
                "analytics_by_type": analytics(1, census["topology"]["full_head_dim"], n),
            }
        )
    population = {
        "decode_forwards": total_forwards,
        "min": min(context_lengths),
        "q25": observed_quantile(0.25),
        "p50": observed_quantile(0.5),
        "q75": observed_quantile(0.75),
        "max": max(context_lengths),
        "unique_context_lengths": len(set(context_lengths)),
        "rule": "Reconstruct each request as c02_prompt+1 through c02_prompt+incremental_decode_forwards, then select nearest observed min/q25/p50/q75/max.",
    }
    return result, population


def main() -> int:
    args = parse_args()
    census = json.loads(args.census.read_text(encoding="utf-8"))
    ceiling_receipt = json.loads(args.bandwidth_ceiling.read_text(encoding="utf-8"))
    if census.get("status") != "PASS":
        raise ValueError("source census is not PASS")
    if not str(ceiling_receipt.get("status", "")).startswith("PASS"):
        raise ValueError("OPE0 bandwidth ceiling is not an accepted PASS status")

    row_by_variant = decode_rows(census)
    anchors = prompt_anchors(census)
    decode_forwards = census["token_census"]["incremental_decode_forwards"]
    linear_semantics = census["op_families"]["nf4_linear"]["op_semantics_sha"]

    main_shapes: list[dict[str, Any]] = []
    supplemental_shapes: list[dict[str, Any]] = []
    group_cache: dict[str, dict[str, Any]] = {}
    for group_id, variants in MAIN_GROUPS + PLE_GROUPS:
        group_cache[group_id] = group_facts(row_by_variant, variants)

    for group_id, _ in MAIN_GROUPS:
        facts = group_cache[group_id]
        main_shapes.append(
            make_linear_shape(
                f"decode_{group_id}",
                "formal_main_projection",
                "decode",
                1,
                facts,
                decode_forwards,
                None,
            )
        )
        for anchor in anchors:
            main_shapes.append(
                make_linear_shape(
                    f"prefill_m{anchor['M']}_{group_id}",
                    "prefill_probe_main_projection",
                    "prefill",
                    anchor["M"],
                    facts,
                    None,
                    anchor,
                )
            )

    for group_id, _ in PLE_GROUPS:
        facts = group_cache[group_id]
        supplemental_shapes.append(
            make_linear_shape(
                f"decode_{group_id}",
                "supplemental_ple_projection",
                "decode",
                1,
                facts,
                decode_forwards,
                None,
            )
        )
        for anchor in anchors:
            supplemental_shapes.append(
                make_linear_shape(
                    f"prefill_m{anchor['M']}_{group_id}",
                    "supplemental_ple_projection",
                    "prefill",
                    anchor["M"],
                    facts,
                    None,
                    anchor,
                )
            )

    core_variants = {variant for _, variants in MAIN_GROUPS for variant in variants}
    ple_variants = {variant for _, variants in PLE_GROUPS for variant in variants}
    core_rows = [row_by_variant[variant] for variant in sorted(core_variants)]
    ple_rows = [row_by_variant[variant] for variant in sorted(ple_variants)]
    core_calls = sum(row["calls_per_forward"] for row in core_rows)
    ple_calls = sum(row["calls_per_forward"] for row in ple_rows)
    core_weight_bytes = sum(
        row["dims"][1] * row["dims"][2] * row["calls_per_forward"]
        for row in core_rows
    )
    ple_weight_bytes = sum(
        row["dims"][1] * row["dims"][2] * row["calls_per_forward"]
        for row in ple_rows
    )
    all_linear_weight_bytes = core_weight_bytes + ple_weight_bytes
    if core_weight_bytes != 3_890_216_960:
        raise ValueError(f"card account mismatch: {core_weight_bytes}")
    if core_calls + ple_calls != census["topology"]["linear4bit_calls_per_forward"]:
        raise ValueError("decode linear call count does not close to census")

    hidden = census["topology"]["hidden_size"]
    vocab = census["topology"]["vocab_size"]
    lm_head_weight_elements = hidden * vocab
    ceiling_gbps = float(ceiling_receipt["ceiling_gbps"])

    lm_head = {
        "shape_id": "decode_lm_head_boundary",
        "category": "boundary_probe",
        "phase": "decode",
        "logical_mkn": [1, hidden, vocab],
        "calls_per_output_token": census["topology"]["lm_head_calls_per_output_token"],
        "census_ptr": "$.op_families.lm_head.shapes[phase=decode,variant=tied_cpu_lm_head]",
        "analytics_by_type": analytics(1, hidden, vocab),
        "boundary_reason": "N=262144 is aligned but deliberately remains a boundary probe because API allocation or implementation limits may require blocking.",
        "weight_storage_hypotheses_bytes": {
            "int8_compute_path_unproven": lm_head_weight_elements,
            "serialized_16bit_fact_compute_path_unproven": lm_head_weight_elements * 2,
        },
    }

    attention, attention_population = attention_shapes(census)
    all_linear_dims = {
        dimension
        for row in main_shapes + supplemental_shapes + [lm_head]
        for dimension in row["logical_mkn"][1:]
    }

    manifest = {
        "schema_version": 1,
        "stage": "OPE1",
        "status": "PASS_MANIFEST_FROZEN",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "shape_census_ref": str(args.census),
            "shape_census_status": census["status"],
            "shape_census_stage": census["stage"],
            "linear_op_semantics_sha": linear_semantics,
            "ope0_bandwidth_ceiling_ref": str(args.bandwidth_ceiling),
            "ope0_tier_a_ceiling_gbps": ceiling_gbps,
            "git_sha": git_head(),
        },
        "rknn_matmul_contract": {
            "header_ref": "/home/orangepi/rknn-llm/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api/include/rknn_matmul_api.h",
            "k_max": 10240,
            "alignment_by_type": {
                name: {
                    "K": spec["k_alignment"],
                    "N": spec["n_alignment"],
                }
                for name, spec in TYPE_SPECS.items()
            },
            "note": "RK3588 header contract: int8 K/N=32/32, int4 K/N=32/64, fp16 K/N=32/16 elements.",
        },
        "prefill_anchors": {
            "values": anchors,
            "lineage_rule": "The execution-card anchors intentionally mix the parallel B0 reference distribution with the C02-rendered maximum. Only M=5142 is an observed formal C02 shape; the other three remain B0-reference probes and must not be silently relabeled.",
        },
        "linear_shapes": {
            "formal_main_and_card_prefill_probes": main_shapes,
            "supplemental_ple": supplemental_shapes,
            "lm_head_boundary": lm_head,
        },
        "decode_attention_standby": {
            "population": attention_population,
            "shapes": attention,
            "scope": "QK^T one-head primitive shapes for OPE5; layer-call and equivalent eight-head invocation counts are kept separate.",
        },
        "decode_weight_traffic_accounts": {
            "unit": "bytes_per_decode_token_at_int8_weight_width",
            "core_projection_scope": {
                "included": "q/k/v/o plus MLP gate/up/down",
                "linear_calls_per_forward": core_calls,
                "weight_bytes": core_weight_bytes,
                "reproduces_execution_card_3890216960": core_weight_bytes == 3_890_216_960,
                "predicted_lower_bound_ms_at_ope0_tier_a": core_weight_bytes
                / (ceiling_gbps * 1e9)
                * 1e3,
            },
            "ple_scope": {
                "included": "PLE context projection, input gate, residual projection",
                "linear_calls_per_forward": ple_calls,
                "weight_bytes": ple_weight_bytes,
            },
            "all_343_linear_excluding_lm_head": {
                "linear_calls_per_forward": core_calls + ple_calls,
                "weight_bytes": all_linear_weight_bytes,
                "predicted_lower_bound_ms_at_ope0_tier_a": all_linear_weight_bytes
                / (ceiling_gbps * 1e9)
                * 1e3,
            },
            "lm_head_separate_hypotheses": {
                "int8_weight_bytes_compute_path_unproven": lm_head_weight_elements,
                "serialized_16bit_weight_bytes_compute_path_unproven": lm_head_weight_elements
                * 2,
            },
            "adjudication": "The card's 3,890,216,960-byte value is reproduced exactly but is not all 343 linear calls: it excludes 82,575,360 bytes across 85 PLE calls in addition to excluding lm_head. OPE4 must show both scopes and use the 3,972,792,320-byte account when claiming all non-lm-head linear weights.",
        },
        "alignment_summary": {
            "linear_k_n_dimensions": sorted(all_linear_dims),
            "all_linear_and_lm_head_shapes_zero_padding_all_types": all(
                not type_result["alignment"]["padding_required"]
                for row in main_shapes + supplemental_shapes + [lm_head]
                for type_result in row["analytics_by_type"].values()
            ),
            "attention_full_context_shapes_may_require_n_padding": True,
        },
        "counts": {
            "main_linear_shape_rows": len(main_shapes),
            "supplemental_ple_shape_rows": len(supplemental_shapes),
            "lm_head_boundary_rows": 1,
            "attention_standby_rows": len(attention),
        },
        "claim_limits": [
            "Analytic IO bytes are compulsory A/B reads plus C write, not measured DRAM traffic.",
            "B0-reference prefill anchors are not observed C02 rendered shapes.",
            "The 3,890,216,960-byte legacy/card account excludes PLE and lm_head; the all-343-linear account is 3,972,792,320 bytes before lm_head.",
            "The 3,972,792,320-byte analytic weight account is numerically equal to an M0 IOVA mapping size; arithmetic equality alone does not prove that allocation's identity or contents.",
            "lm_head 16-bit serialization is a storage fact; its board compute-path width remains unproven and is therefore split into hypotheses.",
            "Attention standby rows describe one-head QK^T primitives; eight-head multiplicity is explicit and not folded into the shape dimensions.",
        ],
    }

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output),
                "core_projection_weight_bytes": core_weight_bytes,
                "all_343_linear_weight_bytes": all_linear_weight_bytes,
                "main_shape_rows": len(main_shapes),
                "supplemental_shape_rows": len(supplemental_shapes),
                "attention_rows": len(attention),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
