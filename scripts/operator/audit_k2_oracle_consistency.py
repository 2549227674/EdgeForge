#!/usr/bin/env python3
"""Audit whether K2's two frozen correctness oracles admit any common output."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback
from typing import Any

import torch

from op0_common import atomic_write_json, now_iso, sha256
from run_k2_day import (
    DEFAULT_PREREG,
    ROOT,
    generated_input,
    vendor_call,
    verify_preregistration,
)
from run_op5_selection import environment_identity, git_sha, load_formal_carrier, prepare_modules


DEFAULT_OUTPUT = ROOT / "results/operator/failures/k2_oracle_consistency_audit_20260916.json"


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def intersection_audit(
    eager: torch.Tensor,
    reference: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    eager_values = eager.float().reshape(-1)
    reference_values = reference.float().reshape(-1)
    eager_radius = atol + rtol * eager_values.abs()
    reference_radius = atol + rtol * reference_values.abs()
    lower = torch.maximum(eager_values - eager_radius, reference_values - reference_radius)
    upper = torch.minimum(eager_values + eager_radius, reference_values + reference_radius)
    gaps = (lower - upper).clamp_min(0.0)
    disjoint = gaps > 0
    disjoint_indices = torch.where(disjoint)[0]
    examples: list[dict[str, Any]] = []
    for tensor_index in disjoint_indices[:8]:
        index = int(tensor_index.item())
        examples.append(
            {
                "flat_output_index": index,
                "eager": float(eager_values[index].item()),
                "reference": float(reference_values[index].item()),
                "eager_allowed_interval": [
                    float((eager_values[index] - eager_radius[index]).item()),
                    float((eager_values[index] + eager_radius[index]).item()),
                ],
                "reference_allowed_interval": [
                    float((reference_values[index] - reference_radius[index]).item()),
                    float((reference_values[index] + reference_radius[index]).item()),
                ],
                "uncovered_gap": float(gaps[index].item()),
            }
        )
    direct_close = torch.isclose(eager, reference, atol=atol, rtol=rtol)
    return {
        "output_elements": eager.numel(),
        "eager_reference_not_close_elements": int((~direct_close).sum().item()),
        "disjoint_allowed_interval_elements": int(disjoint.sum().item()),
        "maximum_uncovered_gap": float(gaps.max().item()) if gaps.numel() else 0.0,
        "examples": examples,
    }


def execute(prereg_path: Path, output_path: Path, device: int) -> dict[str, Any]:
    prereg_path = prereg_path.resolve()
    prereg_sha = sha256(prereg_path)
    verified = verify_preregistration(prereg_path, prereg_sha)
    prereg = verified["prereg"]
    selection = verified["selection_config"]
    log("K2 oracle audit: loading the full verified C02 NF4 carrier")
    model, model_load = load_formal_carrier(selection, device)
    prepared, layout = prepare_modules(model, selection, verified["decode_rows"], device)
    representatives: dict[str, Any] = {}
    for record in prepared:
        if record.variant not in representatives or record.name < representatives[record.variant].name:
            representatives[record.variant] = record
    if len(representatives) != 14:
        raise AssertionError("K2 oracle audit did not resolve 14 representatives")

    protocol = prereg["correctness_protocol"]
    atol = float(protocol["tolerances"]["bfloat16"]["atol"])
    rtol = float(protocol["tolerances"]["bfloat16"]["rtol"])
    distributions = [row["name"] for row in protocol["input_distributions"]]
    seeds = [int(value) for value in protocol["seeds"]]
    rows: list[dict[str, Any]] = []
    conflict_cases = 0
    conflict_elements = 0
    log("K2 oracle audit: checking 14 variants x 3 distributions x 3 seeds")
    import bitsandbytes as bnb

    with torch.inference_mode():
        for variant in sorted(representatives):
            record = representatives[variant]
            dequantized_bf16 = bnb.functional.dequantize_4bit(
                record.module.weight.data,
                quant_state=record.module.weight.quant_state,
            )
            if dequantized_bf16.dtype != torch.bfloat16:
                raise AssertionError(
                    f"literal frozen reference did not dequantize to BF16 for {variant}"
                )
            for distribution in distributions:
                for seed in seeds:
                    activation = generated_input(record.geometry[0], distribution, seed, device)
                    eager = vendor_call(record, activation)
                    reference = torch.nn.functional.linear(
                        activation.float(), dequantized_bf16.float()
                    ).to(torch.bfloat16)
                    torch.cuda.synchronize(device)
                    audit = intersection_audit(eager, reference, atol, rtol)
                    impossible = audit["disjoint_allowed_interval_elements"] > 0
                    conflict_cases += int(impossible)
                    conflict_elements += int(audit["disjoint_allowed_interval_elements"])
                    rows.append(
                        {
                            "variant": variant,
                            "module": record.name,
                            "shape_n_k": [record.geometry[1], record.geometry[0]],
                            "distribution": distribution,
                            "seed": seed,
                            "mathematically_impossible_for_any_candidate": impossible,
                            **audit,
                        }
                    )
                    del activation, eager, reference
            del dequantized_bf16
            torch.cuda.empty_cache()

    prereg_sha_end = sha256(prereg_path)
    if prereg_sha_end != prereg_sha:
        raise AssertionError("K2 preregistration changed during oracle audit")
    result = {
        "schema_version": 1,
        "stage": "OP5_K2_FROZEN_ORACLE_CONSISTENCY_AUDIT",
        "timestamp": now_iso(),
        "purpose": "Determine whether any candidate output can simultaneously satisfy both frozen allclose assertions before further implementation attempts or performance measurement.",
        "preregistration": {
            "path": str(prereg_path),
            "sha256_start": prereg_sha,
            "sha256_end": prereg_sha_end,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "candidate_observation": {
            "path": str(ROOT / "scripts/operator/k2_nf4_gemv.py"),
            "sha256": sha256(ROOT / "scripts/operator/k2_nf4_gemv.py"),
            "note": "Candidate output is not needed for the impossibility proof; only the two frozen oracle intervals are compared.",
        },
        "reference_definition": "bitsandbytes dequantize_4bit (C02 QuantState dtype BF16), cast the dequantized weight to FP32, explicit torch F.linear with FP32 activation, then cast output to BF16",
        "eager_definition": prereg["baselines"]["eager"]["definition"],
        "allclose_semantics": "For candidate c and oracle o, abs(c-o) <= atol + rtol*abs(o). If the eager and reference allowed intervals are disjoint, no real-valued candidate can satisfy both assertions.",
        "tolerances": {"atol": atol, "rtol": rtol, "output_dtype": "torch.bfloat16"},
        "population": {
            "variants": len(representatives),
            "distributions": distributions,
            "seeds": seeds,
            "cases": len(rows),
            "layout": layout,
        },
        "model_load": model_load,
        "environment": environment_identity(),
        "git_sha": verified["git_sha"],
        "rows": rows,
        "summary": {
            "cases_checked": len(rows),
            "cases_with_disjoint_frozen_oracle_intervals": conflict_cases,
            "total_output_elements_with_disjoint_intervals": conflict_elements,
            "frozen_correctness_gate_satisfiable": conflict_cases == 0,
            "performance_measurement_allowed": False if conflict_cases else None,
        },
        "formal_candidate_performance_measurement_started": False,
        "status": "FROZEN_CORRECTNESS_GATE_UNSATISFIABLE" if conflict_cases else "ORACLES_COMPATIBLE",
    }
    atomic_write_json(output_path, result)
    return result


def main() -> int:
    prereg_path = DEFAULT_PREREG
    output_path = DEFAULT_OUTPUT
    try:
        result = execute(prereg_path, output_path, 0)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "sha256": sha256(output_path),
                    "summary": result["summary"],
                    "status": result["status"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "stage": "OP5_K2_FROZEN_ORACLE_CONSISTENCY_AUDIT",
            "failed_at": now_iso(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "git_sha": git_sha(),
            "status": "AUDIT_EXECUTION_FAIL",
        }
        atomic_write_json(output_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
