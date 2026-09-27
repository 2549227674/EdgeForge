#!/usr/bin/env python3
"""Run the single frozen K2-R2 semantic-correctness holdout attempt."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any

import torch

from k2_nf4_gemv import (
    autotune_choices,
    candidate_or_linear4bit_fallback,
    eligible_linear4bit_decode,
)
from op0_common import atomic_write_json, now_iso, sha256
from run_k2_day import (
    ROOT,
    candidate_call,
    copied_proxy,
    error_metrics,
    semantic_microcase,
    vendor_call,
)
from run_op5_selection import (
    environment_identity,
    load_formal_carrier,
    load_json,
    prepare_modules,
)


DEFAULT_PROTOCOL = ROOT / "configs/operator/k2_nf4_decode_gemv_prereg_r2.json"
DATE_TAG = datetime.now().astimezone().strftime("%Y%m%d")
DEFAULT_RESULT = ROOT / f"results/operator/pool/k2_r2_correctness_{DATE_TAG}.json"
DEFAULT_RECEIPT = ROOT / f"results/operator/pool/k2_r2_correctness_receipt_{DATE_TAG}.json"
LOCKED_CANDIDATE = ROOT / "scripts/operator/k2_nf4_gemv.py"
PRIOR_HARNESS_FAILURE = (
    ROOT
    / "results/operator/failures/k2_r2_correctness_failure_20260916T164041.json"
)
PRIOR_HARNESS_FAILURE_SHA256 = (
    "cad059a35b9171158219d343b45d96817876423433849d3f8d31a0736eee40ad"
)


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def verify_prior_harness_interruption(
    protocol_sha: str,
    candidate_sha: str,
) -> dict[str, Any]:
    if not PRIOR_HARNESS_FAILURE.is_file():
        raise AssertionError("prior K2-R2 harness interruption evidence is missing")
    observed_sha = sha256(PRIOR_HARNESS_FAILURE)
    if observed_sha != PRIOR_HARNESS_FAILURE_SHA256:
        raise AssertionError("prior K2-R2 harness interruption evidence hash drift")
    failure = load_json(PRIOR_HARNESS_FAILURE)
    expected = {
        "stage": "OP5_K2_R2_CORRECTNESS",
        "protocol_sha256": protocol_sha,
        "locked_candidate_sha256": candidate_sha,
        "error_type": "IndexError",
        "error": "slice() cannot be applied to a 0-dim tensor.",
        "formal_candidate_performance_started": False,
        "status": "FAIL",
    }
    for field, value in expected.items():
        if failure.get(field) != value:
            raise AssertionError(
                f"prior K2-R2 harness interruption field drift: {field}"
            )
    traceback_text = str(failure.get("traceback", ""))
    if "state.offset[:0]" not in traceback_text or "run_fallback_checks" not in traceback_text:
        raise AssertionError("prior interruption is not the frozen fallback-harness defect")
    return {
        "path": str(PRIOR_HARNESS_FAILURE),
        "sha256": observed_sha,
        "classification": "HARNESS_ONLY_0D_OFFSET_EMPTY_TENSOR_CONSTRUCTION",
        "candidate_source_changed_after_interruption": False,
        "protocol_changed_after_interruption": False,
        "formal_candidate_performance_started_before_interruption": False,
        "original_runner_sha256": failure["runner_sha256"],
        "repair_scope": "R2 correctness runner fallback harness only",
    }


def verify_protocol(protocol_path: Path, protocol_sha_start: str) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    if protocol.get("stage") != "OP5_K2_R2" or protocol.get("protocol_revision") != "R2":
        raise AssertionError("unexpected K2-R2 protocol identity")
    disclosure = protocol["revision_disclosure"]
    if disclosure.get("adaptive_followup_to_failed_r1") is not True:
        raise AssertionError("K2-R2 does not disclose its adaptive R1 ancestry")
    if disclosure.get("r1_remains_immutable_and_failed") is not True:
        raise AssertionError("K2-R2 does not preserve the R1 failure")
    if disclosure.get("r2_holdout_inputs_evaluated_before_freeze") is not False:
        raise AssertionError("K2-R2 holdout was not unobserved at freeze")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("K2-R2 formal correctness requires local sm_89 CUDA")

    observed_sources: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for name, identity in protocol["source_identity"].items():
        if name == "git_sha":
            continue
        path = absolute(identity["path"])
        observed = sha256(path)
        if observed != identity["sha256"]:
            raise AssertionError(
                f"K2-R2 source hash mismatch for {name}: {observed} != {identity['sha256']}"
            )
        observed_sources[name] = {
            "path": str(path),
            "sha256": observed,
            "status": "PASS",
        }
        if path.suffix == ".json":
            loaded[name] = load_json(path)

    current_git_sha = git_sha()
    if current_git_sha != protocol["source_identity"]["git_sha"]:
        raise AssertionError("Git HEAD differs from frozen K2-R2 identity")
    if loaded["r1_correctness_gate_receipt"].get("status") != "BLOCKED_FROZEN_CORRECTNESS_ORACLE_CONFLICT":
        raise AssertionError("R1 failure receipt identity drift")
    if loaded["r1_oracle_consistency_audit"].get("status") != "FROZEN_CORRECTNESS_GATE_UNSATISFIABLE":
        raise AssertionError("R1 oracle audit identity drift")
    if loaded["op5_receipt"].get("status") != "PASS":
        raise AssertionError("OP5 selection gate is not PASS")
    if loaded["op5_selection_result"].get("k2_selected") is not True:
        raise AssertionError("OP5 did not select K2")
    if sha256(LOCKED_CANDIDATE) != disclosure["candidate_identity_locked_for_single_r2_attempt"]["sha256"]:
        raise AssertionError("locked K2-R2 candidate source changed")

    r1_prereg = loaded["r1_preregistration"]
    selection_identity = r1_prereg["source_identity"]["op5_selection_config"]
    selection_path = absolute(selection_identity["path"])
    if sha256(selection_path) != selection_identity["sha256"]:
        raise AssertionError("inherited OP5 selection config hash drift")
    selection = load_json(selection_path)
    census = loaded["shape_census"]
    decode_rows = [
        row
        for row in census["op_families"]["nf4_linear"]["shapes"]
        if row["phase"] == "decode"
    ]
    target = protocol["target_population"]
    if len(decode_rows) != int(target["expected_decode_rows"]):
        raise AssertionError("K2-R2 decode-row count drift")
    if sum(int(row["calls_per_forward"]) for row in decode_rows) != int(target["expected_modules"]):
        raise AssertionError("K2-R2 module count drift")
    if sum(int(row["invocations"]) for row in decode_rows) != int(
        target["expected_projected_decode_invocations"]
    ):
        raise AssertionError("K2-R2 projected invocation count drift")
    holdout = protocol["r2_holdout_correctness_protocol"]
    expected_trials = (
        int(target["expected_decode_rows"])
        * len(holdout["representative_matrix"]["distributions"])
        * len(holdout["representative_matrix"]["seeds"])
    )
    if expected_trials != int(holdout["representative_matrix"]["expected_trials"]):
        raise AssertionError("K2-R2 representative trial count declaration drift")
    if protocol_sha_start != sha256(protocol_path):
        raise AssertionError("K2-R2 protocol changed during verification")
    return {
        "protocol": protocol,
        "selection": selection,
        "decode_rows": decode_rows,
        "sources": observed_sources,
        "git_sha": current_git_sha,
    }


def holdout_input(
    k_dim: int,
    distribution: str,
    seed: int,
    device: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (1, 1, k_dim)
    if distribution == "unit_normal":
        values = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
    elif distribution == "wide_normal":
        values = (
            torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * 8.0
        )
    elif distribution == "near_zero_uniform":
        values = (
            torch.rand(shape, dtype=torch.float32, device=device, generator=generator) * 0.002
            - 0.001
        )
    elif distribution == "alternating_wide":
        perturbation = (
            torch.rand(shape, dtype=torch.float32, device=device, generator=generator) * 0.25
            - 0.125
        )
        signs = torch.where(
            torch.arange(k_dim, device=device) % 2 == 0,
            torch.ones(k_dim, dtype=torch.float32, device=device),
            -torch.ones(k_dim, dtype=torch.float32, device=device),
        ).reshape(1, 1, k_dim)
        values = 8.0 * signs * (1.0 + perturbation)
    else:
        raise AssertionError(f"unknown K2-R2 holdout distribution: {distribution}")
    return values.to(torch.bfloat16)


def explicit_weight_fp32(record: Any, codebook: torch.Tensor) -> torch.Tensor:
    k_dim, n_dim = record.geometry
    element_count = n_dim * k_dim
    packed = record.packed_weight.reshape(-1)
    if packed.numel() * 2 != element_count:
        raise AssertionError(f"odd or inconsistent packed weight for {record.name}")
    nibble_indices = torch.empty(element_count, dtype=torch.uint8, device=packed.device)
    nibble_indices[0::2] = packed >> 4
    nibble_indices[1::2] = packed & 0xF

    first_indices = record.first_absmax.reshape(-1).long()
    scale_positions = torch.arange(first_indices.numel(), device=packed.device)
    block_scales = (
        record.scale_code[first_indices]
        * record.second_absmax.reshape(-1)[scale_positions // 256]
        + record.scale_offset.reshape(())
    ).float()
    if block_scales.numel() * 64 != element_count:
        raise AssertionError(f"non-exact block-64 coverage for {record.name}")
    weights = codebook[nibble_indices.long()] * block_scales.repeat_interleave(64)
    return weights.reshape(n_dim, k_dim)


def comparison(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    exact = actual == expected
    return {
        **error_metrics(actual, expected),
        "allclose": bool(close.all().item()),
        "exact_elements": int(exact.sum().item()),
        "allclose_elements": int(close.sum().item()),
        "non_allclose_elements": int((~close).sum().item()),
        "output_elements": actual.numel(),
    }


def validate_choices(protocol: dict[str, Any], prepared: list[Any]) -> list[dict[str, Any]]:
    choices = autotune_choices()
    allowed = {
        (
            int(row["BLOCK_N"]),
            int(row["BLOCK_K"]),
            int(row["num_warps"]),
            int(row["num_stages"]),
        )
        for row in protocol["candidate_design"]["frozen_search_space_inherited_unchanged_from_r1"]
    }
    observed_geometries: set[tuple[int, int]] = set()
    for row in choices:
        selected = (
            int(row["BLOCK_N"]),
            int(row["BLOCK_K"]),
            int(row["num_warps"]),
            int(row["num_stages"]),
        )
        if selected not in allowed:
            raise AssertionError(f"non-frozen K2-R2 autotune choice: {selected}")
        observed_geometries.add((int(row["key"][0]), int(row["key"][1])))
    expected_geometries = set(record.geometry for record in prepared)
    missing = sorted(expected_geometries - observed_geometries)
    if missing:
        raise AssertionError(f"missing K2-R2 autotune geometry choices: {missing}")
    return choices


def run_r2_fallback_checks(record: Any, device: int) -> dict[str, Any]:
    """Exercise every dispatch rejection without mutating the formal module."""
    k_dim, _ = record.geometry
    checks: list[dict[str, Any]] = []

    def actual_case(name: str, activation: torch.Tensor, grad_enabled: bool = False) -> None:
        context = torch.enable_grad() if grad_enabled else torch.inference_mode()
        with context:
            if eligible_linear4bit_decode(record.module, activation):
                raise AssertionError(f"rejected condition unexpectedly eligible: {name}")
            expected = record.module(activation)
            observed = candidate_or_linear4bit_fallback(record.module, activation)
            torch.cuda.synchronize(device)
        exact = torch.equal(observed, expected)
        if not exact:
            raise AssertionError(f"actual Linear4bit fallback mismatch: {name}")
        checks.append({"condition": name, "carrier": "actual_Linear4bit", "exact": exact})

    generator = torch.Generator(device="cuda")
    generator.manual_seed(7)
    m_gt_one = torch.randn(
        (1, 2, k_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    fp32_input = torch.randn(
        (1, 1, k_dim), dtype=torch.float32, device=device, generator=generator
    )
    noncontiguous_base = torch.randn(
        (1, 1, k_dim * 2), dtype=torch.bfloat16, device=device, generator=generator
    )
    actual_case("M_gt_1", m_gt_one)
    actual_case("activation_dtype_not_bfloat16", fp32_input)
    actual_case("activation_not_contiguous", noncontiguous_base[..., ::2])
    actual_case("grad_enabled", record.input_tensor, grad_enabled=True)

    mutations: list[tuple[str, Any]] = [
        (
            "bias_present",
            lambda proxy, state, state2: setattr(
                proxy, "bias", torch.zeros(1, device=device)
            ),
        ),
        (
            "quant_type_not_nf4",
            lambda proxy, state, state2: setattr(state, "quant_type", "fp4"),
        ),
        (
            "quant_state_not_nested",
            lambda proxy, state, state2: setattr(state, "nested", False),
        ),
        (
            "quant_blocksize_not_64",
            lambda proxy, state, state2: setattr(state, "blocksize", 128),
        ),
        (
            "second_blocksize_not_256",
            lambda proxy, state, state2: setattr(state2, "blocksize", 128),
        ),
        (
            "quant_shape_mismatch",
            lambda proxy, state, state2: setattr(
                state,
                "shape",
                [record.module.out_features + 1, record.module.in_features],
            ),
        ),
        (
            "packed_dtype_mismatch",
            lambda proxy, state, state2: setattr(
                proxy.weight, "data", proxy.weight.data.view(torch.int8)
            ),
        ),
        (
            "packed_size_mismatch",
            lambda proxy, state, state2: setattr(
                proxy.weight, "data", proxy.weight.data.reshape(-1)[:-1]
            ),
        ),
        (
            "first_scale_dtype_mismatch",
            lambda proxy, state, state2: setattr(
                state, "absmax", state.absmax.view(torch.int8)
            ),
        ),
        (
            "first_scale_size_mismatch",
            lambda proxy, state, state2: setattr(state, "absmax", state.absmax[:-1]),
        ),
        (
            "second_scale_dtype_mismatch",
            lambda proxy, state, state2: setattr(
                state2, "absmax", state2.absmax.to(torch.bfloat16)
            ),
        ),
        (
            "second_scale_size_mismatch",
            lambda proxy, state, state2: setattr(
                state2, "absmax", state2.absmax[:-1]
            ),
        ),
        (
            "scale_code_dtype_mismatch",
            lambda proxy, state, state2: setattr(
                state2, "code", state2.code.to(torch.bfloat16)
            ),
        ),
        (
            "scale_code_size_mismatch",
            lambda proxy, state, state2: setattr(state2, "code", state2.code[:-1]),
        ),
        (
            "scale_offset_dtype_mismatch",
            lambda proxy, state, state2: setattr(
                state, "offset", state.offset.to(torch.bfloat16)
            ),
        ),
        (
            "scale_offset_size_mismatch",
            lambda proxy, state, state2: setattr(
                state, "offset", state.offset.reshape(1)[:0]
            ),
        ),
    ]
    with torch.inference_mode():
        expected = record.module(record.input_tensor)
        for name, mutation in mutations:
            proxy = copied_proxy(record, mutation)
            if eligible_linear4bit_decode(proxy, record.input_tensor):
                raise AssertionError(
                    f"synthetic rejected condition unexpectedly eligible: {name}"
                )
            observed = candidate_or_linear4bit_fallback(proxy, record.input_tensor)
            exact = torch.equal(observed, expected)
            if not exact:
                raise AssertionError(f"synthetic dispatch fallback mismatch: {name}")
            checks.append(
                {"condition": name, "carrier": "delegating_proxy", "exact": exact}
            )

    class CpuFallbackProxy:
        in_features = k_dim
        out_features = k_dim
        bias = None
        weight = SimpleNamespace(data=torch.empty(0), quant_state=None)

        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            return value.clone()

    cpu_activation = torch.ones((1, 1, k_dim), dtype=torch.bfloat16)
    cpu_proxy = CpuFallbackProxy()
    if eligible_linear4bit_decode(cpu_proxy, cpu_activation):
        raise AssertionError("CPU input unexpectedly eligible")
    cpu_exact = torch.equal(
        candidate_or_linear4bit_fallback(cpu_proxy, cpu_activation),
        cpu_proxy(cpu_activation),
    )
    if not cpu_exact:
        raise AssertionError("CPU-device fallback mismatch")
    checks.append(
        {"condition": "device_not_cuda_sm89", "carrier": "delegating_proxy", "exact": cpu_exact}
    )

    class RankFallbackProxy:
        in_features = k_dim
        out_features = k_dim
        bias = None
        weight = SimpleNamespace(data=torch.empty(0, device=device), quant_state=None)

        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            return value.clone()

    rank_input = torch.ones((k_dim,), dtype=torch.bfloat16, device=device)
    rank_proxy = RankFallbackProxy()
    with torch.inference_mode():
        if eligible_linear4bit_decode(rank_proxy, rank_input):
            raise AssertionError("rank-1 input unexpectedly eligible")
        rank_exact = torch.equal(
            candidate_or_linear4bit_fallback(rank_proxy, rank_input),
            rank_proxy(rank_input),
        )
    if not rank_exact:
        raise AssertionError("activation-rank fallback mismatch")
    checks.append(
        {"condition": "activation_rank_not_2_or_3", "carrier": "delegating_proxy", "exact": rank_exact}
    )
    return {"checks": checks, "count": len(checks), "all_exact": True, "status": "PASS"}


def summarize_comparisons(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row[field] for row in rows]
    return {
        "cases": len(values),
        "allclose_cases": sum(int(value["allclose"]) for value in values),
        "non_allclose_cases": sum(int(not value["allclose"]) for value in values),
        "exact_elements": sum(int(value["exact_elements"]) for value in values),
        "allclose_elements": sum(int(value["allclose_elements"]) for value in values),
        "non_allclose_elements": sum(int(value["non_allclose_elements"]) for value in values),
        "maximum_abs_error": max(float(value["max_abs"]) for value in values),
        "maximum_relative_error": max(float(value["max_rel"]) for value in values),
    }


def run_correctness(
    protocol: dict[str, Any],
    prepared: list[Any],
    layout: dict[str, Any],
    device: int,
) -> dict[str, Any]:
    if torch.backends.cuda.matmul.allow_tf32:
        raise AssertionError("K2-R2 explicit FP32 oracle requires TF32 matmul disabled")
    if torch.get_float32_matmul_precision() != "highest":
        raise AssertionError("K2-R2 explicit FP32 oracle requires highest matmul precision")
    semantic_alias = {"semantics": protocol["semantic_contract"]}
    log("K2-R2: semantic microcase")
    semantic = semantic_microcase(semantic_alias, device)

    holdout = protocol["r2_holdout_correctness_protocol"]
    atol = float(holdout["tolerances"]["bfloat16"]["atol"])
    rtol = float(holdout["tolerances"]["bfloat16"]["rtol"])
    codebook = torch.tensor(
        protocol["semantic_contract"]["nf4_codebook_index_order"],
        dtype=torch.float32,
        device=device,
    )
    actual_seed = int(holdout["all_actual_weights"]["seed"])
    actual_inputs = {
        k_dim: holdout_input(k_dim, "unit_normal", actual_seed, device)
        for k_dim in sorted({record.geometry[0] for record in prepared})
    }

    log("K2-R2: all 343 actual weights against the single FP32 semantic oracle")
    actual_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, record in enumerate(prepared):
            activation = actual_inputs[record.geometry[0]]
            source_tensors = [
                activation,
                record.packed_weight,
                record.first_absmax,
                record.second_absmax,
                record.scale_code,
                record.scale_offset,
            ]
            snapshots = [value.clone() for value in source_tensors]
            pointers = [value.data_ptr() for value in source_tensors]
            versions = [value._version for value in source_tensors]
            weight_fp32 = explicit_weight_fp32(record, codebook)
            reference = torch.nn.functional.linear(activation.float(), weight_fp32).to(
                torch.bfloat16
            )
            candidate = candidate_call(record, activation)
            eager = vendor_call(record, activation)
            torch.cuda.synchronize(device)
            candidate_reference = comparison(candidate, reference, atol, rtol)
            candidate_eager = comparison(candidate, eager, atol, rtol)
            eager_reference = comparison(eager, reference, atol, rtol)
            unmodified = all(
                torch.equal(value, snapshot)
                for value, snapshot in zip(source_tensors, snapshots)
            )
            identity_unchanged = all(
                value.data_ptr() == pointer and value._version == version
                for value, pointer, version in zip(source_tensors, pointers, versions)
            )
            output_contract = bool(
                candidate.shape == reference.shape
                and candidate.dtype == torch.bfloat16
                and candidate.device == activation.device
                and candidate.is_contiguous()
                and torch.isfinite(candidate).all().item()
            )
            if not candidate_reference["allclose"]:
                raise AssertionError(
                    f"K2-R2 primary oracle failure for {record.name}: {candidate_reference}"
                )
            if not unmodified or not identity_unchanged or not output_contract:
                raise AssertionError(
                    f"K2-R2 source/output contract failure for {record.name}: "
                    f"unmodified={unmodified}, identity={identity_unchanged}, output={output_contract}"
                )
            actual_rows.append(
                {
                    "index": index,
                    "module": record.name,
                    "variant": record.variant,
                    "shape_n_k": [record.geometry[1], record.geometry[0]],
                    "candidate_vs_primary_reference": candidate_reference,
                    "candidate_vs_vendor_eager_non_gating": candidate_eager,
                    "vendor_eager_vs_primary_reference_non_gating": eager_reference,
                    "source_contents_unmodified": unmodified,
                    "source_identity_and_versions_unmodified": identity_unchanged,
                    "output_contract": output_contract,
                }
            )
            del weight_fp32, reference, candidate, eager, snapshots
            if (index + 1) % 50 == 0:
                log(f"K2-R2: completed {index + 1}/343 actual weights")
        torch.cuda.synchronize(device)

    representatives: dict[str, Any] = {}
    for record in prepared:
        if record.variant not in representatives or record.name < representatives[record.variant].name:
            representatives[record.variant] = record
    if len(representatives) != int(protocol["target_population"]["expected_decode_rows"]):
        raise AssertionError("K2-R2 representative coverage drift")

    distributions = [
        row["name"] for row in holdout["representative_matrix"]["distributions"]
    ]
    seeds = [int(value) for value in holdout["representative_matrix"]["seeds"]]
    log("K2-R2: 14 representatives x 4 fresh distributions x 3 fresh seeds")
    import bitsandbytes as bnb

    representative_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for variant in sorted(representatives):
            record = representatives[variant]
            weight_fp32 = explicit_weight_fp32(record, codebook)
            bnb_bf16_weight = bnb.functional.dequantize_4bit(
                record.module.weight.data,
                quant_state=record.module.weight.quant_state,
            )
            for distribution in distributions:
                for seed in seeds:
                    activation = holdout_input(record.geometry[0], distribution, seed, device)
                    reference = torch.nn.functional.linear(
                        activation.float(), weight_fp32
                    ).to(torch.bfloat16)
                    candidate = candidate_call(record, activation)
                    eager = vendor_call(record, activation)
                    bnb_diagnostic = torch.nn.functional.linear(
                        activation.float(), bnb_bf16_weight.float()
                    ).to(torch.bfloat16)
                    torch.cuda.synchronize(device)
                    candidate_reference = comparison(candidate, reference, atol, rtol)
                    if not candidate_reference["allclose"]:
                        raise AssertionError(
                            f"K2-R2 holdout primary oracle failure for "
                            f"{variant}/{distribution}/{seed}: {candidate_reference}"
                        )
                    representative_rows.append(
                        {
                            "variant": variant,
                            "module": record.name,
                            "shape_n_k": [record.geometry[1], record.geometry[0]],
                            "distribution": distribution,
                            "seed": seed,
                            "candidate_vs_primary_reference": candidate_reference,
                            "candidate_vs_vendor_eager_non_gating": comparison(
                                candidate, eager, atol, rtol
                            ),
                            "vendor_eager_vs_primary_reference_non_gating": comparison(
                                eager, reference, atol, rtol
                            ),
                            "bnb_bf16_dequant_diagnostic_vs_primary_reference_non_gating": comparison(
                                bnb_diagnostic, reference, atol, rtol
                            ),
                        }
                    )
                    del activation, reference, candidate, eager, bnb_diagnostic
            del weight_fp32, bnb_bf16_weight
            torch.cuda.empty_cache()
            log(f"K2-R2: completed representative variant {variant}")

    expected_trials = int(holdout["representative_matrix"]["expected_trials"])
    if len(representative_rows) != expected_trials:
        raise AssertionError(
            f"K2-R2 holdout trial count {len(representative_rows)} != {expected_trials}"
        )
    log("K2-R2: exhaustive fallback rejection matrix")
    fallback_record = representatives[sorted(representatives)[0]]
    fallback = run_r2_fallback_checks(fallback_record, device)
    with torch.inference_mode():
        for record in prepared:
            if not eligible_linear4bit_decode(record.module, actual_inputs[record.geometry[0]]):
                raise AssertionError(f"valid K2-R2 module unexpectedly ineligible: {record.name}")
    choices = validate_choices(protocol, prepared)
    if sha256(LOCKED_CANDIDATE) != protocol["source_identity"]["candidate_source"]["sha256"]:
        raise AssertionError("locked candidate changed during K2-R2 correctness")

    candidate_primary_summary = {
        "actual_weights": summarize_comparisons(
            actual_rows, "candidate_vs_primary_reference"
        ),
        "representative_holdout": summarize_comparisons(
            representative_rows, "candidate_vs_primary_reference"
        ),
    }
    return {
        "oracle_policy": protocol["oracle_policy"],
        "tolerances": {"atol": atol, "rtol": rtol, "dtype": "torch.bfloat16"},
        "semantic_microcase": semantic,
        "all_actual_weights": {
            "seed": actual_seed,
            "distribution": "unit_normal",
            "modules": len(actual_rows),
            "variants": len({row["variant"] for row in actual_rows}),
            "rows": actual_rows,
        },
        "representative_holdout": {
            "variants": len(representatives),
            "distributions": distributions,
            "seeds": seeds,
            "trials": len(representative_rows),
            "rows": representative_rows,
        },
        "gating_candidate_vs_primary_reference": candidate_primary_summary,
        "non_gating_numerical_comparators": {
            "actual_candidate_vs_vendor": summarize_comparisons(
                actual_rows, "candidate_vs_vendor_eager_non_gating"
            ),
            "actual_vendor_vs_primary_reference": summarize_comparisons(
                actual_rows, "vendor_eager_vs_primary_reference_non_gating"
            ),
            "holdout_candidate_vs_vendor": summarize_comparisons(
                representative_rows, "candidate_vs_vendor_eager_non_gating"
            ),
            "holdout_vendor_vs_primary_reference": summarize_comparisons(
                representative_rows, "vendor_eager_vs_primary_reference_non_gating"
            ),
            "holdout_bnb_bf16_dequant_vs_primary_reference": summarize_comparisons(
                representative_rows,
                "bnb_bf16_dequant_diagnostic_vs_primary_reference_non_gating",
            ),
        },
        "fallback": fallback,
        "autotune_choices": choices,
        "status": "PASS",
    }


def execute(protocol_path: Path, result_path: Path, receipt_path: Path, device: int) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol_sha_start = sha256(protocol_path)
    candidate_sha_start = sha256(LOCKED_CANDIDATE)
    prior_interruption = verify_prior_harness_interruption(
        protocol_sha_start,
        candidate_sha_start,
    )
    verified = verify_protocol(protocol_path, protocol_sha_start)
    protocol = verified["protocol"]
    environment = environment_identity()
    log("K2-R2: re-verifying and loading the formal C02 carrier")
    model, model_load = load_formal_carrier(verified["selection"], device)
    prepared, layout = prepare_modules(
        model, verified["selection"], verified["decode_rows"], device
    )
    correctness = run_correctness(protocol, prepared, layout, device)
    protocol_sha_end = sha256(protocol_path)
    if protocol_sha_end != protocol_sha_start:
        raise AssertionError("K2-R2 protocol changed during correctness")
    candidate_sha_end = sha256(LOCKED_CANDIDATE)
    if candidate_sha_end != protocol["source_identity"]["candidate_source"]["sha256"]:
        raise AssertionError("locked K2-R2 candidate changed during correctness")
    result = {
        "schema_version": 1,
        "stage": "OP5_K2_R2_CORRECTNESS",
        "timestamp": now_iso(),
        "evidence_scope": protocol["host_scope"],
        "adaptive_revision_disclosure": protocol["revision_disclosure"],
        "protocol": {
            "path": str(protocol_path),
            "sha256_start": protocol_sha_start,
            "sha256_end": protocol_sha_end,
        },
        "locked_candidate": {
            "path": str(LOCKED_CANDIDATE),
            "sha256": candidate_sha_end,
        },
        "candidate_identity_attempt": 1,
        "harness_execution_attempt": 2,
        "prior_harness_interruption": prior_interruption,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "verified_sources": verified["sources"],
        "git_sha": verified["git_sha"],
        "environment": environment,
        "model_load": model_load,
        "layout": layout,
        "correctness": correctness,
        "formal_candidate_performance_started": False,
        "status": "PASS",
    }
    atomic_write_json(result_path, result)
    receipt = {
        "schema_version": 1,
        "stage": "OP5_K2_R2_CORRECTNESS",
        "timestamp": now_iso(),
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha_start},
        "locked_candidate": {"path": str(LOCKED_CANDIDATE), "sha256": candidate_sha_end},
        "candidate_identity_attempt": 1,
        "harness_execution_attempt": 2,
        "prior_harness_interruption": prior_interruption,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "result": {"path": str(result_path), "sha256": sha256(result_path)},
        "gates": {
            "adaptive_r1_disclosure": "PASS",
            "prior_harness_failure_preserved_and_classified": "PASS",
            "protocol_immutable": "PASS",
            "candidate_source_locked": "PASS",
            "source_and_git_identity": "PASS",
            "semantic_microcase": "PASS",
            "all_343_actual_weights_vs_single_primary_oracle": "PASS",
            "fresh_14_x_4_x_3_holdout_vs_single_primary_oracle": "PASS",
            "source_tensors_unmodified": "PASS",
            "fallback_rejection_matrix": "PASS",
            "frozen_autotune_space": "PASS",
            "vendor_numerical_comparator_reported_non_gating": "PASS",
            "r2_day1_performance": "NOT_STARTED"
        },
        "next_authorized_action": "Run R2 isolated day1 performance with the unchanged protocol, candidate, and correctness receipt; continue beyond day1 only if the frozen positive gate passes.",
        "status": "PASS"
    }
    atomic_write_json(receipt_path, receipt)
    if sha256(protocol_path) != protocol_sha_start:
        raise AssertionError("K2-R2 protocol changed through receipt write")
    if sha256(LOCKED_CANDIDATE) != candidate_sha_end:
        raise AssertionError("K2-R2 candidate changed through receipt write")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = now_iso()
    protocol_path = args.protocol.resolve()
    try:
        receipt = execute(
            protocol_path,
            args.result.resolve(),
            args.receipt.resolve(),
            args.device,
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        failure_path = ROOT / (
            "results/operator/failures/"
            f"k2_r2_correctness_failure_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}.json"
        )
        failure = {
            "schema_version": 1,
            "stage": "OP5_K2_R2_CORRECTNESS",
            "started_at": started,
            "failed_at": now_iso(),
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path) if protocol_path.is_file() else None,
            "locked_candidate": str(LOCKED_CANDIDATE),
            "locked_candidate_sha256": sha256(LOCKED_CANDIDATE),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "git_sha": git_sha(),
            "environment": environment_identity(),
            "formal_candidate_performance_started": False,
            "r2_candidate_attempt_disposition": "CLOSE_K2_IF_CANDIDATE_CORRECTNESS_FAILURE; FIX_HARNESS_ONLY_IF_FAILURE_IS_PROVEN_NON_CANDIDATE",
            "status": "FAIL"
        }
        atomic_write_json(failure_path, failure)
        print(
            json.dumps(
                {**failure, "failure_path": str(failure_path)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
