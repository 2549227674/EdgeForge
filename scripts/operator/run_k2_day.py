#!/usr/bin/env python3
"""Run one frozen K2 isolated correctness/performance calendar day."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable

import torch

from k2_nf4_gemv import (
    autotune_choices,
    candidate_nf4_gemv,
    candidate_or_linear4bit_fallback,
    eligible_linear4bit_decode,
)
from load_smoke_nf4 import memory_snapshot
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, sha256, summarize
from run_op5_selection import (
    PreparedModule,
    absolute,
    environment_identity,
    git_sha,
    load_formal_carrier,
    load_json,
    make_vendor_operations,
    prepare_modules,
    run_same_day_bandwidth,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREREG = ROOT / "configs/operator/k2_nf4_decode_gemv_prereg.json"
DATE_TAG = datetime.now().astimezone().strftime("%Y%m%d")
DEFAULT_CORRECTNESS = ROOT / f"results/operator/pool/k2_correctness_{DATE_TAG}.json"
DEFAULT_CORRECTNESS_RECEIPT = (
    ROOT / f"results/operator/pool/k2_correctness_receipt_{DATE_TAG}.json"
)
DEFAULT_RESULT = ROOT / f"results/operator/pool/k2_day1_{DATE_TAG}.json"
DEFAULT_RECEIPT = ROOT / f"results/operator/pool/k2_day1_receipt_{DATE_TAG}.json"


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    delta = (actual_fp32 - expected_fp32).abs()
    denominator = expected_fp32.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "max_rel": float((delta / denominator).max().item()) if delta.numel() else 0.0,
    }


def verify_preregistration(
    prereg_path: Path,
    prereg_sha_start: str,
) -> dict[str, Any]:
    prereg = load_json(prereg_path)
    if prereg.get("stage") != "OP5_K2":
        raise AssertionError("unexpected K2 preregistration identity")
    if prereg.get("kernel") != "k2_nf4_double_quant_decode_gemv":
        raise AssertionError("unexpected K2 kernel identity")
    if prereg.get("frozen_before_any_k2_candidate_implementation_or_measurement") is not True:
        raise AssertionError("K2 preregistration was not frozen before implementation")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("K2 formal measurement requires the local sm_89 CUDA device")

    sources: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for name, identity in prereg["source_identity"].items():
        if name == "git_sha":
            continue
        path = absolute(identity["path"])
        observed = sha256(path)
        if observed != identity["sha256"]:
            raise AssertionError(
                f"frozen source hash mismatch for {name}: {observed} != {identity['sha256']}"
            )
        sources[name] = {"path": str(path), "sha256": observed, "status": "PASS"}
        if path.suffix == ".json":
            loaded[name] = load_json(path)

    observed_git_sha = git_sha()
    if observed_git_sha != prereg["source_identity"]["git_sha"]:
        raise AssertionError("Git HEAD differs from the frozen K2 identity")
    for source_name in ("op0_receipt", "op1_receipt", "p0_nf4_receipt", "op4_receipt", "op5_receipt"):
        if loaded[source_name].get("status") != "PASS":
            raise AssertionError(f"upstream {source_name} is not PASS")
    if loaded["p0_nf4_receipt"].get("formal_evidence_eligible") is not True:
        raise AssertionError("formal C02 NF4 carrier is not evidence eligible")
    if loaded["op5_selection_result"].get("k2_selected") is not True:
        raise AssertionError("OP5 selection did not select K2")

    selection_config = loaded["op5_selection_config"]
    census = loaded["shape_census"]
    decode_rows = [
        row
        for row in census["op_families"]["nf4_linear"]["shapes"]
        if row["phase"] == "decode"
    ]
    target = prereg["target_population"]
    if len(decode_rows) != int(target["expected_decode_rows"]):
        raise AssertionError("K2 decode-row population drift")
    if sum(int(row["calls_per_forward"]) for row in decode_rows) != int(
        target["expected_modules"]
    ):
        raise AssertionError("K2 formal module population drift")
    if sum(int(row["invocations"]) for row in decode_rows) != int(
        target["expected_projected_decode_invocations"]
    ):
        raise AssertionError("K2 projected invocation population drift")
    if prereg_sha_start != sha256(prereg_path):
        raise AssertionError("K2 preregistration changed while inputs were verified")
    return {
        "prereg": prereg,
        "selection_config": selection_config,
        "decode_rows": decode_rows,
        "sources": sources,
        "git_sha": observed_git_sha,
    }


def candidate_call(record: PreparedModule, activation: torch.Tensor | None = None) -> torch.Tensor:
    k_dim, n_dim = record.geometry
    return candidate_nf4_gemv(
        record.input_tensor if activation is None else activation,
        record.packed_weight,
        record.first_absmax,
        record.second_absmax,
        record.scale_code,
        record.scale_offset,
        (n_dim, k_dim),
    )


def vendor_call(record: PreparedModule, activation: torch.Tensor | None = None) -> torch.Tensor:
    k_dim, n_dim = record.geometry
    return torch.ops.bitsandbytes.gemm_4bit.default(
        record.input_tensor if activation is None else activation,
        record.packed_weight,
        [n_dim, k_dim],
        record.second_absmax,
        64,
        "nf4",
        bias=None,
        absmax_8bit=record.first_absmax,
        absmax_code=record.scale_code,
        absmax_offset=record.scale_offset,
    )


def semantic_microcase(prereg: dict[str, Any], device: int) -> dict[str, Any]:
    n_dim, k_dim = 64, 512
    element_count = n_dim * k_dim
    byte_indices = torch.arange(element_count // 2, device=device)
    high = (byte_indices % 16).to(torch.uint8)
    low = (15 - byte_indices % 16).to(torch.uint8)
    packed = ((high << 4) | low).contiguous()
    flattened_codes = torch.empty(element_count, dtype=torch.uint8, device=device)
    flattened_codes[0::2] = high
    flattened_codes[1::2] = low
    first_scale_count = element_count // 64
    first_absmax = (torch.arange(first_scale_count, device=device) % 256).to(torch.uint8)
    second_absmax = torch.tensor([0.75, 1.25], dtype=torch.float32, device=device)
    second_code = torch.linspace(0.25, 1.25, 256, dtype=torch.float32, device=device)
    scale_offset = torch.tensor([0.125], dtype=torch.float32, device=device)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260916)
    activation = torch.randn(
        (1, 1, k_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    sources = [activation, packed, first_absmax, second_absmax, second_code, scale_offset]
    snapshots = [value.clone() for value in sources]
    pointers = [value.data_ptr() for value in sources]
    versions = [value._version for value in sources]

    candidate = candidate_nf4_gemv(
        activation,
        packed,
        first_absmax,
        second_absmax,
        second_code,
        scale_offset,
        (n_dim, k_dim),
    )
    codebook = torch.tensor(
        prereg["semantics"]["nf4_codebook_index_order"],
        dtype=torch.float32,
        device=device,
    )
    flat_offsets = torch.arange(element_count, device=device)
    scale_indices = flat_offsets // 64
    scales = (
        second_code[first_absmax[scale_indices].long()]
        * second_absmax[scale_indices // 256]
        + scale_offset
    )
    explicit_weight = (codebook[flattened_codes.long()] * scales).reshape(n_dim, k_dim)
    reference = (activation.float() @ explicit_weight.t()).to(torch.bfloat16)
    torch.cuda.synchronize(device)
    close = torch.allclose(candidate, reference, atol=0.03125, rtol=0.02)
    unmodified = all(torch.equal(value, snapshot) for value, snapshot in zip(sources, snapshots))
    identity_unchanged = all(
        value.data_ptr() == pointer and value._version == version
        for value, pointer, version in zip(sources, pointers, versions)
    )
    high_codes = sorted(set(int(value) for value in high[:16].cpu().tolist()))
    low_codes = sorted(set(int(value) for value in low[:16].cpu().tolist()))
    boundaries = {
        "flat_63_scale_index": int(scale_indices[63].item()),
        "flat_64_scale_index": int(scale_indices[64].item()),
        "scale_255_second_group": int((scale_indices[255 * 64] // 256).item()),
        "scale_256_second_group": int((scale_indices[256 * 64] // 256).item()),
    }
    if not close:
        raise AssertionError(f"semantic microcase mismatch: {error_metrics(candidate, reference)}")
    if high_codes != list(range(16)) or low_codes != list(range(16)):
        raise AssertionError("semantic microcase did not exhaust both nibble positions")
    if boundaries != {
        "flat_63_scale_index": 0,
        "flat_64_scale_index": 1,
        "scale_255_second_group": 0,
        "scale_256_second_group": 1,
    }:
        raise AssertionError(f"semantic boundary coverage drift: {boundaries}")
    if not unmodified or not identity_unchanged:
        raise AssertionError("candidate modified a semantic-microcase source tensor")
    return {
        "shape_n_k": [n_dim, k_dim],
        "high_nibble_codes_covered": high_codes,
        "low_nibble_codes_covered": low_codes,
        "nested_boundaries": boundaries,
        "candidate_vs_explicit": {**error_metrics(candidate, reference), "allclose": close},
        "source_contents_unmodified": unmodified,
        "source_identity_and_versions_unmodified": identity_unchanged,
        "output": {
            "shape": list(candidate.shape),
            "dtype": str(candidate.dtype),
            "device": str(candidate.device),
            "contiguous": candidate.is_contiguous(),
            "finite": bool(torch.isfinite(candidate).all().item()),
        },
        "status": "PASS",
    }


def generated_input(
    k_dim: int,
    distribution: str,
    seed: int,
    device: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (1, 1, k_dim)
    if distribution == "unit_normal":
        return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
    if distribution == "wide_normal":
        return (
            torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * 8.0
        ).to(torch.bfloat16)
    if distribution == "near_zero_uniform":
        return (
            torch.rand(shape, dtype=torch.float32, device=device, generator=generator) * 0.002
            - 0.001
        ).to(torch.bfloat16)
    raise AssertionError(f"unknown frozen input distribution: {distribution}")


class FallbackProxy:
    def __init__(self, original: Any, weight: Any, **overrides: Any) -> None:
        self.original = original
        self.weight = weight
        self.in_features = overrides.get("in_features", original.in_features)
        self.out_features = overrides.get("out_features", original.out_features)
        self.bias = overrides.get("bias", original.bias)

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        return self.original(activation)


def copied_proxy(record: PreparedModule, mutation: Callable[[Any, Any, Any], None]) -> FallbackProxy:
    original = record.module
    state = copy.copy(original.weight.quant_state)
    state2 = copy.copy(state.state2)
    state.state2 = state2
    weight = SimpleNamespace(data=original.weight.data, quant_state=state)
    proxy = FallbackProxy(original, weight)
    mutation(proxy, state, state2)
    return proxy


def run_fallback_checks(record: PreparedModule, device: int) -> dict[str, Any]:
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
    generator.manual_seed(20260919)
    m_gt_one = torch.randn(
        (1, 2, k_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    fp32_input = torch.randn(
        (1, 1, k_dim), dtype=torch.float32, device=device, generator=generator
    )
    noncontiguous_base = torch.randn(
        (1, 1, k_dim * 2), dtype=torch.bfloat16, device=device, generator=generator
    )
    noncontiguous = noncontiguous_base[..., ::2]
    actual_case("M_gt_1", m_gt_one)
    actual_case("activation_dtype_not_bfloat16", fp32_input)
    actual_case("activation_not_contiguous", noncontiguous)
    actual_case("grad_enabled", record.input_tensor, grad_enabled=True)

    mutations: list[tuple[str, Callable[[Any, Any, Any], None]]] = [
        ("bias_present", lambda proxy, state, state2: setattr(proxy, "bias", torch.zeros(1, device=device))),
        ("quant_type_not_nf4", lambda proxy, state, state2: setattr(state, "quant_type", "fp4")),
        ("quant_state_not_nested", lambda proxy, state, state2: setattr(state, "nested", False)),
        ("quant_blocksize_not_64", lambda proxy, state, state2: setattr(state, "blocksize", 128)),
        ("second_blocksize_not_256", lambda proxy, state, state2: setattr(state2, "blocksize", 128)),
        ("quant_shape_mismatch", lambda proxy, state, state2: setattr(state, "shape", [record.module.out_features + 1, record.module.in_features])),
        ("packed_dtype_mismatch", lambda proxy, state, state2: setattr(proxy.weight, "data", proxy.weight.data.view(torch.int8))),
        ("packed_size_mismatch", lambda proxy, state, state2: setattr(proxy.weight, "data", proxy.weight.data.view(-1)[:-1])),
        ("first_scale_dtype_mismatch", lambda proxy, state, state2: setattr(state, "absmax", state.absmax.view(torch.int8))),
        ("first_scale_size_mismatch", lambda proxy, state, state2: setattr(state, "absmax", state.absmax[:-1])),
        ("second_scale_dtype_mismatch", lambda proxy, state, state2: setattr(state2, "absmax", state2.absmax.to(torch.bfloat16))),
        ("second_scale_size_mismatch", lambda proxy, state, state2: setattr(state2, "absmax", state2.absmax[:-1])),
        ("scale_code_dtype_mismatch", lambda proxy, state, state2: setattr(state2, "code", state2.code.to(torch.bfloat16))),
        ("scale_code_size_mismatch", lambda proxy, state, state2: setattr(state2, "code", state2.code[:-1])),
        ("scale_offset_dtype_mismatch", lambda proxy, state, state2: setattr(state, "offset", state.offset.to(torch.bfloat16))),
        ("scale_offset_size_mismatch", lambda proxy, state, state2: setattr(state, "offset", state.offset[:0])),
    ]
    with torch.inference_mode():
        expected = record.module(record.input_tensor)
        for name, mutation in mutations:
            proxy = copied_proxy(record, mutation)
            if eligible_linear4bit_decode(proxy, record.input_tensor):
                raise AssertionError(f"synthetic rejected condition unexpectedly eligible: {name}")
            observed = candidate_or_linear4bit_fallback(proxy, record.input_tensor)
            exact = torch.equal(observed, expected)
            if not exact:
                raise AssertionError(f"synthetic dispatch fallback mismatch: {name}")
            checks.append({"condition": name, "carrier": "delegating_proxy", "exact": exact})

    cpu_activation = torch.ones((1, 1, k_dim), dtype=torch.bfloat16)
    class CpuFallbackProxy:
        in_features = k_dim
        out_features = k_dim
        bias = None
        weight = SimpleNamespace(data=torch.empty(0), quant_state=None)

        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            return value.clone()

    concrete_cpu_proxy = CpuFallbackProxy()
    if eligible_linear4bit_decode(concrete_cpu_proxy, cpu_activation):
        raise AssertionError("CPU input unexpectedly eligible")
    cpu_expected = concrete_cpu_proxy(cpu_activation)
    cpu_observed = candidate_or_linear4bit_fallback(concrete_cpu_proxy, cpu_activation)
    cpu_exact = torch.equal(cpu_expected, cpu_observed)
    if not cpu_exact:
        raise AssertionError("CPU-device fallback mismatch")
    checks.append({"condition": "device_not_cuda_sm89", "carrier": "delegating_proxy", "exact": cpu_exact})

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
    checks.append({"condition": "activation_rank_not_2_or_3", "carrier": "delegating_proxy", "exact": rank_exact})
    return {"checks": checks, "count": len(checks), "all_exact": True, "status": "PASS"}


def validate_autotune_choices(
    prereg: dict[str, Any], prepared: list[PreparedModule]
) -> list[dict[str, Any]]:
    choices = autotune_choices()
    expected_geometries = sorted(set(record.geometry for record in prepared))
    observed_geometries = sorted(
        (int(row["key"][0]), int(row["key"][1])) for row in choices
    )
    if observed_geometries != expected_geometries:
        raise AssertionError(
            f"autotune geometry coverage drift: {observed_geometries} != {expected_geometries}"
        )
    allowed = {
        (
            int(row["BLOCK_N"]),
            int(row["BLOCK_K"]),
            int(row["num_warps"]),
            int(row["num_stages"]),
        )
        for row in prereg["candidate_design"]["frozen_search_space"]
    }
    for row in choices:
        selected = (
            int(row["BLOCK_N"]),
            int(row["BLOCK_K"]),
            int(row["num_warps"]),
            int(row["num_stages"]),
        )
        if selected not in allowed:
            raise AssertionError(f"Triton selected a non-frozen K2 config: {selected}")
    return choices


def run_correctness(
    prereg: dict[str, Any],
    prepared: list[PreparedModule],
    layout: dict[str, Any],
    device: int,
) -> dict[str, Any]:
    log("K2 correctness: semantic nibble/nested-scale boundary microcase")
    semantic = semantic_microcase(prereg, device)
    atol = float(prereg["correctness_protocol"]["tolerances"]["bfloat16"]["atol"])
    rtol = float(prereg["correctness_protocol"]["tolerances"]["bfloat16"]["rtol"])

    log("K2 correctness: all 343 actual weights against bitsandbytes eager")
    input_by_k = {
        k_dim: generated_input(k_dim, "unit_normal", 20260916, device)
        for k_dim in sorted({record.geometry[0] for record in prepared})
    }
    actual_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, record in enumerate(prepared):
            activation = input_by_k[record.geometry[0]]
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
            eager = vendor_call(record, activation)
            candidate = candidate_call(record, activation)
            torch.cuda.synchronize(device)
            close = torch.allclose(candidate, eager, atol=atol, rtol=rtol)
            unmodified = all(
                torch.equal(value, snapshot)
                for value, snapshot in zip(source_tensors, snapshots)
            )
            identity_unchanged = all(
                value.data_ptr() == pointer and value._version == version
                for value, pointer, version in zip(source_tensors, pointers, versions)
            )
            contract = bool(
                candidate.shape == eager.shape
                and candidate.dtype == torch.bfloat16
                and candidate.device == activation.device
                and candidate.is_contiguous()
                and torch.isfinite(candidate).all().item()
            )
            if not close or not unmodified or not identity_unchanged or not contract:
                raise AssertionError(
                    f"actual-weight correctness failed for {record.name}: "
                    f"close={close}, unmodified={unmodified}, identity={identity_unchanged}, "
                    f"contract={contract}, error={error_metrics(candidate, eager)}"
                )
            actual_rows.append(
                {
                    "index": index,
                    "module": record.name,
                    "variant": record.variant,
                    "shape_n_k": [record.geometry[1], record.geometry[0]],
                    "candidate_vs_eager": {**error_metrics(candidate, eager), "allclose": close},
                    "source_contents_unmodified": unmodified,
                    "source_identity_and_versions_unmodified": identity_unchanged,
                    "output_contract": contract,
                }
            )

    representatives: dict[str, PreparedModule] = {}
    for record in prepared:
        if record.variant not in representatives or record.name < representatives[record.variant].name:
            representatives[record.variant] = record
    expected_variants = sorted(layout["counts_by_variant"])
    if sorted(representatives) != expected_variants:
        raise AssertionError("representative variant coverage drift")

    log("K2 correctness: 14 representatives x 3 distributions x 3 seeds")
    import bitsandbytes as bnb

    randomized_rows: list[dict[str, Any]] = []
    distributions = [row["name"] for row in prereg["correctness_protocol"]["input_distributions"]]
    seeds = [int(seed) for seed in prereg["correctness_protocol"]["seeds"]]
    with torch.inference_mode():
        for variant in expected_variants:
            record = representatives[variant]
            state = record.module.weight.quant_state
            explicit_weight = bnb.functional.dequantize_4bit(
                record.module.weight.data,
                quant_state=state,
            ).float()
            for distribution in distributions:
                for seed in seeds:
                    activation = generated_input(record.geometry[0], distribution, seed, device)
                    eager = vendor_call(record, activation)
                    candidate = candidate_call(record, activation)
                    reference = torch.nn.functional.linear(
                        activation.float(), explicit_weight
                    ).to(torch.bfloat16)
                    torch.cuda.synchronize(device)
                    candidate_eager_close = torch.allclose(candidate, eager, atol=atol, rtol=rtol)
                    candidate_reference_close = torch.allclose(
                        candidate, reference, atol=atol, rtol=rtol
                    )
                    if not candidate_eager_close or not candidate_reference_close:
                        raise AssertionError(
                            f"randomized correctness failed for {variant}/{distribution}/{seed}: "
                            f"eager={error_metrics(candidate, eager)}, "
                            f"reference={error_metrics(candidate, reference)}"
                        )
                    randomized_rows.append(
                        {
                            "variant": variant,
                            "module": record.name,
                            "distribution": distribution,
                            "seed": seed,
                            "candidate_vs_eager": {
                                **error_metrics(candidate, eager),
                                "allclose": candidate_eager_close,
                            },
                            "candidate_vs_explicit_fp32_reference": {
                                **error_metrics(candidate, reference),
                                "allclose": candidate_reference_close,
                            },
                        }
                    )
                    del activation, eager, candidate, reference
            del explicit_weight
            torch.cuda.empty_cache()

    log("K2 correctness: exhaustive dispatch/fallback rejection matrix")
    fallback = run_fallback_checks(representatives[expected_variants[0]], device)
    with torch.inference_mode():
        for record in prepared:
            if not eligible_linear4bit_decode(record.module, input_by_k[record.geometry[0]]):
                raise AssertionError(f"valid K2 module unexpectedly ineligible: {record.name}")
    choices = validate_autotune_choices(prereg, prepared)
    return {
        "tolerances": {"atol": atol, "rtol": rtol, "dtype": "torch.bfloat16"},
        "semantic_microcase": semantic,
        "actual_weights": {
            "seed": 20260916,
            "distribution": "unit_normal",
            "modules_checked": len(actual_rows),
            "variants_checked": len({row["variant"] for row in actual_rows}),
            "rows": actual_rows,
            "status": "PASS",
        },
        "randomized_representatives": {
            "representatives": len(representatives),
            "distributions": distributions,
            "seeds": seeds,
            "trials": len(randomized_rows),
            "rows": randomized_rows,
            "status": "PASS",
        },
        "fallback": fallback,
        "autotune_choices": choices,
        "status": "PASS",
    }


def write_correctness_checkpoint(
    path: Path,
    receipt_path: Path,
    prereg_path: Path,
    prereg_sha: str,
    verified: dict[str, Any],
    correctness: dict[str, Any],
    layout: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    if sha256(prereg_path) != prereg_sha:
        raise AssertionError("K2 preregistration changed during correctness")
    result = {
        "schema_version": 1,
        "stage": "OP5_K2_ISOLATED_CORRECTNESS",
        "timestamp": now_iso(),
        "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
        "candidate": {
            "path": str(ROOT / "scripts/operator/k2_nf4_gemv.py"),
            "sha256": sha256(ROOT / "scripts/operator/k2_nf4_gemv.py"),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "verified_sources": verified["sources"],
        "git_sha": verified["git_sha"],
        "environment": environment,
        "layout": layout,
        "correctness": correctness,
        "preregistration_sha256_start": prereg_sha,
        "preregistration_sha256_end": sha256(prereg_path),
        "status": "PASS",
    }
    atomic_write_json(path, result)
    receipt = {
        "schema_version": 1,
        "stage": "OP5_K2_ISOLATED_CORRECTNESS",
        "timestamp": now_iso(),
        "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
        "result": {"path": str(path), "sha256": sha256(path)},
        "gates": {
            "frozen_preregistration_immutable": "PASS",
            "semantic_microcases": "PASS",
            "all_343_actual_weights_vs_eager": "PASS",
            "representative_14_x_3_x_3_vs_eager_and_fp32_reference": "PASS",
            "source_tensors_unmodified": "PASS",
            "fallback_rejection_matrix": "PASS",
            "frozen_autotune_space": "PASS",
        },
        "formal_performance_conclusion": "PENDING_TWO_VALID_CALENDAR_DAYS",
        "status": "PASS",
    }
    atomic_write_json(receipt_path, receipt)
    if sha256(prereg_path) != prereg_sha:
        raise AssertionError("K2 preregistration changed through correctness receipt write")
    return receipt


def make_performance_operations(
    prepared: list[PreparedModule],
) -> tuple[dict[tuple[int, int], dict[str, Callable[..., torch.Tensor]]], dict[str, float]]:
    operations: dict[tuple[int, int], dict[str, Callable[..., torch.Tensor]]] = {}
    wrapper_seconds: dict[str, float] = {}
    for geometry in sorted(set(record.geometry for record in prepared)):
        eager, compiled, seconds = make_vendor_operations(geometry, 64)
        operations[geometry] = {"eager": eager, "compile": compiled}
        wrapper_seconds[f"K{geometry[0]}_N{geometry[1]}"] = seconds
    return operations, wrapper_seconds


def operation_call(
    implementation: str,
    record: PreparedModule,
    operations: dict[tuple[int, int], dict[str, Callable[..., torch.Tensor]]],
) -> torch.Tensor:
    if implementation == "candidate":
        return candidate_call(record)
    return operations[record.geometry][implementation](
        record.input_tensor,
        record.packed_weight,
        record.second_absmax,
        record.first_absmax,
        record.scale_code,
        record.scale_offset,
    )


def benchmark_day(
    prereg: dict[str, Any],
    prepared: list[PreparedModule],
    decode_rows: list[dict[str, Any]],
    monitor: NvidiaSmiMonitor,
    ceiling_gbps: float,
    device: int,
) -> dict[str, Any]:
    protocol = prereg["measurement_protocol"]
    torch._dynamo.config.recompile_limit = 16
    log("K2 day: creating and materializing 11 fullgraph vendor-compile baselines")
    operations, wrapper_seconds = make_performance_operations(prepared)
    compile_started = time.perf_counter()
    with torch.inference_mode():
        for record in prepared:
            eager = operation_call("eager", record, operations)
            compiled = operation_call("compile", record, operations)
            if not torch.equal(eager, compiled):
                raise AssertionError(f"compiled vendor baseline mismatch for {record.name}")
    torch.cuda.synchronize(device)
    compile_materialization_seconds = time.perf_counter() - compile_started

    def invoke_sequence(implementation: str) -> None:
        for record in prepared:
            operation_call(implementation, record, operations)

    implementations = ["eager", "compile", "candidate"]
    warmup_sequences = int(protocol["warmup_full_343_call_sequences_per_implementation"])
    minimum_seconds = float(protocol["steady_state_warmup_min_seconds"])
    log(f"K2 day: warming {warmup_sequences} full 343-call sequences per implementation")
    monitor.set_phase("k2_day_warmup")
    monitor.sample_now()
    warmup_started = time.perf_counter()
    with torch.inference_mode():
        for repetition in range(warmup_sequences):
            rotated = implementations[repetition % 3 :] + implementations[: repetition % 3]
            for implementation in rotated:
                invoke_sequence(implementation)
        torch.cuda.synchronize(device)
        extra_rounds = 0
        while time.perf_counter() - warmup_started < minimum_seconds:
            for implementation in implementations:
                invoke_sequence(implementation)
            extra_rounds += 1
        torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    measured_sequences = int(protocol["measured_full_343_call_sequences_per_implementation"])
    samples: dict[str, dict[str, list[float]]] = {
        implementation: defaultdict(list) for implementation in implementations
    }
    sequence_samples: dict[str, list[float]] = {
        implementation: [] for implementation in implementations
    }
    log(f"K2 day: timing {measured_sequences} rotated full 343-call sequences x 3")
    monitor.set_phase("k2_day_measurement")
    monitor.sample_now()
    with torch.inference_mode():
        for repetition in range(measured_sequences):
            rotated = implementations[repetition % 3 :] + implementations[: repetition % 3]
            for implementation in rotated:
                sequence_start = torch.cuda.Event(enable_timing=True)
                sequence_end = torch.cuda.Event(enable_timing=True)
                pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
                sequence_start.record()
                for record in prepared:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    operation_call(implementation, record, operations)
                    end.record()
                    pending.append((record.variant, start, end))
                sequence_end.record()
                torch.cuda.synchronize(device)
                sequence_samples[implementation].append(
                    sequence_start.elapsed_time(sequence_end) * 1000.0
                )
                for variant, start, end in pending:
                    samples[implementation][variant].append(start.elapsed_time(end) * 1000.0)
            if (repetition + 1) % 25 == 0:
                log(f"K2 day: completed {repetition + 1}/{measured_sequences} measured rotations")
    monitor.sample_now()

    census_by_variant = {row["variant"]: row for row in decode_rows}
    bytes_by_variant: dict[str, set[int]] = defaultdict(set)
    for record in prepared:
        bytes_by_variant[record.variant].add(record.analytic_bytes)
    rows: list[dict[str, Any]] = []
    for variant in sorted(census_by_variant):
        if len(bytes_by_variant[variant]) != 1:
            raise AssertionError(f"analytic byte count varies within {variant}")
        analytic_bytes = next(iter(bytes_by_variant[variant]))
        summaries = {
            implementation: summarize(samples[implementation][variant])
            for implementation in implementations
        }
        expected_samples = int(census_by_variant[variant]["calls_per_forward"]) * measured_sequences
        if any(int(summary["count"]) != expected_samples for summary in summaries.values()):
            raise AssertionError(f"sample count drift for {variant}")
        eager_us = float(summaries["eager"]["p50"])
        compile_us = float(summaries["compile"]["p50"])
        candidate_us = float(summaries["candidate"]["p50"])
        best_baseline_us = min(eager_us, compile_us)
        candidate_gbps = analytic_bytes / (candidate_us * 1000.0)
        census_row = census_by_variant[variant]
        rows.append(
            {
                "variant": variant,
                "dims": census_row["dims"],
                "calls_per_forward": census_row["calls_per_forward"],
                "projected_invocations": census_row["invocations"],
                "invocation_weight": census_row["weight"],
                "work_weight": census_row["work_weight"],
                "analytic_bytes": analytic_bytes,
                "eager": summaries["eager"],
                "compile": summaries["compile"],
                "candidate": summaries["candidate"],
                "best_baseline": "eager" if eager_us <= compile_us else "compile",
                "best_baseline_p50_us": best_baseline_us,
                "candidate_speedup_vs_eager": eager_us / candidate_us,
                "candidate_speedup_vs_compile": compile_us / candidate_us,
                "candidate_speedup_vs_best_baseline": best_baseline_us / candidate_us,
                "candidate_measured_gbps_decimal": candidate_gbps,
                "candidate_tier_a_relative_score": candidate_gbps / ceiling_gbps,
                "candidate_headroom_fraction": 1.0 - candidate_gbps / ceiling_gbps,
            }
        )

    total_invocations = sum(int(row["projected_invocations"]) for row in rows)

    def weighted(field: str, implementation: str | None = None) -> float:
        return sum(
            float(row[implementation][field] if implementation else row[field])
            * int(row["projected_invocations"])
            for row in rows
        ) / total_invocations

    weighted_bytes = sum(
        int(row["analytic_bytes"]) * int(row["projected_invocations"]) for row in rows
    ) / total_invocations
    weighted_eager_us = weighted("p50", "eager")
    weighted_compile_us = weighted("p50", "compile")
    weighted_candidate_us = weighted("p50", "candidate")
    best_baseline_us = min(weighted_eager_us, weighted_compile_us)
    best_baseline_name = "eager" if weighted_eager_us <= weighted_compile_us else "compile"
    speedup = best_baseline_us / weighted_candidate_us
    mlp_variants = ("mlp_gate_proj", "mlp_up_proj", "mlp_down_proj")
    mlp_speedups = {
        row["variant"]: float(row["candidate_speedup_vs_best_baseline"])
        for row in rows
        if row["variant"] in mlp_variants
    }
    if len(mlp_speedups) != 3:
        raise AssertionError("MLP regression guard coverage drift")
    mlp_guard_pass = all(value >= 0.95 for value in mlp_speedups.values())
    if speedup >= 1.05 and mlp_guard_pass:
        classification = "POSITIVE"
    elif speedup < 0.95:
        classification = "NEGATIVE"
    elif speedup < 1.05:
        classification = "PARITY"
    else:
        classification = "MLP_REGRESSION_GUARD_FAIL"
    candidate_gbps = weighted_bytes / (weighted_candidate_us * 1000.0)
    return {
        "baseline_identity": {
            "eager": prereg["baselines"]["eager"]["definition"],
            "compile": prereg["baselines"]["compile"]["definition"],
            "candidate": prereg["baselines"]["candidate"]["definition"],
            "compile_fullgraph": True,
            "compile_mode": "reduce-overhead",
            "compile_dynamic": False,
            "compile_graph_breaks": 0,
            "torch_dynamo_recompile_limit": 16,
            "wrapper_creation_seconds": wrapper_seconds,
            "compile_materialization_and_343_exact_checks_seconds": compile_materialization_seconds,
        },
        "measurement_protocol": {
            "warmup_full_sequences_per_implementation": warmup_sequences,
            "warmup_extra_rounds_for_minimum_seconds": extra_rounds,
            "warmup_seconds": warmup_seconds,
            "measured_full_sequences_per_implementation": measured_sequences,
            "module_calls_per_sequence": len(prepared),
            "interleave": protocol["interleave_order"],
        },
        "sequence_latency_us": {
            implementation: summarize(values)
            for implementation, values in sequence_samples.items()
        },
        "rows": rows,
        "aggregate": {
            "projected_decode_invocations": total_invocations,
            "invocation_weighted_analytic_bytes_per_call": weighted_bytes,
            "invocation_weighted_eager_p50_us_per_call": weighted_eager_us,
            "invocation_weighted_compile_p50_us_per_call": weighted_compile_us,
            "invocation_weighted_candidate_p50_us_per_call": weighted_candidate_us,
            "best_baseline": best_baseline_name,
            "invocation_weighted_best_baseline_p50_us_per_call": best_baseline_us,
            "candidate_speedup_vs_eager": weighted_eager_us / weighted_candidate_us,
            "candidate_speedup_vs_compile": weighted_compile_us / weighted_candidate_us,
            "candidate_speedup_vs_best_baseline": speedup,
            "candidate_measured_gbps_decimal": candidate_gbps,
            "same_day_ceiling_gbps_decimal": ceiling_gbps,
            "candidate_tier_a_relative_score": candidate_gbps / ceiling_gbps,
            "candidate_headroom_fraction": 1.0 - candidate_gbps / ceiling_gbps,
            "mlp_speedups_vs_best_baseline": mlp_speedups,
            "mlp_no_more_than_5pct_regression": mlp_guard_pass,
            "classification": classification,
        },
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    prereg_path = args.prereg.resolve()
    prereg_sha = sha256(prereg_path)
    verified = verify_preregistration(prereg_path, prereg_sha)
    prereg = verified["prereg"]
    selection = verified["selection_config"]
    environment = environment_identity()
    monitor = NvidiaSmiMonitor(
        interval_seconds=float(prereg["measurement_protocol"]["gpu_telemetry_interval_seconds"])
    )
    monitor.start()
    telemetry: dict[str, Any] | None = None
    try:
        bandwidth = None
        if args.mode == "day":
            bandwidth = run_same_day_bandwidth(selection, monitor, args.device)
        model, model_load = load_formal_carrier(selection, args.device)
        prepared, layout = prepare_modules(
            model, selection, verified["decode_rows"], args.device
        )
        monitor.set_phase("k2_correctness")
        monitor.sample_now()
        correctness = run_correctness(prereg, prepared, layout, args.device)
        correctness_receipt = write_correctness_checkpoint(
            args.correctness.resolve(),
            args.correctness_receipt.resolve(),
            prereg_path,
            prereg_sha,
            verified,
            correctness,
            layout,
            environment,
        )
        log(f"K2 correctness PASS: {correctness_receipt['result']['path']}")
        if args.mode == "correctness-only":
            telemetry = monitor.stop()
            return correctness_receipt

        assert bandwidth is not None
        benchmark = benchmark_day(
            prereg,
            prepared,
            verified["decode_rows"],
            monitor,
            float(bandwidth["ceiling_gbps_decimal"]),
            args.device,
        )
        telemetry = monitor.stop()
        measurement_telemetry = monitor.summary_for_phase("k2_day_measurement")
        bandwidth_telemetry = monitor.summary_for_phase("op5_same_day_bandwidth")
        memory_summary = measurement_telemetry["numeric"]["memory_mhz"]
        if memory_summary is None:
            raise AssertionError("K2 measurement has no memory-clock witness")
        memory_clock_ratio = float(memory_summary["p50"]) / float(
            prereg["measurement_protocol"]["op0_memory_clock_mhz_p50"]
        )
        thermal_valid = not telemetry["thermal_or_hardware_throttle"]
        memory_clock_valid = memory_clock_ratio >= float(
            prereg["measurement_protocol"]["minimum_memory_clock_ratio_vs_op0"]
        )
        day_valid = thermal_valid and memory_clock_valid
        prereg_sha_end = sha256(prereg_path)
        if prereg_sha_end != prereg_sha:
            raise AssertionError("K2 preregistration changed during formal day measurement")
        result = {
            "schema_version": 1,
            "stage": "OP5_K2_ISOLATED_DAY",
            "calendar_day": datetime.now().astimezone().date().isoformat(),
            "timestamp": now_iso(),
            "evidence_scope": "local RTX 4060 Laptop GPU / sm_89 / WSL2 only",
            "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
            "candidate": {
                "path": str(ROOT / "scripts/operator/k2_nf4_gemv.py"),
                "sha256": sha256(ROOT / "scripts/operator/k2_nf4_gemv.py"),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "verified_sources": verified["sources"],
            "git_sha": verified["git_sha"],
            "environment": environment,
            "same_day_bandwidth": bandwidth,
            "model_load": model_load,
            "layout": layout,
            "correctness_checkpoint": correctness_receipt["result"],
            "benchmark": benchmark,
            "telemetry": telemetry,
            "bandwidth_telemetry": bandwidth_telemetry,
            "measurement_telemetry": measurement_telemetry,
            "telemetry_gates": {
                "thermal_or_hardware_throttle_absent": thermal_valid,
                "memory_clock_ratio_vs_op0": memory_clock_ratio,
                "minimum_memory_clock_ratio": prereg["measurement_protocol"]["minimum_memory_clock_ratio_vs_op0"],
                "memory_clock_gate": memory_clock_valid,
            },
            "memory_final": memory_snapshot(torch, args.device),
            "cross_day_replication": {
                "required_valid_calendar_days": prereg["cross_day_replication"]["minimum_valid_calendar_days"],
                "valid_days_in_this_result": 1 if day_valid else 0,
                "final_performance_conclusion": "PENDING_SECOND_VALID_CALENDAR_DAY",
            },
            "ncu_tier_b": "PENDING",
            "reconnection": "PENDING_ISOLATED_PASS",
            "preregistration_sha256_start": prereg_sha,
            "preregistration_sha256_end": prereg_sha_end,
            "day_valid": day_valid,
            "performance_outcome": benchmark["aggregate"]["classification"],
            "status": "PROVISIONAL_DAY1_VALID" if day_valid else "DAY1_INVALID_TELEMETRY",
        }
        atomic_write_json(args.result.resolve(), result)
        receipt = {
            "schema_version": 1,
            "stage": "OP5_K2_ISOLATED_DAY1",
            "calendar_day": result["calendar_day"],
            "timestamp": now_iso(),
            "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
            "correctness": correctness_receipt["result"],
            "result": {"path": str(args.result.resolve()), "sha256": sha256(args.result.resolve())},
            "gates": {
                "frozen_preregistration_immutable": "PASS",
                "upstream_and_source_hashes": "PASS",
                "formal_c02_full_hash": model_load["integrity"]["status"],
                "correctness": correctness["status"],
                "all_343_weights_and_14_variants": "PASS",
                "same_day_bandwidth": "PASS",
                "three_way_200_full_sequence_measurement": "PASS",
                "thermal_or_hardware_throttle_absent": "PASS" if thermal_valid else "FAIL",
                "memory_clock_gate": "PASS" if memory_clock_valid else "FAIL",
                "cross_day_replication": "PENDING",
                "ncu_tier_b": "PENDING",
                "reconnection": "PENDING",
            },
            "performance_outcome": benchmark["aggregate"]["classification"],
            "candidate_speedup_vs_best_baseline": benchmark["aggregate"]["candidate_speedup_vs_best_baseline"],
            "formal_final_conclusion": False,
            "status": result["status"],
        }
        atomic_write_json(args.receipt.resolve(), receipt)
        if sha256(prereg_path) != prereg_sha:
            raise AssertionError("K2 preregistration changed through day receipt write")
        return receipt
    finally:
        if telemetry is None:
            monitor.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--correctness", type=Path, default=DEFAULT_CORRECTNESS)
    parser.add_argument("--correctness-receipt", type=Path, default=DEFAULT_CORRECTNESS_RECEIPT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--mode", choices=("correctness-only", "day"), default="day")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = now_iso()
    try:
        receipt = execute(args)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if receipt["status"] in ("PASS", "PROVISIONAL_DAY1_VALID") else 2
    except BaseException as exc:
        failure_path = ROOT / (
            "results/operator/failures/"
            f"k2_day_failure_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}.json"
        )
        prereg_path = args.prereg.resolve()
        failure = {
            "schema_version": 1,
            "stage": "OP5_K2_ISOLATED_DAY",
            "implementation_fix_iteration": 1,
            "started_at": started,
            "failed_at": now_iso(),
            "preregistration": str(prereg_path),
            "preregistration_sha256": sha256(prereg_path) if prereg_path.is_file() else None,
            "candidate": str(ROOT / "scripts/operator/k2_nf4_gemv.py"),
            "candidate_sha256": sha256(ROOT / "scripts/operator/k2_nf4_gemv.py"),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "git_sha": git_sha(),
            "environment": environment_identity(),
            "status": "FAIL",
        }
        atomic_write_json(failure_path, failure)
        print(json.dumps({**failure, "failure_path": str(failure_path)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
