#!/usr/bin/env python3
"""Execute the frozen formal correctness gate for locked K3 GQA attention."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import torch

from k3_gqa_attention import (
    autotune_choices,
    candidate_gqa_attention,
    candidate_or_eager_attention,
    eligible_gqa_decode,
)
from op0_common import atomic_write_json, now_iso, sha256
from run_op5_selection import environment_identity, load_json
from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward


ROOT = Path(__file__).resolve().parents[2]
DATE_TAG = datetime.now().astimezone().strftime("%Y%m%d")
DEFAULT_PROTOCOL = ROOT / "configs/operator/k3_gqa_decode_attention_prereg.json"
DEFAULT_RESULT = ROOT / f"results/operator/pool/k3_gqa_correctness_{DATE_TAG}.json"
DEFAULT_RECEIPT = ROOT / f"results/operator/pool/k3_gqa_correctness_receipt_{DATE_TAG}.json"
PREREG_RECEIPT = ROOT / "results/operator/pool/k3_gqa_prereg_receipt_20260916.json"
PREREG_RECEIPT_SHA256 = "7516c79668bffb164101d3a80af2c2270dd69ff7a17b598cda57fd1193a57ebf"
CANDIDATE = ROOT / "scripts/operator/k3_gqa_attention.py"
CANDIDATE_SHA256 = "3d3d57ac20e390e9f117ac763af7d5ffcfabe084fce861461e8e62e183ac76f0"
LOCK_RECEIPT = ROOT / "results/operator/pool/k3_gqa_candidate_lock_receipt_20260916.json"
LOCK_RECEIPT_SHA256 = "261e6b443bb9adfb36403edc49d2328f6ac2426dc4af967c417794884ca8eb39"


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class AttentionProxy(torch.nn.Module):
    num_key_value_groups = 4
    head_dim = 256

    def __init__(self) -> None:
        super().__init__()
        self.eval()


def eager_output(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    **kwargs: Any,
) -> torch.Tensor:
    output, _ = eager_attention_forward(module, query, key, value, mask, **kwargs)
    return output


def make_inputs(
    key_tokens: int,
    head_dim: int,
    seed: int,
    scale: float,
    device: int,
    mask_kind: str = "all-zero visible mask",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=f"cuda:{device}")
    generator.manual_seed(seed)

    def random(shape: tuple[int, ...]) -> torch.Tensor:
        value = torch.randn(
            shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        return (value * scale).to(torch.bfloat16)

    query = random((1, 8, 1, head_dim))
    key = random((1, 2, key_tokens, head_dim))
    value = random((1, 2, key_tokens, head_dim))
    mask = torch.zeros(
        (1, 1, 1, key_tokens), dtype=torch.bfloat16, device=device
    )
    if mask_kind == "first element set to BF16 minimum":
        mask[..., 0] = torch.finfo(torch.bfloat16).min
    elif mask_kind == "last element set to BF16 minimum":
        mask[..., -1] = torch.finfo(torch.bfloat16).min
    elif mask_kind != "all-zero visible mask":
        raise AssertionError(f"unknown frozen mask case: {mask_kind}")
    return query, key, value, mask


def comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    delta = (actual.float() - expected.float()).abs()
    close = torch.isclose(actual, expected, atol=atol, rtol=rtol)
    exact = actual == expected
    return {
        "allclose": bool(close.all().item()),
        "max_abs": float(delta.max().item()),
        "exact_elements": int(exact.sum().item()),
        "allclose_elements": int(close.sum().item()),
        "non_allclose_elements": int((~close).sum().item()),
        "output_elements": actual.numel(),
    }


def verify_frozen_identity(
    protocol_path: Path,
    protocol_sha_start: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    if protocol.get("stage") != "OP5_K3":
        raise AssertionError("unexpected K3 protocol identity")
    if protocol.get("frozen_before_candidate_implementation") is not True:
        raise AssertionError("K3 protocol was not frozen before implementation")
    if protocol.get("frozen_before_candidate_measurement") is not True:
        raise AssertionError("K3 protocol was not frozen before measurement")
    if sha256(PREREG_RECEIPT) != PREREG_RECEIPT_SHA256:
        raise AssertionError("K3 preregistration receipt identity drift")
    if sha256(LOCK_RECEIPT) != LOCK_RECEIPT_SHA256:
        raise AssertionError("K3 candidate-lock receipt identity drift")
    if sha256(CANDIDATE) != CANDIDATE_SHA256:
        raise AssertionError("locked K3 candidate source identity drift")
    prereg = load_json(PREREG_RECEIPT)
    lock = load_json(LOCK_RECEIPT)
    if prereg.get("status") != "PASS":
        raise AssertionError("K3 preregistration receipt is not PASS")
    if lock.get("status") != "PASS_CANDIDATE_LOCKED":
        raise AssertionError("K3 candidate-lock receipt is not PASS")
    if lock.get("candidate", {}).get("sha256") != CANDIDATE_SHA256:
        raise AssertionError("K3 candidate-lock SHA field drift")
    if lock.get("protocol", {}).get("sha256") != protocol_sha_start:
        raise AssertionError("K3 lock receipt protocol SHA drift")
    if lock.get("lock_assertions", {}).get("formal_fresh_seeds_not_executed") is not True:
        raise AssertionError("K3 formal holdout was not unobserved at candidate lock")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("formal K3 correctness requires the local sm_89 CUDA device")

    verified_sources: dict[str, Any] = {}
    for name, identity in protocol["source_identity"].items():
        if name == "git_sha":
            continue
        path = absolute(identity["path"])
        observed = sha256(path)
        if observed != identity["sha256"]:
            raise AssertionError(f"K3 frozen source hash mismatch: {name}")
        verified_sources[name] = {
            "path": str(path),
            "sha256": observed,
            "status": "PASS",
        }
    if git_sha() != protocol["source_identity"]["git_sha"]:
        raise AssertionError("Git HEAD differs from frozen K3 identity")
    if sha256(protocol_path) != protocol_sha_start:
        raise AssertionError("K3 protocol changed during identity verification")
    return protocol, verified_sources


def run_trial(
    proxy: AttentionProxy,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    label: str,
    category: str,
    key_tokens: int,
    head_dim: int,
    seed: int | None,
    distribution: str,
    scale: float,
    mask_kind: str,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    proxy.head_dim = head_dim
    snapshots = [tensor.clone() for tensor in inputs]
    pointers = [tensor.data_ptr() for tensor in inputs]
    versions = [tensor._version for tensor in inputs]
    with torch.inference_mode():
        expected = eager_output(
            proxy,
            *inputs,
            dropout=0.0,
            scaling=1.0,
        ).clone()
        actual = candidate_gqa_attention(*inputs).clone()
        torch.cuda.synchronize(inputs[0].device)
    expected_shape = (1, 1, 8, head_dim)
    output_contract = bool(
        tuple(actual.shape) == expected_shape
        and actual.dtype == torch.bfloat16
        and actual.device == inputs[0].device
        and actual.is_contiguous()
        and bool(torch.isfinite(actual).all().item())
    )
    if not output_contract:
        raise AssertionError(f"K3 output contract failure: {label}")
    metrics = comparison(actual, expected, atol, rtol)
    if not metrics["allclose"]:
        raise AssertionError(
            f"locked K3 mismatch: {label}; max_abs={metrics['max_abs']}; "
            f"nonclose={metrics['non_allclose_elements']}"
        )
    source_unmodified = all(
        torch.equal(tensor, snapshot)
        and tensor.data_ptr() == pointer
        and tensor._version == version
        for tensor, snapshot, pointer, version in zip(
            inputs, snapshots, pointers, versions
        )
    )
    if not source_unmodified:
        raise AssertionError(f"K3 source tensor mutation: {label}")
    return {
        "label": label,
        "category": category,
        "K": key_tokens,
        "D": head_dim,
        "seed": seed,
        "distribution": distribution,
        "scale": scale,
        "mask": mask_kind,
        "comparison_vs_transformers_eager": metrics,
        "output_contract": output_contract,
        "source_tensors_unmodified": source_unmodified,
    }


def run_mapping_microcase(
    proxy: AttentionProxy,
    device: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    key_tokens, head_dim = 512, 256
    proxy.head_dim = head_dim
    query = torch.zeros(
        (1, 8, 1, head_dim), dtype=torch.bfloat16, device=device
    )
    key = torch.zeros(
        (1, 2, key_tokens, head_dim), dtype=torch.bfloat16, device=device
    )
    value = torch.empty(
        (1, 2, key_tokens, head_dim), dtype=torch.bfloat16, device=device
    )
    value[:, 0].fill_(1.0)
    value[:, 1].fill_(4.0)
    mask = torch.zeros(
        (1, 1, 1, key_tokens), dtype=torch.bfloat16, device=device
    )
    row = run_trial(
        proxy,
        (query, key, value, mask),
        label="gqa_head_mapping_0_3_to_kv0_4_7_to_kv1",
        category="semantic_microcase",
        key_tokens=key_tokens,
        head_dim=head_dim,
        seed=None,
        distribution="deterministic_zero_qk_constant_v",
        scale=0.0,
        mask_kind="all-zero visible mask",
        atol=atol,
        rtol=rtol,
    )
    with torch.inference_mode():
        output = candidate_gqa_attention(query, key, value, mask)
        first_group = output[0, 0, :4].float()
        second_group = output[0, 0, 4:].float()
    mapping_pass = bool(
        torch.allclose(first_group, torch.ones_like(first_group), atol=atol, rtol=rtol)
        and torch.allclose(
            second_group,
            torch.full_like(second_group, 4.0),
            atol=atol,
            rtol=rtol,
        )
    )
    if not mapping_pass:
        raise AssertionError("K3 GQA query-to-KV head mapping microcase failed")
    row["head_mapping"] = {
        "query_heads_0_through_3_expected_kv_head": 0,
        "query_heads_4_through_7_expected_kv_head": 1,
        "status": "PASS",
    }
    return row


def exact_fallback_case(
    name: str,
    module: AttentionProxy,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    query, key, value, mask = inputs
    eligibility = eligible_gqa_decode(
        module,
        query,
        key,
        value,
        mask,
        dropout=kwargs["dropout"],
        scaling=kwargs["scaling"],
        softcap=kwargs["softcap"],
        output_attentions=kwargs["output_attentions"],
    )
    if eligibility:
        raise AssertionError(f"K3 rejected fallback case remained eligible: {name}")
    actual = candidate_or_eager_attention(module, query, key, value, mask, **kwargs)
    expected = eager_attention_forward(module, query, key, value, mask, **kwargs)
    torch.cuda.synchronize() if query.is_cuda else None
    exact = bool(
        len(actual) == len(expected)
        and all(torch.equal(left, right) for left, right in zip(actual, expected))
    )
    if not exact:
        raise AssertionError(f"K3 fallback was not exact: {name}")
    return {"case": name, "eligible": eligibility, "exact_eager_fallback": exact}


def run_fallback_matrix(device: int) -> dict[str, Any]:
    base_proxy = AttentionProxy()
    base = make_inputs(512, 256, 20261011, 1.0, device)
    default = {
        "dropout": 0.0,
        "scaling": 1.0,
        "softcap": None,
        "output_attentions": False,
    }
    cases: list[tuple[str, AttentionProxy, tuple[Any, ...], dict[str, Any], bool]] = []

    cases.append(("grad_enabled", base_proxy, base, default, True))

    training_proxy = AttentionProxy()
    training_proxy.train()
    cases.append(("module_training", training_proxy, base, default, False))

    for name, updates in (
        ("dropout_nonzero", {"dropout": 0.25}),
        ("scaling_not_one", {"scaling": 0.5}),
        ("softcap_present", {"softcap": 10.0}),
        ("output_attentions_true", {"output_attentions": True}),
    ):
        cases.append((name, base_proxy, base, {**default, **updates}, False))
    cases.append(("attention_mask_none", base_proxy, (*base[:3], None), default, False))

    q_base = torch.empty(
        (1, 8, 1, 512), dtype=torch.bfloat16, device=device
    )
    noncontiguous_query = q_base[..., ::2]
    noncontiguous_query.copy_(base[0])
    cases.append(
        (
            "query_noncontiguous",
            base_proxy,
            (noncontiguous_query, *base[1:]),
            default,
            False,
        )
    )
    mask_base = torch.zeros(
        (1, 1, 1, 1024), dtype=torch.bfloat16, device=device
    )
    noncontiguous_mask = mask_base[..., ::2]
    cases.append(
        (
            "mask_noncontiguous",
            base_proxy,
            (*base[:3], noncontiguous_mask),
            default,
            False,
        )
    )

    float_inputs = tuple(tensor.float() for tensor in base)
    cases.append(("float32_dtype", base_proxy, float_inputs, default, False))

    unsupported_k = make_inputs(513, 256, 20261011, 1.0, device)
    cases.append(("unsupported_k_for_d256", base_proxy, unsupported_k, default, False))

    unsupported_d = make_inputs(512, 128, 20261011, 1.0, device)
    d128_proxy = AttentionProxy()
    d128_proxy.head_dim = 128
    cases.append(("unsupported_head_dim", d128_proxy, unsupported_d, default, False))

    group2_generator = torch.Generator(device=f"cuda:{device}")
    group2_generator.manual_seed(20261011)
    group2_inputs = (
        torch.randn(
            (1, 4, 1, 256),
            dtype=torch.float32,
            device=device,
            generator=group2_generator,
        ).to(torch.bfloat16),
        torch.randn(
            (1, 2, 512, 256),
            dtype=torch.float32,
            device=device,
            generator=group2_generator,
        ).to(torch.bfloat16),
        torch.randn(
            (1, 2, 512, 256),
            dtype=torch.float32,
            device=device,
            generator=group2_generator,
        ).to(torch.bfloat16),
        torch.zeros((1, 1, 1, 512), dtype=torch.bfloat16, device=device),
    )
    group2_proxy = AttentionProxy()
    group2_proxy.num_key_value_groups = 2
    cases.append(("non_frozen_gqa_group", group2_proxy, group2_inputs, default, False))

    rows: list[dict[str, Any]] = []
    for name, module, inputs, kwargs, needs_grad_enabled in cases:
        if needs_grad_enabled:
            row = exact_fallback_case(name, module, inputs, kwargs)
        else:
            with torch.inference_mode():
                row = exact_fallback_case(name, module, inputs, kwargs)
        rows.append(row)
    return {"cases": len(rows), "rows": rows, "status": "PASS"}


def validate_autotune(protocol: dict[str, Any]) -> dict[str, Any]:
    choices = autotune_choices()
    design = protocol["candidate_design"]
    frozen = {
        "score": design["score_search_space"],
        "softmax": design["softmax_search_space"],
        "pv": design["pv_search_space"],
    }
    expected_geometries = {
        (int(row["K"]), int(row["D"]))
        for row in protocol["target_population"]["performance_cases"]
    }
    formal_lengths = protocol["formal_correctness_protocol"]["all_length_cases"]
    expected_geometries |= {
        (int(key_tokens), 256) for key_tokens in formal_lengths["sliding_K"]
    }
    expected_geometries |= {
        (int(key_tokens), 512) for key_tokens in formal_lengths["full_K"]
    }
    allowed_fields = {
        "score": ("BLOCK_K", "BLOCK_D", "num_warps", "num_stages"),
        "softmax": ("num_warps", "num_stages"),
        "pv": ("BLOCK_D", "BLOCK_K", "num_warps", "num_stages"),
    }
    for stage, rows in choices.items():
        allowed = {
            tuple(config[field] for field in allowed_fields[stage])
            for config in frozen[stage]
        }
        if not rows:
            raise AssertionError(f"K3 has no recorded autotune choices for {stage}")
        for row in rows:
            observed = tuple(row.get(field) for field in allowed_fields[stage])
            if observed not in allowed:
                raise AssertionError(f"non-frozen K3 {stage} autotune choice: {row}")
    score_geometries = {(int(row["key"][0]), int(row["key"][1])) for row in choices["score"]}
    pv_geometries = {(int(row["key"][0]), int(row["key"][1])) for row in choices["pv"]}
    softmax_lengths = {int(row["key"][0]) for row in choices["softmax"]}
    expected_lengths = {key_tokens for key_tokens, _ in expected_geometries}
    if score_geometries != expected_geometries or pv_geometries != expected_geometries:
        raise AssertionError("K3 score/PV autotune geometry coverage drift")
    if softmax_lengths != expected_lengths:
        raise AssertionError("K3 softmax autotune length coverage drift")
    return {
        "choices": choices,
        "expected_geometries": [
            {"K": key_tokens, "D": head_dim}
            for key_tokens, head_dim in sorted(expected_geometries)
        ],
        "status": "PASS",
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol_sha_start = sha256(protocol_path)
    protocol, verified_sources = verify_frozen_identity(
        protocol_path, protocol_sha_start
    )
    torch.cuda.set_device(args.device)
    settings = protocol["formal_correctness_protocol"]
    atol = float(settings["tolerance"]["atol"])
    rtol = float(settings["tolerance"]["rtol"])
    proxy = AttentionProxy()
    rows: list[dict[str, Any]] = []

    all_lengths = settings["all_length_cases"]
    length_cases = [
        (int(key_tokens), 256) for key_tokens in all_lengths["sliding_K"]
    ] + [(int(key_tokens), 512) for key_tokens in all_lengths["full_K"]]
    if len(length_cases) != int(all_lengths["expected_cases"]):
        raise AssertionError("K3 formal all-length case count drift")
    log(f"K3 correctness: {len(length_cases)} frozen all-length cases")
    for index, (key_tokens, head_dim) in enumerate(length_cases, 1):
        inputs = make_inputs(
            key_tokens,
            head_dim,
            int(all_lengths["seed"]),
            1.0,
            args.device,
        )
        rows.append(
            run_trial(
                proxy,
                inputs,
                label=f"all_length_k{key_tokens}_d{head_dim}",
                category="all_length",
                key_tokens=key_tokens,
                head_dim=head_dim,
                seed=int(all_lengths["seed"]),
                distribution=all_lengths["distribution"],
                scale=1.0,
                mask_kind="all-zero visible mask",
                atol=atol,
                rtol=rtol,
            )
        )
        del inputs
        torch.cuda.empty_cache()
        log(f"K3 correctness: all-length {index}/{len(length_cases)} PASS")

    matrix = settings["representative_matrix"]
    expected_matrix_trials = int(matrix["expected_trials"])
    matrix_count = 0
    log(f"K3 correctness: {expected_matrix_trials} representative holdout trials")
    for case in matrix["cases"]:
        for distribution in matrix["distributions"]:
            for seed in matrix["seeds"]:
                key_tokens = int(case["K"])
                head_dim = int(case["D"])
                scale = float(distribution["scale"])
                inputs = make_inputs(
                    key_tokens,
                    head_dim,
                    int(seed),
                    scale,
                    args.device,
                )
                rows.append(
                    run_trial(
                        proxy,
                        inputs,
                        label=(
                            f"matrix_{case['family']}_k{key_tokens}_d{head_dim}_"
                            f"{distribution['name']}_seed{seed}"
                        ),
                        category="representative_matrix",
                        key_tokens=key_tokens,
                        head_dim=head_dim,
                        seed=int(seed),
                        distribution=distribution["name"],
                        scale=scale,
                        mask_kind="all-zero visible mask",
                        atol=atol,
                        rtol=rtol,
                    )
                )
                matrix_count += 1
                del inputs
        torch.cuda.empty_cache()
        log(f"K3 correctness: representative {case['family']} K={case['K']} PASS")
    if matrix_count != expected_matrix_trials:
        raise AssertionError("K3 representative matrix trial count drift")

    mask_count = 0
    log("K3 correctness: frozen additive-mask cases")
    for case in matrix["cases"]:
        for mask_kind in settings["mask_cases"]:
            key_tokens = int(case["K"])
            head_dim = int(case["D"])
            seed = int(settings["fresh_seeds"][0])
            inputs = make_inputs(
                key_tokens,
                head_dim,
                seed,
                1.0,
                args.device,
                mask_kind,
            )
            rows.append(
                run_trial(
                    proxy,
                    inputs,
                    label=f"mask_{case['family']}_k{key_tokens}_{mask_kind}",
                    category="mask_matrix",
                    key_tokens=key_tokens,
                    head_dim=head_dim,
                    seed=seed,
                    distribution="unit_normal",
                    scale=1.0,
                    mask_kind=mask_kind,
                    atol=atol,
                    rtol=rtol,
                )
            )
            mask_count += 1
            del inputs
        torch.cuda.empty_cache()

    mapping = run_mapping_microcase(proxy, args.device, atol, rtol)
    rows.append(mapping)
    fallback = run_fallback_matrix(args.device)
    tuning = validate_autotune(protocol)

    max_abs = max(
        float(row["comparison_vs_transformers_eager"]["max_abs"])
        for row in rows
    )
    exact_trials = sum(
        row["comparison_vs_transformers_eager"]["exact_elements"]
        == row["comparison_vs_transformers_eager"]["output_elements"]
        for row in rows
    )
    protocol_sha_end = sha256(protocol_path)
    candidate_sha_end = sha256(CANDIDATE)
    if protocol_sha_end != protocol_sha_start:
        raise AssertionError("K3 protocol changed during formal correctness")
    if candidate_sha_end != CANDIDATE_SHA256:
        raise AssertionError("K3 candidate changed during formal correctness")

    result = {
        "schema_version": 1,
        "stage": "OP5_K3_GQA_DECODE_ATTENTION_CORRECTNESS",
        "timestamp": now_iso(),
        "evidence_scope": protocol["host_scope"],
        "protocol": {
            "path": str(protocol_path),
            "sha256_start": protocol_sha_start,
            "sha256_end": protocol_sha_end,
        },
        "preregistration_receipt": {
            "path": str(PREREG_RECEIPT),
            "sha256": PREREG_RECEIPT_SHA256,
        },
        "candidate_lock_receipt": {
            "path": str(LOCK_RECEIPT),
            "sha256": LOCK_RECEIPT_SHA256,
        },
        "locked_candidate": {
            "path": str(CANDIDATE),
            "sha256_start": CANDIDATE_SHA256,
            "sha256_end": candidate_sha_end,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "verified_sources": verified_sources,
        "git_sha": git_sha(),
        "environment": environment_identity(),
        "correctness": {
            "tolerance": {"atol": atol, "rtol": rtol},
            "all_length_cases": len(length_cases),
            "representative_matrix_trials": matrix_count,
            "mask_matrix_trials": mask_count,
            "semantic_microcases": 1,
            "total_candidate_trials": len(rows),
            "exact_output_trials": exact_trials,
            "maximum_absolute_error": max_abs,
            "rows": rows,
            "status": "PASS",
        },
        "fallback": fallback,
        "autotune": tuning,
        "formal_candidate_performance_started": False,
        "next_authorized_action": "Run K3 isolated Day-1 performance with the unchanged protocol, candidate, and correctness receipt; apply the frozen bounded continuation gate.",
        "status": "PASS",
    }
    result_path = args.result.resolve()
    atomic_write_json(result_path, result)
    receipt = {
        "schema_version": 1,
        "stage": "OP5_K3_GQA_DECODE_ATTENTION_CORRECTNESS",
        "timestamp": now_iso(),
        "protocol": {"path": str(protocol_path), "sha256": protocol_sha_start},
        "locked_candidate": {"path": str(CANDIDATE), "sha256": candidate_sha_end},
        "candidate_lock_receipt": {
            "path": str(LOCK_RECEIPT),
            "sha256": LOCK_RECEIPT_SHA256,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "result": {"path": str(result_path), "sha256": sha256(result_path)},
        "gates": {
            "protocol_immutable": "PASS",
            "candidate_source_locked": "PASS",
            "source_and_git_identity": "PASS",
            "all_length_cases": "PASS",
            "fresh_representative_matrix": "PASS",
            "mask_matrix": "PASS",
            "gqa_head_mapping_microcase": "PASS",
            "source_tensors_unmodified": "PASS",
            "fallback_rejection_matrix": "PASS",
            "frozen_autotune_space": "PASS",
            "day1_performance": "NOT_STARTED"
        },
        "counts": {
            "candidate_trials": len(rows),
            "fallback_cases": fallback["cases"],
            "exact_output_trials": exact_trials,
            "maximum_absolute_error": max_abs
        },
        "next_authorized_action": result["next_authorized_action"],
        "status": "PASS"
    }
    receipt_path = args.receipt.resolve()
    atomic_write_json(receipt_path, receipt)
    if sha256(protocol_path) != protocol_sha_start:
        raise AssertionError("K3 protocol changed through receipt write")
    if sha256(CANDIDATE) != candidate_sha_end:
        raise AssertionError("K3 candidate changed through receipt write")
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
        receipt = execute(args)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        failure_path = ROOT / (
            "results/operator/failures/"
            f"k3_gqa_correctness_failure_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}.json"
        )
        failure = {
            "schema_version": 1,
            "stage": "OP5_K3_GQA_DECODE_ATTENTION_CORRECTNESS",
            "started_at": started,
            "failed_at": now_iso(),
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path) if protocol_path.is_file() else None,
            "locked_candidate": str(CANDIDATE),
            "locked_candidate_sha256": sha256(CANDIDATE),
            "candidate_lock_receipt": str(LOCK_RECEIPT),
            "candidate_lock_receipt_sha256": sha256(LOCK_RECEIPT),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "command": sys.argv,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "git_sha": git_sha(),
            "environment": environment_identity(),
            "formal_candidate_performance_started": False,
            "candidate_attempt_disposition": "CLOSE_K3_IF_LOCKED_CANDIDATE_CORRECTNESS_FAILED; REPAIR_AND_RERUN_ONLY_IF_PROVEN_HARNESS_ONLY",
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
