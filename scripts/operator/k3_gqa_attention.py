#!/usr/bin/env python3
"""Locked-design K3 GQA decode attention candidate for the local sm_89 carrier.

The candidate implements the frozen output-only Gemma4 eager semantics without
materializing the 2-to-8 KV-head repeat.  It deliberately keeps the three
pre-registered stages separate: QK, FP32 softmax with a BF16 probability
boundary, and PV.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward


K3_OP_SEMANTICS = (
    "decode M=1; BF16 Q[1,8,1,D], K/V[1,2,K,D], mask[1,1,1,K]; "
    "GQA group size 4; scale 1; FP32 softmax rounded to BF16 before PV; "
    "contiguous BF16 output[1,1,8,D]"
)


_SCORE_CONFIGS = [
    triton.Config({"BLOCK_K": 8, "BLOCK_D": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_K": 16, "BLOCK_D": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_K": 16, "BLOCK_D": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_K": 32, "BLOCK_D": 64}, num_warps=8, num_stages=3),
]

_SOFTMAX_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
]

_PV_CONFIGS = [
    triton.Config({"BLOCK_D": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_D": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_D": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_D": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_SCORE_CONFIGS, key=["K", "D"])
@triton.jit
def _gqa_score_kernel(
    query_ptr: tl.tensor,
    key_ptr: tl.tensor,
    mask_ptr: tl.tensor,
    score_ptr: tl.tensor,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    query_head = tl.program_id(0)
    key_block = tl.program_id(1)
    key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    key_mask = key_offsets < K
    kv_head = query_head // 4
    accumulator = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for dimension_start in range(0, D, BLOCK_D):
        dimension_offsets = dimension_start + tl.arange(0, BLOCK_D)
        dimension_mask = dimension_offsets < D
        query = tl.load(
            query_ptr + query_head * D + dimension_offsets,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)
        key = tl.load(
            key_ptr
            + (kv_head * K + key_offsets[:, None]) * D
            + dimension_offsets[None, :],
            mask=key_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(key * query[None, :], axis=1)

    # torch.matmul(BF16, BF16) returns BF16 in the frozen eager path, and the
    # subsequent BF16 mask addition has another BF16 carrier boundary.  Keep
    # those two roundings even though the workspace itself is FP32.
    qk_bf16 = accumulator.to(tl.bfloat16)
    additive_mask = tl.load(
        mask_ptr + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(tl.bfloat16)
    masked_bf16 = (qk_bf16 + additive_mask).to(tl.bfloat16)
    tl.store(
        score_ptr + query_head * K + key_offsets,
        masked_bf16.to(tl.float32),
        mask=key_mask,
    )


@triton.autotune(configs=_SOFTMAX_CONFIGS, key=["K"])
@triton.jit
def _gqa_softmax_kernel(
    score_ptr: tl.tensor,
    probability_ptr: tl.tensor,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    query_head = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    valid = offsets < K
    logits = tl.load(
        score_ptr + query_head * K + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    logits -= tl.max(logits, axis=0)
    numerator = tl.exp(logits)
    denominator = tl.sum(numerator, axis=0)
    probability = (numerator / denominator).to(tl.bfloat16)
    tl.store(
        probability_ptr + query_head * K + offsets,
        probability,
        mask=valid,
    )


@triton.autotune(configs=_PV_CONFIGS, key=["K", "D"])
@triton.jit
def _gqa_pv_kernel(
    probability_ptr: tl.tensor,
    value_ptr: tl.tensor,
    output_ptr: tl.tensor,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    query_head = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    dimension_mask = dimension_offsets < D
    kv_head = query_head // 4
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for key_start in range(0, K, BLOCK_K):
        key_offsets = key_start + tl.arange(0, BLOCK_K)
        key_mask = key_offsets < K
        probability = tl.load(
            probability_ptr + query_head * K + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            value_ptr
            + (kv_head * K + key_offsets[:, None]) * D
            + dimension_offsets[None, :],
            mask=key_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(probability[:, None] * value, axis=0)

    tl.store(
        output_ptr + query_head * D + dimension_offsets,
        accumulator,
        mask=dimension_mask,
    )


def _validate_candidate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[int, int]:
    if not torch.cuda.is_available() or query.device.type != "cuda":
        raise ValueError("K3 requires CUDA inputs")
    if torch.cuda.get_device_capability(query.device) != (8, 9):
        raise ValueError("K3 is restricted to sm_89")
    if torch.is_grad_enabled():
        raise ValueError("K3 is restricted to inference/no-grad execution")
    if query.ndim != 4 or tuple(query.shape[:3]) != (1, 8, 1):
        raise ValueError("K3 query must have shape [1,8,1,D]")
    head_dim = int(query.shape[3])
    if key.ndim != 4 or value.ndim != 4 or tuple(key.shape[:2]) != (1, 2):
        raise ValueError("K3 key/value must have shape [1,2,K,D]")
    if tuple(value.shape) != tuple(key.shape) or int(key.shape[3]) != head_dim:
        raise ValueError("K3 key/value geometry does not match the query")
    key_tokens = int(key.shape[2])
    if not (
        (head_dim == 256 and key_tokens == 512)
        or (head_dim == 512 and 807 <= key_tokens <= 7359)
    ):
        raise ValueError("K3 geometry is outside the frozen decode population")
    expected_mask_shape = (1, 1, 1, key_tokens)
    if tuple(attention_mask.shape) != expected_mask_shape:
        raise ValueError("K3 mask must have shape [1,1,1,K]")
    tensors = (query, key, value, attention_mask)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("K3 tensors must share one CUDA device")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("K3 tensors must all be BF16")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("K3 tensors must be contiguous")
    return key_tokens, head_dim


def candidate_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen three-stage K3 candidate and return output only."""
    key_tokens, head_dim = _validate_candidate_inputs(
        query, key, value, attention_mask
    )
    scores = torch.empty(
        (8, key_tokens), dtype=torch.float32, device=query.device
    )
    probabilities = torch.empty(
        (8, key_tokens), dtype=torch.bfloat16, device=query.device
    )
    output = torch.empty(
        (1, 1, 8, head_dim), dtype=torch.bfloat16, device=query.device
    )

    score_grid = lambda meta: (8, triton.cdiv(key_tokens, meta["BLOCK_K"]))
    _gqa_score_kernel[score_grid](
        query,
        key,
        attention_mask,
        scores,
        K=key_tokens,
        D=head_dim,
    )
    _gqa_softmax_kernel[(8,)](
        scores,
        probabilities,
        K=key_tokens,
        BLOCK_SIZE=triton.next_power_of_2(key_tokens),
    )
    pv_grid = lambda meta: (8, triton.cdiv(head_dim, meta["BLOCK_D"]))
    _gqa_pv_kernel[pv_grid](
        probabilities,
        value,
        output,
        K=key_tokens,
        D=head_dim,
    )
    return output


def eligible_gqa_decode(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dropout: float | int,
    scaling: float | None,
    softcap: float | None,
    output_attentions: bool,
) -> bool:
    """Return whether a Gemma4 attention call exactly matches K3's contract."""
    if attention_mask is None:
        return False
    try:
        _validate_candidate_inputs(query, key, value, attention_mask)
    except (TypeError, ValueError):
        return False
    return bool(
        not torch.is_grad_enabled()
        and not bool(getattr(module, "training", True))
        and int(getattr(module, "num_key_value_groups", -1)) == 4
        and int(getattr(module, "head_dim", -1)) == int(query.shape[-1])
        and float(dropout) == 0.0
        and scaling is not None
        and float(scaling) == 1.0
        and softcap is None
        and output_attentions is False
    )


def candidate_or_eager_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    output_attentions: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch K3 only on its exact contract; otherwise call eager unchanged."""
    if eligible_gqa_decode(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        softcap=softcap,
        output_attentions=output_attentions,
    ):
        assert attention_mask is not None
        return candidate_gqa_attention(query, key, value, attention_mask), None
    return eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        softcap=softcap,
        output_attentions=output_attentions,
        **kwargs,
    )


def _selected_config(kernel: Any) -> list[dict[str, Any]]:
    cache = getattr(kernel, "cache", {})
    rows: list[dict[str, Any]] = []
    for key, config in cache.items():
        kwargs = dict(getattr(config, "kwargs", {}))
        rows.append(
            {
                "key": list(key) if isinstance(key, tuple) else [key],
                **kwargs,
                "num_warps": getattr(config, "num_warps", None),
                "num_stages": getattr(config, "num_stages", None),
            }
        )
    return sorted(rows, key=lambda row: tuple(str(value) for value in row["key"]))


def autotune_choices() -> dict[str, list[dict[str, Any]]]:
    """Return the selected frozen Triton configurations for audit output."""
    return {
        "score": _selected_config(_gqa_score_kernel),
        "softmax": _selected_config(_gqa_softmax_kernel),
        "pv": _selected_config(_gqa_pv_kernel),
    }
