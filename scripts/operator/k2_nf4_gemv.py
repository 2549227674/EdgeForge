#!/usr/bin/env python3
"""Triton K2 fused nested-NF4 decode GEMV for the frozen sm_89 contract."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


K2_OP_SEMANTICS = (
    "M=1 BF16 activation; row-major nested NF4 weight; blocksize=64; "
    "second-level blocksize=256; high nibble first; fp32 accumulation; BF16 output"
)


@triton.jit
def _decode_nf4(index: tl.tensor) -> tl.tensor:
    bit3 = (index & 0b1000) != 0
    bit2 = (index & 0b0100) != 0
    bit1 = (index & 0b0010) != 0
    bit0 = (index & 0b0001) != 0

    positive = tl.where(
        bit2,
        tl.where(
            bit1,
            tl.where(bit0, 1.0, 0.7229568362236023),
            tl.where(bit0, 0.5626170039176941, 0.44070982933044434),
        ),
        tl.where(
            bit1,
            tl.where(bit0, 0.33791524171829224, 0.24611230194568634),
            tl.where(bit0, 0.16093020141124725, 0.07958029955625534),
        ),
    )
    negative = tl.where(
        bit2,
        tl.where(
            bit1,
            tl.where(bit0, 0.0, -0.09105003625154495),
            tl.where(bit0, -0.18477343022823334, -0.28444138169288635),
        ),
        tl.where(
            bit1,
            tl.where(bit0, -0.39491748809814453, -0.5250730514526367),
            tl.where(bit0, -0.6961928009986877, -1.0),
        ),
    )
    return tl.where(bit3, positive, negative)


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 4, "BLOCK_K": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_K": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 16, "BLOCK_K": 512}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["K", "N"])
@triton.jit
def _nested_nf4_gemv_kernel(
    activation_ptr: tl.tensor,
    packed_weight_ptr: tl.tensor,
    first_absmax_ptr: tl.tensor,
    second_absmax_ptr: tl.tensor,
    second_code_ptr: tl.tensor,
    scale_offset_ptr: tl.tensor,
    output_ptr: tl.tensor,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    output_rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = output_rows < N
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    scale_offset = tl.load(scale_offset_ptr).to(tl.float32)

    for k_start in range(0, K, BLOCK_K):
        reduction_columns = k_start + tl.arange(0, BLOCK_K)
        column_mask = reduction_columns < K
        flat_offsets = output_rows[:, None] * K + reduction_columns[None, :]
        weight_mask = row_mask[:, None] & column_mask[None, :]

        packed = tl.load(
            packed_weight_ptr + flat_offsets // 2,
            mask=weight_mask,
            other=0,
        ).to(tl.uint8)
        high_nibble = packed >> 4
        low_nibble = packed & 0xF
        nibble = tl.where((flat_offsets & 1) == 0, high_nibble, low_nibble)

        scale_indices = flat_offsets // 64
        compressed_scale = tl.load(
            first_absmax_ptr + scale_indices,
            mask=weight_mask,
            other=0,
        ).to(tl.int32)
        second_code = tl.load(
            second_code_ptr + compressed_scale,
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)
        second_absmax = tl.load(
            second_absmax_ptr + scale_indices // 256,
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)
        scale = second_code * second_absmax + scale_offset
        weight = _decode_nf4(nibble) * scale

        activation = tl.load(
            activation_ptr + reduction_columns,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(weight * activation[None, :], axis=1)

    tl.store(output_ptr + output_rows, accumulator, mask=row_mask)


def candidate_nf4_gemv(
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    first_absmax: torch.Tensor,
    second_absmax: torch.Tensor,
    second_code: torch.Tensor,
    scale_offset: torch.Tensor,
    shape_n_k: tuple[int, int] | list[int],
) -> torch.Tensor:
    """Run the frozen K2 candidate for one M=1 nested-NF4 projection."""
    n_dim, k_dim = (int(value) for value in shape_n_k)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(activation.device) != (8, 9):
        raise ValueError("K2 candidate is restricted to sm_89")
    if activation.device.type != "cuda" or activation.dtype != torch.bfloat16:
        raise ValueError("K2 candidate requires a CUDA BF16 activation")
    if not activation.is_contiguous() or activation.numel() != k_dim or activation.shape[-1] != k_dim:
        raise ValueError("K2 candidate requires one contiguous M=1 activation row")
    if packed_weight.device != activation.device or packed_weight.dtype != torch.uint8:
        raise ValueError("packed NF4 weight must be uint8 on the activation device")
    if packed_weight.numel() != (n_dim * k_dim + 1) // 2:
        raise ValueError("packed NF4 weight size does not match [N,K]")
    expected_scales = triton.cdiv(n_dim * k_dim, 64)
    if (
        first_absmax.device != activation.device
        or first_absmax.dtype != torch.uint8
        or first_absmax.numel() != expected_scales
    ):
        raise ValueError("first-level nested scale indices do not match the contract")
    if (
        second_absmax.device != activation.device
        or second_absmax.dtype != torch.float32
        or second_absmax.numel() != triton.cdiv(expected_scales, 256)
    ):
        raise ValueError("second-level absmax tensor does not match the contract")
    if (
        second_code.device != activation.device
        or second_code.dtype != torch.float32
        or second_code.numel() != 256
    ):
        raise ValueError("second-level codebook does not match the contract")
    if (
        scale_offset.device != activation.device
        or scale_offset.dtype != torch.float32
        or scale_offset.numel() != 1
    ):
        raise ValueError("nested scale offset does not match the contract")
    if k_dim % 64 != 0:
        raise ValueError("K must be divisible by the frozen quantization blocksize")

    output = torch.empty((*activation.shape[:-1], n_dim), dtype=torch.bfloat16, device=activation.device)
    grid = lambda meta: (triton.cdiv(n_dim, meta["BLOCK_N"]),)
    _nested_nf4_gemv_kernel[grid](
        activation,
        packed_weight.view(-1),
        first_absmax.view(-1),
        second_absmax.view(-1),
        second_code.view(-1),
        scale_offset.reshape(()),
        output,
        K=k_dim,
        N=n_dim,
    )
    return output


def eligible_linear4bit_decode(module: Any, activation: torch.Tensor) -> bool:
    weight = getattr(module, "weight", None)
    packed_weight = getattr(weight, "data", None)
    state = getattr(weight, "quant_state", None)
    state2 = getattr(state, "state2", None)
    in_features = int(getattr(module, "in_features", -1))
    out_features = int(getattr(module, "out_features", -1))
    activation_shape_ok = bool(
        activation.ndim in (2, 3)
        and activation.shape[-1] == in_features
        and activation.numel() == in_features
    )
    expected_weight_elements = out_features * in_features
    expected_first_scales = (expected_weight_elements + 63) // 64
    expected_second_scales = (expected_first_scales + 255) // 256

    def exact_tensor(
        value: Any,
        dtype: torch.dtype,
        numel: int,
    ) -> bool:
        return bool(
            isinstance(value, torch.Tensor)
            and value.device == activation.device
            and value.dtype == dtype
            and value.numel() == numel
        )

    return bool(
        torch.cuda.is_available()
        and activation.device.type == "cuda"
        and torch.cuda.get_device_capability(activation.device) == (8, 9)
        and activation.dtype == torch.bfloat16
        and activation.is_contiguous()
        and activation_shape_ok
        and not torch.is_grad_enabled()
        and in_features > 0
        and out_features > 0
        and getattr(module, "bias", None) is None
        and state is not None
        and state2 is not None
        and bool(getattr(state, "nested", False))
        and getattr(state, "quant_type", None) == "nf4"
        and int(getattr(state, "blocksize", -1)) == 64
        and int(getattr(state2, "blocksize", -1)) == 256
        and list(getattr(state, "shape", []))
        == [out_features, in_features]
        and exact_tensor(
            packed_weight,
            torch.uint8,
            (expected_weight_elements + 1) // 2,
        )
        and exact_tensor(
            getattr(state, "absmax", None),
            torch.uint8,
            expected_first_scales,
        )
        and exact_tensor(
            getattr(state2, "absmax", None),
            torch.float32,
            expected_second_scales,
        )
        and exact_tensor(getattr(state2, "code", None), torch.float32, 256)
        and exact_tensor(getattr(state, "offset", None), torch.float32, 1)
    )


def candidate_or_linear4bit_fallback(module: Any, activation: torch.Tensor) -> torch.Tensor:
    """Use K2 only for the frozen decode contract; otherwise call the module unchanged."""
    if not eligible_linear4bit_decode(module, activation):
        return module(activation)
    state = module.weight.quant_state
    return candidate_nf4_gemv(
        activation,
        module.weight.data,
        state.absmax,
        state.state2.absmax,
        state.state2.code,
        state.offset,
        tuple(state.shape),
    )


def autotune_choices() -> list[dict[str, Any]]:
    """Return Triton's selected frozen config per populated (K,N) key."""
    cache = getattr(_nested_nf4_gemv_kernel, "cache", {})
    rows: list[dict[str, Any]] = []
    for key, config in cache.items():
        kwargs = dict(getattr(config, "kwargs", {}))
        rows.append(
            {
                "key": list(key) if isinstance(key, tuple) else [key],
                "BLOCK_N": kwargs.get("BLOCK_N"),
                "BLOCK_K": kwargs.get("BLOCK_K"),
                "num_warps": getattr(config, "num_warps", None),
                "num_stages": getattr(config, "num_stages", None),
            }
        )
    return sorted(rows, key=lambda row: tuple(str(value) for value in row["key"]))
