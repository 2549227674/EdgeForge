#!/usr/bin/env python3
"""K1 Triton implementation of the Transformers 5.15 Gemma4 RMSNorm.

The formal carrier uses BF16 inputs/outputs and FP32 reduction.  Gemma4 has
two real variants: most modules multiply by a learned *direct* scale, while
``v_norm`` has no scale at all.  There is deliberately no ``1 + weight``
path here; OP1 proved that premise from the execution-card draft incorrect.
"""

from __future__ import annotations

from typing import Callable

import torch
import triton
import triton.language as tl


SCALED_SEMANTICS = (
    "y = cast_input_dtype(float32(x) * pow(mean(float32(x)^2) + eps, -0.5) "
    "* float32(weight)); direct scale, no implicit +1"
)
SCALE_FREE_SEMANTICS = (
    "y = cast_input_dtype(float32(x) * pow(mean(float32(x)^2) + eps, -0.5)); "
    "no learned scale"
)


@triton.jit
def k1_gemma4_rmsnorm_kernel(
    input_ptr: tl.tensor,
    weight_ptr: tl.tensor,
    output_ptr: tl.tensor,
    n_cols: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WITH_SCALE: tl.constexpr,
) -> None:
    """Normalize one contiguous logical row per Triton program."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    offsets = row * n_cols + columns

    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / n_cols
    inverse_rms = tl.rsqrt(mean_square + epsilon)
    normalized = values * inverse_rms
    if WITH_SCALE:
        scale = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        normalized *= scale
    tl.store(output_ptr + offsets, normalized, mask=mask)


def reference_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    """Explicit FP32 semantic oracle, before the carrier's final dtype cast."""
    values = hidden_states.float()
    mean_square = values.pow(2).mean(-1, keepdim=True) + epsilon
    normalized = values * torch.pow(mean_square, -0.5)
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized


def eager_scaled(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Exact scaled Gemma4RMSNorm formula used by Transformers 5.15."""
    return reference_fp32(hidden_states, weight, epsilon).type_as(hidden_states)


def eager_scale_free(hidden_states: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Exact scale-free Gemma4RMSNorm formula used by Gemma4 v_norm."""
    return reference_fp32(hidden_states, None, epsilon).type_as(hidden_states)


def make_compiled_baselines() -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
    """Compile both semantic variants before any timed region."""
    compiled_scaled = torch.compile(
        eager_scaled,
        fullgraph=True,
        mode="reduce-overhead",
        dynamic=True,
    )
    compiled_scale_free = torch.compile(
        eager_scale_free,
        fullgraph=True,
        mode="reduce-overhead",
        dynamic=True,
    )
    return compiled_scaled, compiled_scale_free


def candidate_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    """Launch K1 for a contiguous tensor whose normalized axis is last."""
    if not hidden_states.is_cuda:
        raise ValueError("K1 requires a CUDA input")
    if hidden_states.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported K1 dtype: {hidden_states.dtype}")
    if not hidden_states.is_contiguous() or hidden_states.stride(-1) != 1:
        raise ValueError("K1 requires contiguous storage and last-axis stride 1")
    if hidden_states.ndim < 1 or hidden_states.shape[-1] <= 0:
        raise ValueError("K1 requires a non-empty normalized axis")

    n_cols = int(hidden_states.shape[-1])
    n_rows = hidden_states.numel() // n_cols
    if n_rows <= 0:
        raise ValueError("K1 requires at least one logical row")
    if weight is not None:
        if not weight.is_cuda or weight.device != hidden_states.device:
            raise ValueError("K1 scale must be on the input CUDA device")
        if weight.dtype != hidden_states.dtype:
            raise TypeError("K1 scale dtype must match the input dtype")
        if not weight.is_contiguous() or tuple(weight.shape) != (n_cols,):
            raise ValueError("K1 scale must be a contiguous vector of normalized width")

    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError(f"K1 normalized width exceeds the supported block: {n_cols}")
    num_warps = 4 if block_size <= 1024 else 8
    output = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    # Triton needs a pointer in the signature even when WITH_SCALE specializes
    # the load away. Reusing input_ptr is safe because it is never dereferenced
    # as a scale on the scale-free specialization.
    weight_or_dummy = weight if weight is not None else hidden_states
    k1_gemma4_rmsnorm_kernel[(n_rows,)](
        hidden_states,
        weight_or_dummy,
        output,
        n_cols=n_cols,
        epsilon=float(epsilon),
        BLOCK_SIZE=block_size,
        WITH_SCALE=weight is not None,
        num_warps=num_warps,
    )
    return output
