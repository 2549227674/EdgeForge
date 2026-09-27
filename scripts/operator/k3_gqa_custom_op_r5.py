#!/usr/bin/env python3
"""R5 integration adapter for the locked K3 candidate.

R5 keeps the R4 torch.library op and bounded-length adapter, but normalizes the
real Gemma 4 decode convention ``attention_mask=None`` to the mathematically
equivalent explicit BF16 zero mask required by the locked candidate.
"""

from __future__ import annotations

from typing import Any

import torch

import k3_gqa_custom_op as r4
from k3_gqa_attention import eager_attention_forward, eligible_gqa_decode


ATTENTION_REGISTRY_KEY = "edgeforge_k3_gqa_r5"
FULL_UPPER_BUCKETS = r4.FULL_UPPER_BUCKETS
OP_SCHEMA = r4.OP_SCHEMA
QUALIFIED_OP_NAME = r4.QUALIFIED_OP_NAME
prewarm_candidate = r4.prewarm_candidate


class R5InvocationTracker(r4.K3InvocationTracker):
    def begin_run(self, *, request_id: str, tape_id: str) -> None:
        super().begin_run(request_id=request_id, tape_id=tape_id)
        assert self._active is not None
        self._active["synthetic_zero_mask_calls"] = 0

    def note_synthetic_zero_mask(self) -> None:
        if self._active is not None:
            self._active["synthetic_zero_mask_calls"] += 1

    def _serialize(self, active: dict[str, Any]) -> dict[str, Any]:
        result = super()._serialize(active)
        result["synthetic_zero_mask_calls"] = active[
            "synthetic_zero_mask_calls"
        ]
        return result


INVOCATION_TRACKER = R5InvocationTracker()
# The already-registered torch.library CUDA implementation closes over the R4
# module global. Point that global at the R5 tracker so actual op dispatches and
# interface decisions reconcile in one counter set.
r4.INVOCATION_TRACKER = INVOCATION_TRACKER


def _decode_shaped(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> bool:
    return bool(
        query.ndim == 4
        and tuple(query.shape[:3]) == (1, 8, 1)
        and key.ndim == 4
        and value.ndim == 4
        and tuple(key.shape[:2]) == (1, 2)
        and tuple(value.shape) == tuple(key.shape)
        and int(key.shape[-1]) == int(query.shape[-1])
    )


def k3_attention_interface_r5(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    normalized_mask = attention_mask
    synthesized_zero_mask = False
    if attention_mask is None and _decode_shaped(query, key, value):
        normalized_mask = torch.zeros(
            (1, 1, 1, int(key.shape[2])),
            dtype=query.dtype,
            device=query.device,
        )
        synthesized_zero_mask = True

    output_attentions = bool(kwargs.get("output_attentions", False))
    eligible = eligible_gqa_decode(
        module,
        query,
        key,
        value,
        normalized_mask,
        dropout=dropout,
        scaling=scaling,
        softcap=softcap,
        output_attentions=output_attentions,
    )
    if not eligible:
        phase = str(getattr(INVOCATION_TRACKER._local, "phase", "unassigned"))
        reason = "prefill_policy" if phase == "prefill" else "dispatch_contract_miss"
        INVOCATION_TRACKER.record_decision(
            module=module,
            candidate=False,
            fallback_reason=reason,
            original_k=int(key.shape[2]),
            dispatched_k=int(key.shape[2]),
        )
        return eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            softcap=softcap,
            **kwargs,
        )

    assert normalized_mask is not None
    original_tensors = (query, key, value) + (
        (attention_mask,) if attention_mask is not None else ()
    )
    original_descriptors = tuple(
        r4._tensor_descriptor(tensor) for tensor in original_tensors
    )
    normalized_mask_descriptor = r4._tensor_descriptor(normalized_mask)
    prepared = r4.prepare_bounded_inputs(
        query, key, value, normalized_mask
    )
    INVOCATION_TRACKER.record_decision(
        module=module,
        candidate=True,
        fallback_reason=None,
        original_k=prepared.original_k,
        dispatched_k=prepared.dispatched_k,
    )
    if synthesized_zero_mask:
        INVOCATION_TRACKER.note_synthetic_zero_mask()
    INVOCATION_TRACKER.validate_padding_once(prepared)
    prepared_descriptors = tuple(
        r4._tensor_descriptor(tensor)
        for tensor in (prepared.query, prepared.key, prepared.value, prepared.mask)
    )
    output = r4.call_k3_custom_op(prepared)

    if original_descriptors != tuple(
        r4._tensor_descriptor(tensor) for tensor in original_tensors
    ):
        INVOCATION_TRACKER.record_input_violation(
            f"original input structure changed in {INVOCATION_TRACKER.module_name(module)}"
        )
    if normalized_mask_descriptor != r4._tensor_descriptor(normalized_mask):
        INVOCATION_TRACKER.record_input_violation(
            f"normalized mask structure changed in {INVOCATION_TRACKER.module_name(module)}"
        )
    if prepared_descriptors != tuple(
        r4._tensor_descriptor(tensor)
        for tensor in (prepared.query, prepared.key, prepared.value, prepared.mask)
    ):
        INVOCATION_TRACKER.record_input_violation(
            f"custom-op input structure changed in {INVOCATION_TRACKER.module_name(module)}"
        )
    expected_output = (
        (1, 1, 8, int(query.shape[-1])),
        torch.bfloat16,
        query.device,
        True,
    )
    observed_output = (
        tuple(output.shape),
        output.dtype,
        output.device,
        bool(output.is_contiguous()),
    )
    if observed_output != expected_output:
        INVOCATION_TRACKER.record_output_violation(
            f"unexpected output contract in {INVOCATION_TRACKER.module_name(module)}: "
            f"{observed_output}"
        )
    return output, None


class K3Reconnector(r4.K3Reconnector):
    def __init__(self, model: Any, *, expected_modules: int = 42) -> None:
        super().__init__(model, expected_modules=expected_modules)
        INVOCATION_TRACKER.configure_modules(self._modules)

    def activate(self) -> dict[str, Any]:
        if self._active:
            raise RuntimeError("K3 R5 reconnection is already active")
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self._registry_had_key = ATTENTION_REGISTRY_KEY in ALL_ATTENTION_FUNCTIONS
        self._previous_registry_value = ALL_ATTENTION_FUNCTIONS.get(
            ATTENTION_REGISTRY_KEY
        )
        ALL_ATTENTION_FUNCTIONS.register(
            ATTENTION_REGISTRY_KEY, k3_attention_interface_r5
        )
        for config in self._configs.values():
            config._attn_implementation = ATTENTION_REGISTRY_KEY
        self._active = True
        return {
            "status": "ACTIVE",
            "patched_config_objects": len(self._configs),
            "attention_modules": len(self._modules),
            "registry_key": ATTENTION_REGISTRY_KEY,
            "torch_op": QUALIFIED_OP_NAME,
            "none_mask_normalization": "ENABLED_FOR_DECODE_SHAPES",
        }

    def restore(self) -> dict[str, Any]:
        if not self._active:
            return {"status": "ALREADY_INACTIVE", "restored_config_objects": 0}
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        for config_id, config in self._configs.items():
            config._attn_implementation = self._original_implementations[config_id]
        if self._registry_had_key:
            ALL_ATTENTION_FUNCTIONS.register(
                ATTENTION_REGISTRY_KEY, self._previous_registry_value
            )
        else:
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(
                ATTENTION_REGISTRY_KEY, None
            )
        self._active = False
        return {
            "status": "RESTORED",
            "restored_config_objects": len(self._configs),
            "registry_key_removed": not self._registry_had_key,
        }

    def assert_restored(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if self._active:
            raise AssertionError("K3 R5 reconnector remains active")
        for config_id, config in self._configs.items():
            if config._attn_implementation != self._original_implementations[config_id]:
                raise AssertionError("attention implementation was not restored")
        if (
            not self._registry_had_key
            and ATTENTION_REGISTRY_KEY in ALL_ATTENTION_FUNCTIONS
        ):
            raise AssertionError("temporary K3 R5 registry key remains installed")
