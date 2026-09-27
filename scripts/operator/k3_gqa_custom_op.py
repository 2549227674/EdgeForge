#!/usr/bin/env python3
"""K3 torch.library bridge, bounded-length adapter, and reversible dispatch.

The locked K3 kernels stay in ``k3_gqa_attention.py``.  This module only
connects them to the Gemma 4 eager attention interface for the exact decode
contract, counts every formal dispatch, and restores the original attention
configuration after replay.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Any, Iterator

import torch

from k3_gqa_attention import (
    candidate_gqa_attention,
    eager_attention_forward,
    eligible_gqa_decode,
)


OP_NAMESPACE = "edgeforge"
OP_NAME = "k3_gqa_attention"
QUALIFIED_OP_NAME = f"{OP_NAMESPACE}::{OP_NAME}"
OP_SCHEMA = (
    "k3_gqa_attention(Tensor query, Tensor key, Tensor value, "
    "Tensor attention_mask) -> Tensor"
)
ATTENTION_REGISTRY_KEY = "edgeforge_k3_gqa"

FULL_UPPER_BUCKETS = (
    807,
    1264,
    1436,
    1569,
    1714,
    1843,
    1973,
    2153,
    2334,
    2514,
    2695,
    2897,
    3123,
    3349,
    3575,
    3801,
    4011,
    4191,
    4440,
    4728,
    5029,
    5331,
    5632,
    6455,
    7359,
)


def _tensor_descriptor(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        value.data_ptr(),
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
        bool(value.is_contiguous()),
    )


def _family(head_dim: int) -> str:
    if head_dim == 256:
        return "sliding"
    if head_dim == 512:
        return "full"
    return "other"


def _full_upper_bucket(key_tokens: int) -> int:
    for bucket in FULL_UPPER_BUCKETS:
        if key_tokens <= bucket:
            return bucket
    raise ValueError(f"full-attention K={key_tokens} exceeds the frozen maximum")


@dataclass
class PreparedInputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    mask: torch.Tensor
    original_k: int
    dispatched_k: int
    padded: bool


def prepare_bounded_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> PreparedInputs:
    """Pad full attention to the smallest frozen correctness-covered K."""
    original_k = int(key.shape[2])
    head_dim = int(query.shape[-1])
    dispatched_k = original_k if head_dim == 256 else _full_upper_bucket(original_k)
    if dispatched_k == original_k:
        return PreparedInputs(
            query=query,
            key=key,
            value=value,
            mask=attention_mask,
            original_k=original_k,
            dispatched_k=dispatched_k,
            padded=False,
        )

    padded_key = torch.zeros(
        (1, 2, dispatched_k, head_dim), dtype=key.dtype, device=key.device
    )
    padded_value = torch.zeros_like(padded_key)
    padded_mask = torch.full(
        (1, 1, 1, dispatched_k),
        torch.finfo(attention_mask.dtype).min,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    padded_key[:, :, :original_k, :].copy_(key)
    padded_value[:, :, :original_k, :].copy_(value)
    padded_mask[:, :, :, :original_k].copy_(attention_mask)
    return PreparedInputs(
        query=query,
        key=padded_key,
        value=padded_value,
        mask=padded_mask,
        original_k=original_k,
        dispatched_k=dispatched_k,
        padded=True,
    )


class K3InvocationTracker:
    """Count formal attention decisions without retaining tensor contents."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._module_names: dict[int, str] = {}
        self._active: dict[str, Any] | None = None
        self._completed: list[dict[str, Any]] = []
        self._validated_padding_buckets: set[int] = set()

    @property
    def active(self) -> bool:
        return self._active is not None

    def configure_modules(self, modules: list[tuple[str, Any]]) -> None:
        self._module_names = {id(module): name for name, module in modules}

    def begin_run(self, *, request_id: str, tape_id: str) -> None:
        if self._active is not None:
            raise RuntimeError("a K3 tracker run is already active")
        self._active = {
            "request_id": request_id,
            "tape_id": tape_id,
            "total_attention_calls": 0,
            "candidate_calls": 0,
            "eager_calls": 0,
            "torch_op_calls": 0,
            "phase_counts": Counter(),
            "candidate_phase_counts": Counter(),
            "eager_phase_counts": Counter(),
            "candidate_family_counts": Counter(),
            "candidate_module_counts": Counter(),
            "fallback_reason_counts": Counter(),
            "original_k_counts": Counter(),
            "dispatched_k_counts": Counter(),
            "bucket_pair_counts": Counter(),
            "padded_calls": 0,
            "input_structure_violations": [],
            "output_structure_violations": [],
            "padding_validation_failures": [],
        }
        self._local.phase = "unassigned"

    def set_phase(self, phase: str) -> None:
        self._local.phase = phase

    def module_name(self, module: Any) -> str:
        name = self._module_names.get(id(module))
        if name is None:
            raise RuntimeError("K3 dispatch reached an unregistered attention module")
        return name

    def record_decision(
        self,
        *,
        module: Any,
        candidate: bool,
        fallback_reason: str | None,
        original_k: int,
        dispatched_k: int,
    ) -> None:
        if self._active is None:
            return
        phase = str(getattr(self._local, "phase", "unassigned"))
        if phase not in ("prefill", "decode"):
            raise RuntimeError(f"formal K3 dispatch has invalid phase: {phase}")
        active = self._active
        active["total_attention_calls"] += 1
        active["phase_counts"][phase] += 1
        if candidate:
            name = self.module_name(module)
            family = _family(int(module.head_dim))
            active["candidate_calls"] += 1
            active["candidate_phase_counts"][phase] += 1
            active["candidate_family_counts"][family] += 1
            active["candidate_module_counts"][name] += 1
            active["original_k_counts"][str(original_k)] += 1
            active["dispatched_k_counts"][str(dispatched_k)] += 1
            active["bucket_pair_counts"][f"{original_k}->{dispatched_k}"] += 1
            if original_k != dispatched_k:
                active["padded_calls"] += 1
        else:
            active["eager_calls"] += 1
            active["eager_phase_counts"][phase] += 1
            active["fallback_reason_counts"][fallback_reason or "unknown"] += 1

    def note_torch_op(self) -> None:
        if self._active is not None:
            self._active["torch_op_calls"] += 1

    def record_input_violation(self, message: str) -> None:
        if self._active is not None:
            self._active["input_structure_violations"].append(message)

    def record_output_violation(self, message: str) -> None:
        if self._active is not None:
            self._active["output_structure_violations"].append(message)

    def validate_padding_once(self, prepared: PreparedInputs) -> None:
        if (
            self._active is None
            or not prepared.padded
            or prepared.dispatched_k in self._validated_padding_buckets
        ):
            return
        start = prepared.original_k
        failures: list[str] = []
        if int(torch.count_nonzero(prepared.key[:, :, start:, :]).item()) != 0:
            failures.append("padded key suffix is not zero")
        if int(torch.count_nonzero(prepared.value[:, :, start:, :]).item()) != 0:
            failures.append("padded value suffix is not zero")
        expected = torch.finfo(torch.bfloat16).min
        if not bool(torch.all(prepared.mask[:, :, :, start:] == expected).item()):
            failures.append("padded mask suffix is not BF16 minimum")
        if failures:
            self._active["padding_validation_failures"].extend(failures)
            raise AssertionError("; ".join(failures))
        self._validated_padding_buckets.add(prepared.dispatched_k)

    def _serialize(self, active: dict[str, Any]) -> dict[str, Any]:
        return {
            "request_id": active["request_id"],
            "tape_id": active["tape_id"],
            "total_attention_calls": active["total_attention_calls"],
            "candidate_calls": active["candidate_calls"],
            "eager_calls": active["eager_calls"],
            "torch_op_calls": active["torch_op_calls"],
            "phase_counts": dict(sorted(active["phase_counts"].items())),
            "candidate_phase_counts": dict(
                sorted(active["candidate_phase_counts"].items())
            ),
            "eager_phase_counts": dict(sorted(active["eager_phase_counts"].items())),
            "candidate_family_counts": dict(
                sorted(active["candidate_family_counts"].items())
            ),
            "candidate_module_counts": dict(
                sorted(active["candidate_module_counts"].items())
            ),
            "fallback_reason_counts": dict(
                sorted(active["fallback_reason_counts"].items())
            ),
            "original_k_counts": dict(sorted(active["original_k_counts"].items())),
            "dispatched_k_counts": dict(
                sorted(active["dispatched_k_counts"].items())
            ),
            "bucket_pair_counts": dict(sorted(active["bucket_pair_counts"].items())),
            "padded_calls": active["padded_calls"],
            "input_structure_violations": active["input_structure_violations"],
            "output_structure_violations": active["output_structure_violations"],
            "padding_validation_failures": active["padding_validation_failures"],
        }

    def end_run(self) -> dict[str, Any]:
        if self._active is None:
            raise RuntimeError("no active K3 tracker run")
        result = self._serialize(self._active)
        self._completed.append(result)
        self._active = None
        self._local.phase = "unassigned"
        return result

    def abort_run(self) -> dict[str, Any] | None:
        if self._active is None:
            return None
        result = self._serialize(self._active)
        result["aborted"] = True
        self._active = None
        self._local.phase = "unassigned"
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "completed_runs": len(self._completed),
            "padding_buckets_content_validated": sorted(
                self._validated_padding_buckets
            ),
        }


INVOCATION_TRACKER = K3InvocationTracker()


def _cuda_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    INVOCATION_TRACKER.note_torch_op()
    return candidate_gqa_attention(query, key, value, attention_mask)


def _fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    del key, value, attention_mask
    return torch.empty(
        (int(query.shape[0]), int(query.shape[2]), int(query.shape[1]), int(query.shape[3])),
        dtype=query.dtype,
        device=query.device,
    )


_LIBRARY_DEF = torch.library.Library(OP_NAMESPACE, "FRAGMENT")
_LIBRARY_DEF.define(OP_SCHEMA)
_LIBRARY_CUDA = torch.library.Library(OP_NAMESPACE, "IMPL", "CUDA")
_LIBRARY_CUDA.impl(OP_NAME, _cuda_impl)
_LIBRARY_META = torch.library.Library(OP_NAMESPACE, "IMPL", "Meta")
_LIBRARY_META.impl(OP_NAME, _fake_impl)


def call_k3_custom_op(prepared: PreparedInputs) -> torch.Tensor:
    return torch.ops.edgeforge.k3_gqa_attention.default(
        prepared.query, prepared.key, prepared.value, prepared.mask
    )


def k3_attention_interface(
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
    output_attentions = bool(kwargs.get("output_attentions", False))
    eligible = eligible_gqa_decode(
        module,
        query,
        key,
        value,
        attention_mask,
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

    assert attention_mask is not None
    original_descriptors = tuple(
        _tensor_descriptor(tensor)
        for tensor in (query, key, value, attention_mask)
    )
    prepared = prepare_bounded_inputs(query, key, value, attention_mask)
    INVOCATION_TRACKER.record_decision(
        module=module,
        candidate=True,
        fallback_reason=None,
        original_k=prepared.original_k,
        dispatched_k=prepared.dispatched_k,
    )
    INVOCATION_TRACKER.validate_padding_once(prepared)
    prepared_descriptors = tuple(
        _tensor_descriptor(tensor)
        for tensor in (prepared.query, prepared.key, prepared.value, prepared.mask)
    )
    output = call_k3_custom_op(prepared)
    if original_descriptors != tuple(
        _tensor_descriptor(tensor)
        for tensor in (query, key, value, attention_mask)
    ):
        INVOCATION_TRACKER.record_input_violation(
            f"original input structure changed in {INVOCATION_TRACKER.module_name(module)}"
        )
    if prepared_descriptors != tuple(
        _tensor_descriptor(tensor)
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


class K3Reconnector:
    """Temporarily route the language-model attention registry to K3."""

    def __init__(self, model: Any, *, expected_modules: int = 42) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        self.model = model
        self.expected_modules = expected_modules
        self._modules = [
            (name, module)
            for name, module in model.named_modules()
            if name.startswith("model.language_model.")
            and isinstance(module, Gemma4TextAttention)
        ]
        if len(self._modules) != expected_modules:
            raise AssertionError(
                f"discovered {len(self._modules)} text attention modules, "
                f"expected {expected_modules}"
            )
        self._configs = {id(module.config): module.config for _, module in self._modules}
        self._original_implementations = {
            config_id: config._attn_implementation
            for config_id, config in self._configs.items()
        }
        self._previous_registry_value: Any = None
        self._registry_had_key = False
        self._active = False
        INVOCATION_TRACKER.configure_modules(self._modules)

    @property
    def active(self) -> bool:
        return self._active

    def discovery_receipt(self) -> dict[str, Any]:
        return {
            "module_count": len(self._modules),
            "module_names": [name for name, _ in self._modules],
            "module_object_ids": {
                name: id(module) for name, module in self._modules
            },
            "config_object_count": len(self._configs),
            "attention_implementations": {
                str(config_id): value
                for config_id, value in self._original_implementations.items()
            },
        }

    def activate(self) -> dict[str, Any]:
        if self._active:
            raise RuntimeError("K3 reconnection is already active")
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self._registry_had_key = ATTENTION_REGISTRY_KEY in ALL_ATTENTION_FUNCTIONS
        self._previous_registry_value = ALL_ATTENTION_FUNCTIONS.get(
            ATTENTION_REGISTRY_KEY
        )
        ALL_ATTENTION_FUNCTIONS.register(
            ATTENTION_REGISTRY_KEY, k3_attention_interface
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
            ALL_ATTENTION_FUNCTIONS.pop(ATTENTION_REGISTRY_KEY, None)
        self._active = False
        return {
            "status": "RESTORED",
            "restored_config_objects": len(self._configs),
            "registry_key_removed": not self._registry_had_key,
        }

    def assert_restored(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if self._active:
            raise AssertionError("K3 reconnector remains active")
        for config_id, config in self._configs.items():
            if config._attn_implementation != self._original_implementations[config_id]:
                raise AssertionError("attention implementation was not restored")
        if not self._registry_had_key and ATTENTION_REGISTRY_KEY in ALL_ATTENTION_FUNCTIONS:
            raise AssertionError("temporary K3 attention registry key remains installed")


@contextmanager
def activated(reconnector: K3Reconnector) -> Iterator[dict[str, Any]]:
    event = reconnector.activate()
    try:
        yield event
    finally:
        reconnector.restore()
        reconnector.assert_restored()


def prewarm_candidate(device: int = 0) -> dict[str, Any]:
    """Compile/autotune the 26 bounded shapes before formal request timing."""
    import time

    cuda_device = torch.device(f"cuda:{device}")
    cases = [(512, 256), *((value, 512) for value in FULL_UPPER_BUCKETS)]
    timings: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for key_tokens, head_dim in cases:
            query = torch.zeros(
                (1, 8, 1, head_dim), dtype=torch.bfloat16, device=cuda_device
            )
            key = torch.zeros(
                (1, 2, key_tokens, head_dim),
                dtype=torch.bfloat16,
                device=cuda_device,
            )
            value = torch.zeros_like(key)
            mask = torch.zeros(
                (1, 1, 1, key_tokens),
                dtype=torch.bfloat16,
                device=cuda_device,
            )
            case_started = time.perf_counter()
            output = torch.ops.edgeforge.k3_gqa_attention.default(
                query, key, value, mask
            )
            torch.cuda.synchronize(device)
            timings.append(
                {
                    "K": key_tokens,
                    "D": head_dim,
                    "seconds": time.perf_counter() - case_started,
                    "output_contract": {
                        "shape": list(output.shape),
                        "dtype": str(output.dtype),
                        "contiguous": bool(output.is_contiguous()),
                    },
                }
            )
            del query, key, value, mask, output
    torch.cuda.empty_cache()
    return {
        "cases": timings,
        "case_count": len(timings),
        "total_seconds": time.perf_counter() - started,
        "tracker_active_during_prewarm": False,
    }
