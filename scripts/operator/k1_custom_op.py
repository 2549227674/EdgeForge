"""torch.library registration and reversible Gemma 4 RMSNorm reconnection.

The formal CUDA implementation delegates to the OP2 Triton candidate.  A CPU
implementation exists only to make the dispatch boundary explicit and useful
for non-formal unit tests; OP3 treats any CPU dispatch as a hard failure.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import threading
from types import MethodType
from typing import Any, Iterator

import torch

from k1_rmsnorm import candidate_rmsnorm, eager_scale_free, eager_scaled
from op0_common import summarize


OP_NAMESPACE = "edgeforge"
OP_NAME = "k1_gemma4_rmsnorm"
QUALIFIED_OP_NAME = f"{OP_NAMESPACE}::{OP_NAME}"
OP_SCHEMA = f"{OP_NAME}(Tensor hidden_states, Tensor? weight, float epsilon) -> Tensor"


def _validate_contract(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
) -> None:
    if hidden_states.ndim < 1 or hidden_states.shape[-1] <= 0:
        raise ValueError("K1 requires a non-empty normalized axis")
    if hidden_states.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported K1 dtype: {hidden_states.dtype}")
    if not hidden_states.is_contiguous() or hidden_states.stride(-1) != 1:
        raise ValueError("K1 requires contiguous storage and final-axis stride 1")
    if weight is not None:
        if weight.device != hidden_states.device:
            raise ValueError("K1 scale must be on the input device")
        if weight.dtype != hidden_states.dtype:
            raise TypeError("K1 scale dtype must match the input dtype")
        if not weight.is_contiguous() or tuple(weight.shape) != (
            int(hidden_states.shape[-1]),
        ):
            raise ValueError("K1 scale must be a contiguous normalized-width vector")


class K1InvocationTracker:
    """Bounded instrumentation for millions of custom-op invocations.

    Counts are retained for every formal call. CUDA events are retained only
    for a configured number of samples per runtime signature and only during
    warm pass 1, keeping pass-2 end-to-end timing free of event instrumentation.
    """

    def __init__(self, event_samples_per_signature: int = 32) -> None:
        if event_samples_per_signature < 0:
            raise ValueError("event sample cap cannot be negative")
        self.event_samples_per_signature = event_samples_per_signature
        self._local = threading.local()
        self._active_run: dict[str, Any] | None = None
        self._event_scheduled: Counter[str] = Counter()
        self._event_samples_us: dict[str, list[float]] = defaultdict(list)
        self._signature_metadata: dict[str, dict[str, Any]] = {}
        self._pending_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._completed_runs: list[dict[str, Any]] = []

    @property
    def active(self) -> bool:
        return self._active_run is not None

    def begin_run(
        self,
        *,
        request_id: str,
        tape_id: str,
        pass_number: int,
        sample_cuda_events: bool,
    ) -> None:
        if self._active_run is not None:
            raise RuntimeError("a K1 tracker run is already active")
        self._active_run = {
            "request_id": request_id,
            "tape_id": tape_id,
            "pass": int(pass_number),
            "sample_cuda_events": bool(sample_cuda_events),
            "total_invocations": 0,
            "backend_counts": Counter(),
            "phase_counts": Counter(),
            "scale_counts": Counter(),
            "signature_counts": Counter(),
            "module_counts": Counter(),
        }
        self._local.phase = "unassigned"
        self._local.module_name = None

    def abort_run(self) -> dict[str, Any] | None:
        if self._active_run is None:
            return None
        try:
            self.flush_pending_events(require_synchronized=True)
        except BaseException:
            # Preserve the original replay failure. Incomplete timing samples
            # are diagnostic-only warm-pass evidence and can be discarded.
            self._pending_events.clear()
        record = self._serialize_active_run()
        record["aborted"] = True
        self._active_run = None
        self._local.phase = "unassigned"
        self._local.module_name = None
        return record

    def end_run(self) -> dict[str, Any]:
        if self._active_run is None:
            raise RuntimeError("no active K1 tracker run")
        self.flush_pending_events(require_synchronized=False)
        if getattr(self._local, "module_name", None) is not None:
            raise RuntimeError("K1 module context leaked at end of run")
        record = self._serialize_active_run()
        self._completed_runs.append(record)
        self._active_run = None
        self._local.phase = "unassigned"
        return record

    def set_phase(self, phase: str) -> None:
        self._local.phase = phase

    @contextmanager
    def module_context(self, module_name: str) -> Iterator[None]:
        previous = getattr(self._local, "module_name", None)
        if previous is not None:
            raise RuntimeError(f"nested K1 module context: {previous} -> {module_name}")
        self._local.module_name = module_name
        try:
            yield
        finally:
            self._local.module_name = None

    def _signature(
        self, hidden_states: torch.Tensor, weight: torch.Tensor | None
    ) -> tuple[str, dict[str, Any]]:
        n_cols = int(hidden_states.shape[-1])
        n_rows = int(hidden_states.numel() // n_cols)
        phase = str(getattr(self._local, "phase", "unassigned"))
        with_scale = weight is not None
        dtype = str(hidden_states.dtype)
        key = (
            f"{phase}|rows={n_rows}|cols={n_cols}|dtype={dtype}|"
            f"with_scale={str(with_scale).lower()}"
        )
        analytic_bytes = 4 * int(hidden_states.numel())
        if with_scale:
            analytic_bytes += 2 * n_cols
        metadata = {
            "phase": phase,
            "flattened_rows": n_rows,
            "normalized_width": n_cols,
            "input_shape_examples": [list(hidden_states.shape)],
            "dtype": dtype,
            "with_scale": with_scale,
            "analytic_bytes_per_invocation": analytic_bytes,
        }
        return key, metadata

    def before_dispatch(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor | None,
        backend: str,
    ) -> tuple[str | None, torch.cuda.Event | None]:
        if self._active_run is None:
            return None, None
        module_name = getattr(self._local, "module_name", None)
        phase = str(getattr(self._local, "phase", "unassigned"))
        if module_name is None:
            raise RuntimeError("formal K1 dispatch has no RMSNorm module identity")
        if phase not in ("prefill", "decode"):
            raise RuntimeError(f"formal K1 dispatch has invalid phase: {phase}")
        key, metadata = self._signature(hidden_states, weight)
        existing = self._signature_metadata.setdefault(key, metadata)
        shape = list(hidden_states.shape)
        if shape not in existing["input_shape_examples"] and len(
            existing["input_shape_examples"]
        ) < 8:
            existing["input_shape_examples"].append(shape)

        active = self._active_run
        active["total_invocations"] += 1
        active["backend_counts"][backend] += 1
        active["phase_counts"][phase] += 1
        active["scale_counts"]["with_scale" if weight is not None else "scale_free"] += 1
        active["signature_counts"][key] += 1
        active["module_counts"][module_name] += 1

        should_sample = (
            backend == "cuda"
            and active["sample_cuda_events"]
            and self._event_scheduled[key] < self.event_samples_per_signature
        )
        if not should_sample:
            return key, None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._event_scheduled[key] += 1
        return key, start

    def after_dispatch(
        self,
        signature: str | None,
        start: torch.cuda.Event | None,
    ) -> None:
        if signature is None or start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending_events.append((signature, start, end))

    def flush_pending_events(self, *, require_synchronized: bool = True) -> None:
        if not self._pending_events:
            return
        if require_synchronized:
            torch.cuda.synchronize()
        pending, self._pending_events = self._pending_events, []
        for signature, start, end in pending:
            self._event_samples_us[signature].append(
                float(start.elapsed_time(end) * 1000.0)
            )

    def _serialize_active_run(self) -> dict[str, Any]:
        if self._active_run is None:
            raise RuntimeError("no active tracker record")
        active = self._active_run
        return {
            "request_id": active["request_id"],
            "tape_id": active["tape_id"],
            "pass": active["pass"],
            "sample_cuda_events": active["sample_cuda_events"],
            "total_invocations": active["total_invocations"],
            "backend_counts": dict(sorted(active["backend_counts"].items())),
            "phase_counts": dict(sorted(active["phase_counts"].items())),
            "scale_counts": dict(sorted(active["scale_counts"].items())),
            "signature_counts": dict(sorted(active["signature_counts"].items())),
            "module_counts": dict(sorted(active["module_counts"].items())),
        }

    def kernel_summary(self) -> dict[str, Any]:
        signatures: list[dict[str, Any]] = []
        for key, metadata in sorted(self._signature_metadata.items()):
            values = self._event_samples_us.get(key, [])
            record = {
                "signature": key,
                **metadata,
                "scheduled_samples": self._event_scheduled[key],
                "sample_count": len(values),
            }
            if values:
                distribution = summarize(values)
                p50_us = float(distribution["p50"])
                analytic_bytes = int(metadata["analytic_bytes_per_invocation"])
                record.update(
                    {
                        "p50_us": p50_us,
                        "iqr_us": distribution["iqr"],
                        "distribution_us": distribution,
                        "analytic_gbps": (
                            analytic_bytes / (p50_us / 1_000_000.0) / 1_000_000_000.0
                        ),
                    }
                )
            else:
                record.update({"p50_us": None, "iqr_us": None, "analytic_gbps": None})
            signatures.append(record)
        return {
            "event_samples_per_signature_cap": self.event_samples_per_signature,
            "timing_source": "torch.cuda.Event around the actual torch.library CUDA implementation dispatch in B pass 1",
            "pass_2_end_to_end_contains_cuda_events": False,
            "signatures": signatures,
            "completed_run_count": len(self._completed_runs),
        }


INVOCATION_TRACKER = K1InvocationTracker()


def _cuda_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    _validate_contract(hidden_states, weight)
    signature, start = INVOCATION_TRACKER.before_dispatch(
        hidden_states, weight, "cuda"
    )
    output = candidate_rmsnorm(hidden_states, weight, float(epsilon))
    INVOCATION_TRACKER.after_dispatch(signature, start)
    return output


def _cpu_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    _validate_contract(hidden_states, weight)
    signature, start = INVOCATION_TRACKER.before_dispatch(
        hidden_states, weight, "cpu_fallback"
    )
    if weight is None:
        output = eager_scale_free(hidden_states, float(epsilon))
    else:
        output = eager_scaled(hidden_states, weight, float(epsilon))
    INVOCATION_TRACKER.after_dispatch(signature, start)
    return output


def _fake_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    del epsilon
    _validate_contract(hidden_states, weight)
    return torch.empty_like(hidden_states, memory_format=torch.contiguous_format)


# Keep all Library objects alive for the process lifetime; registrations are
# removed when their Library owner is garbage-collected.
_LIBRARY_DEF = torch.library.Library(OP_NAMESPACE, "DEF")
_LIBRARY_DEF.define(OP_SCHEMA)
_LIBRARY_CUDA = torch.library.Library(OP_NAMESPACE, "IMPL", "CUDA")
_LIBRARY_CUDA.impl(OP_NAME, _cuda_impl)
_LIBRARY_CPU = torch.library.Library(OP_NAMESPACE, "IMPL", "CPU")
_LIBRARY_CPU.impl(OP_NAME, _cpu_impl)
_LIBRARY_META = torch.library.Library(OP_NAMESPACE, "IMPL", "Meta")
_LIBRARY_META.impl(OP_NAME, _fake_impl)


def call_k1_custom_op(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    return torch.ops.edgeforge.k1_gemma4_rmsnorm.default(
        hidden_states, weight, float(epsilon)
    )


class K1Reconnector:
    """Reversibly bind the K1 op to language-model Gemma4RMSNorm instances."""

    def __init__(
        self,
        model: Any,
        *,
        expected_modules: int = 302,
        expected_scaled: int = 278,
        expected_scale_free: int = 24,
    ) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

        self.model = model
        self.expected_modules = expected_modules
        self.expected_scaled = expected_scaled
        self.expected_scale_free = expected_scale_free
        self._modules: list[tuple[str, Any]] = []
        for name, module in model.named_modules():
            if not name.startswith("model.language_model."):
                continue
            if isinstance(module, Gemma4RMSNorm):
                self._modules.append((name, module))
        self._baseline_instance_forwards: dict[str, tuple[bool, Any]] = {
            name: ("forward" in module.__dict__, module.__dict__.get("forward"))
            for name, module in self._modules
        }
        self._original_instance_forwards: dict[str, tuple[bool, Any]] = {}
        self._active = False
        self._validate_discovery()

    def _validate_discovery(self) -> None:
        scaled = sum(bool(module.with_scale) for _, module in self._modules)
        scale_free = len(self._modules) - scaled
        if len(self._modules) != self.expected_modules:
            raise AssertionError(
                f"discovered {len(self._modules)} language RMSNorm modules, "
                f"expected {self.expected_modules}"
            )
        if scaled != self.expected_scaled or scale_free != self.expected_scale_free:
            raise AssertionError(
                "RMSNorm semantic topology mismatch: "
                f"scaled={scaled}/{self.expected_scaled}, "
                f"scale_free={scale_free}/{self.expected_scale_free}"
            )
        forbidden = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, type(self._modules[0][1]))
            and not name.startswith("model.language_model.")
        ]
        if forbidden:
            # Multimodal towers may have their own RMSNorm implementation, but
            # no Gemma4RMSNorm instance may be accidentally selected here.
            self.forbidden_same_class_modules = forbidden
        else:
            self.forbidden_same_class_modules = []
        for name, module in self._modules:
            has_weight = hasattr(module, "weight")
            if bool(module.with_scale) != has_weight:
                raise AssertionError(f"weight topology disagrees with with_scale: {name}")

    def discovery_receipt(self) -> dict[str, Any]:
        scaled = [name for name, module in self._modules if module.with_scale]
        scale_free = [name for name, module in self._modules if not module.with_scale]
        weights = {
            name: {
                "object_id": id(module.weight),
                "data_ptr": module.weight.data_ptr(),
                "shape": list(module.weight.shape),
                "dtype": str(module.weight.dtype),
                "device": str(module.weight.device),
            }
            for name, module in self._modules
            if module.with_scale
        }
        return {
            "path_prefix": "model.language_model.",
            "module_count": len(self._modules),
            "scaled_count": len(scaled),
            "scale_free_count": len(scale_free),
            "scaled_module_names": scaled,
            "scale_free_module_names": scale_free,
            "weight_objects": weights,
            "preexisting_instance_forward_count": sum(
                had_forward
                for had_forward, _ in self._baseline_instance_forwards.values()
            ),
            "preexisting_instance_forward_objects_sha256": canonical_sha256(
                [
                    {
                        "name": name,
                        "present": had_forward,
                        "object_id": id(forward) if had_forward else None,
                        "type": (
                            f"{type(forward).__module__}.{type(forward).__qualname__}"
                            if had_forward
                            else None
                        ),
                    }
                    for name, (had_forward, forward) in sorted(
                        self._baseline_instance_forwards.items()
                    )
                ]
            ),
            "same_class_modules_outside_language_model": self.forbidden_same_class_modules,
        }

    def activate(self) -> dict[str, Any]:
        if self._active:
            raise RuntimeError("K1 reconnection is already active")
        if self._original_instance_forwards:
            raise RuntimeError("stale K1 forward restoration state")

        for name, module in self._modules:
            had_instance_forward = "forward" in module.__dict__
            previous = module.__dict__.get("forward")
            baseline_had_forward, baseline_forward = self._baseline_instance_forwards[name]
            if had_instance_forward != baseline_had_forward or (
                had_instance_forward and previous is not baseline_forward
            ):
                raise AssertionError(
                    f"RMSNorm forward baseline drift before activation: {name}"
                )
            self._original_instance_forwards[name] = (had_instance_forward, previous)

            def custom_forward(
                current_module: Any,
                hidden_states: torch.Tensor,
                *,
                _module_name: str = name,
            ) -> torch.Tensor:
                weight = current_module.weight if current_module.with_scale else None
                with INVOCATION_TRACKER.module_context(_module_name):
                    return call_k1_custom_op(
                        hidden_states, weight, float(current_module.eps)
                    )

            module.forward = MethodType(custom_forward, module)
        self._active = True
        return {
            "status": "ACTIVE",
            "patched_modules": len(self._modules),
            "torch_op": QUALIFIED_OP_NAME,
        }

    def restore(self) -> dict[str, Any]:
        if not self._active:
            if self._original_instance_forwards:
                raise RuntimeError("inactive reconnector retained restoration state")
            return {"status": "ALREADY_INACTIVE", "restored_modules": 0}
        for name, module in self._modules:
            had_instance_forward, previous = self._original_instance_forwards[name]
            if had_instance_forward:
                module.forward = previous
            else:
                delattr(module, "forward")
        restored = len(self._modules)
        self._original_instance_forwards.clear()
        self._active = False
        return {"status": "RESTORED", "restored_modules": restored}

    @property
    def active(self) -> bool:
        return self._active

    def assert_restored(self) -> None:
        if self._active or self._original_instance_forwards:
            raise AssertionError("K1 forward overrides were not fully restored")
        mismatches = []
        for name, module in self._modules:
            baseline_had_forward, baseline_forward = self._baseline_instance_forwards[name]
            observed_had_forward = "forward" in module.__dict__
            observed_forward = module.__dict__.get("forward")
            if observed_had_forward != baseline_had_forward or (
                observed_had_forward and observed_forward is not baseline_forward
            ):
                mismatches.append(name)
        if mismatches:
            raise AssertionError(
                f"RMSNorm forward resolution was not restored to its load-time baseline: {mismatches}"
            )


def canonical_sha256(value: Any) -> str:
    import hashlib

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
