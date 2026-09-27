#!/usr/bin/env python3
"""Reusable OP0 K0 vector-add harness and Nsight Systems target.

The direct harness validates eager, torch.compile, and Triton implementations,
then measures them with CUDA events under a rotating interleave.  The profiling
target emits a same-run CUDA-event receipt for comparison with the GPU timeline
reported by Nsight Systems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
import triton
import triton.language as tl

from op0_common import (
    NvidiaSmiMonitor,
    atomic_write_json,
    now_iso,
    summarize,
    text_sha256,
)


@triton.jit
def k0_vector_add_kernel(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    output_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def candidate_add(
    x: torch.Tensor, y: torch.Tensor, output: torch.Tensor, block_size: int
) -> torch.Tensor:
    grid = (triton.cdiv(x.numel(), block_size),)
    k0_vector_add_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=block_size)
    return output


def eager_add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    torch.add(x, y, out=output)
    return output


def make_compiled_add() -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return torch.compile(eager_add, fullgraph=True, mode="reduce-overhead")


def cuda_event_measure(
    operations: dict[str, Callable[[], torch.Tensor]],
    order: list[str],
    warmup_repetitions: int,
    measured_repetitions: int,
    before_synchronize: Callable[[], None] | None = None,
) -> dict[str, list[float]]:
    """Measure operations in microseconds without host-side launch timing."""
    for _ in range(warmup_repetitions):
        for name in order:
            operations[name]()
    torch.cuda.synchronize()

    pending: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in order
    }
    for repetition in range(measured_repetitions):
        rotated = order[repetition % len(order) :] + order[: repetition % len(order)]
        for name in rotated:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operations[name]()
            end.record()
            pending[name].append((start, end))
    if before_synchronize is not None:
        before_synchronize()
    torch.cuda.synchronize()
    return {
        name: [start.elapsed_time(end) * 1000.0 for start, end in pairs]
        for name, pairs in pending.items()
    }


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (actual - expected).abs()
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
    relative = delta / denominator
    max_rel = float(relative.max().item()) if relative.numel() else 0.0
    return max_abs, max_rel


def run_correctness(
    config: dict[str, Any],
    compiled_add: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    settings = config["k0_selfcheck"]
    atol = float(settings["correctness_atol"])
    rtol = float(settings["correctness_rtol"])
    block_size = int(settings["triton_block_size"])
    trials: list[dict[str, Any]] = []

    for seed in settings["correctness_seeds"]:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(seed))
        for numel in settings["correctness_shapes"]:
            x = torch.randn(int(numel), device="cuda", dtype=torch.float32, generator=generator)
            y = torch.randn(int(numel), device="cuda", dtype=torch.float32, generator=generator)
            expected = x + y
            implementations: dict[str, dict[str, Any]] = {}
            for name in ("eager", "compile", "candidate"):
                output = torch.empty_like(x)
                if name == "eager":
                    eager_add(x, y, output)
                elif name == "compile":
                    compiled_add(x, y, output)
                else:
                    candidate_add(x, y, output, block_size)
                torch.cuda.synchronize()
                max_abs, max_rel = _error_metrics(output, expected)
                passed = bool(torch.allclose(output, expected, atol=atol, rtol=rtol))
                implementations[name] = {
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "pass": passed,
                }
            trials.append(
                {
                    "seed": int(seed),
                    "dims": [int(numel)],
                    "implementations": implementations,
                    "pass": all(item["pass"] for item in implementations.values()),
                }
            )
            del x, y, expected, output

    max_abs = max(
        implementation["max_abs"]
        for trial in trials
        for implementation in trial["implementations"].values()
    )
    max_rel = max(
        implementation["max_rel"]
        for trial in trials
        for implementation in trial["implementations"].values()
    )
    return {
        "tolerance_policy": {
            "formula": "torch.allclose(actual, reference, atol + rtol * abs(reference))",
            "atol": atol,
            "rtol": rtol,
            "dtype": "float32",
        },
        "max_abs": max_abs,
        "max_rel": max_rel,
        "seeds": [int(value) for value in settings["correctness_seeds"]],
        "n_trials": len(trials),
        "trials": trials,
        "pass": all(trial["pass"] for trial in trials),
    }


def _compact_clocks(telemetry: dict[str, Any]) -> dict[str, Any]:
    numeric = telemetry["numeric"]
    return {
        "sm_mhz_p50": numeric["sm_mhz"]["p50"] if numeric["sm_mhz"] else None,
        "sm_mhz_min": numeric["sm_mhz"]["min"] if numeric["sm_mhz"] else None,
        "memory_mhz_p50": (
            numeric["memory_mhz"]["p50"] if numeric["memory_mhz"] else None
        ),
        "temp_c_max": (
            numeric["temperature_c"]["max"] if numeric["temperature_c"] else None
        ),
        "power_w_p50": numeric["power_w"]["p50"] if numeric["power_w"] else None,
        "pstates": telemetry["pstates"],
        "throttle_flags": telemetry["throttle_flags"],
        "thermal_or_hardware_throttle": telemetry["thermal_or_hardware_throttle"],
        "sample_count": telemetry["sample_count"],
        "monitor_errors": telemetry["errors"],
    }


def run_direct_harness(
    config: dict[str, Any],
    env_identity_ref: str,
    git_sha: str,
    ceiling_gbps: float,
    ceiling_memory_mhz: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K0 requires CUDA")
    settings = config["k0_selfcheck"]
    block_size = int(settings["triton_block_size"])
    numel = int(settings["timed_numel"])
    compiled_add = make_compiled_add()

    monitor = NvidiaSmiMonitor(config["power_protocol"]["sample_interval_seconds"])
    monitor.set_phase("k0_correctness")
    monitor.start()
    try:
        correctness = run_correctness(config, compiled_add)
        if not correctness["pass"]:
            raise AssertionError("K0 randomized correctness check failed")

        monitor.set_phase("k0_stabilization")
        x = torch.randn(numel, device="cuda", dtype=torch.float32)
        y = torch.randn(numel, device="cuda", dtype=torch.float32)
        outputs = {
            name: torch.empty_like(x) for name in ("eager", "compile", "candidate")
        }
        operations: dict[str, Callable[[], torch.Tensor]] = {
            "eager": lambda: eager_add(x, y, outputs["eager"]),
            "compile": lambda: compiled_add(x, y, outputs["compile"]),
            "candidate": lambda: candidate_add(x, y, outputs["candidate"], block_size),
        }
        # Trigger any remaining specialization before resetting peak statistics.
        for operation in operations.values():
            operation()
        torch.cuda.synchronize()
        stabilization_started = time.monotonic()
        stabilization_launches = 0
        while (
            time.monotonic() - stabilization_started
            < float(settings["steady_state_warmup_min_seconds"])
        ):
            for _ in range(128):
                operations["candidate"]()
                stabilization_launches += 1
            torch.cuda.synchronize()
        stabilization_seconds = time.monotonic() - stabilization_started

        monitor.set_phase("k0_timing")
        torch.cuda.reset_peak_memory_stats()
        samples = cuda_event_measure(
            operations,
            ["eager", "compile", "candidate"],
            int(settings["warmup_repetitions_per_baseline"]),
            int(settings["measured_repetitions_per_baseline"]),
            before_synchronize=monitor.sample_now,
        )
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    finally:
        telemetry = monitor.stop()

    timing_telemetry = monitor.summary_for_phase("k0_timing")

    baselines: dict[str, Any] = {}
    implementations = {
        "eager": "torch.add(out=)",
        "compile": "torch.compile(torch.add(out=), fullgraph=True, mode=reduce-overhead)",
        "candidate": "Triton k0_vector_add_kernel",
    }
    for name, values in samples.items():
        summary = summarize(values)
        baselines[name] = {
            "implementation": implementations[name],
            "p50_us": summary["p50"],
            "iqr_us": summary["iqr"],
            "reps": summary["count"],
            "distribution_us": summary,
        }
    baselines["reference"] = {
        "implementation": "torch.add float32 semantic reference",
        "timed": False,
        "p50_us": None,
        "iqr_us": None,
        "reps": 0,
        "reason": "correctness-only reference; three timed baselines are frozen in config",
    }

    analytic_bytes = numel * 3 * torch.tensor([], dtype=torch.float32).element_size()
    candidate_seconds = baselines["candidate"]["p50_us"] / 1_000_000.0
    measured_gbps = analytic_bytes / candidate_seconds / 1_000_000_000.0
    relative_score = measured_gbps / ceiling_gbps
    clocks = _compact_clocks(timing_telemetry)
    memory_clock_ratio = (
        clocks["memory_mhz_p50"] / ceiling_memory_mhz
        if clocks["memory_mhz_p50"] is not None and ceiling_memory_mhz > 0
        else None
    )
    minimum_memory_clock_ratio = float(
        settings["minimum_memory_clock_ratio_vs_bandwidth"]
    )
    clock_comparable = (
        memory_clock_ratio is not None
        and memory_clock_ratio >= minimum_memory_clock_ratio
    )
    valid = (
        correctness["pass"]
        and not clocks["thermal_or_hardware_throttle"]
        and clock_comparable
    )
    return {
        "schema_version": 1,
        "stage": "OP0-c",
        "kernel": "k0_vector_add",
        "op_semantics_sha": text_sha256(settings["semantics"]),
        "carrier": "synthetic_op0_harness_selfcheck",
        "shapes": [
            {
                "dims": [numel],
                "dtype": "float32",
                "phase": "synthetic_selfcheck",
                "weight": 1.0,
            }
        ],
        "baselines": baselines,
        "correctness": correctness,
        "roofline": {
            "tier": "Tier-A analytic",
            "analytic_bytes": analytic_bytes,
            "measured_gbps": round(measured_gbps, 6),
            "ceiling_gbps": round(ceiling_gbps, 6),
            "relative_score": round(relative_score, 6),
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "clocks": clocks,
        "telemetry": telemetry,
        "timing_phase_telemetry": timing_telemetry,
        "backlink": {
            "carrier": "synthetic",
            "tape_ids": [],
            "tokens_per_s_delta": None,
            "ttft_delta": None,
            "side_effects": [
                "synthetic OP0 shape; excluded from formal carrier conclusions"
            ],
        },
        "env_identity_ref": env_identity_ref,
        "git_sha": git_sha,
        "timestamp": now_iso(),
        "receipt": {
            "warmup_repetitions_per_baseline": int(
                settings["warmup_repetitions_per_baseline"]
            ),
            "measured_repetitions_per_baseline": int(
                settings["measured_repetitions_per_baseline"]
            ),
            "interleave_order": settings["interleave_order"],
            "event_timer": "torch.cuda.Event(enable_timing=True)",
            "steady_state_warmup": {
                "minimum_seconds": float(settings["steady_state_warmup_min_seconds"]),
                "observed_seconds": round(stabilization_seconds, 6),
                "untimed_candidate_launches": stabilization_launches,
            },
            "memory_clock_comparability": {
                "bandwidth_memory_mhz_p50": ceiling_memory_mhz,
                "k0_timing_memory_mhz_p50": clocks["memory_mhz_p50"],
                "observed_ratio": round(memory_clock_ratio, 6)
                if memory_clock_ratio is not None
                else None,
                "minimum_ratio": minimum_memory_clock_ratio,
                "pass": clock_comparable,
            },
            "status": "PASS" if valid else "INVALID",
        },
    }


def run_nsys_target(config: dict[str, Any], output_path: Path) -> int:
    settings = config["k0_selfcheck"]
    numel = int(settings["timed_numel"])
    block_size = int(settings["triton_block_size"])
    repetitions = int(settings["nsys_measured_repetitions"])
    monitor = NvidiaSmiMonitor(config["power_protocol"]["sample_interval_seconds"])
    monitor.set_phase("k0_nsys_stabilization")
    monitor.start()
    try:
        x = torch.empty(numel, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        output = torch.empty_like(x)
        for _ in range(int(settings["warmup_repetitions_per_baseline"])):
            candidate_add(x, y, output, block_size)
        torch.cuda.synchronize()

        stabilization_started = time.monotonic()
        stabilization_launches = 0
        while (
            time.monotonic() - stabilization_started
            < float(settings["steady_state_warmup_min_seconds"])
        ):
            for _ in range(128):
                candidate_add(x, y, output, block_size)
                stabilization_launches += 1
            torch.cuda.synchronize()
        stabilization_seconds = time.monotonic() - stabilization_started

        pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        monitor.set_phase("k0_nsys_measure")
        torch.cuda.nvtx.range_push(settings["nsys_nvtx_range"])
        try:
            for _ in range(repetitions):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                candidate_add(x, y, output, block_size)
                end.record()
                pairs.append((start, end))
            monitor.sample_now()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
        event_us = [start.elapsed_time(end) * 1000.0 for start, end in pairs]
        payload = {
            "schema_version": 1,
            "stage": "OP0-c-nsys-target",
            "kernel": "k0_vector_add",
            "nvtx_range": settings["nsys_nvtx_range"],
            "cuda_event_us": summarize(event_us),
            "steady_state_warmup": {
                "minimum_seconds": float(settings["steady_state_warmup_min_seconds"]),
                "observed_seconds": round(stabilization_seconds, 6),
                "untimed_candidate_launches": stabilization_launches,
            },
            "telemetry": monitor.stop(),
            "timestamp": now_iso(),
            "status": "PASS" if len(event_us) == repetitions else "FAIL",
        }
        atomic_write_json(output_path, payload)
        return 0
    except BaseException as exc:
        payload = {
            "schema_version": 1,
            "stage": "OP0-c-nsys-target",
            "kernel": "k0_vector_add",
            "timestamp": now_iso(),
            "status": "FAIL",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "telemetry": monitor.stop(),
        }
        atomic_write_json(output_path, payload)
        raise


def run_ncu_target(config: dict[str, Any]) -> int:
    settings = config["k0_selfcheck"]
    numel = min(int(settings["timed_numel"]), 65536)
    block_size = int(settings["triton_block_size"])
    x = torch.empty(numel, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    output = torch.empty_like(x)
    candidate_add(x, y, output, block_size)
    torch.cuda.synchronize()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--nsys-target", action="store_true")
    parser.add_argument("--ncu-target", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.nsys_target:
        if args.output is None:
            raise ValueError("--output is required with --nsys-target")
        return run_nsys_target(config, args.output)
    if args.ncu_target:
        return run_ncu_target(config)
    raise ValueError("invoke the direct harness through run_op0.py")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
