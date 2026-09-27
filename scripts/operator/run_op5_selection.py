#!/usr/bin/env python3
"""Execute the frozen OP5 K2 selection gate on the formal C02 NF4 carrier."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import torch

from k0_harness import cuda_event_measure
from load_smoke_nf4 import (
    CPU_MODULES,
    CUDA_MODULES,
    materialize_persistent_cpu_modules,
    memory_snapshot,
    verify_formal_artifact,
)
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, sha256, summarize
from run_op0 import launch_triad


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/operator/op5_k2_selection_probe.json"
DEFAULT_RESULT = ROOT / "results/operator/pool/op5_k2_selection.json"
DEFAULT_RECEIPT = ROOT / "results/operator/pool/op5_receipt.json"
DEFAULT_FAILURE = ROOT / "results/operator/failures/op5_k2_selection_failure.json"


@dataclass
class PreparedModule:
    name: str
    variant: str
    geometry: tuple[int, int]
    input_tensor: torch.Tensor
    packed_weight: torch.Tensor
    second_absmax: torch.Tensor
    first_absmax: torch.Tensor
    scale_code: torch.Tensor
    scale_offset: torch.Tensor
    module: Any
    analytic_bytes: int


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def environment_identity() -> dict[str, Any]:
    command = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,clocks.max.memory,power.max_limit",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "triton",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "safetensors",
            )
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device_name": props.name if props else None,
            "compute_capability": (
                f"sm_{props.major}{props.minor}" if props else None
            ),
            "total_memory_bytes": props.total_memory if props else None,
            "multi_processor_count": props.multi_processor_count if props else None,
            "l2_cache_size_bytes": props.L2_cache_size if props else None,
        },
        "nvidia_smi": (command.stdout + command.stderr).strip(),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_inputs(
    config: dict[str, Any], config_path: Path, config_sha_start: str
) -> dict[str, Any]:
    if config.get("stage") != "OP5_K2_SELECTION_GATE":
        raise AssertionError("unexpected OP5 selection config identity")
    if config.get("frozen_before_any_selection_probe_measurement") is not True:
        raise AssertionError("selection probe was not frozen before measurement")
    if config.get("candidate") != "k2_nf4_double_quant_decode_gemv":
        raise AssertionError("unexpected K2 candidate")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise RuntimeError("OP5 selection requires the local sm_89 CUDA device")

    sources: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for name, identity in config["source_identity"].items():
        if name == "git_sha":
            continue
        path = absolute(identity["path"])
        observed = sha256(path)
        if observed != identity["sha256"]:
            raise AssertionError(
                f"source hash mismatch for {name}: {observed} != {identity['sha256']}"
            )
        sources[name] = {
            "path": str(path),
            "sha256": observed,
            "status": "PASS",
        }
        if path.suffix == ".json":
            loaded[name] = load_json(path)

    current_git_sha = git_sha()
    if current_git_sha != config["source_identity"]["git_sha"]:
        raise AssertionError("Git HEAD differs from the frozen selection identity")
    if loaded["op0_receipt"].get("status") != "PASS":
        raise AssertionError("OP0 is not PASS")
    if loaded["op1_receipt"].get("status") != "PASS":
        raise AssertionError("OP1 is not PASS")
    if loaded["p0_nf4_receipt"].get("status") != "PASS":
        raise AssertionError("formal C02 NF4 carrier is not PASS")
    if loaded["p0_nf4_receipt"].get("formal_evidence_eligible") is not True:
        raise AssertionError("formal C02 NF4 carrier is not evidence eligible")
    if loaded["op4_receipt"].get("status") != "PASS":
        raise AssertionError("OP4 is not PASS")

    census = loaded["shape_census"]
    if census.get("status") != "PASS":
        raise AssertionError("shape census is not PASS")
    decode_rows = [
        row
        for row in census["op_families"]["nf4_linear"]["shapes"]
        if row["phase"] == "decode"
    ]
    target = config["target_population"]
    if len(decode_rows) != int(target["expected_rows"]):
        raise AssertionError("decode NF4 census row count drift")
    if sum(int(row["invocations"]) for row in decode_rows) != int(
        target["expected_projected_invocations"]
    ):
        raise AssertionError("decode NF4 projected invocation count drift")
    observed_weight = sum(float(row["weight"]) for row in decode_rows)
    if not math.isclose(
        observed_weight,
        float(target["expected_invocation_weight"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("decode NF4 invocation weight drift")

    return {
        "config": {
            "path": str(config_path),
            "sha256": config_sha_start,
            "frozen_at": config["frozen_at"],
        },
        "sources": sources,
        "git_sha": current_git_sha,
        "census": census,
        "decode_rows": decode_rows,
        "op1_profiler": loaded["op1_channel_b_profiler"],
        "upstream_states": {
            "p0_nf4": loaded["p0_nf4_receipt"]["status"],
            "op0": loaded["op0_receipt"]["status"],
            "op1": loaded["op1_receipt"]["status"],
            "op4": loaded["op4_receipt"]["status"],
        },
    }


def run_same_day_bandwidth(
    config: dict[str, Any], monitor: NvidiaSmiMonitor, device: int
) -> dict[str, Any]:
    settings = config["analytic_bytes"]["same_day_ceiling"]
    size_mib = int(settings["array_size_mib"])
    numel = size_mib * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
    block_size = int(settings["block_size"])
    log(f"OP5 selection: measuring same-day {size_mib} MiB triad ceiling")
    monitor.set_phase("op5_same_day_bandwidth")
    monitor.sample_now()
    torch.cuda.empty_cache()
    a = torch.ones(numel, dtype=torch.float32, device=device)
    b = torch.full((numel,), 2.0, dtype=torch.float32, device=device)
    output = torch.empty_like(a)
    operation = lambda: launch_triad(a, b, output, block_size)
    timings = cuda_event_measure(
        {"triad": operation},
        ["triad"],
        int(settings["warmup_repetitions"]),
        int(settings["measured_repetitions"]),
        before_synchronize=monitor.sample_now,
    )["triad"]
    operation()
    sample_indices = [0, numel // 2, numel - 1]
    observed = output[sample_indices].cpu().tolist()
    if observed != [2.0, 2.0, 2.0]:
        raise AssertionError(f"same-day triad correctness failed: {observed}")
    analytic_bytes = int(settings["analytic_bytes_per_repetition"])
    if analytic_bytes != 3 * size_mib * 1024 * 1024:
        raise AssertionError("same-day triad analytic byte declaration is inconsistent")
    gbps = [analytic_bytes / (value * 1000.0) for value in timings]
    latency = summarize(timings)
    throughput = summarize(gbps)
    del output, b, a
    torch.cuda.empty_cache()
    return {
        "operation": "Triton float32 triad",
        "array_size_mib": size_mib,
        "analytic_bytes_per_repetition": analytic_bytes,
        "latency_us": latency,
        "gbps_decimal": throughput,
        "ceiling_gbps_decimal": float(throughput["p50"]),
        "correctness": {"sample_indices": sample_indices, "observed": observed, "pass": True},
        "protocol": {
            "warmup_repetitions": int(settings["warmup_repetitions"]),
            "measured_repetitions": int(settings["measured_repetitions"]),
            "block_size": block_size,
        },
    }


def load_formal_carrier(
    config: dict[str, Any], device: int
) -> tuple[Any, dict[str, Any]]:
    import bitsandbytes as bnb
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    carrier = config["carrier"]
    model_path = absolute(carrier["model"])
    log("OP5 selection: re-verifying the complete C02 artifact inventory")
    integrity = verify_formal_artifact(
        model_path, carrier["model_manifest_sha256"]
    )
    torch.manual_seed(int(config["baseline_probe"]["seed"]))
    torch.cuda.manual_seed_all(int(config["baseline_probe"]["seed"]))
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    device_map: dict[str, int | str] = {
        **{name: "cpu" for name in CPU_MODULES},
        **{name: device for name in CUDA_MODULES},
    }
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    log("OP5 selection: loading the formal C02 carrier as double-quantized NF4")
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quantization,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    cpu_materialization = materialize_persistent_cpu_modules(model, torch)
    return model, {
        "integrity": integrity,
        "load_seconds": load_seconds,
        "device_map_requested": device_map,
        "hf_device_map_observed": getattr(model, "hf_device_map", None),
        "cpu_materialization": cpu_materialization,
        "memory_after_load": memory_snapshot(torch, device),
        "bitsandbytes_version": bnb.__version__,
    }


def variant_for_module(name: str, module: Any) -> str:
    if name == "model.language_model.per_layer_model_projection":
        return "ple_context_projection"
    if name.endswith(".per_layer_input_gate"):
        return "ple_input_gate"
    if name.endswith(".per_layer_projection"):
        return "ple_residual_projection"
    if name.endswith(".mlp.gate_proj"):
        return "mlp_gate_proj"
    if name.endswith(".mlp.up_proj"):
        return "mlp_up_proj"
    if name.endswith(".mlp.down_proj"):
        return "mlp_down_proj"
    if name.endswith(".self_attn.q_proj"):
        return "q_proj_sliding" if module.out_features == 2048 else "q_proj_full"
    if name.endswith(".self_attn.k_proj"):
        return (
            "k_proj_sliding_nonshared"
            if module.out_features == 512
            else "k_proj_full_nonshared"
        )
    if name.endswith(".self_attn.v_proj"):
        return (
            "v_proj_sliding_nonshared"
            if module.out_features == 512
            else "v_proj_full_nonshared"
        )
    if name.endswith(".self_attn.o_proj"):
        return "o_proj_sliding" if module.in_features == 2048 else "o_proj_full"
    raise AssertionError(f"unclassified active Linear4bit module: {name}")


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def prepare_modules(
    model: Any, config: dict[str, Any], decode_rows: list[dict[str, Any]], device: int
) -> tuple[list[PreparedModule], dict[str, Any]]:
    import bitsandbytes as bnb

    quantization = config["carrier"]["quantization"]
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("model.language_model.")
        and isinstance(module, bnb.nn.Linear4bit)
    ]
    if len(modules) != int(quantization["expected_language_linear4bit_modules"]):
        raise AssertionError(f"active Linear4bit count drift: {len(modules)}")
    census_by_variant = {row["variant"]: row for row in decode_rows}
    inputs: dict[int, torch.Tensor] = {}
    prepared: list[PreparedModule] = []
    layout_rows: list[dict[str, Any]] = []
    count_by_variant: Counter[str] = Counter()

    for name, module in modules:
        variant = variant_for_module(name, module)
        if variant not in census_by_variant:
            raise AssertionError(f"module variant missing from decode census: {variant}")
        census_row = census_by_variant[variant]
        _, expected_k, expected_n = (int(value) for value in census_row["dims"])
        if (module.in_features, module.out_features) != (expected_k, expected_n):
            raise AssertionError(f"geometry mismatch for {name}")

        state = module.weight.quant_state
        if state is None or not state.nested or state.state2 is None:
            raise AssertionError(f"{name} is not nested-quantized")
        if state.quant_type != quantization["quant_type"]:
            raise AssertionError(f"quant type drift for {name}")
        if state.blocksize != int(quantization["expected_quant_blocksize"]):
            raise AssertionError(f"quant blocksize drift for {name}")
        if state.state2.blocksize != int(
            quantization["expected_second_level_blocksize"]
        ):
            raise AssertionError(f"second-level blocksize drift for {name}")
        if list(state.shape) != [expected_n, expected_k]:
            raise AssertionError(f"quant-state shape drift for {name}: {state.shape}")
        if module.bias is not None:
            raise AssertionError(f"unexpected bias on {name}")
        packed = module.weight.data.view(-1, 1)
        expected_packed = (expected_n * expected_k + 1) // 2
        expected_first_absmax = math.ceil(expected_n * expected_k / state.blocksize)
        expected_second_absmax = math.ceil(
            expected_first_absmax / state.state2.blocksize
        )
        if packed.dtype != torch.uint8 or packed.numel() != expected_packed:
            raise AssertionError(f"packed NF4 storage drift for {name}")
        if state.absmax.dtype != torch.uint8 or state.absmax.numel() != expected_first_absmax:
            raise AssertionError(f"first-level scale storage drift for {name}")
        if state.state2.absmax.dtype != torch.float32 or state.state2.absmax.numel() != expected_second_absmax:
            raise AssertionError(f"second-level scale storage drift for {name}")
        if state.state2.code.dtype != torch.float32 or state.state2.code.numel() != 256:
            raise AssertionError(f"scale codebook drift for {name}")
        if state.offset.dtype != torch.float32 or state.offset.numel() != 1:
            raise AssertionError(f"scale offset drift for {name}")
        if expected_k not in inputs:
            inputs[expected_k] = torch.randn(
                (1, 1, expected_k), dtype=torch.bfloat16, device=device
            )
        analytic_bytes = (
            tensor_bytes(inputs[expected_k])
            + tensor_bytes(packed)
            + tensor_bytes(state.absmax)
            + tensor_bytes(state.state2.absmax)
            + tensor_bytes(state.state2.code)
            + tensor_bytes(state.offset)
            + expected_n * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
        prepared.append(
            PreparedModule(
                name=name,
                variant=variant,
                geometry=(expected_k, expected_n),
                input_tensor=inputs[expected_k],
                packed_weight=packed,
                second_absmax=state.state2.absmax,
                first_absmax=state.absmax,
                scale_code=state.state2.code,
                scale_offset=state.offset,
                module=module,
                analytic_bytes=analytic_bytes,
            )
        )
        count_by_variant[variant] += 1
        layout_rows.append(
            {
                "name": name,
                "variant": variant,
                "shape_n_k": [expected_n, expected_k],
                "packed_weight": {
                    "shape": list(packed.shape),
                    "dtype": str(packed.dtype),
                    "bytes": tensor_bytes(packed),
                },
                "first_absmax": {
                    "shape": list(state.absmax.shape),
                    "dtype": str(state.absmax.dtype),
                    "bytes": tensor_bytes(state.absmax),
                },
                "second_absmax": {
                    "shape": list(state.state2.absmax.shape),
                    "dtype": str(state.state2.absmax.dtype),
                    "bytes": tensor_bytes(state.state2.absmax),
                },
                "scale_code": {
                    "shape": list(state.state2.code.shape),
                    "dtype": str(state.state2.code.dtype),
                    "bytes": tensor_bytes(state.state2.code),
                },
                "scale_offset": {
                    "shape": list(state.offset.shape),
                    "dtype": str(state.offset.dtype),
                    "bytes": tensor_bytes(state.offset),
                },
                "quant_type": state.quant_type,
                "blocksize": state.blocksize,
                "nested": state.nested,
                "second_level_blocksize": state.state2.blocksize,
                "bias": None,
                "analytic_bytes": analytic_bytes,
            }
        )

    for variant, row in census_by_variant.items():
        if count_by_variant[variant] != int(row["calls_per_forward"]):
            raise AssertionError(
                f"module count for {variant} is {count_by_variant[variant]}, "
                f"expected {row['calls_per_forward']}"
            )
    representatives = {
        variant: min((row for row in layout_rows if row["variant"] == variant), key=lambda row: row["name"])
        for variant in sorted(census_by_variant)
    }
    return prepared, {
        "module_count": len(prepared),
        "variant_count": len(count_by_variant),
        "counts_by_variant": dict(sorted(count_by_variant.items())),
        "all_layout_invariants_pass": True,
        "representatives": representatives,
        "module_order": [record.name for record in prepared],
    }


def make_vendor_operations(
    geometry: tuple[int, int], blocksize: int
) -> tuple[
    Callable[..., torch.Tensor],
    Callable[..., torch.Tensor],
    float,
]:
    k_dim, n_dim = geometry
    shape_b = [n_dim, k_dim]

    def operation(
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        second_absmax: torch.Tensor,
        first_absmax: torch.Tensor,
        scale_code: torch.Tensor,
        scale_offset: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.bitsandbytes.gemm_4bit.default(
            activation,
            packed_weight,
            shape_b,
            second_absmax,
            blocksize,
            "nf4",
            bias=None,
            absmax_8bit=first_absmax,
            absmax_code=scale_code,
            absmax_offset=scale_offset,
        )

    started = time.perf_counter()
    compiled = torch.compile(
        operation,
        fullgraph=True,
        mode="reduce-overhead",
        dynamic=False,
    )
    compile_wrapper_seconds = time.perf_counter() - started
    return operation, compiled, compile_wrapper_seconds


def call_record(
    record: PreparedModule,
    operation: Callable[..., torch.Tensor],
) -> torch.Tensor:
    return operation(
        record.input_tensor,
        record.packed_weight,
        record.second_absmax,
        record.first_absmax,
        record.scale_code,
        record.scale_offset,
    )


def benchmark_baselines(
    prepared: list[PreparedModule],
    decode_rows: list[dict[str, Any]],
    config: dict[str, Any],
    monitor: NvidiaSmiMonitor,
    ceiling_gbps: float,
) -> dict[str, Any]:
    protocol = config["measurement_protocol"]
    blocksize = int(config["carrier"]["quantization"]["expected_quant_blocksize"])
    unique_geometries = sorted({record.geometry for record in prepared})
    recompile_limit = int(config["baseline_probe"]["torch_dynamo_recompile_limit"])
    if recompile_limit < len(unique_geometries):
        raise AssertionError("frozen Dynamo recompile limit cannot cover all geometries")
    torch._dynamo.config.recompile_limit = recompile_limit
    operations: dict[tuple[int, int], dict[str, Callable[..., torch.Tensor]]] = {}
    compile_seconds: dict[str, float] = {}
    log(f"OP5 selection: compiling {len(unique_geometries)} tensor-only vendor baselines")
    for geometry in unique_geometries:
        eager, compiled, wrapper_seconds = make_vendor_operations(geometry, blocksize)
        operations[geometry] = {"eager": eager, "compile": compiled}
        compile_seconds[f"K{geometry[0]}_N{geometry[1]}"] = wrapper_seconds

    correctness: list[dict[str, Any]] = []
    compile_materialization_started = time.perf_counter()
    with torch.inference_mode():
        for record in prepared:
            eager_output = call_record(record, operations[record.geometry]["eager"])
            compiled_output = call_record(record, operations[record.geometry]["compile"])
            module_output = record.module(record.input_tensor)
            torch.cuda.synchronize()
            eager_compile_exact = torch.equal(eager_output, compiled_output)
            eager_module_exact = torch.equal(eager_output, module_output)
            if not eager_compile_exact or not eager_module_exact:
                raise AssertionError(f"vendor baseline correctness drift for {record.name}")
            correctness.append(
                {
                    "module": record.name,
                    "variant": record.variant,
                    "eager_compile_exact": eager_compile_exact,
                    "eager_module_forward_exact": eager_module_exact,
                    "output_shape": list(eager_output.shape),
                    "output_dtype": str(eager_output.dtype),
                }
            )
    compile_materialization_seconds = time.perf_counter() - compile_materialization_started

    def invoke_sequence(implementation: str) -> None:
        for record in prepared:
            call_record(record, operations[record.geometry][implementation])

    warmup_sequences = int(
        protocol["warmup_full_343_call_sequences_per_implementation"]
    )
    log(
        f"OP5 selection: warming {warmup_sequences} full 343-call sequences per baseline"
    )
    monitor.set_phase("op5_k2_baseline_warmup")
    monitor.sample_now()
    warmup_started = time.perf_counter()
    warmup_executed = 0
    with torch.inference_mode():
        for repetition in range(warmup_sequences):
            order = ["eager", "compile"] if repetition % 2 == 0 else ["compile", "eager"]
            for implementation in order:
                invoke_sequence(implementation)
            warmup_executed += 1
        torch.cuda.synchronize()
        minimum_seconds = float(protocol["steady_state_warmup_min_seconds"])
        while time.perf_counter() - warmup_started < minimum_seconds:
            invoke_sequence("eager")
            invoke_sequence("compile")
            warmup_executed += 1
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    measured_sequences = int(
        protocol["measured_full_343_call_sequences_per_implementation"]
    )
    samples: dict[str, dict[str, list[float]]] = {
        "eager": defaultdict(list),
        "compile": defaultdict(list),
    }
    sequence_samples: dict[str, list[float]] = {"eager": [], "compile": []}
    log(
        f"OP5 selection: timing {measured_sequences} interleaved full 343-call sequences"
    )
    monitor.set_phase("op5_k2_baseline_measurement")
    monitor.sample_now()
    with torch.inference_mode():
        for repetition in range(measured_sequences):
            order = ["eager", "compile"] if repetition % 2 == 0 else ["compile", "eager"]
            for implementation in order:
                sequence_start = torch.cuda.Event(enable_timing=True)
                sequence_end = torch.cuda.Event(enable_timing=True)
                pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
                sequence_start.record()
                for record in prepared:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    call_record(record, operations[record.geometry][implementation])
                    end.record()
                    pending.append((record.variant, start, end))
                sequence_end.record()
                torch.cuda.synchronize()
                sequence_samples[implementation].append(
                    sequence_start.elapsed_time(sequence_end) * 1000.0
                )
                for variant, start, end in pending:
                    samples[implementation][variant].append(
                        start.elapsed_time(end) * 1000.0
                    )
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
        eager_summary = summarize(samples["eager"][variant])
        compile_summary = summarize(samples["compile"][variant])
        compile_p50 = float(compile_summary["p50"])
        measured_gbps = analytic_bytes / (compile_p50 * 1000.0)
        relative_score = measured_gbps / ceiling_gbps
        census_row = census_by_variant[variant]
        expected_samples = int(census_row["calls_per_forward"]) * measured_sequences
        if int(compile_summary["count"]) != expected_samples:
            raise AssertionError(f"sample count drift for {variant}")
        rows.append(
            {
                "variant": variant,
                "dims": census_row["dims"],
                "calls_per_forward": census_row["calls_per_forward"],
                "projected_invocations": census_row["invocations"],
                "invocation_weight": census_row["weight"],
                "work_weight": census_row["work_weight"],
                "analytic_bytes": analytic_bytes,
                "eager": eager_summary,
                "compile": compile_summary,
                "compile_speedup_vs_eager": float(eager_summary["p50"]) / compile_p50,
                "compile_measured_gbps_decimal": measured_gbps,
                "compile_tier_a_relative_score": relative_score,
                "compile_headroom_fraction": 1.0 - relative_score,
            }
        )

    total_invocations = sum(int(row["projected_invocations"]) for row in rows)
    weighted_bytes = sum(
        int(row["analytic_bytes"]) * int(row["projected_invocations"])
        for row in rows
    ) / total_invocations
    weighted_eager_us = sum(
        float(row["eager"]["p50"]) * int(row["projected_invocations"])
        for row in rows
    ) / total_invocations
    weighted_compile_us = sum(
        float(row["compile"]["p50"]) * int(row["projected_invocations"])
        for row in rows
    ) / total_invocations
    weighted_gbps = weighted_bytes / (weighted_compile_us * 1000.0)
    weighted_relative_score = weighted_gbps / ceiling_gbps
    threshold = float(config["bottleneck_gate"]["maximum_relative_score_to_pass"])
    bottleneck_pass = weighted_relative_score <= threshold
    return {
        "baseline_identity": {
            "eager": config["baseline_probe"]["eager"],
            "compile": config["baseline_probe"]["compile"],
            "compile_fullgraph": True,
            "compile_mode": "reduce-overhead",
            "compile_dynamic": False,
            "compile_graph_breaks": 0,
            "torch_dynamo_recompile_limit": recompile_limit,
            "unique_geometries": [
                {"k": geometry[0], "n": geometry[1]}
                for geometry in unique_geometries
            ],
            "compile_wrapper_creation_seconds": compile_seconds,
            "compile_materialization_and_343_module_correctness_seconds": compile_materialization_seconds,
        },
        "correctness": {
            "modules_checked": len(correctness),
            "all_eager_compile_exact": all(
                row["eager_compile_exact"] for row in correctness
            ),
            "all_eager_module_forward_exact": all(
                row["eager_module_forward_exact"] for row in correctness
            ),
            "pass": True,
        },
        "measurement_protocol": {
            "warmup_full_sequences_required": warmup_sequences,
            "warmup_full_sequences_executed": warmup_executed,
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
            "compile_speedup_vs_eager": weighted_eager_us / weighted_compile_us,
            "compile_measured_gbps_decimal": weighted_gbps,
            "same_day_ceiling_gbps_decimal": ceiling_gbps,
            "compile_tier_a_relative_score": weighted_relative_score,
            "compile_headroom_fraction": 1.0 - weighted_relative_score,
            "maximum_relative_score_to_pass": threshold,
            "bottleneck_gate": "PASS" if bottleneck_pass else "FAIL",
        },
    }


def profiler_hotspot_evidence(profiler: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in profiler["selected_runs"]:
        matches = [
            item
            for item in run["decode"]["profiler"]["interesting_key_averages"]
            if "gemm_4bit_simt" in item["operator"]
        ]
        if len(matches) != 1 or int(matches[0]["count"]) != 343:
            raise AssertionError("OP1 profiler lacks the expected decode gemm_4bit_simt row")
        rows.append(
            {
                "request_id": run["request_id"],
                "kernel": matches[0]["operator"],
                "calls": matches[0]["count"],
                "self_device_time_total_us_observer_only": matches[0][
                    "self_device_time_total_us"
                ],
            }
        )
    return {
        "rows": rows,
        "performance_numbers_eligible": False,
        "use": "topology and qualitative hotspot ordering only",
    }


def build_backlog(
    k2_bottleneck_pass: bool, census: dict[str, Any]
) -> list[dict[str, Any]]:
    attention = census["op_families"]["attention"]
    lm_head = census["op_families"]["lm_head"]
    attention_invocations = sum(
        int(row["invocations"]) for row in attention["shapes"]
    )
    lm_head_invocations = sum(int(row["invocations"]) for row in lm_head["shapes"])
    return [
        {
            "rank": 1,
            "item": "NF4 double-quant decode GEMV",
            "disposition": "SELECTED_AS_K2" if k2_bottleneck_pass else "REJECTED_BY_B_GATE",
            "evidence": "343 calls/forward; 99.9493204332% of projected NF4 invocations are decode; all 14 decode rows measured in the frozen selection probe.",
        },
        {
            "rank": 2,
            "item": "GQA decode attention (SWA 512 + full <=10k)",
            "disposition": "NEXT_CANDIDATE_REQUIRES_OWN_B_PROBE",
            "evidence": f"{attention['calls_per_model_forward']} calls/forward and {attention_invocations} projected invocations; two real census geometries.",
        },
        {
            "rank": 3,
            "item": "Inductor generated-code comparison",
            "disposition": "SUPPORTING_BASELINE_NOT_STANDALONE_KERNEL",
            "evidence": "Useful for K2/K3 comparison, but has no independent census frequency and cannot itself answer Authenticity.",
        },
        {
            "rank": 4,
            "item": "lm_head 262k vocab + softcap fusion",
            "disposition": "HOLD_CURRENT_CARRIER_DELIVERABILITY",
            "evidence": f"Real path ({lm_head_invocations} projected calls) but the tied BF16 lm_head is intentionally CPU-resident under the 8 GiB carrier contract; GPU Triton reconnection is not presently deliverable.",
        },
        {
            "rank": 5,
            "item": "KV dequantization",
            "disposition": "BLOCKED_AUTHENTICITY",
            "evidence": "The frozen C02 PyTorch carrier stores BF16 KV; there is no quantized-KV dequant op in the observed replay path.",
        },
    ]


def execute(
    config_path: Path, result_path: Path, receipt_path: Path, device: int
) -> dict[str, Any]:
    config_sha_start = sha256(config_path)
    config = load_json(config_path)
    verified = verify_frozen_inputs(config, config_path, config_sha_start)
    env = environment_identity()
    monitor = NvidiaSmiMonitor(
        interval_seconds=float(config["measurement_protocol"]["gpu_telemetry_interval_seconds"])
    )
    monitor.start()
    model = None
    try:
        bandwidth = run_same_day_bandwidth(config, monitor, device)
        model, model_load = load_formal_carrier(config, device)
        prepared, layout = prepare_modules(
            model, config, verified["decode_rows"], device
        )
        benchmark = benchmark_baselines(
            prepared,
            verified["decode_rows"],
            config,
            monitor,
            float(bandwidth["ceiling_gbps_decimal"]),
        )
    finally:
        telemetry = monitor.stop()

    config_sha_end = sha256(config_path)
    if config_sha_end != config_sha_start:
        raise AssertionError("selection config changed during execution")
    bandwidth_telemetry = monitor.summary_for_phase("op5_same_day_bandwidth")
    measurement_telemetry = monitor.summary_for_phase(
        "op5_k2_baseline_measurement"
    )
    if telemetry["thermal_or_hardware_throttle"]:
        raise AssertionError("thermal or hardware throttle invalidated OP5 selection")
    op0_memory_clock = float(
        config["measurement_protocol"]["op0_memory_clock_mhz_p50"]
    )
    minimum_ratio = float(
        config["measurement_protocol"]["minimum_memory_clock_ratio_vs_op0"]
    )
    measurement_memory = measurement_telemetry["numeric"]["memory_mhz"]
    if measurement_memory is None:
        raise AssertionError("selection measurement has no memory-clock witness")
    memory_clock_ratio = float(measurement_memory["p50"]) / op0_memory_clock
    if memory_clock_ratio < minimum_ratio:
        raise AssertionError("selection measurement memory clock is below the frozen gate")

    bottleneck_pass = benchmark["aggregate"]["bottleneck_gate"] == "PASS"
    four_questions = {
        "authenticity": {
            "answer": "PASS",
            "basis": "Formal OP1 C02 replay observed 343 Linear4bit calls per model forward; the frozen census projects 7,444,815 NF4 calls, of which 7,441,042 (99.9493204332%) are M=1 decode calls across all 11 requests.",
            "census_pointer": "results/operator/shape_census.json#/op_families/nf4_linear",
        },
        "bottleneck": {
            "answer": "PASS" if bottleneck_pass else "FAIL",
            "basis": "Frozen selection probe measured the full 343-weight streaming sequence for eager and fullgraph compile against a same-day Triton triad ceiling.",
            "compile_tier_a_relative_score": benchmark["aggregate"][
                "compile_tier_a_relative_score"
            ],
            "compile_headroom_fraction": benchmark["aggregate"][
                "compile_headroom_fraction"
            ],
            "maximum_relative_score_to_pass": benchmark["aggregate"][
                "maximum_relative_score_to_pass"
            ],
        },
        "capability": {
            "answer": "PASS_DESIGN",
            "basis": "All 343 exact C02 modules share the predeclared row-major nested-NF4 invariants (block 64, second-level block 256, uint8 packed/code indices, BF16 activation/output). Triton 3.7.1 on sm_89 can express nibble decode, nested scale lookup, and fused reduction without materializing BF16 weights.",
            "layout_invariants": layout["all_layout_invariants_pass"],
        },
        "deliverability": {
            "answer": "PASS_DESIGN",
            "basis": "A torch.library op can consume the existing packed tensors without changing parameters/state_dict; Linear4bit can dispatch only for M=1 BF16 no-grad nested NF4 and fall back unchanged for prefill or any invariant miss.",
            "formal_reconnection_deferred_to": "K2 implementation/reconnection within OP5",
        },
    }
    k2_selected = all(
        answer["answer"].startswith("PASS") for answer in four_questions.values()
    )
    backlog = build_backlog(k2_selected, verified["census"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "stage": "OP5_K2_SELECTION_GATE",
        "timestamp": now_iso(),
        "evidence_scope": config["evidence_scope"],
        "verified_inputs": {
            key: value
            for key, value in verified.items()
            if key not in ("census", "decode_rows", "op1_profiler")
        },
        "environment": env,
        "same_day_bandwidth": bandwidth,
        "model_load": model_load,
        "nf4_layout": layout,
        "op1_profiler_hotspot": profiler_hotspot_evidence(
            verified["op1_profiler"]
        ),
        "baseline_probe": benchmark,
        "four_questions": four_questions,
        "backlog": backlog,
        "telemetry": telemetry,
        "bandwidth_telemetry": bandwidth_telemetry,
        "measurement_telemetry": measurement_telemetry,
        "memory_clock_ratio_vs_op0": memory_clock_ratio,
        "memory_final": memory_snapshot(torch, device),
        "config_sha256_start": config_sha_start,
        "config_sha256_end": config_sha_end,
        "k2_selected": k2_selected,
        "status": "PASS" if k2_selected else "K2_REJECTED",
    }
    atomic_write_json(result_path, result)
    result_sha = sha256(result_path)
    receipt = {
        "schema_version": 1,
        "stage": "OP5",
        "timestamp": now_iso(),
        "config": {"path": str(config_path), "sha256": config_sha_start},
        "result": {"path": str(result_path), "sha256": result_sha},
        "gates": {
            "frozen_probe_config_immutable": "PASS",
            "upstream_p0_op0_op1_op4": "PASS",
            "formal_c02_full_hash": model_load["integrity"]["status"],
            "all_343_nf4_layout_invariants": "PASS",
            "same_day_bandwidth": "PASS",
            "compile_fullgraph_no_break": "PASS",
            "eager_compile_exact_all_343": "PASS",
            "thermal_or_hardware_throttle_absent": "PASS",
            "memory_clock_gate": "PASS",
            "backlog_reordered": "PASS",
            "k2_four_questions": "PASS" if k2_selected else "FAIL",
        },
        "selected_k2": (
            "k2_nf4_double_quant_decode_gemv" if k2_selected else None
        ),
        "formal_k2_measurement_started": False,
        "status": "PASS" if k2_selected else "K2_REJECTED",
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--failure", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    result_path = args.result.resolve()
    receipt_path = args.receipt.resolve()
    failure_path = args.failure.resolve()
    started = now_iso()
    try:
        receipt = execute(config_path, result_path, receipt_path, args.device)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if receipt["status"] == "PASS" else 2
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "stage": "OP5_K2_SELECTION_GATE",
            "started_at": started,
            "failed_at": now_iso(),
            "config": str(config_path),
            "config_sha256": sha256(config_path) if config_path.is_file() else None,
            "result": str(result_path),
            "receipt": str(receipt_path),
            "command": sys.argv,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "git_sha": git_sha(),
            "environment": environment_identity(),
            "status": "FAIL",
        }
        atomic_write_json(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
