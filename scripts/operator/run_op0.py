#!/usr/bin/env python3
"""Execute and receipt the EdgeForge OP0 measurement foundation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from importlib import metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import traceback
from typing import Any, Callable

import torch
import triton
import triton.language as tl

from k0_harness import cuda_event_measure, run_direct_harness
from op0_common import (
    NvidiaSmiMonitor,
    atomic_write_json,
    now_iso,
    sha256,
    summarize,
    text_sha256,
)


@triton.jit
def bandwidth_triad_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    output_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, a + 0.5 * b, mask=mask)


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def run_command(
    command: list[str], cwd: Path | None = None, timeout: int | None = None
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "command": command,
            "exit_code": None,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "timed_out": True,
        }


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _first_line_containing(value: str, pattern: str) -> str | None:
    return next((line for line in value.splitlines() if pattern in line), None)


def collect_environment_identity(
    repo_root: Path, config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    git_sha_result = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_status_result = run_command(["git", "status", "--short"], cwd=repo_root)
    os_release = Path("/etc/os-release")
    proc_version = Path("/proc/version")
    powercfg = Path("/mnt/c/Windows/System32/powercfg.exe")
    powershell = Path(
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    power_plan = (
        run_command([str(powercfg), "/GETACTIVESCHEME"], timeout=30)
        if powercfg.is_file()
        else {"command": [str(powercfg)], "exit_code": None, "error": "not found"}
    )
    battery = (
        run_command(
            [
                str(powershell),
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | ConvertTo-Json -Compress",
            ],
            timeout=30,
        )
        if powershell.is_file()
        else {"command": [str(powershell)], "exit_code": None, "error": "not found"}
    )
    smi = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,clocks.max.memory,power.max_limit,pci.bus_id",
            "--format=csv,noheader",
        ]
    )
    props = torch.cuda.get_device_properties(0)
    resolved_tools = {
        name: shutil.which(name) for name in ("nvcc", "ncu", "nsys", "nvidia-smi")
    }
    configured_nsys = Path(config["k0_selfcheck"]["nsys_binary"])
    versions = {
        "nvcc": run_command([resolved_tools["nvcc"], "--version"])
        if resolved_tools["nvcc"]
        else None,
        "ncu": run_command([resolved_tools["ncu"], "--version"])
        if resolved_tools["ncu"]
        else None,
        "path_nsys": run_command([resolved_tools["nsys"], "--version"])
        if resolved_tools["nsys"]
        else None,
        "configured_nsys": run_command([str(configured_nsys), "--version"])
        if configured_nsys.is_file()
        else None,
    }
    battery_status = None
    if battery.get("exit_code") == 0 and battery.get("stdout"):
        try:
            battery_status = json.loads(battery["stdout"])
        except json.JSONDecodeError:
            battery_status = None

    return {
        "schema_version": 1,
        "stage": "OP0-a",
        "timestamp": now_iso(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "wsl_interop_present": "WSL_INTEROP" in os.environ,
            "proc_version": proc_version.read_text(encoding="utf-8").strip()
            if proc_version.is_file()
            else None,
            "os_release": os_release.read_text(encoding="utf-8").strip()
            if os_release.is_file()
            else None,
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "packages": {
                name: package_version(name)
                for name in (
                    "torch",
                    "triton",
                    "numpy",
                    "transformers",
                    "bitsandbytes",
                )
            },
        },
        "cuda": {
            "torch_cuda_target": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "device": {
                "name": props.name,
                "compute_capability": f"sm_{props.major}{props.minor}",
                "total_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "l2_cache_size_bytes": props.L2_cache_size,
            },
            "nvidia_smi": smi,
        },
        "power_identity": {
            "active_plan": power_plan,
            "battery_query": battery,
            "battery_status": battery_status,
            "ac_power_observed": isinstance(battery_status, dict)
            and battery_status.get("BatteryStatus") in (2, 6, 7, 8, 9, 11),
            "plan_changed_by_script": False,
        },
        "tool_resolution": {
            "path": os.environ.get("PATH"),
            "cuda_bin_present_in_path": "/usr/local/cuda/bin"
            in os.environ.get("PATH", "").split(":"),
            "resolved": resolved_tools,
            "configured_nsys": str(configured_nsys),
            "versions": versions,
        },
        "repository": {
            "root": str(repo_root),
            "git_sha": git_sha_result.get("stdout"),
            "git_status_short": git_status_result.get("stdout", "").splitlines(),
        },
        "measurement_config": {
            "path": str(config_path),
            "sha256": sha256(config_path),
            "frozen_at": config["frozen_at"],
        },
        "scope_guard": {
            "host_scope": config["host_scope"],
            "formal_numbers_must_not_be_extrapolated": config[
                "formal_numbers_must_not_be_extrapolated"
            ],
        },
        "status": "PASS",
    }


def run_toolchain_probe(
    repo_root: Path, script_path: Path, config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    nsys_path = Path(config["k0_selfcheck"]["nsys_binary"])
    nvcc_version = run_command([nvcc, "--version"]) if Path(nvcc).is_file() else None
    ncu_version = run_command([ncu, "--version"]) if Path(ncu).is_file() else None

    ncu_command = [
        ncu,
        "--set",
        "basic",
        "--kernel-name",
        "regex:k0_vector_add_kernel",
        "--launch-count",
        "1",
        "--target-processes",
        "all",
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--ncu-target",
    ]
    ncu_smoke = (
        run_command(ncu_command, cwd=repo_root, timeout=180)
        if Path(ncu).is_file()
        else {
            "command": ncu_command,
            "exit_code": None,
            "stdout": "",
            "stderr": "ncu not found",
            "timed_out": False,
        }
    )
    combined_ncu_output = ncu_smoke.get("stdout", "") + "\n" + ncu_smoke.get(
        "stderr", ""
    )
    if "ERR_NVGPUCTRPERM" in combined_ncu_output:
        ncu_state = "BLOCKED_WINDOWS_PERFORMANCE_COUNTER_PERMISSION"
    elif ncu_smoke.get("exit_code") == 0:
        ncu_state = "PASS"
    else:
        ncu_state = "FAIL"

    nvcc_release = None
    if nvcc_version:
        release_line = _first_line_containing(nvcc_version["stdout"], "release")
        match = re.search(r"release\s+(\d+\.\d+)", release_line or "")
        nvcc_release = match.group(1) if match else None
    torch_cuda_major_minor = ".".join(str(torch.version.cuda).split(".")[:2])
    version_match = nvcc_release == torch_cuda_major_minor
    return {
        "schema_version": 1,
        "stage": "OP0-toolchain-probe",
        "timestamp": now_iso(),
        "path_fix": {
            "cuda_bin": "/usr/local/cuda/bin",
            "present_in_path": "/usr/local/cuda/bin"
            in os.environ.get("PATH", "").split(":"),
            "nvcc_resolved": shutil.which("nvcc"),
            "ncu_resolved": shutil.which("ncu"),
            "status": "PASS"
            if shutil.which("nvcc") and shutil.which("ncu")
            else "FAIL",
        },
        "nvcc": {
            "installed": Path(nvcc).is_file(),
            "path": nvcc,
            "version_probe": nvcc_version,
            "release": nvcc_release,
            "torch_cuda_target": torch.version.cuda,
            "major_minor_matches_torch": version_match,
            "cuda_cpp_policy": "standalone compile available; PyTorch extension requires an ABI/version smoke before use",
            "status": "AVAILABLE_WITH_VERSION_MISMATCH"
            if Path(nvcc).is_file() and not version_match
            else "PASS",
        },
        "ncu": {
            "installed": Path(ncu).is_file(),
            "path": ncu,
            "version_probe": ncu_version,
            "smoke": ncu_smoke,
            "status": ncu_state,
            "owner_action": (
                "Windows NVIDIA Control Panel > Developer > Manage GPU Performance Counters > Allow access to the NVIDIA GPU Performance Counters to all users"
                if ncu_state == "BLOCKED_WINDOWS_PERFORMANCE_COUNTER_PERMISSION"
                else None
            ),
            "blocks_op0": False,
            "blocks_op4_tier_b": ncu_state != "PASS",
        },
        "nsys": {
            "configured_path": str(nsys_path),
            "installed": nsys_path.is_file(),
            "version_probe": run_command([str(nsys_path), "--version"])
            if nsys_path.is_file()
            else None,
            "status": "PASS" if nsys_path.is_file() else "FAIL",
        },
        "status": "PASS" if Path(nvcc).is_file() and nsys_path.is_file() else "FAIL",
    }


def launch_triad(
    a: torch.Tensor, b: torch.Tensor, output: torch.Tensor, block_size: int
) -> torch.Tensor:
    grid = (triton.cdiv(a.numel(), block_size),)
    bandwidth_triad_kernel[grid](
        a, b, output, a.numel(), BLOCK_SIZE=block_size
    )
    return output


def _throughput_samples(latency_us: list[float], analytic_bytes: int) -> list[float]:
    return [analytic_bytes / (latency * 1000.0) for latency in latency_us]


def run_bandwidth_ceiling(
    config: dict[str, Any], env_identity_ref: str, git_sha: str
) -> dict[str, Any]:
    settings = config["bandwidth"]
    if settings["dtype"] != "float32":
        raise ValueError("the implemented OP0 bandwidth harness is frozen to float32")
    warmups = int(settings["warmup_repetitions_per_operation"])
    repetitions = int(settings["measured_repetitions_per_operation"])
    block_size = int(settings["operations"]["triad"]["block_size"])
    monitor = NvidiaSmiMonitor(config["power_protocol"]["sample_interval_seconds"])
    monitor.set_phase("bandwidth_setup")
    monitor.start()
    size_results: list[dict[str, Any]] = []
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for size_mib in settings["array_sizes_mib"]:
            size_mib = int(size_mib)
            monitor.set_phase(f"bandwidth_{size_mib}_mib")
            array_bytes = size_mib * 1024 * 1024
            numel = array_bytes // torch.tensor([], dtype=torch.float32).element_size()
            log(f"bandwidth: allocating three {size_mib} MiB arrays")
            a = torch.empty(numel, device="cuda", dtype=torch.float32)
            b = torch.empty_like(a)
            output = torch.empty_like(a)
            a.fill_(1.0)
            b.fill_(2.0)
            torch.cuda.synchronize()
            operations: dict[str, Callable[[], torch.Tensor]] = {
                "memcpy": lambda: output.copy_(a),
                "triad": lambda: launch_triad(a, b, output, block_size),
            }
            samples = cuda_event_measure(
                operations,
                ["memcpy", "triad"],
                warmups,
                repetitions,
            )

            operations["memcpy"]()
            torch.cuda.synchronize()
            sample_indices = torch.tensor(
                [0, numel // 2, numel - 1], device="cuda", dtype=torch.long
            )
            memcpy_values = output[sample_indices].cpu().tolist()
            operations["triad"]()
            torch.cuda.synchronize()
            triad_values = output[sample_indices].cpu().tolist()
            correctness = {
                "sample_indices": [0, numel // 2, numel - 1],
                "memcpy_observed": memcpy_values,
                "memcpy_expected": 1.0,
                "triad_observed": triad_values,
                "triad_expected": 2.0,
                "pass": all(value == 1.0 for value in memcpy_values)
                and all(value == 2.0 for value in triad_values),
            }
            operations_result: dict[str, Any] = {}
            for name, latency_values in samples.items():
                analytic_bytes = numel * int(
                    settings["operations"][name]["analytic_bytes_per_element"]
                )
                latency_summary = summarize(latency_values)
                throughput_summary = summarize(
                    _throughput_samples(latency_values, analytic_bytes)
                )
                operations_result[name] = {
                    "analytic_bytes": analytic_bytes,
                    "latency_us": latency_summary,
                    "gbps_decimal": throughput_summary,
                    "latency_iqr_fraction": round(
                        latency_summary["iqr"] / latency_summary["p50"], 6
                    ),
                }
            size_results.append(
                {
                    "array_size_mib": size_mib,
                    "array_bytes": array_bytes,
                    "numel": numel,
                    "operations": operations_result,
                    "correctness": correctness,
                }
            )
            del a, b, output, sample_indices
            torch.cuda.empty_cache()
    finally:
        telemetry = monitor.stop()

    stable_sizes = {int(value) for value in settings["stable_segment_sizes_mib"]}
    stable_rows = [
        result for result in size_results if result["array_size_mib"] in stable_sizes
    ]
    if {row["array_size_mib"] for row in stable_rows} != stable_sizes:
        raise AssertionError("not every frozen stable-segment size was measured")
    max_iqr_fraction = float(settings["stable_segment_max_iqr_fraction"])
    stability_checks = {
        str(row["array_size_mib"]): {
            name: values["latency_iqr_fraction"] <= max_iqr_fraction
            for name, values in row["operations"].items()
        }
        for row in stable_rows
    }
    ceiling_operation = settings["formal_ceiling_operation"]
    per_size_ceilings = [
        float(row["operations"][ceiling_operation]["gbps_decimal"]["p50"])
        for row in stable_rows
    ]
    ceiling_gbps = statistics.median(per_size_ceilings)
    datasheet_gbps = float(settings["datasheet_crosscheck_gbps_decimal"])
    ratio = ceiling_gbps / datasheet_gbps
    if 0.80 <= ratio <= 1.20:
        interpretation = "same scale and within ±20% of the 256 GB/s cross-check"
    elif ratio < 0.80:
        interpretation = (
            "below the cross-check; laptop dynamic power/clock behavior, software path, "
            "and analytic versus physical DRAM traffic remain candidate causes"
        )
    else:
        interpretation = (
            "above the cross-check; caching/traffic-accounting effects must be excluded "
            "before treating this as physical DRAM bandwidth"
        )
    thermal_invalid = bool(telemetry["thermal_or_hardware_throttle"])
    correctness_pass = all(row["correctness"]["pass"] for row in size_results)
    stable_pass = all(
        passed
        for size_check in stability_checks.values()
        for passed in size_check.values()
    )
    status = "PASS" if correctness_pass and stable_pass and not thermal_invalid else "INVALID"
    return {
        "schema_version": 1,
        "stage": "OP0-b",
        "kernel": "bandwidth_memcpy_and_triad",
        "op_semantics_sha": text_sha256(
            json.dumps(settings["operations"], sort_keys=True)
        ),
        "carrier": "RTX_4060_Laptop_GPU_sm89_WSL2",
        "dtype": settings["dtype"],
        "byte_accounting": settings["byte_accounting"],
        "sizes": size_results,
        "stable_segment": {
            "sizes_mib": sorted(stable_sizes),
            "maximum_latency_iqr_fraction": max_iqr_fraction,
            "checks": stability_checks,
            "pass": stable_pass,
        },
        "ceiling": {
            "operation": ceiling_operation,
            "per_size_p50_gbps_decimal": per_size_ceilings,
            "p50_across_stable_sizes_gbps_decimal": round(ceiling_gbps, 6),
        },
        "datasheet_crosscheck": {
            "reference_gbps_decimal": datasheet_gbps,
            "observed_to_reference_ratio": round(ratio, 6),
            "interpretation": interpretation,
            "is_acceptance_gate": False,
        },
        "correctness": {"pass": correctness_pass},
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "clocks": {
            "sm_mhz_p50": telemetry["numeric"]["sm_mhz"]["p50"]
            if telemetry["numeric"]["sm_mhz"]
            else None,
            "sm_mhz_min": telemetry["numeric"]["sm_mhz"]["min"]
            if telemetry["numeric"]["sm_mhz"]
            else None,
            "memory_mhz_p50": telemetry["numeric"]["memory_mhz"]["p50"]
            if telemetry["numeric"]["memory_mhz"]
            else None,
            "temp_c_max": telemetry["numeric"]["temperature_c"]["max"]
            if telemetry["numeric"]["temperature_c"]
            else None,
            "power_w_p50": telemetry["numeric"]["power_w"]["p50"]
            if telemetry["numeric"]["power_w"]
            else None,
            "throttle_flags": telemetry["throttle_flags"],
        },
        "telemetry": telemetry,
        "env_identity_ref": env_identity_ref,
        "git_sha": git_sha,
        "timestamp": now_iso(),
        "receipt": {
            "warmup_repetitions_per_operation": warmups,
            "measured_repetitions_per_operation": repetitions,
            "interleave_order": settings["interleave_order"],
            "thermal_or_hardware_throttle": thermal_invalid,
            "status": status,
        },
    }


def parse_nsys_cuda_trace(csv_text: str, kernel_fragment: str) -> list[float]:
    lines = csv_text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("Start (ns),")),
        None,
    )
    if header_index is None:
        raise ValueError("nsys cuda_gpu_trace output has no CSV header")
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    durations_us: list[float] = []
    for row in reader:
        if kernel_fragment in row.get("Name", ""):
            durations_us.append(float(row["Duration (ns)"]) / 1000.0)
    return durations_us


def run_nsys_crosscheck(
    repo_root: Path,
    harness_path: Path,
    config_path: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    settings = config["k0_selfcheck"]
    nsys = Path(settings["nsys_binary"])
    if not nsys.is_file():
        raise FileNotFoundError(f"configured nsys binary is missing: {nsys}")
    report_base = output_dir / "k0_timeline"
    report_path = output_dir / "k0_timeline.nsys-rep"
    target_receipt_path = output_dir / "k0_nsys_target.json"
    profile_command = [
        str(nsys),
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        f"--output={report_base}",
        sys.executable,
        str(harness_path),
        "--config",
        str(config_path),
        "--nsys-target",
        "--output",
        str(target_receipt_path),
    ]
    log("K0 nsys: capturing the full process and filtering the frozen NVTX range")
    profile = run_command(profile_command, cwd=repo_root, timeout=300)
    if profile["exit_code"] != 0:
        raise RuntimeError(f"nsys profile failed: {profile['stderr']}")
    if not report_path.is_file():
        raise FileNotFoundError(f"nsys did not create {report_path}")
    target = json.loads(target_receipt_path.read_text(encoding="utf-8"))
    stats_command = [
        str(nsys),
        "stats",
        "--force-export=true",
        "--quiet",
        "--filter-nvtx",
        settings["nsys_nvtx_range"],
        "--report",
        "cuda_gpu_trace",
        "--format",
        "csv",
        "--output",
        "-",
        str(report_path),
    ]
    stats = run_command(stats_command, cwd=repo_root, timeout=180)
    if stats["exit_code"] != 0:
        raise RuntimeError(f"nsys stats failed: {stats['stderr']}")
    durations_us = parse_nsys_cuda_trace(stats["stdout"], "k0_vector_add_kernel")
    expected_count = int(settings["nsys_measured_repetitions"])
    if not durations_us:
        raise AssertionError("nsys timeline contains no K0 Triton kernel in the NVTX range")
    nsys_summary = summarize(durations_us)
    event_p50_us = float(target["cuda_event_us"]["p50"])
    nsys_p50_us = float(nsys_summary["p50"])
    relative_bias = abs(event_p50_us - nsys_p50_us) / nsys_p50_us
    limit = float(settings["nsys_cuda_event_relative_bias_limit"])
    count_pass = len(durations_us) == expected_count
    bias_pass = relative_bias <= limit
    target_throttle = target["telemetry"]["thermal_or_hardware_throttle"]
    status = (
        "PASS"
        if count_pass and bias_pass and target["status"] == "PASS" and not target_throttle
        else "FAIL"
    )
    return {
        "schema_version": 1,
        "stage": "OP0-c-timer-crosscheck",
        "kernel": "k0_vector_add",
        "timestamp": now_iso(),
        "nvtx_range": settings["nsys_nvtx_range"],
        "cuda_event_us": target["cuda_event_us"],
        "nsys_gpu_timeline_us": nsys_summary,
        "kernel_launch_count": {
            "expected": expected_count,
            "observed": len(durations_us),
            "pass": count_pass,
        },
        "relative_bias": round(relative_bias, 9),
        "relative_bias_formula": "abs(cuda_event_p50 - nsys_p50) / nsys_p50",
        "relative_bias_limit": limit,
        "bias_pass": bias_pass,
        "timeline_artifact": {
            "path": str(report_path),
            "bytes": report_path.stat().st_size,
            "sha256": sha256(report_path),
            "version": run_command([str(nsys), "--version"]),
        },
        "commands": {
            "profile": profile,
            "stats": {
                "command": stats["command"],
                "exit_code": stats["exit_code"],
                "stderr": stats["stderr"],
                "stdout_sha256": text_sha256(stats["stdout"]),
            },
        },
        "target_receipt_ref": str(target_receipt_path),
        "target_telemetry": target["telemetry"],
        "thermal_or_hardware_throttle": target_throttle,
        "status": status,
    }


def write_failure(
    repo_root: Path,
    config_path: Path,
    phase: str,
    error: BaseException,
    receipt_path: Path,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    failure_path = repo_root / "results/operator/failures" / f"op0_{phase}_{timestamp}.json"
    payload = {
        "schema_version": 1,
        "stage": "OP0",
        "phase": phase,
        "timestamp": now_iso(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "command": sys.argv,
        "python_executable": sys.executable,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "status": "FAIL",
    }
    atomic_write_json(failure_path, payload)
    atomic_write_json(
        receipt_path,
        {
            "schema_version": 1,
            "stage": "OP0",
            "timestamp": now_iso(),
            "phase": phase,
            "failure_ref": str(failure_path),
            "status": "FAIL",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/operator/op0_measurement.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/operator/op0")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "op0_receipt.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("stage") != "OP0" or config.get("schema_version") != 1:
        raise ValueError("unexpected OP0 measurement config identity")
    if not torch.cuda.is_available():
        raise RuntimeError("OP0 requires a CUDA device")
    harness_path = repo_root / "scripts/operator/k0_harness.py"
    phase = "initialization"
    try:
        atomic_write_json(
            receipt_path,
            {
                "schema_version": 1,
                "stage": "OP0",
                "timestamp": now_iso(),
                "config_path": str(config_path),
                "config_sha256": sha256(config_path),
                "status": "RUNNING",
            },
        )

        phase = "environment_identity"
        log("OP0-a: freezing environment identity")
        environment = collect_environment_identity(repo_root, config_path, config)
        env_path = output_dir / f"env_identity_{datetime.now().strftime('%Y%m%d')}.json"
        atomic_write_json(env_path, environment)
        git_sha = environment["repository"]["git_sha"]

        phase = "toolchain_probe"
        log("OP0: recording nvcc/ncu/nsys availability and ncu permission smoke")
        toolchain = run_toolchain_probe(
            repo_root, harness_path, config_path, config
        )
        toolchain_path = output_dir / "toolchain_probe.json"
        atomic_write_json(toolchain_path, toolchain)
        if toolchain["path_fix"]["status"] != "PASS":
            raise AssertionError(
                "the formal OP0 process does not contain /usr/local/cuda/bin in PATH"
            )

        phase = "bandwidth_ceiling"
        log("OP0-b: measuring memcpy + triad bandwidth ceiling")
        env_ref = str(env_path.relative_to(repo_root))
        bandwidth = run_bandwidth_ceiling(config, env_ref, git_sha)
        bandwidth_path = output_dir / "bandwidth_ceiling.json"
        atomic_write_json(bandwidth_path, bandwidth)
        if bandwidth["receipt"]["status"] != "PASS":
            raise AssertionError("bandwidth measurement invalidated by an OP0 gate")

        phase = "k0_direct_harness"
        log("OP0-c: validating correctness and three CUDA-event baselines")
        ceiling_gbps = bandwidth["ceiling"][
            "p50_across_stable_sizes_gbps_decimal"
        ]
        ceiling_memory_mhz = bandwidth["clocks"]["memory_mhz_p50"]
        if ceiling_memory_mhz is None:
            raise AssertionError("bandwidth run has no memory-clock witness sample")
        k0 = run_direct_harness(
            config, env_ref, git_sha, ceiling_gbps, ceiling_memory_mhz
        )
        k0_path = output_dir / "k0_harness_selfcheck.json"
        atomic_write_json(k0_path, k0)
        if k0["receipt"]["status"] != "PASS":
            raise AssertionError("K0 direct harness was invalidated")

        phase = "k0_nsys_crosscheck"
        crosscheck = run_nsys_crosscheck(
            repo_root, harness_path, config_path, config, output_dir
        )
        crosscheck_path = output_dir / "k0_nsys_crosscheck.json"
        atomic_write_json(crosscheck_path, crosscheck)
        if crosscheck["status"] != "PASS":
            raise AssertionError(
                "K0 CUDA-event versus nsys timeline self-check exceeded its gate"
            )

        phase = "final_receipt"
        receipt = {
            "schema_version": 1,
            "stage": "OP0",
            "timestamp": now_iso(),
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "artifacts": {
                "environment_identity": str(env_path),
                "toolchain_probe": str(toolchain_path),
                "bandwidth_ceiling": str(bandwidth_path),
                "k0_harness": str(k0_path),
                "k0_nsys_crosscheck": str(crosscheck_path),
                "k0_timeline": crosscheck["timeline_artifact"]["path"],
            },
            "gates": {
                "environment_identity": environment["status"],
                "cuda_path": toolchain["path_fix"]["status"],
                "bandwidth": bandwidth["receipt"]["status"],
                "k0_correctness": "PASS" if k0["correctness"]["pass"] else "FAIL",
                "k0_measurement": k0["receipt"]["status"],
                "k0_nsys_relative_bias": crosscheck["status"],
                "ncu_op4_soft_gate": toolchain["ncu"]["status"],
            },
            "summary": {
                "bandwidth_ceiling_gbps_decimal": ceiling_gbps,
                "bandwidth_crosscheck_ratio": bandwidth["datasheet_crosscheck"][
                    "observed_to_reference_ratio"
                ],
                "k0_candidate_p50_us": k0["baselines"]["candidate"]["p50_us"],
                "k0_nsys_p50_us": crosscheck["nsys_gpu_timeline_us"]["p50"],
                "k0_event_to_nsys_relative_bias": crosscheck["relative_bias"],
            },
            "scope": {
                "formal_host": config["host_scope"],
                "synthetic_k0_excluded_from_carrier_conclusions": True,
                "ncu_is_op4_soft_gate_and_does_not_block_op0": True,
            },
            "git_sha": git_sha,
            "status": "PASS",
        }
        atomic_write_json(receipt_path, receipt)
        log(f"OP0 complete: {receipt_path}")
        return 0
    except BaseException as error:
        write_failure(repo_root, config_path, phase, error, receipt_path)
        log(f"OP0 failed in {phase}: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
