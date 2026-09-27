#!/usr/bin/env python3
"""Run the frozen OP4 Nsight Compute attribution for K1 RMSNorm."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from typing import Any

import torch
import triton

from k1_rmsnorm import candidate_rmsnorm
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, sha256
from run_op2 import expand_formal_cases, verify_model_and_load_weights


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/operator/op4_ncu_attribution.json"
DEFAULT_RESULT = ROOT / "results/operator/k1/k1_rmsnorm_op4.json"
DEFAULT_RECEIPT = ROOT / "results/operator/k1/op4_receipt.json"
DEFAULT_RAW = ROOT / "results/operator/k1/op4_ncu_raw.csv"
DEFAULT_TARGET_RECEIPT = ROOT / "results/operator/k1/op4_target_receipt.json"


def _absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha(value: Any) -> str:
    import hashlib

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _verify_artifact(name: str, identity: dict[str, Any]) -> dict[str, Any]:
    path = _absolute(identity["path"])
    observed = sha256(path)
    expected = identity["sha256"]
    if observed != expected:
        raise AssertionError(
            f"frozen source hash mismatch for {name}: {observed} != {expected}"
        )
    return {"path": str(path), "sha256": observed, "status": "PASS"}


def _power_supply_snapshot() -> dict[str, Any]:
    paths = {
        "ac_online": Path("/sys/class/power_supply/AC1/online"),
        "battery_status": Path("/sys/class/power_supply/BAT1/status"),
        "battery_capacity_percent": Path("/sys/class/power_supply/BAT1/capacity"),
    }
    snapshot: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.is_file():
            snapshot[name] = None
            continue
        value = path.read_text(encoding="utf-8").strip()
        snapshot[name] = int(value) if name != "battery_status" else value
    snapshot["captured_at"] = now_iso()
    return snapshot


def _ncu_identity(ncu_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(ncu_path), "--version"], check=True, capture_output=True, text=True
    )
    output = (completed.stdout + completed.stderr).strip()
    version = re.search(r"Version\s+([^\s]+)", output)
    build = re.search(r"build\s+(\d+)", output)
    if version is None or build is None:
        raise AssertionError(f"could not parse ncu identity: {output}")
    return {
        "path": str(ncu_path),
        "version": version.group(1),
        "build": build.group(1),
        "raw": output,
    }


def _query_ncu_metrics(ncu_path: Path, required: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(ncu_path), "--query-metrics", "--query-metrics-mode", "all"],
        check=True,
        capture_output=True,
        text=True,
    )
    available = {
        line.split(maxsplit=1)[0]
        for line in (completed.stdout + completed.stderr).splitlines()
        if line.strip()
    }
    missing = sorted(set(required) - available)
    if missing:
        raise AssertionError(f"required ncu metrics are unavailable: {missing}")
    return {"required": required, "missing": missing, "status": "PASS"}


def _analytic_bytes(case: dict[str, Any], element_bytes: int) -> int:
    m_rows, n_cols = (int(value) for value in case["dims"])
    return 2 * m_rows * n_cols * element_bytes + (
        n_cols * element_bytes if case["with_scale"] else 0
    )


def _case_projection(case: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "case_id",
        "source_census_row",
        "source_variant",
        "variant",
        "with_scale",
        "dims",
        "phase",
        "token_count",
        "invocations",
        "weight",
        "work_weight",
    )
    return {key: case[key] for key in keys}


def _prepare_cases(
    config: dict[str, Any], verified: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    op2_prereg = _json(Path(verified["op2_preregistration"]["path"]))
    census = _json(Path(verified["shape_census"]["path"]))
    op2_result = _json(Path(verified["op2_result"]["path"]))
    op3_receipt = _json(Path(verified["op3_receipt"]["path"]))
    if op2_result.get("status") != "PASS":
        raise AssertionError("bound OP2 result is not PASS")
    if op3_receipt.get("status") != "PASS":
        raise AssertionError("bound OP3 receipt is not PASS")

    expanded = sorted(
        expand_formal_cases(census, op2_prereg), key=lambda item: item["case_id"]
    )
    op2_cases = sorted(
        op2_result["formal_case_measurements"], key=lambda item: item["case_id"]
    )
    if len(expanded) != len(op2_cases):
        raise AssertionError("OP2 expanded/result case counts differ")
    element_bytes = int(config["attribution_protocol"]["formal_element_bytes"])
    for expanded_case, measured_case in zip(expanded, op2_cases, strict=True):
        if _case_projection(expanded_case) != _case_projection(measured_case):
            raise AssertionError(
                f"OP2 formal case drift at {expanded_case['case_id']}"
            )
        expected_bytes = _analytic_bytes(expanded_case, element_bytes)
        if expected_bytes != int(measured_case["analytic_bytes"]):
            raise AssertionError(
                f"OP2 analytic-byte drift at {expanded_case['case_id']}"
            )
        expanded_case["analytic_bytes"] = expected_bytes
        expanded_case["op2_candidate_p50_us"] = float(
            measured_case["baselines"]["candidate"]["p50_us"]
        )
        expanded_case["op2_candidate_iqr_us"] = float(
            measured_case["baselines"]["candidate"]["iqr_us"]
        )

    expected_count = int(config["formal_population"]["expected_case_count"])
    if len(expanded) != expected_count:
        raise AssertionError(f"expected {expected_count} cases, got {len(expanded)}")
    observed_cases_sha = _canonical_sha([_case_projection(case) for case in expanded])
    expected_cases_sha = config["formal_population"]["case_projection_sha256"]
    if observed_cases_sha != expected_cases_sha:
        raise AssertionError(
            f"formal case projection hash mismatch: {observed_cases_sha} != {expected_cases_sha}"
        )
    return expanded, op2_prereg, census, op2_result


def verify_frozen_inputs(
    config: dict[str, Any], config_path: Path, config_sha: str
) -> dict[str, Any]:
    if config.get("stage") != "OP4" or config.get("kernel") != "k1_gemma4_rmsnorm":
        raise AssertionError("unexpected OP4 configuration identity")
    if config.get("frozen_before_formal_measurement") is not True:
        raise AssertionError("OP4 configuration is not marked frozen")

    verified = {
        name: _verify_artifact(name, identity)
        for name, identity in config["source_identity"].items()
    }
    cases, op2_prereg, census, op2_result = _prepare_cases(config, verified)

    readiness = _json(Path(verified["op4_readiness_receipt"]["path"]))
    if readiness.get("status") != "PASS":
        raise AssertionError("OP4 counter readiness receipt is not PASS")
    if not readiness.get("assessment", {}).get("op4_executable_now", False):
        raise AssertionError("OP4 readiness does not authorize formal profiling")

    ncu_path = _absolute(config["toolchain"]["ncu_path"])
    ncu = _ncu_identity(ncu_path)
    if ncu["version"] != config["toolchain"]["ncu_version"]:
        raise AssertionError("live Nsight Compute version differs from frozen version")
    if ncu["build"] != str(config["toolchain"]["ncu_build"]):
        raise AssertionError("live Nsight Compute build differs from frozen build")
    metrics = _query_ncu_metrics(
        ncu_path, list(config["attribution_protocol"]["metrics"])
    )

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise AssertionError("OP4 requires the frozen sm_89 CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if device_name != config["host_scope"]["cuda_device_name"]:
        raise AssertionError(f"unexpected CUDA device: {device_name}")
    power = _power_supply_snapshot()
    if config["power_protocol"]["require_ac_online"] and power["ac_online"] != 1:
        raise AssertionError(f"AC power gate failed: {power}")

    return {
        "config": {
            "path": str(config_path),
            "sha256": config_sha,
            "frozen_at": config["frozen_at"],
        },
        "artifacts": verified,
        "cases": cases,
        "op2_prereg": op2_prereg,
        "census": census,
        "op2_result": op2_result,
        "readiness": readiness,
        "ncu": ncu,
        "metrics": metrics,
        "power_start": power,
        "device": {
            "name": device_name,
            "compute_capability": "sm_89",
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
    }


def _target_run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    observed_sha_start = sha256(config_path)
    if args.expected_config_sha and observed_sha_start != args.expected_config_sha:
        raise AssertionError("target observed an unexpected OP4 config hash")
    config = _json(config_path)
    identities = {
        name: _verify_artifact(name, identity)
        for name, identity in config["source_identity"].items()
        if name
        in {
            "shape_census",
            "op2_preregistration",
            "op2_result",
            "op3_receipt",
            "implementation",
            "op2_runner",
            "op4_runner",
        }
    }
    cases, op2_prereg, census, _ = _prepare_cases(config, identities)
    weights, model_evidence = verify_model_and_load_weights(census, op2_prereg)
    warmup = int(config["attribution_protocol"]["warmup_launches_per_case"])
    seed_base = int(config["attribution_protocol"]["input_seed_base"])

    torch.cuda.reset_peak_memory_stats()
    inputs: list[torch.Tensor] = []
    started = time.monotonic()
    for index, case in enumerate(cases, start=1):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed_base + int(case["source_census_row"]))
        hidden_states = torch.empty(
            tuple(case["dims"]), device="cuda", dtype=torch.bfloat16
        )
        hidden_states.normal_(mean=0.0, std=1.0, generator=generator)
        weight = weights[case["source_variant"]] if case["with_scale"] else None
        warmed_output: torch.Tensor | None = None
        for _ in range(warmup):
            warmed_output = candidate_rmsnorm(hidden_states, weight, float(op2_prereg["semantics"]["epsilon"]))
        torch.cuda.synchronize()
        del warmed_output
        inputs.append(hidden_states)
        if index % 12 == 0:
            print(f"OP4_TARGET_WARMUP {index}/{len(cases)}", flush=True)

    outputs: list[torch.Tensor] = []
    launch_manifest: list[dict[str, Any]] = []
    start_status = torch.cuda.cudart().cudaProfilerStart()
    try:
        for launch_index, (case, hidden_states) in enumerate(
            zip(cases, inputs, strict=True)
        ):
            weight = (
                weights[case["source_variant"]] if case["with_scale"] else None
            )
            outputs.append(
                candidate_rmsnorm(
                    hidden_states,
                    weight,
                    float(op2_prereg["semantics"]["epsilon"]),
                )
            )
            launch_manifest.append(
                {
                    "launch_index": launch_index,
                    "case_id": case["case_id"],
                    "dims": case["dims"],
                    "with_scale": case["with_scale"],
                    "expected_grid_size": int(case["dims"][0]),
                    "analytic_bytes": int(case["analytic_bytes"]),
                }
            )
        torch.cuda.synchronize()
    finally:
        stop_status = torch.cuda.cudart().cudaProfilerStop()

    witness = float(outputs[0][0, 0].float().item()) if outputs else None
    observed_sha_end = sha256(config_path)
    target_receipt = {
        "schema_version": 1,
        "stage": "OP4-target",
        "kernel": "k1_gemma4_rmsnorm",
        "timestamp": now_iso(),
        "config": {
            "path": str(config_path),
            "sha256_at_start": observed_sha_start,
            "sha256_at_end": observed_sha_end,
            "immutable": observed_sha_start == observed_sha_end,
        },
        "protocol": {
            "warmup_launches_per_case": warmup,
            "profiled_launches_per_case": 1,
            "formal_case_count": len(cases),
            "input_seed_base": seed_base,
            "formal_dtype": "bfloat16",
            "cuda_profiler_start_status": str(start_status),
            "cuda_profiler_stop_status": str(stop_status),
        },
        "launch_manifest": launch_manifest,
        "memory": {
            "retained_input_bytes": sum(tensor.numel() * tensor.element_size() for tensor in inputs),
            "retained_output_bytes": sum(tensor.numel() * tensor.element_size() for tensor in outputs),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "output_materialization_witness": witness,
        "model_weight_evidence": model_evidence,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "status": "PASS"
        if (
            len(launch_manifest) == int(config["formal_population"]["expected_case_count"])
            and observed_sha_start == observed_sha_end
            and str(start_status) == "cudaError.success"
            and str(stop_status) == "cudaError.success"
        )
        else "FAIL",
    }
    atomic_write_json(args.target_receipt.resolve(), target_receipt)
    print(
        f"OP4_TARGET_{target_receipt['status']} launches={len(launch_manifest)}",
        flush=True,
    )
    return 0 if target_receipt["status"] == "PASS" else 2


def _parse_ncu_csv(raw_output: str, metrics: list[str]) -> list[dict[str, Any]]:
    lines = raw_output.splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith('"ID",')
        )
    except StopIteration as error:
        raise AssertionError("ncu raw CSV header was not found") from error
    parsed = list(csv.reader(lines[header_index:]))
    if len(parsed) < 3:
        raise AssertionError("ncu raw CSV is incomplete")
    header = parsed[0]
    if len(set(header)) != len(header):
        raise AssertionError("ncu raw CSV has duplicate column names")
    missing = sorted(set(metrics) - set(header))
    if missing:
        raise AssertionError(f"ncu raw CSV is missing metrics: {missing}")
    records: list[dict[str, Any]] = []
    for row in parsed[2:]:
        if len(row) != len(header) or not row[0].isdigit():
            continue
        raw = dict(zip(header, row, strict=True))
        values: dict[str, int] = {}
        for metric in metrics:
            numeric = float(raw[metric])
            if not numeric.is_integer():
                raise AssertionError(f"metric {metric} is not integral: {numeric}")
            values[metric] = int(numeric)
        records.append(
            {
                "ncu_id": int(raw["ID"]),
                "process_id": int(raw["Process ID"]),
                "kernel_name": raw["Kernel Name"],
                "context": int(raw["Context"]),
                "stream": int(raw["Stream"]),
                "block_size": raw["Block Size"],
                "grid_size": raw["Grid Size"],
                "device": raw["Device"],
                "compute_capability": raw["CC"],
                "metrics": values,
                "launch_grid_size": int(float(raw["launch__grid_size"])),
                "launch_block_size": int(float(raw["launch__block_size"])),
                "replayer_passes": int(float(raw["profiler__replayer_passes"])),
            }
        )
    records.sort(key=lambda item: item["ncu_id"])
    return records


def _compact_telemetry(summary: dict[str, Any]) -> dict[str, Any]:
    numeric = summary["numeric"]
    return {
        "sample_count": summary["sample_count"],
        "interval_seconds": summary["interval_seconds"],
        "sm_mhz_p50": numeric["sm_mhz"]["p50"] if numeric["sm_mhz"] else None,
        "sm_mhz_min": numeric["sm_mhz"]["min"] if numeric["sm_mhz"] else None,
        "memory_mhz_p50": numeric["memory_mhz"]["p50"] if numeric["memory_mhz"] else None,
        "memory_mhz_min": numeric["memory_mhz"]["min"] if numeric["memory_mhz"] else None,
        "temperature_c_max": numeric["temperature_c"]["max"] if numeric["temperature_c"] else None,
        "power_w_p50": numeric["power_w"]["p50"] if numeric["power_w"] else None,
        "pstates": summary["pstates"],
        "throttle_flags": summary["throttle_flags"],
        "thermal_or_hardware_throttle": summary["thermal_or_hardware_throttle"],
        "errors": summary["errors"],
    }


def _attribution(
    cases: list[dict[str, Any]],
    ncu_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(cases) != len(ncu_rows):
        raise AssertionError(
            f"ncu returned {len(ncu_rows)} launches for {len(cases)} cases"
        )
    threshold = float(config["attribution_protocol"]["analytic_tolerance_fraction"])
    expected_kernel = config["attribution_protocol"]["kernel_name"]
    ceiling = float(config["roofline"]["ceiling_gbps_decimal"])
    measurements: list[dict[str, Any]] = []
    for case, row in zip(cases, ncu_rows, strict=True):
        if row["kernel_name"] != expected_kernel:
            raise AssertionError(f"unexpected kernel name: {row['kernel_name']}")
        if row["launch_grid_size"] != int(case["dims"][0]):
            raise AssertionError(
                f"launch order/grid mismatch for {case['case_id']}: "
                f"{row['launch_grid_size']} != {case['dims'][0]}"
            )
        measured = int(row["metrics"]["dram__bytes.sum"])
        measured_read = int(row["metrics"]["dram__bytes_read.sum"])
        measured_write = int(row["metrics"]["dram__bytes_write.sum"])
        if measured <= 0 or measured != measured_read + measured_write:
            raise AssertionError(f"invalid DRAM byte decomposition for {case['case_id']}")
        analytic = int(case["analytic_bytes"])
        relative_delta = measured / analytic - 1.0
        within = abs(relative_delta) <= threshold
        accounting_bytes = analytic if within else measured
        accounting_source = "Tier-A analytic (ncu-backed within ±15%)" if within else "Tier-B ncu measured (supersedes analytic)"
        seconds = float(case["op2_candidate_p50_us"]) / 1_000_000.0
        measurements.append(
            {
                **_case_projection(case),
                "analytic_bytes": analytic,
                "ncu": row,
                "measured_dram_bytes": measured,
                "measured_dram_read_bytes": measured_read,
                "measured_dram_write_bytes": measured_write,
                "measured_to_analytic_ratio": round(measured / analytic, 9),
                "relative_delta": round(relative_delta, 9),
                "absolute_relative_delta": round(abs(relative_delta), 9),
                "within_analytic_tolerance": within,
                "accounting_bytes": accounting_bytes,
                "accounting_source": accounting_source,
                "op2_timing": {
                    "candidate_p50_us": case["op2_candidate_p50_us"],
                    "candidate_iqr_us": case["op2_candidate_iqr_us"],
                    "source": "frozen OP2; ncu duration is not used as performance timing",
                },
                "corrected_roofline": {
                    "bytes": accounting_bytes,
                    "candidate_gbps_decimal": round(accounting_bytes / seconds / 1e9, 6),
                    "ceiling_gbps_decimal": ceiling,
                    "relative_score": round(accounting_bytes / seconds / 1e9 / ceiling, 9),
                },
            }
        )

    superseded = [item for item in measurements if not item["within_analytic_tolerance"]]
    backed = [item for item in measurements if item["within_analytic_tolerance"]]
    projected_analytic = sum(
        item["analytic_bytes"] * item["invocations"] for item in measurements
    )
    projected_measured = sum(
        item["measured_dram_bytes"] * item["invocations"] for item in measurements
    )
    projected_accounting = sum(
        item["accounting_bytes"] * item["invocations"] for item in measurements
    )
    projected_seconds = sum(
        item["op2_timing"]["candidate_p50_us"] * item["invocations"]
        for item in measurements
    ) / 1_000_000.0
    by_phase: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        selected = [item for item in measurements if item["phase"] == phase]
        by_phase[phase] = {
            "formal_cases": len(selected),
            "analytic_backed_cases": sum(
                item["within_analytic_tolerance"] for item in selected
            ),
            "ncu_superseded_cases": sum(
                not item["within_analytic_tolerance"] for item in selected
            ),
            "projected_invocations": sum(item["invocations"] for item in selected),
            "projected_analytic_bytes": sum(
                item["analytic_bytes"] * item["invocations"] for item in selected
            ),
            "projected_measured_dram_bytes": sum(
                item["measured_dram_bytes"] * item["invocations"] for item in selected
            ),
            "projected_accounting_bytes": sum(
                item["accounting_bytes"] * item["invocations"] for item in selected
            ),
        }
    aggregate = {
        "formal_cases": len(measurements),
        "analytic_tolerance_fraction": threshold,
        "analytic_backed_cases": len(backed),
        "ncu_superseded_cases": len(superseded),
        "all_cases_within_analytic_tolerance": not superseded,
        "superseded_case_ids": [item["case_id"] for item in superseded],
        "max_absolute_relative_delta": max(
            item["absolute_relative_delta"] for item in measurements
        ),
        "projected_workload": {
            "invocations": sum(item["invocations"] for item in measurements),
            "analytic_bytes": projected_analytic,
            "measured_dram_bytes": projected_measured,
            "accounting_bytes": projected_accounting,
            "accounting_rule": "analytic when within ±15%; otherwise per-case ncu measured bytes",
            "candidate_gpu_seconds_from_op2_p50": round(projected_seconds, 6),
            "analytic_gbps_decimal": round(projected_analytic / projected_seconds / 1e9, 6),
            "measured_dram_gbps_decimal": round(projected_measured / projected_seconds / 1e9, 6),
            "corrected_accounting_gbps_decimal": round(projected_accounting / projected_seconds / 1e9, 6),
            "corrected_relative_score": round(
                projected_accounting / projected_seconds / 1e9 / ceiling, 9
            ),
            "ceiling_gbps_decimal": ceiling,
        },
        "by_phase": by_phase,
        "interpretation": {
            "performance_timing_source": "OP2 CUDA-event 200-repetition interleaved p50/IQR",
            "ncu_duration_use": "diagnostic only; excluded from throughput and roofline timing",
            "counter_repetitions": "one profiled launch per formal case after 50 warmups; Nsight kernel replay collected the requested deterministic counters",
            "disagreement_policy": "A >15% difference does not fail OP4; it makes the measured ncu byte count authoritative for that case.",
        },
    }
    return measurements, aggregate


def _environment() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config_sha_start = sha256(config_path)
    config = _json(config_path)
    verified = verify_frozen_inputs(config, config_path, config_sha_start)
    expected_count = int(config["formal_population"]["expected_case_count"])
    metrics = list(config["attribution_protocol"]["metrics"])
    target_receipt_path = args.target_receipt.resolve()
    raw_path = args.raw.resolve()

    command = [
        config["toolchain"]["ncu_path"],
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(metrics),
        "--disable-extra-suffixes",
        "--cache-control",
        config["attribution_protocol"]["cache_control"],
        "--replay-mode",
        config["attribution_protocol"]["replay_mode"],
        "--profile-from-start",
        "off",
        "--kernel-name",
        f"regex:^{config['attribution_protocol']['kernel_name']}$",
        "--launch-count",
        str(expected_count),
        "--target-processes",
        "application-only",
        sys.executable,
        str(Path(__file__).resolve()),
        "--target",
        "--config",
        str(config_path),
        "--expected-config-sha",
        config_sha_start,
        "--target-receipt",
        str(target_receipt_path),
    ]

    monitor = NvidiaSmiMonitor(
        float(config["power_protocol"]["gpu_telemetry_interval_seconds"])
    )
    monitor.set_phase("op4_ncu_formal")
    monitor.start()
    monitor.sample_now()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(config["attribution_protocol"]["timeout_seconds"]),
        )
    finally:
        monitor.sample_now()
        telemetry = monitor.stop()
    elapsed = time.monotonic() - started
    raw_output = completed.stdout
    if completed.stderr:
        raw_output += (
            "\n# --- STDERR (captured verbatim after stdout) ---\n" + completed.stderr
        )
    _write_text_atomic(raw_path, raw_output)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ncu formal command exited {completed.returncode}; raw output: {raw_path}"
        )

    target_receipt = _json(target_receipt_path)
    if target_receipt.get("status") != "PASS":
        raise AssertionError("OP4 target receipt is not PASS")
    ncu_rows = _parse_ncu_csv(completed.stdout, metrics)
    measurements, aggregate = _attribution(verified["cases"], ncu_rows, config)

    config_sha_end = sha256(config_path)
    config_immutable = config_sha_start == config_sha_end
    power_end = _power_supply_snapshot()
    ac_pass = (
        verified["power_start"]["ac_online"] == 1 and power_end["ac_online"] == 1
    )
    compact = _compact_telemetry(telemetry)
    telemetry_pass = telemetry["sample_count"] >= int(
        config["power_protocol"]["minimum_telemetry_samples"]
    ) and not telemetry["errors"]
    thermal_pass = not telemetry["thermal_or_hardware_throttle"]
    memory_clock = compact["memory_mhz_p50"]
    op0_clock = float(config["power_protocol"]["op0_memory_clock_mhz_p50"])
    memory_clock_ratio = (
        float(memory_clock) / op0_clock if memory_clock is not None else None
    )
    clock_pass = (
        memory_clock_ratio is not None
        and memory_clock_ratio
        >= float(config["power_protocol"]["minimum_memory_clock_ratio_vs_op0"])
    )
    exact_rows = len(ncu_rows) == expected_count
    one_pass = all(row["replayer_passes"] == 1 for row in ncu_rows)
    attribution_closed = (
        len(measurements) == expected_count
        and all(item["measured_dram_bytes"] > 0 for item in measurements)
    )
    gates = {
        "readiness_counter_smoke": "PASS",
        "frozen_source_identity": "PASS",
        "config_immutable_during_measurement": "PASS" if config_immutable else "FAIL",
        "formal_population_all_96_cases": "PASS" if exact_rows else "FAIL",
        "warmup_at_least_50_per_case": "PASS"
        if int(target_receipt["protocol"]["warmup_launches_per_case"]) >= 50
        else "FAIL",
        "ncu_required_metrics_complete": "PASS" if attribution_closed else "FAIL",
        "ncu_single_replay_pass": "PASS" if one_pass else "FAIL",
        "analytic_plus_minus_15_percent_closed": "PASS"
        if aggregate["all_cases_within_analytic_tolerance"]
        else "SUPERSEDED_BY_NCU_PER_PROTOCOL",
        "ac_power": "PASS" if ac_pass else "FAIL",
        "telemetry_1hz": "PASS" if telemetry_pass else "FAIL",
        "memory_clock_comparability": "PASS" if clock_pass else "FAIL",
        "thermal_hardware_throttle": "PASS" if thermal_pass else "FAIL",
        "op2_performance_timing_inherited": "PASS",
        "op3_formal_reconnection_inherited": "PASS",
    }
    blocking = {
        key: value
        for key, value in gates.items()
        if key != "analytic_plus_minus_15_percent_closed"
    }
    status = "PASS" if all(value == "PASS" for value in blocking.values()) else "INVALID"

    result = {
        "schema_version": 1,
        "stage": "OP4",
        "kernel": "k1_gemma4_rmsnorm",
        "timestamp": now_iso(),
        "status": status,
        "preregistration": {
            "path": str(config_path),
            "sha256_at_start": config_sha_start,
            "sha256_at_end": config_sha_end,
            "immutable_during_measurement": config_immutable,
            "frozen_at": config["frozen_at"],
        },
        "source_verification": {
            "artifacts": verified["artifacts"],
            "device": verified["device"],
            "ncu": verified["ncu"],
            "metrics": verified["metrics"],
        },
        "protocol": config["attribution_protocol"],
        "formal_case_measurements": measurements,
        "aggregate": aggregate,
        "measurement_receipt": {
            "command": command,
            "exit_code": completed.returncode,
            "elapsed_seconds": round(elapsed, 6),
            "raw_output": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "target_receipt": {
                "path": str(target_receipt_path),
                "sha256": sha256(target_receipt_path),
            },
            "ncu_launch_rows": len(ncu_rows),
            "ncu_replayer_passes": sorted(
                {row["replayer_passes"] for row in ncu_rows}
            ),
            "power_supply_start": verified["power_start"],
            "power_supply_end": power_end,
            "telemetry": telemetry,
            "clocks": compact,
            "memory_clock_comparability": {
                "op0_memory_mhz_p50": op0_clock,
                "op4_memory_mhz_p50": memory_clock,
                "observed_ratio": round(memory_clock_ratio, 6)
                if memory_clock_ratio is not None
                else None,
                "minimum_ratio": float(
                    config["power_protocol"]["minimum_memory_clock_ratio_vs_op0"]
                ),
                "pass": clock_pass,
            },
        },
        "accounting_effect": {
            "upstream_op2_result_is_immutable": True,
            "op2_tier_a_status": "preserved as historical analytic result",
            "op4_tier_b_status": "authoritative correction layer for cases outside ±15%",
            "consumer_rule": "Use formal_case_measurements[].accounting_bytes and aggregate.projected_workload.accounting_bytes for post-OP4 K1 roofline reporting.",
        },
        "environment": _environment(),
        "implementation": {
            "kernel_source": str(_absolute(config["source_identity"]["implementation"]["path"])),
            "kernel_source_sha256": sha256(
                _absolute(config["source_identity"]["implementation"]["path"])
            ),
            "runner_source": str(Path(__file__).resolve()),
            "runner_source_sha256": sha256(Path(__file__).resolve()),
        },
        "git_sha": _git_sha(),
        "gates": gates,
    }
    atomic_write_json(args.result.resolve(), result)
    result_sha = sha256(args.result.resolve())
    receipt = {
        "schema_version": 1,
        "stage": "OP4",
        "kernel": "k1_gemma4_rmsnorm",
        "timestamp": result["timestamp"],
        "status": status,
        "artifacts": {
            "preregistration": {"path": str(config_path), "sha256": config_sha_end},
            "result": {"path": str(args.result.resolve()), "sha256": result_sha},
            "raw_ncu_csv": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "target_receipt": {
                "path": str(target_receipt_path),
                "sha256": sha256(target_receipt_path),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "summary": {
            "formal_cases": aggregate["formal_cases"],
            "analytic_backed_cases": aggregate["analytic_backed_cases"],
            "ncu_superseded_cases": aggregate["ncu_superseded_cases"],
            "max_absolute_relative_delta": aggregate["max_absolute_relative_delta"],
            "projected_analytic_bytes": aggregate["projected_workload"]["analytic_bytes"],
            "projected_measured_dram_bytes": aggregate["projected_workload"]["measured_dram_bytes"],
            "projected_accounting_bytes": aggregate["projected_workload"]["accounting_bytes"],
            "corrected_relative_score": aggregate["projected_workload"]["corrected_relative_score"],
            "ncu_replayer_passes": sorted(
                {row["replayer_passes"] for row in ncu_rows}
            ),
            "telemetry_samples": telemetry["sample_count"],
            "memory_clock_ratio_vs_op0": round(memory_clock_ratio, 6)
            if memory_clock_ratio is not None
            else None,
            "thermal_or_hardware_throttle": telemetry[
                "thermal_or_hardware_throttle"
            ],
        },
        "scope": {
            "k1_b_complete": status == "PASS",
            "k1_success_definition_closed": status == "PASS",
            "op2_unchanged": sha256(Path(verified["artifacts"]["op2_result"]["path"]))
            == config["source_identity"]["op2_result"]["sha256"],
            "op3_unchanged": sha256(Path(verified["artifacts"]["op3_result"]["path"]))
            == config["source_identity"]["op3_result"]["sha256"],
            "performance_gain_required": False,
            "op5_not_executed": True,
        },
        "gates": gates,
        "git_sha": _git_sha(),
    }
    atomic_write_json(args.receipt.resolve(), receipt)
    return receipt


def _write_failure(
    args: argparse.Namespace,
    started_at: str,
    config_sha: str | None,
    error: BaseException,
) -> Path:
    failure_dir = ROOT / "results/operator/failures"
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = failure_dir / f"k1_op4_{stamp}.json"
    payload = {
        "schema_version": 1,
        "stage": "OP4",
        "kernel": "k1_gemma4_rmsnorm",
        "started_at": started_at,
        "failed_at": now_iso(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "config": str(args.config.resolve()),
        "config_sha256_at_failure": config_sha,
        "environment": _environment(),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "status": "FAIL",
    }
    atomic_write_json(path, payload)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--target-receipt", type=Path, default=DEFAULT_TARGET_RECEIPT)
    parser.add_argument("--target", action="store_true")
    parser.add_argument("--expected-config-sha")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target:
        return _target_run(args)
    started_at = now_iso()
    config_sha: str | None = None
    try:
        config_sha = sha256(args.config.resolve())
        receipt = run(args)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0 if receipt["status"] == "PASS" else 2
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        failure = _write_failure(args, started_at, config_sha, error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        print(f"failure receipt: {failure}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
