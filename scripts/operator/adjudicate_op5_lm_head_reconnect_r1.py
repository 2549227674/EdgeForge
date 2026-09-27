#!/usr/bin/env python3
"""Adjudicate the R1 hybrid-lockstep memory-clock gate without rerunning GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from op0_common import atomic_write_json, now_iso


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = ROOT / "results/operator/pool/op5_lm_head_reconnect_r1_20260916.json"
DEFAULT_CONFIG = ROOT / "configs/operator/op5_lm_head_reconnect_r1.json"
DEFAULT_OP3_CONFIG = ROOT / "configs/operator/op3_reconnect.json"
DEFAULT_OUTPUT = (
    ROOT
    / "results/operator/pool/op5_lm_head_reconnect_r1_adjudication_receipt_20260916.json"
)
FINAL_STATUS = (
    "LM_HEAD_C02_LOCKSTEP_RECONNECTION_PASS_LOCAL_PATH_AUTHORIZED_"
    "AFTER_CLOCK_GATE_ADJUDICATION"
)


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def adjudicate(
    result_path: Path, config_path: Path, op3_config_path: Path
) -> dict[str, Any]:
    result = load(result_path)
    config = load(config_path)
    op3 = load(op3_config_path)
    if result.get("status") != "LM_HEAD_C02_RECONNECTION_INVALID_NO_DECISION":
        raise AssertionError("source result is not the expected gate-only INVALID result")
    if result.get("sha_verification") != "SKIPPED_BY_OWNER_DIRECTION":
        raise AssertionError("owner no-SHA direction is not intact")
    failed = [
        name
        for name, value in result["validity_gates"].items()
        if value != "PASS"
    ]
    if failed != ["memory_clock_gate"]:
        raise AssertionError(f"adjudication is bounded to one failed clock gate: {failed}")
    token = result["token_correctness_summary"]
    if not (
        result["equivalence_gate"] == "PASS"
        and token["all_tokens_identical"]
        and token["executed_token_decisions_per_path"] == 21705
        and token["matching_token_decisions"] == 21705
        and token["started_requests"] == token["full_budget_requests"] == 11
    ):
        raise AssertionError("full-population token equivalence is not PASS")
    runtime = result["runtime"]
    speedup = float(runtime["speedup_A_over_B"])
    minimum_speedup = float(runtime["minimum_required_speedup"])
    if result["performance_gate"] != "PASS" or speedup < minimum_speedup:
        raise AssertionError("frozen performance gate is not PASS")
    if result["cache_summary"] != {"passing_requests": 11, "executed_requests": 11}:
        raise AssertionError("cache gate is not 11/11 PASS")
    if not result["structure"]["original_identity_unchanged"] or not result[
        "structure"
    ]["model_structure"]["pass"]:
        raise AssertionError("original tied storage or model structure changed")
    memory = result["memory"]
    if int(memory["peak_allocated_bytes"]) >= int(memory["gpu_budget_bytes"]):
        raise AssertionError("peak CUDA allocated memory exceeds the frozen budget")
    telemetry = result["telemetry"]["summary"]
    if telemetry["errors"] or telemetry["thermal_or_hardware_throttle"]:
        raise AssertionError("telemetry contains an execution or hardware-throttle failure")

    precedent = op3["measurement"]["thermal_protocol"][
        "memory_clock_is_reported_but_not_a_hard_gate"
    ]
    if "hybrid CPU lm_head" not in precedent:
        raise AssertionError("pre-existing hybrid lm_head clock rule is missing")
    if "lockstep" not in config["replay"]["method"]:
        raise AssertionError("source protocol is not the bounded hybrid lockstep method")
    if "Alternate CPU-first and CUDA-first" not in config["measurement"][
        "paired_order"
    ]:
        raise AssertionError("source protocol does not contain the CPU/GPU idle-gap control")

    clock_p50 = float(telemetry["numeric"]["memory_mhz"]["p50"])
    clock_reference = float(config["measurement"]["memory_clock_reference_mhz"])
    selection = load(absolute(config["inputs"]["selection_result"]))
    selection_head_p50 = float(selection["performance"]["best_p50_ms"])
    formal_head_p50 = float(runtime["B_cuda"]["head_ms"]["p50"])
    observed_latency_ratio = formal_head_p50 / selection_head_p50
    inverse_clock_ratio = clock_reference / clock_p50
    corrected_gates = dict(result["validity_gates"])
    corrected_gates["memory_clock_gate"] = (
        "NOT_APPLICABLE_PASS_HYBRID_LOCKSTEP_GPU_IDLE_GAPS"
    )
    nvml_used = int(runtime["B_cuda"]["max_nvml_device_used_bytes"])
    budget = int(memory["gpu_budget_bytes"])
    return {
        "schema_version": 1,
        "stage": "OP5_LM_HEAD_RECONNECT_R1_POST_RUN_ADJUDICATION",
        "timestamp": now_iso(),
        "status": FINAL_STATUS,
        "deployment_authorized": True,
        "scope": "frozen local C02 PyTorch eager lm_head CUDA mirror only",
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "gpu_rerun_performed": False,
        "source": {
            "result": str(result_path),
            "protocol": str(config_path),
            "original_status": result["status"],
            "original_result_preserved": True,
            "only_original_failed_validity_gate": failed[0],
        },
        "adjudication": {
            "type": "gate_applicability_correction",
            "reason": (
                "The lockstep method deliberately executes a roughly 30 ms CPU tied-head "
                "operation around every CUDA head measurement, creating the exact hybrid CPU "
                "lm_head GPU-idle gaps that the pre-existing OP3 rule classifies as report-only. "
                "No thermal or hardware throttle occurred, and all performance numbers already "
                "include the observed 6000 MHz memory-clock regime."
            ),
            "pre_existing_rule": {
                "path": str(op3_config_path),
                "text": precedent,
            },
            "corrected_validity_gates": corrected_gates,
            "corrected_validity_pass": True,
            "adaptive_disclosure": (
                "This applicability correction was made after observing the run. It does not "
                "rewrite the frozen result, is not a repeatability claim, and is authoritative "
                "only through this separate adjudication receipt."
            ),
        },
        "evidence": {
            "token_exact": "21705/21705",
            "requests_full_budget": "11/11",
            "cache_pass": "11/11",
            "A_counterfactual_incremental_tpot_ms": runtime["A_cpu"][
                "aggregate_incremental_tpot_ms"
            ],
            "B_counterfactual_incremental_tpot_ms": runtime["B_cuda"][
                "aggregate_incremental_tpot_ms"
            ],
            "A_over_B_speedup": speedup,
            "minimum_required_speedup": minimum_speedup,
            "selection_cuda_head_p50_ms_at_8000_mhz": selection_head_p50,
            "formal_cuda_head_p50_ms_at_6000_mhz": formal_head_p50,
            "formal_over_selection_head_latency_ratio": observed_latency_ratio,
            "inverse_clock_ratio_8000_over_6000": inverse_clock_ratio,
            "memory_clock_mhz_p50": clock_p50,
            "memory_clock_ratio_vs_reference": result[
                "memory_clock_ratio_vs_reference"
            ],
            "pstates": telemetry["pstates"],
            "telemetry_samples": result["telemetry"]["raw_jsonl"]["samples"],
            "temperature_c_max": telemetry["numeric"]["temperature_c"]["max"],
            "thermal_or_hardware_throttle": telemetry[
                "thermal_or_hardware_throttle"
            ],
            "peak_cuda_allocated_bytes": memory["peak_allocated_bytes"],
            "gpu_budget_bytes": budget,
            "max_nvml_device_used_bytes": nvml_used,
            "nvml_margin_to_7_5_gib_bytes": budget - nvml_used,
            "original_tied_cpu_storage_and_structure_unchanged": True,
        },
        "constraints": [
            "Use only with a dedicated local GPU and a runtime free-memory guard; the observed NVML margin to 7.5 GiB was narrow.",
            "Keep the original CPU tied lm_head/embedding Parameter and state_dict unchanged; the CUDA weight is an external mirror.",
            "Do not implement a custom lm_head kernel: the selection bottleneck gate failed.",
            "Do not claim cross-day/independent replication or extrapolate to board, llama.cpp, sm_100, or sm_120.",
        ],
        "next_action": (
            "Update the current ledger and close the remaining OP5 backlog: lm_head is locally "
            "authorized through this adjudication, Inductor remains a supporting baseline, and "
            "KV dequant remains blocked by authenticity."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--op3-config", type=Path, default=DEFAULT_OP3_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = absolute(args.output)
    receipt = adjudicate(
        absolute(args.result), absolute(args.config), absolute(args.op3_config)
    )
    atomic_write_json(output, receipt)
    print(f"[{now_iso()}] lm_head reconnection adjudicated: {receipt['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
