#!/usr/bin/env python3
"""Execute K3 R5 with a same-load full eager/K3 token oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback
from typing import Any

import torch

import run_k3_reconnect_r4 as r4
from k3_gqa_attention import autotune_choices
from k3_gqa_custom_op_r5 import (
    ATTENTION_REGISTRY_KEY,
    INVOCATION_TRACKER,
    K3Reconnector,
    OP_SCHEMA,
    QUALIFIED_OP_NAME,
    prewarm_candidate,
)
from load_smoke_nf4 import get_submodule
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso


# Reused request helpers resolve their tracker through the R4 runner module.
# Point that module global at the R5 tracker before any run begins.
r4.INVOCATION_TRACKER = INVOCATION_TRACKER


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/operator/k3_gqa_reconnect_override_r5.json"
DEFAULT_PREREG = (
    ROOT
    / "results/operator/pool/k3_gqa_reconnect_r5_prereg_receipt_20260916.json"
)
DEFAULT_RESULT = ROOT / "results/operator/pool/k3_gqa_reconnect_r5_20260916.json"
DEFAULT_RECEIPT = (
    ROOT / "results/operator/pool/k3_gqa_reconnect_r5_receipt_20260916.json"
)
DEFAULT_PROGRESS = (
    ROOT / "results/operator/pool/k3_gqa_reconnect_r5_progress_20260916.json"
)
DEFAULT_TELEMETRY = (
    ROOT / "results/operator/pool/k3_gqa_reconnect_r5_telemetry_20260916.jsonl"
)


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_inputs(
    config: dict[str, Any], prereg_path: Path
) -> dict[str, Any]:
    if config.get("stage") != "OP5_K3_RECONNECT_R5":
        raise AssertionError("unexpected R5 config identity")
    if config.get("authority", {}).get("sha_verification") != "SKIPPED_BY_OWNER_DIRECTION":
        raise AssertionError("R5 must retain the owner no-SHA direction")
    if config.get("no_sha_policy", {}).get("calculate_or_compare_sha") is not False:
        raise AssertionError("R5 no-SHA policy is not explicit")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise AssertionError("R5 requires the frozen sm_89 CUDA device")

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_FORMAL_REPLAY":
        raise AssertionError("R5 preregistration receipt is not frozen")
    if prereg.get("formal_replay_started") is not False:
        raise AssertionError("R5 preregistration was not captured before replay")

    ncu_path = absolute(config["authority"]["upstream_ncu_receipt"])
    ncu = json.loads(ncu_path.read_text(encoding="utf-8"))
    expected_ncu = "K3_R3_NCU_PASS_RECONNECTION_AUTHORIZED"
    if ncu.get("status") != expected_ncu:
        raise AssertionError(f"upstream ncu status is not {expected_ncu}")

    r4_failure_path = absolute(config["r4_invalidation"]["failure_receipt"])
    r4_failure = json.loads(r4_failure_path.read_text(encoding="utf-8"))
    if r4_failure.get("status") != "EXECUTION_FAILURE":
        raise AssertionError("R4 invalidation does not point to its execution failure")

    model_path = absolute(config["inputs"]["model"])
    inventory = r4.validate_model_inventory_without_sha(model_path)
    tapes = r4.load_tapes_without_sha(
        absolute(config["inputs"]["tapes_manifest"])
    )
    census = json.loads(
        absolute(config["inputs"]["shape_census"]).read_text(encoding="utf-8")
    )
    if census.get("status") != "PASS":
        raise AssertionError("shape census is not PASS")
    archived = r4.load_qualified_baseline(
        absolute(config["inputs"]["archived_baseline_metadata"])
    )
    return {
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "preregistration": {
            "path": str(prereg_path),
            "status": prereg["status"],
        },
        "ncu": {"path": str(ncu_path), "status": ncu["status"]},
        "r4_invalidation": {
            "path": str(r4_failure_path),
            "status": r4_failure["status"],
        },
        "model_inventory": inventory,
        "tapes": tapes,
        "census": census,
        "archived": archived,
    }


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = now_iso()
        self.completed: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.current_path = "unassigned"
        self.state = "INITIALIZING"

    def write(self, state: str | None = None) -> None:
        if state is not None:
            self.state = state
        atomic_write_json(
            self.path,
            {
                "schema_version": 1,
                "stage": "OP5_K3_RECONNECT_R5",
                "status": self.state,
                "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
                "started_at": self.started_at,
                "updated_at": now_iso(),
                "completed_runs": len(self.completed),
                "total_runs": 22,
                "current": self.current,
                "completed": self.completed,
            },
        )

    def token_progress(self, value: dict[str, Any]) -> None:
        self.current = {"path": self.current_path, **value}
        self.write("RUNNING")

    def complete(self, record: dict[str, Any]) -> None:
        entry = {
            "path": record["path"],
            "tape_id": record["tape_id"],
            "request_id": record["request_id"],
            "completion_tokens": record["completion_tokens"],
            "total_seconds": record["total_seconds"],
            "completed_at": now_iso(),
        }
        if record["path"] == "B":
            entry.update(
                {
                    "exact_a_b_token_match": record["token_comparison"][
                        "identical"
                    ],
                    "candidate_calls": record["attention_dispatch"][
                        "candidate_calls"
                    ],
                    "synthetic_zero_mask_calls": record[
                        "attention_dispatch"
                    ]["synthetic_zero_mask_calls"],
                }
            )
        self.completed.append(entry)
        self.current = None
        self.write("RUNNING")


def delta(a: float, b: float) -> dict[str, float | None]:
    return {
        "b_minus_a": b - a,
        "b_vs_a_percent": ((b / a) - 1.0) * 100.0 if a else None,
    }


def run_formal(
    config: dict[str, Any],
    config_path: Path,
    prereg_path: Path,
    progress: ProgressWriter,
    telemetry_path: Path,
    device: int,
) -> dict[str, Any]:
    started_at = now_iso()
    wall_started = time.perf_counter()
    progress.write("VALIDATING_WITHOUT_SHA")
    validated = validate_inputs(config, prereg_path)
    tapes = validated.pop("tapes")
    census = validated.pop("census")
    archived = validated.pop("archived")

    progress.write("LOADING_MODEL")
    model, tokenizer, carrier = r4.load_carrier_without_sha(config, device)
    rendered = r4.render_requests(tokenizer, tapes, census, archived)
    reconnect = K3Reconnector(
        model, expected_modules=int(config["dispatch"]["expected_attention_modules"])
    )
    reconnect_before = reconnect.discovery_receipt()
    structure_before = r4.model_structure_snapshot(model)
    if len(structure_before["nf4"]) != 343:
        raise AssertionError("model structure snapshot does not contain 343 NF4 modules")
    output_head = get_submodule(model, "lm_head")
    token_embedding = get_submodule(model, "model.language_model.embed_tokens")
    if output_head.weight.data_ptr() != token_embedding.weight.data_ptr():
        raise AssertionError("lm_head is not tied to the CPU token embedding")
    attention_modules = {
        int(module.layer_idx): module for _, module in reconnect._modules
    }

    monitor = NvidiaSmiMonitor(
        interval_seconds=float(config["replay"]["telemetry_interval_seconds"])
    )
    monitor.set_phase("post_load_pre_baseline")
    monitor.start()
    a_records: list[dict[str, Any]] = []
    b_records: list[dict[str, Any]] = []
    activation_events: list[dict[str, Any]] = []
    replay_error: BaseException | None = None
    prewarm: dict[str, Any] | None = None
    try:
        progress.write("RUNNING_A_EAGER")
        log("R5 A: starting 11 same-load eager requests")
        for request in rendered:
            progress.current_path = "A"
            monitor.set_phase(f"A:{request['request_id']}")
            record = r4.run_request(
                model,
                request,
                device=device,
                seed=int(config["replay"]["seed"]),
                progress=progress,
                progress_interval=int(config["replay"]["progress_interval_tokens"]),
                attention_modules=attention_modules,
            )
            record["path"] = "A"
            record["archived_cross_process_comparison"] = record.pop(
                "token_comparison"
            )
            a_records.append(record)
            progress.complete(record)

        progress.write("PREWARMING_26_BOUNDED_SHAPES")
        monitor.set_phase("K3_prewarm")
        log("R5: prewarming the locked K3 op on 26 bounded shapes")
        prewarm = prewarm_candidate(device)
        prewarm["autotune_choices"] = autotune_choices()
        if prewarm["case_count"] != 26:
            raise AssertionError("K3 prewarm did not cover 26 bounded shapes")
        torch.cuda.reset_peak_memory_stats(device)

        activation_events.append(reconnect.activate())
        progress.write("RUNNING_B_K3")
        log("R5 B: starting 11 same-load K3 requests")
        a_by_request = {record["request_id"]: record for record in a_records}
        for request in rendered:
            request_b = dict(request)
            request_b["baseline_generated_token_ids"] = a_by_request[
                request["request_id"]
            ]["generated_token_ids"]
            progress.current_path = "B"
            monitor.set_phase(f"B:{request['request_id']}")
            record = r4.run_request(
                model,
                request_b,
                device=device,
                seed=int(config["replay"]["seed"]),
                progress=progress,
                progress_interval=int(config["replay"]["progress_interval_tokens"]),
                attention_modules=attention_modules,
            )
            record["path"] = "B"
            record["token_comparison"][
                "comparison"
            ] = "same-load A eager vs B K3 R5"
            b_records.append(record)
            progress.complete(record)
    except BaseException as exc:
        replay_error = exc
        raise
    finally:
        if reconnect.active:
            activation_events.append(reconnect.restore())
        reconnect.assert_restored()
        monitor.set_phase("post_replay_restored")
        monitor.stop()
        if replay_error is not None:
            log(
                f"R5 replay failed after A={len(a_records)}, B={len(b_records)} complete requests"
            )

    structure_after = r4.model_structure_snapshot(model)
    structure = r4.compare_model_structure(structure_before, structure_after)
    reconnect_after = reconnect.discovery_receipt()
    if reconnect_before != reconnect_after:
        raise AssertionError("attention reconnection discovery changed after restoration")
    telemetry = r4.telemetry_report(monitor, telemetry_path)
    dispatch = r4.reconcile_dispatch(b_records, config)
    synthetic_masks = sum(
        record["attention_dispatch"]["synthetic_zero_mask_calls"]
        for record in b_records
    )
    expected_synthetic = int(config["dispatch"]["expected_synthetic_zero_masks"])
    dispatch["expected"]["synthetic_zero_masks"] = expected_synthetic
    dispatch["observed"]["synthetic_zero_masks"] = synthetic_masks
    dispatch["gates"]["synthetic_zero_masks"] = synthetic_masks == expected_synthetic
    dispatch["pass"] = all(dispatch["gates"].values())

    runtime_a = r4.aggregate_runtime(a_records)
    runtime_b = r4.aggregate_runtime(b_records)
    token_exact_count = sum(
        record["token_comparison"]["identical"] for record in b_records
    )
    prefill_exact_count = sum(
        record["token_comparison"]["first_token_identical"]
        for record in b_records
    )
    cache_pass_count = sum(
        record["final_cache"]["pass"] for record in a_records + b_records
    )
    a_tokens = sum(record["completion_tokens"] for record in a_records)
    b_tokens = sum(record["completion_tokens"] for record in b_records)
    memory_budget = int(float(config["carrier"]["gpu_budget_gib"]) * 1024**3)
    peak_allocated = max(
        runtime_a["peak_cuda_allocated_bytes"],
        runtime_b["peak_cuda_allocated_bytes"],
    )
    telemetry_summary = telemetry["summary"]
    gates = {
        "owner_no_sha_direction_honored": "PASS",
        "upstream_ncu_authorization": "PASS",
        "r4_invalid_attempt_excluded": "PASS",
        "same_load_eager_oracle": "PASS",
        "formal_c02_nf4_carrier_343_modules": "PASS",
        "all_5_tapes_22_ab_runs_full_budget": "PASS"
        if len(a_records) == 11
        and len(b_records) == 11
        and a_tokens == 21705
        and b_tokens == 21705
        else "FAIL",
        "prefill_first_token_exact_11_of_11": "PASS"
        if prefill_exact_count == 11
        else "FAIL",
        "all_greedy_token_ids_exact_11_of_11": "PASS"
        if token_exact_count == 11
        else "FAIL",
        "dispatch_mask_and_invocation_reconciliation": "PASS"
        if dispatch["pass"]
        else "FAIL",
        "input_output_structure_unchanged_per_custom_op": "PASS"
        if all(
            not record["attention_dispatch"]["input_structure_violations"]
            and not record["attention_dispatch"]["output_structure_violations"]
            and not record["attention_dispatch"]["padding_validation_failures"]
            for record in b_records
        )
        else "FAIL",
        "model_parameter_state_nf4_config_restored": "PASS"
        if structure["pass"]
        else "FAIL",
        "kv_cache_structure_22_of_22": "PASS"
        if cache_pass_count == 22
        else "FAIL",
        "gpu_memory_below_7_5_gib": "PASS"
        if peak_allocated < memory_budget
        else "FAIL",
        "telemetry_present": "PASS"
        if int(telemetry["raw_jsonl"]["samples"]) > 0
        else "FAIL",
        "thermal_hardware_throttle_absent": "PASS"
        if not telemetry_summary["thermal_or_hardware_throttle"]
        else "FAIL",
    }
    hard_pass = all(value == "PASS" for value in gates.values())
    status = (
        config["decision"]["on_pass"]
        if hard_pass
        else config["decision"]["on_fail"]
    )
    return {
        "schema_version": 1,
        "stage": "OP5_K3_RECONNECT_R5",
        "kernel": "k3_gqa_decode_attention",
        "status": status,
        "hard_gates_pass": hard_pass,
        "started_at": started_at,
        "finished_at": now_iso(),
        "wall_seconds": time.perf_counter() - wall_started,
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "environment": r4.environment_identity(),
        "protocol": {
            "path": str(config_path),
            "preregistration": str(prereg_path),
            "replication_disclosure": config["authority"][
                "replication_disclosure"
            ],
        },
        "r4_invalidation": config["r4_invalidation"],
        "r5_changes": config["r5_changes_frozen_before_formal_replay"],
        "validated_inputs": validated,
        "carrier": carrier,
        "prewarm": prewarm,
        "reconnection": {
            "torch_library_op": QUALIFIED_OP_NAME,
            "schema": OP_SCHEMA,
            "attention_registry_key": ATTENTION_REGISTRY_KEY,
            "before": reconnect_before,
            "after": reconnect_after,
            "activation_events": activation_events,
            "restored": True,
            "model_structure": structure,
        },
        "replay_protocol": config["replay"],
        "length_bucket_adapter": config["length_bucket_adapter"],
        "records": {"A": a_records, "B": b_records},
        "token_correctness_summary": {
            "oracle": "same-load A eager vs B K3",
            "requests": len(b_records),
            "prefill_first_token_identical_requests": prefill_exact_count,
            "all_tokens_identical_requests": token_exact_count,
            "completion_tokens_per_path": b_tokens,
            "first_divergences": [
                {
                    "request_id": record["request_id"],
                    **record["token_comparison"]["first_divergence"],
                }
                for record in b_records
                if record["token_comparison"]["first_divergence"] is not None
            ],
        },
        "attention_dispatch": dispatch,
        "tracker": INVOCATION_TRACKER.summary(),
        "cache_summary": {
            "passing_runs": cache_pass_count,
            "total_runs": len(a_records) + len(b_records),
        },
        "runtime": {
            "A": runtime_a,
            "B": runtime_b,
            "delta": {
                "aggregate_incremental_tpot_ms": delta(
                    runtime_a["aggregate_incremental_tpot_ms"],
                    runtime_b["aggregate_incremental_tpot_ms"],
                ),
                "aggregate_incremental_tokens_per_second": delta(
                    runtime_a["aggregate_incremental_tokens_per_second"],
                    runtime_b["aggregate_incremental_tokens_per_second"],
                ),
                "total_request_seconds": delta(
                    runtime_a["total_request_seconds"],
                    runtime_b["total_request_seconds"],
                ),
            },
        },
        "telemetry": telemetry,
        "gpu_budget_bytes": memory_budget,
        "peak_cuda_allocated_bytes": peak_allocated,
        "gates": gates,
        "limitations": [
            config["authority"]["replication_disclosure"],
            "R5 uses one same-load A pass and one same-load B pass; it is a deployment equivalence test, not a repeatability claim.",
            "The length-bucket and None-to-zero-mask adapters are judged only by same-load deployment equivalence and receive no isolated/ncu speed claim.",
            "Results are local to the C02 NF4 sm_89 carrier and output_attentions=false decode path.",
            *config["decision"]["non_authorizations"],
        ],
    }


def build_receipt(result: dict[str, Any], result_path: Path) -> dict[str, Any]:
    token = result["token_correctness_summary"]
    dispatch = result["attention_dispatch"]["observed"]
    runtime = result["runtime"]
    return {
        "schema_version": 1,
        "stage": result["stage"],
        "kernel": result["kernel"],
        "timestamp": result["finished_at"],
        "status": result["status"],
        "hard_gates_pass": result["hard_gates_pass"],
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "result": str(result_path),
        "gates": result["gates"],
        "summary": {
            "tapes": 5,
            "requests_per_path": token["requests"],
            "runs": result["cache_summary"]["total_runs"],
            "completion_tokens_per_path": token["completion_tokens_per_path"],
            "prefill_first_token_identical_requests": token[
                "prefill_first_token_identical_requests"
            ],
            "all_tokens_identical_requests": token[
                "all_tokens_identical_requests"
            ],
            "candidate_calls": dispatch["candidate_calls"],
            "synthetic_zero_masks": dispatch["synthetic_zero_masks"],
            "prefill_eager_calls": dispatch["prefill_eager_calls"],
            "candidate_calls_by_family": dispatch["candidate_calls_by_family"],
            "padded_calls": dispatch["padded_calls"],
            "cache_passing_runs": result["cache_summary"]["passing_runs"],
            "A_incremental_tpot_ms": runtime["A"][
                "aggregate_incremental_tpot_ms"
            ],
            "B_incremental_tpot_ms": runtime["B"][
                "aggregate_incremental_tpot_ms"
            ],
            "B_vs_A_tpot_percent": runtime["delta"]
            ["aggregate_incremental_tpot_ms"]["b_vs_a_percent"],
            "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
            "telemetry_samples": result["telemetry"]["raw_jsonl"]["samples"],
            "thermal_or_hardware_throttle": result["telemetry"]["summary"]
            ["thermal_or_hardware_throttle"],
            "wall_seconds": result["wall_seconds"],
        },
        "r4_disposition": "INVALID_EXECUTION_ATTEMPT_NO_DECISION",
        "replication_disclosure": result["protocol"]["replication_disclosure"],
        "next_action": (
            "Accept K3 only for the frozen local C02 output-attentions-false decode path and update the current ledger."
            if result["hard_gates_pass"]
            else "Close the K3 deployment path at C02 reconnection; retain isolated and ncu evidence as bounded diagnostics only."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = absolute(args.config)
    prereg_path = absolute(args.prereg)
    result_path = absolute(args.result)
    receipt_path = absolute(args.receipt)
    progress = ProgressWriter(absolute(args.progress))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        result = run_formal(
            config,
            config_path,
            prereg_path,
            progress,
            absolute(args.telemetry),
            args.device,
        )
        atomic_write_json(result_path, result)
        atomic_write_json(receipt_path, build_receipt(result, result_path))
        progress.write("COMPLETE")
        log(f"formal K3 R5 reconnection finished: {result['status']}")
        return 0 if result["hard_gates_pass"] else 2
    except BaseException as exc:
        failed_at = now_iso()
        failure_path = (
            ROOT
            / "results/operator/failures"
            / ("k3_gqa_reconnect_r5_" + failed_at.replace(":", "").replace("+", "_") + ".json")
        )
        atomic_write_json(
            failure_path,
            {
                "schema_version": 1,
                "stage": "OP5_K3_RECONNECT_R5",
                "status": "EXECUTION_FAILURE",
                "timestamp": failed_at,
                "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "progress": str(progress.path),
            },
        )
        progress.current = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failure_receipt": str(failure_path),
        }
        progress.write("FAILED")
        log(f"formal K3 R5 reconnection failed: {failure_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
