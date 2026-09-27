#!/usr/bin/env python3
"""Correct the R5 KV-cache gate offline without changing raw replay evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from op0_common import atomic_write_json, now_iso


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = ROOT / "results/operator/pool/k3_gqa_reconnect_r5_20260916.json"
DEFAULT_RECEIPT = (
    ROOT
    / "results/operator/pool/k3_gqa_reconnect_r5_adjudication_receipt_20260916.json"
)
DEFAULT_MODEL_CONFIG = ROOT / "models/C02-S.O-bf16-r1/config.json"


def absolute(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def cache_signature(cache: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            record["layer"],
            record["cache_type"],
            record["is_sliding"],
            record["is_kv_shared_layer"],
            record["initialized"],
            record["logical_sequence_length"],
            tuple(record["keys"]["shape"]),
            tuple(record["values"]["shape"]),
            record["keys"]["dtype"],
            record["values"]["dtype"],
            record["keys"]["device"],
            record["values"]["device"],
        )
        for record in cache["records"]
    ]


def validate_cache(
    cache: dict[str, Any], text_config: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    hidden_layers = int(text_config["num_hidden_layers"])
    shared_layers = int(text_config["num_kv_shared_layers"])
    expected_layers = hidden_layers - shared_layers
    sliding_window = int(text_config["sliding_window"])
    layer_types = text_config["layer_types"]
    per_layer = text_config.get("per_layer_config", {})
    default_head_dim = int(text_config["head_dim"])

    if cache["layer_count"] != expected_layers:
        errors.append(
            f"layer_count={cache['layer_count']} expected={expected_layers}"
        )
    if len(cache["records"]) != expected_layers:
        errors.append(
            f"record_count={len(cache['records'])} expected={expected_layers}"
        )
    expected_sequence_length = int(cache["expected_sequence_length"])
    for layer_index, record in enumerate(cache["records"]):
        layer_type = layer_types[layer_index]
        is_sliding = layer_type == "sliding_attention"
        head_dim = int(
            per_layer.get(f"{layer_index:02d}", {}).get(
                "head_dim", default_head_dim
            )
        )
        stored_length = (
            min(expected_sequence_length, sliding_window - 1)
            if is_sliding
            else expected_sequence_length
        )
        expected_shape = [1, 2, stored_length, head_dim]
        if record["layer"] != layer_index:
            errors.append(f"layer {layer_index}: index mismatch")
        if bool(record["is_sliding"]) != is_sliding:
            errors.append(f"layer {layer_index}: sliding classification mismatch")
        if bool(record["is_kv_shared_layer"]):
            errors.append(f"layer {layer_index}: persistent layer marked shared")
        if not bool(record["initialized"]):
            errors.append(f"layer {layer_index}: uninitialized")
        if int(record["logical_sequence_length"]) != expected_sequence_length:
            errors.append(f"layer {layer_index}: logical length mismatch")
        for field in ("keys", "values"):
            tensor = record[field]
            if tensor["shape"] != expected_shape:
                errors.append(f"layer {layer_index}: {field} shape mismatch")
            if tensor["dtype"] != "torch.bfloat16":
                errors.append(f"layer {layer_index}: {field} dtype mismatch")
            if not str(tensor["device"]).startswith("cuda:"):
                errors.append(f"layer {layer_index}: {field} device mismatch")
    return errors


def adjudicate(result_path: Path, model_config_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    text_config = model_config["text_config"]
    hidden_layers = int(text_config["num_hidden_layers"])
    shared_layers = int(text_config["num_kv_shared_layers"])
    persistent_layers = hidden_layers - shared_layers
    old_error = f"cache layer count {persistent_layers} != {hidden_layers}"

    cache_errors: dict[str, list[str]] = {}
    records = result["records"]
    for path in ("A", "B"):
        for record in records[path]:
            request_id = record["request_id"]
            cache = record["final_cache"]
            if cache["errors"] != [old_error]:
                cache_errors[f"{path}:{request_id}:raw"] = list(cache["errors"])
            errors = validate_cache(cache, text_config)
            if errors:
                cache_errors[f"{path}:{request_id}:corrected"] = errors

    pairwise_equal: dict[str, bool] = {}
    for baseline, candidate in zip(records["A"], records["B"], strict=True):
        if baseline["request_id"] != candidate["request_id"]:
            raise AssertionError("A/B request order mismatch")
        request_id = baseline["request_id"]
        pairwise_equal[request_id] = (
            cache_signature(baseline["final_cache"])
            == cache_signature(candidate["final_cache"])
        )

    corrected_cache_pass = not cache_errors and all(pairwise_equal.values())
    corrected_gates = dict(result["gates"])
    corrected_gates["kv_cache_structure_22_of_22"] = (
        "PASS" if corrected_cache_pass else "FAIL"
    )
    corrected_hard_pass = all(value == "PASS" for value in corrected_gates.values())
    if corrected_hard_pass:
        raise AssertionError("cache correction unexpectedly changed the final decision")
    if result["token_correctness_summary"]["all_tokens_identical_requests"] != 0:
        raise AssertionError("unexpected R5 token-correctness population")

    return {
        "schema_version": 1,
        "stage": "OP5_K3_RECONNECT_R5_POSTRUN_ADJUDICATION",
        "timestamp": now_iso(),
        "status": "K3_C02_RECONNECTION_FAIL_CLOSE_DEPLOYMENT_PATH",
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "source_result": str(result_path),
        "source_status": result["status"],
        "correction": {
            "field": "kv_cache_structure_22_of_22",
            "original": "FAIL (0/22)",
            "corrected": "PASS (22/22)",
            "reason": (
                "The original validator compared DynamicCache.layers against all "
                f"{hidden_layers} attention modules. The carrier declares "
                f"num_kv_shared_layers={shared_layers}, so only {persistent_layers} "
                "layers own persistent KV state."
            ),
            "model_config_facts": {
                "num_hidden_layers": hidden_layers,
                "num_kv_shared_layers": shared_layers,
                "expected_persistent_cache_layers": persistent_layers,
            },
            "raw_error_expected_on_all_runs": old_error,
            "raw_error_conformance": not cache_errors,
            "corrected_cache_runs_passing": 22 if corrected_cache_pass else 0,
            "corrected_cache_runs_total": 22,
            "a_b_pairwise_structure_equal": pairwise_equal,
            "a_b_pairwise_structure_equal_count": sum(pairwise_equal.values()),
            "validation_errors": cache_errors,
        },
        "corrected_gates": corrected_gates,
        "corrected_hard_gates_pass": corrected_hard_pass,
        "decision_unchanged": True,
        "independent_failure_basis": {
            "all_greedy_token_ids_exact_11_of_11": "FAIL",
            "identical_requests": result["token_correctness_summary"][
                "all_tokens_identical_requests"
            ],
            "requests": result["token_correctness_summary"]["requests"],
        },
        "next_action": (
            "Close the K3 deployment path at C02 reconnection; retain isolated and "
            "ncu evidence as bounded diagnostics only."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_path = absolute(args.result)
    receipt_path = absolute(args.receipt)
    receipt = adjudicate(result_path, absolute(args.model_config))
    atomic_write_json(receipt_path, receipt)
    print(receipt_path)
    print(receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
