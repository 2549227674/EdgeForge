#!/usr/bin/env python3
"""Execute the no-SHA K3 C02 reconnection authorized by the R3 override."""

from __future__ import annotations

import argparse
from collections import Counter
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable

import torch

from k3_gqa_attention import autotune_choices
from k3_gqa_custom_op import (
    ATTENTION_REGISTRY_KEY,
    FULL_UPPER_BUCKETS,
    INVOCATION_TRACKER,
    K3Reconnector,
    OP_SCHEMA,
    QUALIFIED_OP_NAME,
    prewarm_candidate,
)
from load_smoke_nf4 import (
    CPU_MODULES,
    CUDA_MODULES,
    PERSISTENT_CPU_MODULES,
    get_submodule,
    materialize_persistent_cpu_modules,
    memory_snapshot,
    module_storage_snapshot,
)
from op0_common import NvidiaSmiMonitor, atomic_write_json, now_iso, summarize


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/operator/k3_gqa_reconnect_override_r4.json"
DEFAULT_PREREG = (
    ROOT
    / "results/operator/pool/k3_gqa_reconnect_r4_prereg_receipt_20260916.json"
)
DEFAULT_RESULT = ROOT / "results/operator/pool/k3_gqa_reconnect_r4_20260916.json"
DEFAULT_RECEIPT = (
    ROOT / "results/operator/pool/k3_gqa_reconnect_r4_receipt_20260916.json"
)
DEFAULT_PROGRESS = (
    ROOT / "results/operator/pool/k3_gqa_reconnect_r4_progress_20260916.json"
)
DEFAULT_TELEMETRY = (
    ROOT / "results/operator/pool/k3_gqa_reconnect_r4_telemetry_20260916.jsonl"
)


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def absolute(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def environment_identity() -> dict[str, Any]:
    command = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,power.max_limit",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
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
                "tokenizers",
            )
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "compute_capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None,
            "torch_cuda": torch.version.cuda,
        },
        "nvidia_smi": (command.stdout + command.stderr).strip(),
        "git_and_file_sha": "SKIPPED_BY_OWNER_DIRECTION",
    }


def validate_model_inventory_without_sha(model_path: Path) -> dict[str, Any]:
    manifest_path = model_path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate") != "C02-S.O":
        raise AssertionError("model manifest does not identify C02-S.O")
    files = manifest.get("files_excluding_manifest")
    if not isinstance(files, list) or not files:
        raise AssertionError("model manifest has no file inventory")
    mismatches = []
    observed_bytes = 0
    for entry in files:
        path = model_path / entry["path"]
        if not path.is_file():
            mismatches.append({"path": entry["path"], "reason": "missing"})
            continue
        size = path.stat().st_size
        observed_bytes += size
        if size != int(entry["bytes"]):
            mismatches.append(
                {
                    "path": entry["path"],
                    "reason": "byte_size_mismatch",
                    "expected": int(entry["bytes"]),
                    "observed": size,
                }
            )
    if mismatches:
        raise AssertionError(f"model inventory mismatch without SHA: {mismatches[:8]}")
    return {
        "mode": "presence_and_byte_size_only",
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "manifest_path": str(manifest_path),
        "file_count": len(files),
        "total_bytes": observed_bytes,
        "status": "PASS",
    }


def load_tapes_without_sha(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("tapes", [])
    if len(entries) != 5:
        raise AssertionError(f"expected 5 tapes, observed {len(entries)}")
    tapes: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    for entry in entries:
        tape_path = ROOT / entry["path"]
        tape = json.loads(tape_path.read_text(encoding="utf-8"))
        if tape.get("tape_id") != entry.get("tape_id"):
            raise AssertionError(f"tape ID mismatch: {tape_path}")
        requests = tape.get("requests", [])
        if len(requests) != int(entry["requests"]):
            raise AssertionError(f"request count mismatch: {tape_path}")
        for request in requests:
            request_id = request["request_id"]
            if request_id in request_ids:
                raise AssertionError(f"duplicate request ID: {request_id}")
            request_ids.add(request_id)
        tapes.append({"entry": entry, "tape": tape, "path": str(tape_path)})
    if len(request_ids) != 11:
        raise AssertionError(f"expected 11 requests, observed {len(request_ids)}")
    return tapes


def load_qualified_baseline(path: Path) -> dict[str, dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise AssertionError("inherited OP3 baseline result is not PASS")
    if result.get("gates", {}).get("baseline_self_repeat_tokens") != "PASS":
        raise AssertionError("inherited OP3 A1/A2 baseline self-repeat is not PASS")
    records = [
        record
        for record in result.get("records", [])
        if record.get("path") == "A" and int(record.get("pass", 0)) == 2
    ]
    if len(records) != 11:
        raise AssertionError(f"expected 11 archived A2 records, observed {len(records)}")
    by_request = {record["request_id"]: record for record in records}
    if len(by_request) != 11:
        raise AssertionError("archived A2 request IDs are not unique")
    for record in records:
        if len(record["generated_token_ids"]) != int(record["completion_tokens"]):
            raise AssertionError(f"archived token length mismatch: {record['request_id']}")
    return by_request


def validate_inputs(config: dict[str, Any], prereg_path: Path) -> dict[str, Any]:
    if config.get("stage") != "OP5_K3_RECONNECT_R4":
        raise AssertionError("unexpected R4 config identity")
    if config.get("authority", {}).get("sha_verification") != "SKIPPED_BY_OWNER_DIRECTION":
        raise AssertionError("R4 must retain the owner no-SHA direction")
    if config.get("no_sha_policy", {}).get("calculate_or_compare_sha") is not False:
        raise AssertionError("R4 no-SHA policy is not explicit")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise AssertionError("R4 requires the frozen sm_89 CUDA device")

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_BEFORE_FORMAL_REPLAY":
        raise AssertionError("R4 preregistration receipt is not frozen")
    if prereg.get("formal_candidate_replay_started") is not False:
        raise AssertionError("R4 preregistration was not captured before replay")

    ncu_path = absolute(config["authority"]["upstream_ncu_receipt"])
    ncu = json.loads(ncu_path.read_text(encoding="utf-8"))
    expected_ncu = "K3_R3_NCU_PASS_RECONNECTION_AUTHORIZED"
    if ncu.get("status") != expected_ncu:
        raise AssertionError(f"upstream ncu status is not {expected_ncu}")

    model_path = absolute(config["inputs"]["model"])
    inventory = validate_model_inventory_without_sha(model_path)
    tapes = load_tapes_without_sha(absolute(config["inputs"]["tapes_manifest"]))
    census = json.loads(
        absolute(config["inputs"]["shape_census"]).read_text(encoding="utf-8")
    )
    if census.get("status") != "PASS":
        raise AssertionError("shape census is not PASS")
    baseline = load_qualified_baseline(absolute(config["inputs"]["baseline_result"]))
    return {
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "preregistration": {
            "path": str(prereg_path),
            "status": prereg["status"],
        },
        "ncu": {"path": str(ncu_path), "status": ncu["status"]},
        "model_inventory": inventory,
        "tapes": tapes,
        "census": census,
        "baseline": baseline,
    }


def load_carrier_without_sha(
    config: dict[str, Any], device: int
) -> tuple[Any, Any, dict[str, Any]]:
    import bitsandbytes as bnb
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_path = absolute(config["inputs"]["model"])
    seed = int(config["replay"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    device_map: dict[str, int | str] = {
        **{name: "cpu" for name in CPU_MODULES},
        **{name: device for name in CUDA_MODULES},
    }
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    log("loading C02 once as NF4 (SHA verification skipped by owner direction)")
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    cpu_materialization = materialize_persistent_cpu_modules(model, torch)

    active_quantized = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("model.language_model.")
        and isinstance(module, bnb.nn.Linear4bit)
    ]
    expected = int(
        config["carrier"]["quantization"]["expected_language_linear4bit_modules"]
    )
    quant_types = Counter(
        str(getattr(getattr(module.weight, "quant_state", None), "quant_type", None))
        for _, module in active_quantized
    )
    weight_devices = Counter(str(module.weight.device) for _, module in active_quantized)
    if len(active_quantized) != expected or set(quant_types) != {"nf4"}:
        raise AssertionError(
            f"NF4 topology mismatch: count={len(active_quantized)}, types={quant_types}"
        )
    if any(not value.startswith(f"cuda:{device}") for value in weight_devices):
        raise AssertionError(f"active NF4 weight device drift: {weight_devices}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer, {
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "load_seconds": load_seconds,
        "load_count": 1,
        "device_map_requested": device_map,
        "hf_device_map_observed": getattr(model, "hf_device_map", None),
        "cpu_materialization": cpu_materialization,
        "quantization": {
            "active_language_linear4bit_count": len(active_quantized),
            "quant_types": dict(sorted(quant_types.items())),
            "weight_devices": dict(sorted(weight_devices.items())),
        },
        "memory_after_load": memory_snapshot(torch, device),
    }


def render_requests(
    tokenizer: Any,
    tapes: list[dict[str, Any]],
    census: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    census_by_request = {
        row["request_id"]: row for row in census["token_census"]["requests"]
    }
    rendered: list[dict[str, Any]] = []
    for tape_record in tapes:
        tape = tape_record["tape"]
        for request_number, request in enumerate(tape["requests"], start=1):
            request_id = request["request_id"]
            encoded = tokenizer.apply_chat_template(
                request["messages"], tokenize=True, add_generation_prompt=True
            )
            token_ids = [int(value) for value in encoded["input_ids"]]
            census_row = census_by_request.get(request_id)
            reference = baseline.get(request_id)
            if census_row is None or reference is None:
                raise AssertionError(f"request missing from frozen evidence: {request_id}")
            if len(token_ids) != int(census_row["c02_rendered_prompt_tokens"]):
                raise AssertionError(f"rendered prompt length drift: {request_id}")
            if len(token_ids) != int(reference["rendered_prompt_tokens"]):
                raise AssertionError(f"rendered prompt differs from archived A2 length: {request_id}")
            completion = int(request["reference_completion_tokens"])
            if completion != int(census_row["b0_reference_completion_tokens"]):
                raise AssertionError(f"completion budget differs from census: {request_id}")
            if completion != int(reference["completion_tokens"]):
                raise AssertionError(f"completion budget differs from archived A2: {request_id}")
            rendered.append(
                {
                    "tape_id": tape["tape_id"],
                    "task_id": tape["task_id"],
                    "request_number": request_number,
                    "request_id": request_id,
                    "rendered_token_ids": token_ids,
                    "rendered_prompt_tokens": len(token_ids),
                    "completion_tokens": completion,
                    "baseline_generated_token_ids": [
                        int(value) for value in reference["generated_token_ids"]
                    ],
                }
            )
    if len(rendered) != 11:
        raise AssertionError(f"rendered request count is {len(rendered)}, expected 11")
    if sum(row["completion_tokens"] for row in rendered) != 21705:
        raise AssertionError("rendered completion budget total is not 21705")
    return rendered


def tensor_identity(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        id(value),
        value.data_ptr() if value.device.type != "meta" else None,
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        str(value.device),
    )


def quant_state_identity(module: Any) -> tuple[Any, ...]:
    state = module.weight.quant_state
    state2 = getattr(state, "state2", None)
    tensors = [
        getattr(state, "absmax", None),
        getattr(state, "code", None),
        getattr(state2, "absmax", None),
        getattr(state2, "code", None),
    ]
    return (
        id(state),
        bool(getattr(state, "nested", False)),
        getattr(state, "blocksize", None),
        getattr(state, "quant_type", None),
        id(state2) if state2 is not None else None,
        getattr(state2, "blocksize", None) if state2 is not None else None,
        tuple(tensor_identity(value) if isinstance(value, torch.Tensor) else None for value in tensors),
    )


def model_structure_snapshot(model: Any) -> dict[str, Any]:
    import bitsandbytes as bnb

    modules = {
        name: (f"{type(module).__module__}.{type(module).__qualname__}", id(module))
        for name, module in model.named_modules()
    }
    parameters = {
        name: tensor_identity(value) for name, value in model.named_parameters()
    }
    state_dict_keys = tuple(model.state_dict().keys())
    nf4 = {
        name: (
            id(module),
            tensor_identity(module.weight),
            quant_state_identity(module),
        )
        for name, module in model.named_modules()
        if name.startswith("model.language_model.")
        and isinstance(module, bnb.nn.Linear4bit)
    }
    persistent_cpu = {
        name: module_storage_snapshot(get_submodule(model, name))
        for name in PERSISTENT_CPU_MODULES
    }
    attention_configs = {
        name: (
            id(module.config),
            module.config._attn_implementation,
            int(module.layer_idx),
            int(module.head_dim),
            bool(module.is_sliding),
            bool(module.is_kv_shared_layer),
        )
        for name, module in model.named_modules()
        if name.startswith("model.language_model.")
        and type(module).__name__ == "Gemma4TextAttention"
    }
    return {
        "modules": modules,
        "parameters": parameters,
        "state_dict_keys": state_dict_keys,
        "nf4": nf4,
        "persistent_cpu": persistent_cpu,
        "hf_device_map": getattr(model, "hf_device_map", None),
        "attention_configs": attention_configs,
    }


def compare_named_mapping(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(
        name for name in set(before) & set(after) if before[name] != after[name]
    )
    return {
        "before_count": len(before),
        "after_count": len(after),
        "missing": missing,
        "added": added,
        "changed": changed,
        "pass": not missing and not added and not changed,
    }


def compare_model_structure(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "modules": compare_named_mapping(before["modules"], after["modules"]),
        "parameters": compare_named_mapping(
            before["parameters"], after["parameters"]
        ),
        "nf4": compare_named_mapping(before["nf4"], after["nf4"]),
        "attention_configs": compare_named_mapping(
            before["attention_configs"], after["attention_configs"]
        ),
        "state_dict_keys": {
            "before_count": len(before["state_dict_keys"]),
            "after_count": len(after["state_dict_keys"]),
            "exact_ordered_equality": before["state_dict_keys"]
            == after["state_dict_keys"],
        },
        "persistent_cpu": {
            "exact_equality": before["persistent_cpu"] == after["persistent_cpu"]
        },
        "hf_device_map": {
            "exact_equality": before["hf_device_map"] == after["hf_device_map"]
        },
    }
    passed = (
        checks["modules"]["pass"]
        and checks["parameters"]["pass"]
        and checks["nf4"]["pass"]
        and checks["attention_configs"]["pass"]
        and checks["state_dict_keys"]["exact_ordered_equality"]
        and checks["persistent_cpu"]["exact_equality"]
        and checks["hf_device_map"]["exact_equality"]
    )
    return {
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "comparison": "direct structural equality",
        "checks": checks,
        "pass": passed,
    }


def cache_structure(
    cache: Any, attention_modules: dict[int, Any], expected_sequence_length: int
) -> dict[str, Any]:
    layers = list(getattr(cache, "layers", []))
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    persistent_attention_modules = {
        layer_index: module
        for layer_index, module in attention_modules.items()
        if not bool(module.is_kv_shared_layer)
    }
    shared_attention_modules = {
        layer_index: module
        for layer_index, module in attention_modules.items()
        if bool(module.is_kv_shared_layer)
    }
    if len(layers) != len(persistent_attention_modules):
        errors.append(
            "cache layer count "
            f"{len(layers)} != persistent attention layer count "
            f"{len(persistent_attention_modules)}"
        )
    expected_persistent_indices = list(range(len(layers)))
    if sorted(persistent_attention_modules) != expected_persistent_indices:
        errors.append(
            "persistent cache layer indices do not match DynamicCache indices: "
            f"{sorted(persistent_attention_modules)} != {expected_persistent_indices}"
        )
    for layer_index, layer in enumerate(layers):
        module = persistent_attention_modules.get(layer_index)
        if module is None:
            errors.append(
                f"no persistent attention module for cache layer {layer_index}"
            )
            continue
        initialized = bool(getattr(layer, "is_initialized", False))
        record: dict[str, Any] = {
            "layer": layer_index,
            "cache_type": type(layer).__name__,
            "is_sliding": bool(module.is_sliding),
            "is_kv_shared_layer": False,
            "initialized": initialized,
            "logical_sequence_length": int(layer.get_seq_length()),
        }
        if not initialized:
            errors.append(f"non-shared cache layer {layer_index} is uninitialized")
            records.append(record)
            continue
        keys = layer.keys
        values = layer.values
        record["keys"] = {
            "shape": list(keys.shape),
            "dtype": str(keys.dtype),
            "device": str(keys.device),
            "contiguous": bool(keys.is_contiguous()),
        }
        record["values"] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "device": str(values.device),
            "contiguous": bool(values.is_contiguous()),
        }
        expected_stored = (
            min(expected_sequence_length, int(module.sliding_window) - 1)
            if module.is_sliding
            else expected_sequence_length
        )
        expected_shape = (1, 2, expected_stored, int(module.head_dim))
        if tuple(keys.shape) != expected_shape or tuple(values.shape) != expected_shape:
            errors.append(
                f"cache geometry mismatch layer {layer_index}: "
                f"{tuple(keys.shape)}/{tuple(values.shape)} != {expected_shape}"
            )
        if keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
            errors.append(f"cache dtype mismatch layer {layer_index}")
        if keys.device.type != "cuda" or values.device.type != "cuda":
            errors.append(f"cache device mismatch layer {layer_index}")
        if int(layer.get_seq_length()) != expected_sequence_length:
            errors.append(f"logical cache length mismatch layer {layer_index}")
        records.append(record)
    return {
        "cache_type": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "layer_count": len(layers),
        "attention_module_count": len(attention_modules),
        "persistent_attention_layer_count": len(persistent_attention_modules),
        "kv_shared_attention_layer_count": len(shared_attention_modules),
        "expected_sequence_length": expected_sequence_length,
        "records": records,
        "errors": errors,
        "pass": not errors,
    }


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started_at = now_iso()
        self.completed: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.state = "INITIALIZING"

    def write(self, state: str | None = None) -> None:
        if state is not None:
            self.state = state
        atomic_write_json(
            self.path,
            {
                "schema_version": 1,
                "stage": "OP5_K3_RECONNECT_R4",
                "status": self.state,
                "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
                "started_at": self.started_at,
                "updated_at": now_iso(),
                "completed_requests": len(self.completed),
                "total_requests": 11,
                "current": self.current,
                "completed": self.completed,
            },
        )

    def token_progress(self, value: dict[str, Any]) -> None:
        self.current = value
        self.write("RUNNING")

    def complete(self, record: dict[str, Any]) -> None:
        self.completed.append(
            {
                "tape_id": record["tape_id"],
                "request_id": record["request_id"],
                "completion_tokens": record["completion_tokens"],
                "exact_token_match": record["token_comparison"]["identical"],
                "candidate_calls": record["attention_dispatch"]["candidate_calls"],
                "total_seconds": record["total_seconds"],
                "completed_at": now_iso(),
            }
        )
        self.current = None
        self.write("RUNNING")


def first_divergence(left: list[int], right: list[int]) -> int | None:
    common = min(len(left), len(right))
    for index in range(common):
        if left[index] != right[index]:
            return index
    return common if len(left) != len(right) else None


def compare_token_lists(
    baseline: list[int], candidate: list[int], prompt_tokens: int
) -> dict[str, Any]:
    first = first_divergence(baseline, candidate)
    result: dict[str, Any] = {
        "comparison": "archived qualified OP3 A2 vs K3 R4",
        "left_length": len(baseline),
        "right_length": len(candidate),
        "identical": first is None,
        "matching_prefix_tokens": first if first is not None else len(baseline),
        "first_token_identical": bool(
            baseline and candidate and baseline[0] == candidate[0]
        ),
        "first_divergence": None,
    }
    if first is not None:
        start = max(0, first - 8)
        result["first_divergence"] = {
            "generated_index_zero_based": first,
            "absolute_context_position_zero_based": prompt_tokens + first,
            "baseline_token_id": baseline[first] if first < len(baseline) else None,
            "candidate_token_id": candidate[first] if first < len(candidate) else None,
            "baseline_prefix_context": baseline[start:first],
            "candidate_prefix_context": candidate[start:first],
        }
    return result


def run_request(
    model: Any,
    request: dict[str, Any],
    *,
    device: int,
    seed: int,
    progress: ProgressWriter,
    progress_interval: int,
    attention_modules: dict[int, Any],
) -> dict[str, Any]:
    text_model = get_submodule(model, "model.language_model")
    token_embedding = text_model.embed_tokens
    per_layer_embedding = text_model.embed_tokens_per_layer
    output_head = model.lm_head
    cuda_device = torch.device(f"cuda:{device}")
    num_layers = int(text_model.config.num_hidden_layers)
    per_layer_width = int(text_model.config.hidden_size_per_layer_input)
    softcap = text_model.config.final_logit_softcapping
    completion_tokens = int(request["completion_tokens"])
    prompt_ids = request["rendered_token_ids"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
    attention_mask = torch.ones_like(input_ids, device="cpu")
    current_ids = input_ids
    past_key_values = None
    generated: list[int] = []
    step_seconds: list[float] = []
    embedding_cuda_deltas: list[int] = []
    eos_positions: list[int] = []
    raw_eos_ids = text_model.config.eos_token_id
    eos_ids = (
        {int(raw_eos_ids)}
        if isinstance(raw_eos_ids, int)
        else {int(value) for value in raw_eos_ids}
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    request_started = time.perf_counter()
    INVOCATION_TRACKER.begin_run(
        request_id=request["request_id"], tape_id=request["tape_id"]
    )
    try:
        for token_index in range(completion_tokens):
            phase = "prefill" if token_index == 0 else "decode"
            INVOCATION_TRACKER.set_phase(phase)
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            allocation_before_embedding = torch.cuda.memory_allocated(device)
            inputs_embeds_cpu = token_embedding(current_ids)
            per_layer_inputs_cpu = per_layer_embedding(current_ids).reshape(
                *current_ids.shape, num_layers, per_layer_width
            )
            allocation_after_embedding = torch.cuda.memory_allocated(device)
            embedding_delta = allocation_after_embedding - allocation_before_embedding
            embedding_cuda_deltas.append(int(embedding_delta))
            if embedding_delta != 0:
                raise AssertionError(
                    f"CPU embedding lookup changed CUDA allocation: {embedding_delta}"
                )
            inputs_embeds = inputs_embeds_cpu.to(cuda_device)
            per_layer_inputs = per_layer_inputs_cpu.to(cuda_device)
            step_attention_mask = attention_mask.to(cuda_device)
            del inputs_embeds_cpu, per_layer_inputs_cpu
            with torch.inference_mode():
                output = text_model(
                    inputs_embeds=inputs_embeds,
                    per_layer_inputs=per_layer_inputs,
                    attention_mask=step_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=False,
                )
                past_key_values = output.past_key_values
                last_hidden_cpu = output.last_hidden_state[:, -1:, :].to("cpu")
                logits = output_head(last_hidden_cpu)
                if softcap is not None:
                    logits = torch.tanh(logits / softcap) * softcap
            if not bool(torch.isfinite(last_hidden_cpu).all().item()):
                raise FloatingPointError("non-finite final hidden state")
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("non-finite logits")
            next_token = int(logits[0, -1].argmax().item())
            generated.append(next_token)
            if next_token in eos_ids:
                eos_positions.append(token_index)
            current_ids = torch.tensor([[next_token]], dtype=torch.long, device="cpu")
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype),
                ),
                dim=1,
            )
            del inputs_embeds, per_layer_inputs, step_attention_mask
            del output, last_hidden_cpu, logits
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - step_started
            step_seconds.append(elapsed)
            emitted = token_index + 1
            if (
                emitted == 1
                or emitted % progress_interval == 0
                or emitted == completion_tokens
            ):
                value = {
                    "tape_id": request["tape_id"],
                    "request_id": request["request_id"],
                    "generated_tokens": emitted,
                    "completion_tokens": completion_tokens,
                    "latest_step_seconds": elapsed,
                    "elapsed_seconds": time.perf_counter() - request_started,
                    "cache_sequence_length": int(past_key_values.get_seq_length()),
                    "captured_at": now_iso(),
                }
                progress.token_progress(value)
                log(
                    f"K3 {request['request_id']} token {emitted}/{completion_tokens}; "
                    f"step={elapsed * 1000.0:.3f} ms"
                )
        dispatch = INVOCATION_TRACKER.end_run()
    except BaseException:
        INVOCATION_TRACKER.abort_run()
        raise

    total_seconds = time.perf_counter() - request_started
    final_sequence_length = len(prompt_ids) + completion_tokens - 1
    if int(past_key_values.get_seq_length()) != final_sequence_length:
        raise AssertionError("final cache sequence length mismatch")
    cache = cache_structure(
        past_key_values, attention_modules, final_sequence_length
    )
    comparison = compare_token_lists(
        request["baseline_generated_token_ids"], generated, len(prompt_ids)
    )
    incremental_seconds = sum(step_seconds[1:])
    incremental_tokens = max(0, completion_tokens - 1)
    record = {
        "tape_id": request["tape_id"],
        "task_id": request["task_id"],
        "request_number": request["request_number"],
        "request_id": request["request_id"],
        "seed": seed,
        "greedy": True,
        "eos_does_not_truncate": True,
        "rendered_prompt_tokens": len(prompt_ids),
        "completion_tokens": completion_tokens,
        "generated_token_ids": generated,
        "eos_token_ids": sorted(eos_ids),
        "eos_generated_positions_zero_based": eos_positions,
        "token_comparison": comparison,
        "ttft_ms": step_seconds[0] * 1000.0,
        "incremental_decode_seconds": incremental_seconds,
        "incremental_decode_tokens": incremental_tokens,
        "incremental_tpot_ms": incremental_seconds / incremental_tokens * 1000.0
        if incremental_tokens
        else None,
        "incremental_tokens_per_second": incremental_tokens / incremental_seconds
        if incremental_tokens and incremental_seconds > 0
        else None,
        "total_seconds": total_seconds,
        "total_tokens_per_second": completion_tokens / total_seconds,
        "step_seconds_summary": summarize(step_seconds),
        "embedding_cuda_allocation_delta_max_bytes": max(
            embedding_cuda_deltas, default=0
        ),
        "final_cache": cache,
        "attention_dispatch": dispatch,
        "memory": memory_snapshot(torch, device),
    }
    del current_ids, input_ids, attention_mask, past_key_values
    torch.cuda.empty_cache()
    return record


def aggregate_counter(
    records: Iterable[dict[str, Any]], field: str
) -> Counter[str]:
    total: Counter[str] = Counter()
    for record in records:
        total.update(record["attention_dispatch"].get(field, {}))
    return total


def reconcile_dispatch(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    per_request = []
    for record in records:
        dispatch = record["attention_dispatch"]
        completion = int(record["completion_tokens"])
        expected_candidate = (completion - 1) * 42
        expected_total = completion * 42
        checks = {
            "total": dispatch["total_attention_calls"] == expected_total,
            "candidate": dispatch["candidate_calls"] == expected_candidate,
            "torch_op": dispatch["torch_op_calls"] == expected_candidate,
            "prefill_eager": dispatch["eager_phase_counts"] == {"prefill": 42},
            "decode_candidate": dispatch["candidate_phase_counts"]
            == {"decode": expected_candidate},
            "fallback_reason": dispatch["fallback_reason_counts"]
            == {"prefill_policy": 42},
            "input_structure": not dispatch["input_structure_violations"],
            "output_structure": not dispatch["output_structure_violations"],
            "padding": not dispatch["padding_validation_failures"],
        }
        per_request.append(
            {
                "request_id": record["request_id"],
                "expected_total_attention_calls": expected_total,
                "expected_candidate_calls": expected_candidate,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    total_attention = sum(
        record["attention_dispatch"]["total_attention_calls"] for record in records
    )
    total_candidate = sum(
        record["attention_dispatch"]["candidate_calls"] for record in records
    )
    total_eager = sum(
        record["attention_dispatch"]["eager_calls"] for record in records
    )
    total_torch_op = sum(
        record["attention_dispatch"]["torch_op_calls"] for record in records
    )
    families = aggregate_counter(records, "candidate_family_counts")
    modules = aggregate_counter(records, "candidate_module_counts")
    expected_modules = int(config["dispatch"]["expected_attention_modules"])
    expected_per_module = int(
        config["dispatch"]["expected_candidate_calls_per_module"]
    )
    gates = {
        "per_request": all(item["pass"] for item in per_request),
        "total_attention": total_attention
        == int(config["dispatch"]["expected_total_attention_calls"]),
        "total_candidate": total_candidate
        == int(config["dispatch"]["expected_candidate_calls"]),
        "total_eager": total_eager
        == int(config["dispatch"]["expected_prefill_eager_calls"]),
        "torch_op": total_torch_op
        == int(config["dispatch"]["expected_candidate_calls"]),
        "families": dict(sorted(families.items()))
        == config["dispatch"]["expected_candidate_calls_by_family"],
        "module_cardinality": len(modules) == expected_modules,
        "module_frequency": len(modules) == expected_modules
        and all(value == expected_per_module for value in modules.values()),
    }
    return {
        "expected": {
            "total_attention_calls": config["dispatch"]["expected_total_attention_calls"],
            "candidate_calls": config["dispatch"]["expected_candidate_calls"],
            "prefill_eager_calls": config["dispatch"]["expected_prefill_eager_calls"],
            "candidate_calls_by_family": config["dispatch"]["expected_candidate_calls_by_family"],
            "candidate_calls_per_module": expected_per_module,
        },
        "observed": {
            "total_attention_calls": total_attention,
            "candidate_calls": total_candidate,
            "prefill_eager_calls": total_eager,
            "torch_op_calls": total_torch_op,
            "candidate_calls_by_family": dict(sorted(families.items())),
            "candidate_module_counts": dict(sorted(modules.items())),
            "padded_calls": sum(
                record["attention_dispatch"]["padded_calls"] for record in records
            ),
        },
        "per_request": per_request,
        "gates": gates,
        "pass": all(gates.values()),
    }


def telemetry_report(monitor: NvidiaSmiMonitor, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in monitor.samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)
    summary = monitor.summary()
    return {
        "interval_seconds": monitor.interval_seconds,
        "raw_jsonl": {
            "path": str(path),
            "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
            "samples": len(monitor.samples),
        },
        "summary": {key: value for key, value in summary.items() if key != "samples"},
    }


def aggregate_runtime(records: list[dict[str, Any]]) -> dict[str, Any]:
    incremental_tokens = sum(record["incremental_decode_tokens"] for record in records)
    incremental_seconds = sum(
        record["incremental_decode_seconds"] for record in records
    )
    total_tokens = sum(record["completion_tokens"] for record in records)
    total_seconds = sum(record["total_seconds"] for record in records)
    return {
        "requests": len(records),
        "completion_tokens": total_tokens,
        "ttft_ms": summarize(record["ttft_ms"] for record in records),
        "incremental_decode_tokens": incremental_tokens,
        "incremental_decode_seconds": incremental_seconds,
        "aggregate_incremental_tpot_ms": incremental_seconds
        / incremental_tokens
        * 1000.0,
        "aggregate_incremental_tokens_per_second": incremental_tokens
        / incremental_seconds,
        "total_request_seconds": total_seconds,
        "aggregate_total_tokens_per_second": total_tokens / total_seconds,
        "peak_cuda_allocated_bytes": max(
            record["memory"]["peak_allocated_bytes"] for record in records
        ),
        "peak_cuda_reserved_bytes": max(
            record["memory"]["peak_reserved_bytes"] for record in records
        ),
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
    baseline = validated.pop("baseline")

    progress.write("LOADING_MODEL")
    model, tokenizer, carrier = load_carrier_without_sha(config, device)
    rendered = render_requests(tokenizer, tapes, census, baseline)
    reconnect = K3Reconnector(
        model, expected_modules=int(config["dispatch"]["expected_attention_modules"])
    )
    reconnect_before = reconnect.discovery_receipt()
    structure_before = model_structure_snapshot(model)
    if len(structure_before["nf4"]) != 343:
        raise AssertionError("model structure snapshot does not contain 343 NF4 modules")
    output_head = get_submodule(model, "lm_head")
    token_embedding = get_submodule(model, "model.language_model.embed_tokens")
    if output_head.weight.data_ptr() != token_embedding.weight.data_ptr():
        raise AssertionError("lm_head is not tied to the CPU token embedding")

    progress.write("PREWARMING_26_BOUNDED_SHAPES")
    log("prewarming the locked K3 op on 1 sliding + 25 full upper buckets")
    prewarm = prewarm_candidate(device)
    prewarm["autotune_choices"] = autotune_choices()
    if prewarm["case_count"] != 26:
        raise AssertionError("K3 prewarm did not cover 26 bounded shapes")
    torch.cuda.reset_peak_memory_stats(device)

    records: list[dict[str, Any]] = []
    activation_events: list[dict[str, Any]] = []
    attention_modules = {
        int(module.layer_idx): module for _, module in reconnect._modules
    }
    monitor = NvidiaSmiMonitor(
        interval_seconds=float(config["replay"]["telemetry_interval_seconds"])
    )
    monitor.set_phase("post_prewarm_pre_replay")
    monitor.start()
    replay_error: BaseException | None = None
    try:
        activation_events.append(reconnect.activate())
        progress.write("RUNNING")
        for request in rendered:
            monitor.set_phase(f"K3:{request['request_id']}")
            record = run_request(
                model,
                request,
                device=device,
                seed=int(config["replay"]["seed"]),
                progress=progress,
                progress_interval=int(config["replay"]["progress_interval_tokens"]),
                attention_modules=attention_modules,
            )
            records.append(record)
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
            log(f"replay failed after {len(records)} complete requests")

    structure_after = model_structure_snapshot(model)
    structure = compare_model_structure(structure_before, structure_after)
    reconnect_after = reconnect.discovery_receipt()
    if reconnect_before != reconnect_after:
        raise AssertionError("attention reconnection discovery changed after restoration")
    telemetry = telemetry_report(monitor, telemetry_path)
    dispatch = reconcile_dispatch(records, config)
    runtime = aggregate_runtime(records)
    token_exact_count = sum(
        record["token_comparison"]["identical"] for record in records
    )
    prefill_exact_count = sum(
        record["token_comparison"]["first_token_identical"] for record in records
    )
    cache_exact_count = sum(record["final_cache"]["pass"] for record in records)
    completion_tokens = sum(record["completion_tokens"] for record in records)
    memory_budget = int(float(config["carrier"]["gpu_budget_gib"]) * 1024**3)
    telemetry_summary = telemetry["summary"]
    gates = {
        "owner_no_sha_direction_honored": "PASS",
        "upstream_ncu_authorization": "PASS",
        "qualified_archived_baseline": "PASS",
        "formal_c02_nf4_carrier_343_modules": "PASS",
        "all_5_tapes_11_requests_full_budget": "PASS"
        if len(records) == 11 and completion_tokens == 21705
        else "FAIL",
        "prefill_first_token_exact_11_of_11": "PASS"
        if prefill_exact_count == 11
        else "FAIL",
        "all_greedy_token_ids_exact_11_of_11": "PASS"
        if token_exact_count == 11
        else "FAIL",
        "dispatch_and_invocation_reconciliation": "PASS"
        if dispatch["pass"]
        else "FAIL",
        "input_output_structure_unchanged_per_custom_op": "PASS"
        if all(
            not record["attention_dispatch"]["input_structure_violations"]
            and not record["attention_dispatch"]["output_structure_violations"]
            and not record["attention_dispatch"]["padding_validation_failures"]
            for record in records
        )
        else "FAIL",
        "model_parameter_state_nf4_config_restored": "PASS"
        if structure["pass"]
        else "FAIL",
        "kv_cache_structure_11_of_11": "PASS"
        if cache_exact_count == 11
        else "FAIL",
        "gpu_memory_below_7_5_gib": "PASS"
        if runtime["peak_cuda_allocated_bytes"] < memory_budget
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
        "stage": "OP5_K3_RECONNECT_R4",
        "kernel": "k3_gqa_decode_attention",
        "status": status,
        "hard_gates_pass": hard_pass,
        "started_at": started_at,
        "finished_at": now_iso(),
        "wall_seconds": time.perf_counter() - wall_started,
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "environment": environment_identity(),
        "protocol": {
            "path": str(config_path),
            "preregistration": str(prereg_path),
            "replication_disclosure": config["authority"]["replication_disclosure"],
        },
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
        "records": records,
        "token_correctness_summary": {
            "baseline": config["inputs"]["baseline_result"],
            "baseline_selection": config["inputs"]["baseline_selection"],
            "requests": len(records),
            "prefill_first_token_identical_requests": prefill_exact_count,
            "all_tokens_identical_requests": token_exact_count,
            "total_candidate_tokens": completion_tokens,
            "first_divergences": [
                {
                    "request_id": record["request_id"],
                    **record["token_comparison"]["first_divergence"],
                }
                for record in records
                if record["token_comparison"]["first_divergence"] is not None
            ],
        },
        "attention_dispatch": dispatch,
        "tracker": INVOCATION_TRACKER.summary(),
        "cache_summary": {
            "passing_requests": cache_exact_count,
            "total_requests": len(records),
        },
        "runtime": runtime,
        "telemetry": telemetry,
        "gpu_budget_bytes": memory_budget,
        "gates": gates,
        "limitations": [
            config["authority"]["replication_disclosure"],
            "The archived qualified OP3 A2 token lists are reused; unchanged eager is not rerun in R4.",
            "The length-bucket adapter is judged only for deployment equivalence and receives no isolated/ncu speed claim.",
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
            "requests": token["requests"],
            "completion_tokens": token["total_candidate_tokens"],
            "prefill_first_token_identical_requests": token[
                "prefill_first_token_identical_requests"
            ],
            "all_tokens_identical_requests": token[
                "all_tokens_identical_requests"
            ],
            "candidate_calls": dispatch["candidate_calls"],
            "prefill_eager_calls": dispatch["prefill_eager_calls"],
            "candidate_calls_by_family": dispatch["candidate_calls_by_family"],
            "padded_calls": dispatch["padded_calls"],
            "cache_passing_requests": result["cache_summary"]["passing_requests"],
            "aggregate_incremental_tpot_ms": runtime[
                "aggregate_incremental_tpot_ms"
            ],
            "aggregate_incremental_tokens_per_second": runtime[
                "aggregate_incremental_tokens_per_second"
            ],
            "peak_cuda_allocated_bytes": runtime["peak_cuda_allocated_bytes"],
            "telemetry_samples": result["telemetry"]["raw_jsonl"]["samples"],
            "thermal_or_hardware_throttle": result["telemetry"]["summary"]
            ["thermal_or_hardware_throttle"],
            "wall_seconds": result["wall_seconds"],
        },
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
        receipt = build_receipt(result, result_path)
        atomic_write_json(receipt_path, receipt)
        progress.write("COMPLETE")
        log(f"formal K3 reconnection finished: {result['status']}")
        return 0 if result["hard_gates_pass"] else 2
    except BaseException as exc:
        failed_at = now_iso()
        failure_path = (
            ROOT
            / "results/operator/failures"
            / ("k3_gqa_reconnect_r4_" + failed_at.replace(":", "").replace("+", "_") + ".json")
        )
        failure = {
            "schema_version": 1,
            "stage": "OP5_K3_RECONNECT_R4",
            "status": "EXECUTION_FAILURE",
            "timestamp": failed_at,
            "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "progress": str(progress.path),
        }
        atomic_write_json(failure_path, failure)
        progress.current = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failure_receipt": str(failure_path),
        }
        progress.write("FAILED")
        log(f"formal K3 reconnection failed: {failure_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
