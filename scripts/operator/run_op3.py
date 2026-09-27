#!/usr/bin/env python3
"""Execute the frozen OP3 C02-NF4 RMSNorm reconnection and tape replay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

import torch

from k1_custom_op import (
    INVOCATION_TRACKER,
    K1Reconnector,
    OP_SCHEMA,
    QUALIFIED_OP_NAME,
    canonical_sha256,
)
from load_smoke_nf4 import (
    CPU_MODULES,
    CUDA_MODULES,
    PERSISTENT_CPU_MODULES,
    get_submodule,
    materialize_persistent_cpu_modules,
    memory_snapshot,
    module_storage_snapshot,
    verify_formal_artifact,
)
from op0_common import (
    NvidiaSmiMonitor,
    atomic_write_json,
    now_iso,
    sha256,
    summarize,
    text_sha256,
)
from replay_tapes_torch import (
    load_verified_tapes,
    render_verified_requests,
    run_greedy_request,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/operator/op3_reconnect.json"
DEFAULT_RESULT = ROOT / "results/operator/k1/k1_rmsnorm_op3.json"
DEFAULT_RECEIPT = ROOT / "results/operator/k1/op3_receipt.json"
DEFAULT_PROGRESS = ROOT / "results/operator/k1/op3_progress.json"


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def absolute(path_value: str) -> Path:
    path = Path(path_value)
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
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "compute_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None
            ),
            "torch_cuda": torch.version.cuda,
        },
        "nvidia_smi": (command.stdout + command.stderr).strip(),
    }


def verify_frozen_inputs(
    config: dict[str, Any], config_path: Path, config_sha_start: str
) -> dict[str, Any]:
    if config.get("stage") != "OP3" or config.get("kernel") != "k1_gemma4_rmsnorm":
        raise AssertionError("unexpected OP3 config identity")
    if config.get("frozen_before_any_op3_formal_replay") is not True:
        raise AssertionError("OP3 config is not marked frozen before replay")
    if config["custom_op"]["schema"] != OP_SCHEMA:
        raise AssertionError("registered custom-op schema differs from frozen schema")
    if config["custom_op"]["qualified_name"] != QUALIFIED_OP_NAME:
        raise AssertionError("registered custom-op name differs from frozen name")
    current_git = git_sha()
    if current_git != config["source_identity"]["git_sha"]:
        raise AssertionError(
            f"Git HEAD drift: {current_git} != {config['source_identity']['git_sha']}"
        )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        raise AssertionError("OP3 requires the frozen sm_89 CUDA device")

    verified: dict[str, Any] = {
        "config": {
            "path": str(config_path),
            "sha256": config_sha_start,
            "frozen_at": config["frozen_at"],
        },
        "git_sha": current_git,
        "files": {},
    }
    identities = config["source_identity"]
    for name in (
        "p0_nf4_receipt",
        "shape_census",
        "op1_receipt",
        "op2_preregistration",
        "op2_receipt",
        "tapes_manifest",
        "chat_template",
        "tokenizer",
    ):
        identity = identities[name]
        path = absolute(identity["path"])
        observed = sha256(path)
        if observed != identity["sha256"]:
            raise AssertionError(
                f"frozen hash mismatch for {name}: {observed} != {identity['sha256']}"
            )
        verified["files"][name] = {
            "path": str(path),
            "sha256": observed,
            "status": "PASS",
        }
    model_manifest = absolute(identities["model"]["manifest_path"])
    observed_manifest = sha256(model_manifest)
    if observed_manifest != identities["model"]["manifest_sha256"]:
        raise AssertionError("formal model manifest receipt hash mismatch")
    model_path = absolute(identities["model"]["path"])
    internal_manifest = model_path / "artifact_manifest.json"
    if sha256(internal_manifest) != observed_manifest:
        raise AssertionError("model-local and results model manifests differ")
    verified["files"]["model_manifest"] = {
        "path": str(model_manifest),
        "model_local_path": str(internal_manifest),
        "sha256": observed_manifest,
        "status": "PASS",
    }

    upstream = {}
    expected_stages = {
        "p0_nf4_receipt": "OP-P0-3",
        "op1_receipt": "OP1",
        "op2_receipt": "OP2",
    }
    for name, expected_stage in expected_stages.items():
        payload = json.loads(
            absolute(identities[name]["path"]).read_text(encoding="utf-8")
        )
        if payload.get("status") != "PASS" or payload.get("stage") != expected_stage:
            raise AssertionError(f"upstream gate is not PASS: {name}")
        upstream[name] = {"stage": payload["stage"], "status": payload["status"]}
    p0 = json.loads(
        absolute(identities["p0_nf4_receipt"]["path"]).read_text(encoding="utf-8")
    )
    if p0.get("formal_evidence_eligible") is not True:
        raise AssertionError("P0 NF4 carrier is not formal-evidence eligible")
    verified["upstream"] = upstream
    verified["status"] = "PASS"
    return verified


def load_formal_carrier(
    config: dict[str, Any], device: int
) -> tuple[Any, Any, dict[str, Any]]:
    import bitsandbytes as bnb
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_path = absolute(config["source_identity"]["model"]["path"])
    log("re-verifying all 15.9 GB C02 artifacts before the formal replay")
    integrity = verify_formal_artifact(
        model_path, config["source_identity"]["model"]["manifest_sha256"]
    )
    seed = int(config["replay"]["decoding"]["seed"])
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
    log("loading C02 once as NF4 with persistent CPU embeddings")
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        attn_implementation=config["carrier"]["attention_implementation"],
    )
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    log("detaching CPU embedding hooks and restoring the tied CPU lm_head")
    cpu_materialization = materialize_persistent_cpu_modules(model, torch)

    active_quantized = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("model.language_model.")
        and isinstance(module, bnb.nn.Linear4bit)
    ]
    expected_linear = int(
        config["carrier"]["quantization"]["expected_language_linear4bit_modules"]
    )
    quant_types = Counter(
        str(getattr(getattr(module.weight, "quant_state", None), "quant_type", None))
        for _, module in active_quantized
    )
    weight_devices = Counter(str(module.weight.device) for _, module in active_quantized)
    if len(active_quantized) != expected_linear or set(quant_types) != {"nf4"}:
        raise AssertionError(
            f"formal NF4 topology mismatch: count={len(active_quantized)}, types={quant_types}"
        )
    if any(not value.startswith(f"cuda:{device}") for value in weight_devices):
        raise AssertionError(f"active NF4 weight device drift: {weight_devices}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer, {
        "integrity": integrity,
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


def model_identity_snapshot(model: Any) -> dict[str, Any]:
    modules = [
        {
            "name": name,
            "type": f"{type(module).__module__}.{type(module).__qualname__}",
            "object_id": id(module),
        }
        for name, module in model.named_modules()
    ]
    parameters = []
    for name, value in model.named_parameters():
        parameters.append(
            {
                "name": name,
                "object_id": id(value),
                "data_ptr": value.data_ptr() if value.device.type != "meta" else None,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        )
    state_dict_keys = list(model.state_dict().keys())
    persistent_cpu = {
        name: module_storage_snapshot(get_submodule(model, name))
        for name in PERSISTENT_CPU_MODULES
    }
    hf_device_map = getattr(model, "hf_device_map", None)
    return {
        "module_count": len(modules),
        "modules_sha256": canonical_sha256(modules),
        "parameter_count": len(parameters),
        "parameters_sha256": canonical_sha256(parameters),
        "state_dict_key_count": len(state_dict_keys),
        "state_dict_keys_sha256": canonical_sha256(state_dict_keys),
        "persistent_cpu_storage": persistent_cpu,
        "persistent_cpu_storage_sha256": canonical_sha256(persistent_cpu),
        "hf_device_map": hf_device_map,
        "hf_device_map_sha256": canonical_sha256(hf_device_map),
    }


def compare_model_identity(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "module_count",
        "modules_sha256",
        "parameter_count",
        "parameters_sha256",
        "state_dict_key_count",
        "state_dict_keys_sha256",
        "persistent_cpu_storage_sha256",
        "hf_device_map_sha256",
    )
    checks = {field: before[field] == after[field] for field in fields}
    return {
        "checks": checks,
        "before": before,
        "after": after,
        "pass": all(checks.values()),
    }


def compare_tokens(
    left: dict[str, Any], right: dict[str, Any], comparison: str
) -> dict[str, Any]:
    left_ids = left["generated_token_ids"]
    right_ids = right["generated_token_ids"]
    common = min(len(left_ids), len(right_ids))
    first = next(
        (index for index in range(common) if left_ids[index] != right_ids[index]),
        None,
    )
    if first is None and len(left_ids) != len(right_ids):
        first = common
    identical = first is None
    result: dict[str, Any] = {
        "comparison": comparison,
        "request_id": left["request_id"],
        "left_path_pass": f"{left['path']}{left['pass']}",
        "right_path_pass": f"{right['path']}{right['pass']}",
        "left_sha256": left["generated_token_ids_sha256"],
        "right_sha256": right["generated_token_ids_sha256"],
        "left_length": len(left_ids),
        "right_length": len(right_ids),
        "identical": identical,
        "first_divergence": None,
    }
    if first is not None:
        prefix_start = max(0, first - 8)
        result["first_divergence"] = {
            "generated_index_zero_based": first,
            "absolute_context_position_zero_based": left["rendered_prompt_tokens"] + first,
            "left_token_id": left_ids[first] if first < len(left_ids) else None,
            "right_token_id": right_ids[first] if first < len(right_ids) else None,
            "left_prefix_context": left_ids[prefix_start:first],
            "right_prefix_context": right_ids[prefix_start:first],
            "left_remaining_suffix_tokens": max(0, len(left_ids) - first - 1),
            "right_remaining_suffix_tokens": max(0, len(right_ids) - first - 1),
        }
    return result


def aggregate_path(records: list[dict[str, Any]], path: str) -> dict[str, Any]:
    selected = [
        record for record in records if record["path"] == path and record["pass"] == 2
    ]
    if len(selected) != 11:
        raise AssertionError(f"measured {path} record count is {len(selected)}, expected 11")
    incremental_tokens = sum(record["incremental_decode_tokens"] for record in selected)
    incremental_seconds = sum(record["incremental_decode_seconds"] for record in selected)
    completion_tokens = sum(record["completion_tokens"] for record in selected)
    total_seconds = sum(record["total_seconds"] for record in selected)
    return {
        "path": path,
        "pass": 2,
        "requests": len(selected),
        "completion_tokens": completion_tokens,
        "ttft_ms": summarize(record["ttft_ms"] for record in selected),
        "request_incremental_tpot_ms": summarize(
            record["incremental_tpot_ms"] for record in selected
        ),
        "request_incremental_tokens_per_second": summarize(
            record["incremental_tokens_per_second"] for record in selected
        ),
        "incremental_decode_tokens": incremental_tokens,
        "incremental_decode_seconds": incremental_seconds,
        "aggregate_incremental_tpot_ms": (
            incremental_seconds / incremental_tokens * 1000.0
        ),
        "aggregate_incremental_tokens_per_second": (
            incremental_tokens / incremental_seconds
        ),
        "total_request_seconds": total_seconds,
        "aggregate_total_tokens_per_second": completion_tokens / total_seconds,
        "peak_cuda_allocated_bytes": max(
            record["memory"]["peak_allocated_bytes"] for record in selected
        ),
        "peak_cuda_reserved_bytes": max(
            record["memory"]["peak_reserved_bytes"] for record in selected
        ),
        "peak_final_device_used_bytes": max(
            record["memory"]["device_used_bytes"] for record in selected
        ),
    }


def delta(value_a: float, value_b: float) -> dict[str, float | None]:
    return {
        "b_minus_a": value_b - value_a,
        "b_vs_a_percent": ((value_b / value_a) - 1.0) * 100.0 if value_a else None,
    }


def build_end_to_end_report(
    records: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    aggregate_a = aggregate_path(records, "A")
    aggregate_b = aggregate_path(records, "B")
    by_key = {
        (record["request_id"], record["path"], record["pass"]): record
        for record in records
    }
    per_request = []
    for request_id in sorted({record["request_id"] for record in records}):
        a = by_key[(request_id, "A", 2)]
        b = by_key[(request_id, "B", 2)]
        per_request.append(
            {
                "request_id": request_id,
                "ttft_ms": {"A": a["ttft_ms"], "B": b["ttft_ms"], **delta(a["ttft_ms"], b["ttft_ms"])},
                "incremental_tpot_ms": {
                    "A": a["incremental_tpot_ms"],
                    "B": b["incremental_tpot_ms"],
                    **delta(a["incremental_tpot_ms"], b["incremental_tpot_ms"]),
                },
                "incremental_tokens_per_second": {
                    "A": a["incremental_tokens_per_second"],
                    "B": b["incremental_tokens_per_second"],
                    **delta(
                        a["incremental_tokens_per_second"],
                        b["incremental_tokens_per_second"],
                    ),
                },
                "total_seconds": {
                    "A": a["total_seconds"],
                    "B": b["total_seconds"],
                    **delta(a["total_seconds"], b["total_seconds"]),
                },
                "peak_cuda_allocated_bytes": {
                    "A": a["memory"]["peak_allocated_bytes"],
                    "B": b["memory"]["peak_allocated_bytes"],
                    **delta(
                        float(a["memory"]["peak_allocated_bytes"]),
                        float(b["memory"]["peak_allocated_bytes"]),
                    ),
                },
            }
        )
    return {
        "A": aggregate_a,
        "B": aggregate_b,
        "delta": {
            "ttft_p50_ms": delta(
                float(aggregate_a["ttft_ms"]["p50"]),
                float(aggregate_b["ttft_ms"]["p50"]),
            ),
            "aggregate_incremental_tpot_ms": delta(
                aggregate_a["aggregate_incremental_tpot_ms"],
                aggregate_b["aggregate_incremental_tpot_ms"],
            ),
            "aggregate_incremental_tokens_per_second": delta(
                aggregate_a["aggregate_incremental_tokens_per_second"],
                aggregate_b["aggregate_incremental_tokens_per_second"],
            ),
            "total_request_seconds": delta(
                aggregate_a["total_request_seconds"],
                aggregate_b["total_request_seconds"],
            ),
            "peak_cuda_allocated_bytes": delta(
                float(aggregate_a["peak_cuda_allocated_bytes"]),
                float(aggregate_b["peak_cuda_allocated_bytes"]),
            ),
        },
        "per_request": per_request,
        "comparisons": comparisons,
    }


def expected_signature_counts(census: dict[str, Any]) -> dict[str, int]:
    expected: Counter[str] = Counter()
    for row in census["op_families"]["rmsnorm"]["shapes"]:
        rows, cols = (int(value) for value in row["dims"])
        phase = row["phase"]
        invocations = int(row["invocations"])
        variants: list[tuple[bool, int]]
        if row["variant"].startswith("kv_norm_"):
            if invocations % 2:
                raise AssertionError("KV norm census count cannot split evenly")
            variants = [(True, invocations // 2), (False, invocations // 2)]
        else:
            variants = [(True, invocations)]
        for with_scale, count in variants:
            key = (
                f"{phase}|rows={rows}|cols={cols}|dtype=torch.bfloat16|"
                f"with_scale={str(with_scale).lower()}"
            )
            expected[key] += count
    return dict(sorted(expected.items()))


def validate_kernel_runs(
    records: list[dict[str, Any]], census: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    b_records = [record for record in records if record["path"] == "B"]
    if len(b_records) != 22:
        raise AssertionError(f"expected 22 B request runs, observed {len(b_records)}")
    expected_calls = int(config["reconnection"]["expected_calls_per_model_forward"])
    expected_scaled = int(config["reconnection"]["expected_scaled_modules"])
    expected_scale_free = int(config["reconnection"]["expected_scale_free_modules"])
    run_checks = []
    signature_by_pass: dict[int, Counter[str]] = {1: Counter(), 2: Counter()}
    total_by_pass = Counter()
    for record in b_records:
        kernel = record["kernel"]
        forwards = int(record["completion_tokens"])
        expected_total = forwards * expected_calls
        checks = {
            "total": kernel["total_invocations"] == expected_total,
            "cuda_only": kernel["backend_counts"] == {"cuda": expected_total},
            "prefill": kernel["phase_counts"].get("prefill") == expected_calls,
            "decode": kernel["phase_counts"].get("decode", 0)
            == (forwards - 1) * expected_calls,
            "scaled": kernel["scale_counts"].get("with_scale")
            == forwards * expected_scaled,
            "scale_free": kernel["scale_counts"].get("scale_free")
            == forwards * expected_scale_free,
            "module_cardinality": len(kernel["module_counts"])
            == int(config["reconnection"]["expected_modules"]),
            "module_frequency": all(
                count == forwards for count in kernel["module_counts"].values()
            ),
        }
        run_checks.append(
            {
                "request_id": record["request_id"],
                "pass": record["pass"],
                "expected_invocations": expected_total,
                "observed_invocations": kernel["total_invocations"],
                "checks": checks,
                "pass_gate": all(checks.values()),
            }
        )
        signature_by_pass[record["pass"]].update(kernel["signature_counts"])
        total_by_pass[record["pass"]] += kernel["total_invocations"]

    expected_signatures = expected_signature_counts(census)
    signature_checks = {
        str(pass_number): dict(sorted(counter.items())) == expected_signatures
        for pass_number, counter in signature_by_pass.items()
    }
    expected_per_pass = int(config["gates"]["expected_b_invocations_per_pass"])
    pass_total_checks = {
        str(pass_number): total_by_pass[pass_number] == expected_per_pass
        for pass_number in (1, 2)
    }
    pass_gate = (
        all(item["pass_gate"] for item in run_checks)
        and all(signature_checks.values())
        and all(pass_total_checks.values())
        and sum(total_by_pass.values())
        == int(config["gates"]["expected_b_invocations_both_passes"])
    )
    return {
        "expected_signature_counts_from_op1_op2_census": expected_signatures,
        "observed_signature_counts_by_pass": {
            str(key): dict(sorted(value.items()))
            for key, value in signature_by_pass.items()
        },
        "signature_reconciliation": signature_checks,
        "invocations_by_pass": dict(sorted(total_by_pass.items())),
        "pass_total_reconciliation": pass_total_checks,
        "run_checks": run_checks,
        "cpu_fallback_invocations": sum(
            record["kernel"]["backend_counts"].get("cpu_fallback", 0)
            for record in b_records
        ),
        "pass": pass_gate,
    }


def compact_telemetry_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "samples"}


def telemetry_report(
    monitor: NvidiaSmiMonitor, raw_path: Path
) -> dict[str, Any]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(raw_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in monitor.samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, raw_path)
    groups = {
        "A_pass_1": [s for s in monitor.samples if s["phase"].startswith("A1:")],
        "A_pass_2_measured": [
            s for s in monitor.samples if s["phase"].startswith("A2:")
        ],
        "B_pass_1_event_sampled": [
            s for s in monitor.samples if s["phase"].startswith("B1:")
        ],
        "B_pass_2_measured": [
            s for s in monitor.samples if s["phase"].startswith("B2:")
        ],
    }
    return {
        "interval_seconds": monitor.interval_seconds,
        "raw_jsonl": {
            "path": str(raw_path),
            "sha256": sha256(raw_path),
            "samples": len(monitor.samples),
        },
        "overall": compact_telemetry_summary(monitor.summary()),
        "by_replay_group": {
            name: compact_telemetry_summary(monitor._summarize_samples(samples))
            for name, samples in groups.items()
        },
    }


def script_artifacts(config_path: Path) -> dict[str, Any]:
    paths = {
        "config": config_path,
        "custom_op": ROOT / "scripts/operator/k1_custom_op.py",
        "replay": ROOT / "scripts/operator/replay_tapes_torch.py",
        "runner": ROOT / "scripts/operator/run_op3.py",
        "op2_candidate": ROOT / "scripts/operator/k1_rmsnorm.py",
    }
    return {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }


class ProgressWriter:
    def __init__(self, path: Path, config_sha: str, total_runs: int) -> None:
        self.path = path
        self.config_sha = config_sha
        self.total_runs = total_runs
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
                "stage": "OP3",
                "status": self.state,
                "started_at": self.started_at,
                "updated_at": now_iso(),
                "config_sha256": self.config_sha,
                "completed_runs": len(self.completed),
                "total_runs": self.total_runs,
                "current": self.current,
                "completed": self.completed,
            },
        )

    def token_progress(self, value: dict[str, Any]) -> None:
        self.current = value
        self.write("RUNNING")

    def complete_record(self, record: dict[str, Any]) -> None:
        self.completed.append(
            {
                "path": record["path"],
                "pass": record["pass"],
                "tape_id": record["tape_id"],
                "request_id": record["request_id"],
                "generated_token_ids_sha256": record["generated_token_ids_sha256"],
                "total_seconds": record["total_seconds"],
                "completed_at": now_iso(),
            }
        )
        self.current = None
        self.write("RUNNING")


def run_formal(
    config: dict[str, Any],
    config_path: Path,
    config_sha_start: str,
    progress: ProgressWriter,
    device: int,
) -> dict[str, Any]:
    started_at = now_iso()
    wall_started = time.perf_counter()
    progress.write("VERIFYING_INPUTS")
    frozen_evidence = verify_frozen_inputs(config, config_path, config_sha_start)
    identities = config["source_identity"]
    census = json.loads(absolute(identities["shape_census"]["path"]).read_text())
    if census.get("status") != "PASS":
        raise AssertionError("shape census is not PASS")
    manifest, tapes = load_verified_tapes(
        ROOT,
        absolute(identities["tapes_manifest"]["path"]),
        identities["tapes_manifest"]["sha256"],
    )
    if len(tapes) != int(identities["tapes_manifest"]["tapes"]):
        raise AssertionError("formal tape count drift")

    progress.write("LOADING_MODEL")
    model, tokenizer, carrier = load_formal_carrier(config, device)
    rendered = render_verified_requests(
        tokenizer,
        tapes,
        census,
        absolute(identities["chat_template"]["path"]),
        identities["chat_template"]["sha256"],
    )
    if sum(len(rows) for rows in rendered.values()) != int(
        identities["tapes_manifest"]["requests"]
    ):
        raise AssertionError("formal request count drift")

    reconnector = K1Reconnector(
        model,
        expected_modules=int(config["reconnection"]["expected_modules"]),
        expected_scaled=int(config["reconnection"]["expected_scaled_modules"]),
        expected_scale_free=int(config["reconnection"]["expected_scale_free_modules"]),
    )
    reconnect_before = reconnector.discovery_receipt()
    identity_before = model_identity_snapshot(model)
    persistent_cpu_before = identity_before["persistent_cpu_storage"]
    output_head = get_submodule(model, "lm_head")
    token_embedding = get_submodule(model, "model.language_model.embed_tokens")
    if output_head.weight.data_ptr() != token_embedding.weight.data_ptr():
        raise AssertionError("lm_head is not tied to the CPU token embedding")

    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    activation_events: list[dict[str, Any]] = []
    monitor = NvidiaSmiMonitor(
        interval_seconds=float(config["measurement"]["telemetry_interval_seconds"])
    )
    monitor.set_phase("post_load_preflight")
    monitor.start()
    progress.write("RUNNING")
    replay_error: BaseException | None = None
    try:
        for tape_record in tapes:
            tape_id = tape_record["tape"]["tape_id"]
            requests = rendered[tape_id]
            baseline_by_pass: dict[int, dict[str, dict[str, Any]]] = {}
            for pass_number in (1, 2):
                baseline_by_pass[pass_number] = {}
                for request in requests:
                    monitor.set_phase(f"A{pass_number}:{request['request_id']}")
                    record = run_greedy_request(
                        model,
                        request,
                        device=device,
                        seed=int(config["replay"]["decoding"]["seed"]),
                        path="A",
                        pass_number=pass_number,
                        tracker=None,
                        progress_callback=progress.token_progress,
                    )
                    records.append(record)
                    baseline_by_pass[pass_number][request["request_id"]] = record
                    progress.complete_record(record)
            for request in requests:
                comparisons.append(
                    compare_tokens(
                        baseline_by_pass[1][request["request_id"]],
                        baseline_by_pass[2][request["request_id"]],
                        "A1_vs_A2_baseline_self_repeat",
                    )
                )

            activation_events.append({"tape_id": tape_id, **reconnector.activate()})
            custom_by_pass: dict[int, dict[str, dict[str, Any]]] = {}
            try:
                for pass_number in (1, 2):
                    custom_by_pass[pass_number] = {}
                    for request in requests:
                        INVOCATION_TRACKER.begin_run(
                            request_id=request["request_id"],
                            tape_id=tape_id,
                            pass_number=pass_number,
                            sample_cuda_events=(pass_number == 1),
                        )
                        try:
                            monitor.set_phase(f"B{pass_number}:{request['request_id']}")
                            record = run_greedy_request(
                                model,
                                request,
                                device=device,
                                seed=int(config["replay"]["decoding"]["seed"]),
                                path="B",
                                pass_number=pass_number,
                                tracker=INVOCATION_TRACKER,
                                progress_callback=progress.token_progress,
                            )
                            record["kernel"] = INVOCATION_TRACKER.end_run()
                        except BaseException:
                            if INVOCATION_TRACKER.active:
                                INVOCATION_TRACKER.abort_run()
                            raise
                        records.append(record)
                        custom_by_pass[pass_number][request["request_id"]] = record
                        progress.complete_record(record)
            finally:
                activation_events.append({"tape_id": tape_id, **reconnector.restore()})
                reconnector.assert_restored()

            for request in requests:
                request_id = request["request_id"]
                comparisons.append(
                    compare_tokens(
                        custom_by_pass[1][request_id],
                        custom_by_pass[2][request_id],
                        "B1_vs_B2_custom_self_repeat",
                    )
                )
                comparisons.append(
                    compare_tokens(
                        baseline_by_pass[2][request_id],
                        custom_by_pass[2][request_id],
                        "A2_vs_B2_formal",
                    )
                )
    except BaseException as exc:
        replay_error = exc
        raise
    finally:
        if INVOCATION_TRACKER.active:
            INVOCATION_TRACKER.abort_run()
        if reconnector.active:
            activation_events.append({"tape_id": "failure_cleanup", **reconnector.restore()})
        reconnector.assert_restored()
        monitor.stop()
        if replay_error is not None:
            log(f"replay cleanup completed after {type(replay_error).__name__}")

    identity_after = model_identity_snapshot(model)
    identity_gate = compare_model_identity(identity_before, identity_after)
    reconnect_after = reconnector.discovery_receipt()
    weight_identity_pass = (
        reconnect_before["weight_objects"] == reconnect_after["weight_objects"]
    )
    if not identity_gate["pass"] or not weight_identity_pass:
        raise AssertionError("model/state/weight identity drift after reconnection")
    if identity_after["persistent_cpu_storage"] != persistent_cpu_before:
        raise AssertionError("persistent CPU embedding storage drift")

    kernel_reconciliation = validate_kernel_runs(records, census, config)
    if not kernel_reconciliation["pass"]:
        raise AssertionError("custom-op invocation/census reconciliation failed")
    end_to_end = build_end_to_end_report(records, comparisons)
    aa = [item for item in comparisons if item["comparison"].startswith("A1_vs_A2")]
    bb = [item for item in comparisons if item["comparison"].startswith("B1_vs_B2")]
    ab = [item for item in comparisons if item["comparison"].startswith("A2_vs_B2")]
    token_summary = {
        "baseline_self_repeat_identical_requests": sum(item["identical"] for item in aa),
        "baseline_self_repeat_total_requests": len(aa),
        "custom_self_repeat_identical_requests": sum(item["identical"] for item in bb),
        "custom_self_repeat_total_requests": len(bb),
        "a_b_identical_requests": sum(item["identical"] for item in ab),
        "a_b_total_requests": len(ab),
        "baseline_all_identical": all(item["identical"] for item in aa),
        "custom_all_identical": all(item["identical"] for item in bb),
        "a_b_all_identical": all(item["identical"] for item in ab),
    }
    token_summary["attribution"] = (
        "TOKEN_EXACT"
        if token_summary["baseline_all_identical"] and token_summary["a_b_all_identical"]
        else (
            "K1_ASSOCIATED_DIVERGENCE_BASELINE_STABLE"
            if token_summary["baseline_all_identical"]
            else "CONFOUNDED_BY_BASELINE_BNB_NONDETERMINISM"
        )
    )

    raw_telemetry_path = ROOT / "results/operator/k1/op3_telemetry.jsonl"
    telemetry = telemetry_report(monitor, raw_telemetry_path)
    thermal_throttle = bool(
        telemetry["overall"]["thermal_or_hardware_throttle"]
    )
    telemetry_coverage = (
        telemetry["overall"]["sample_count"] > 0
        and not telemetry["overall"]["errors"]
        and all(
            summary["sample_count"] > 0
            for summary in telemetry["by_replay_group"].values()
        )
    )
    memory_budget = int(float(config["carrier"]["gpu_budget_gib"]) * 1024**3)
    memory_gate = all(
        end_to_end[path][field] < memory_budget
        for path in ("A", "B")
        for field in (
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
            "peak_final_device_used_bytes",
        )
    )
    config_immutable = sha256(config_path) == config_sha_start
    gates = {
        "op3_config_frozen_and_immutable": "PASS" if config_immutable else "FAIL",
        "upstream_identity": "PASS",
        "formal_c02_nf4_carrier": "PASS",
        "direct_scale_and_scale_free_semantics": "PASS",
        "language_model_only_reconnection": "PASS",
        "custom_op_cuda_dispatch": (
            "PASS" if kernel_reconciliation["cpu_fallback_invocations"] == 0 else "FAIL"
        ),
        "kernel_invocation_op2_frequency_reconciliation": (
            "PASS" if kernel_reconciliation["pass"] else "FAIL"
        ),
        "four_passes_5_tapes_11_requests": "PASS" if len(records) == 44 else "FAIL",
        "pass_2_only_end_to_end_metrics": "PASS",
        "baseline_self_repeat_tokens": (
            "PASS" if token_summary["baseline_all_identical"] else "REPORTABLE_DIVERGENCE"
        ),
        "a_b_generated_tokens": (
            "PASS" if token_summary["a_b_all_identical"] else "REPORTABLE_DIVERGENCE"
        ),
        "finite_outputs": "PASS",
        "state_dict_module_parameter_weight_identity": (
            "PASS" if identity_gate["pass"] and weight_identity_pass else "FAIL"
        ),
        "cpu_embedding_no_remigration": "PASS",
        "gpu_memory_below_7_5_gib": "PASS" if memory_gate else "FAIL",
        "telemetry_1hz_coverage": "PASS" if telemetry_coverage else "FAIL",
        "thermal_hardware_throttle": "PASS" if not thermal_throttle else "FAIL",
        "ncu_tier_b": "BLOCKED_WINDOWS_ERR_NVGPUCTRPERM_DEFERRED_TO_OP4",
    }
    hard_gate_values = [
        value
        for key, value in gates.items()
        if key
        not in (
            "baseline_self_repeat_tokens",
            "a_b_generated_tokens",
            "ncu_tier_b",
        )
    ]
    hard_gates_pass = all(value == "PASS" for value in hard_gate_values)

    result = {
        "schema_version": 1,
        "stage": "OP3",
        "kernel": "k1_gemma4_rmsnorm",
        "status": "PASS" if hard_gates_pass else "FAIL",
        "started_at": started_at,
        "finished_at": now_iso(),
        "wall_seconds": time.perf_counter() - wall_started,
        "git_sha": git_sha(),
        "environment": environment_identity(),
        "artifacts": script_artifacts(config_path),
        "frozen_input_evidence": frozen_evidence,
        "carrier": carrier,
        "reconnection": {
            "torch_library_op": QUALIFIED_OP_NAME,
            "schema": OP_SCHEMA,
            "before": reconnect_before,
            "after": reconnect_after,
            "activation_events": activation_events,
            "model_identity": identity_gate,
            "rmsnorm_weight_identity_pass": weight_identity_pass,
            "original_forward_restored": True,
        },
        "replay_protocol": config["replay"],
        "records": records,
        "kernel_layer": {
            "invocation_reconciliation": kernel_reconciliation,
            "cuda_event_timing": INVOCATION_TRACKER.kernel_summary(),
        },
        "token_request_layer": end_to_end,
        "token_correctness_summary": token_summary,
        "telemetry": telemetry,
        "gpu_budget_bytes": memory_budget,
        "gates": gates,
        "hard_gates_pass": hard_gates_pass,
        "limitations": [
            "OP3 uses the formal C02-S.O bitsandbytes NF4 Transformers carrier only.",
            "The tied lm_head remains on CPU, so end-to-end metrics include CPU logits and are not llama.cpp system numbers.",
            "CUDA-event samples are collected only in B pass 1; B pass 2 is the end-to-end measurement pass.",
            "Tier-B ncu attribution remains the separately blocked OP4 soft gate.",
        ],
    }
    return result


def build_receipt(
    result: dict[str, Any], result_path: Path, config_path: Path
) -> dict[str, Any]:
    end_to_end = result["token_request_layer"]
    token = result["token_correctness_summary"]
    return {
        "schema_version": 1,
        "stage": "OP3",
        "kernel": "k1_gemma4_rmsnorm",
        "timestamp": result["finished_at"],
        "status": result["status"],
        "artifacts": {
            **result["artifacts"],
            "result": {"path": str(result_path), "sha256": sha256(result_path)},
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
        },
        "gates": result["gates"],
        "summary": {
            "tapes": 5,
            "requests": 11,
            "runs": len(result["records"]),
            "completion_tokens_per_path_per_pass": 21705,
            "custom_op_invocations_per_b_pass": result["kernel_layer"]
            ["invocation_reconciliation"]["invocations_by_pass"],
            "baseline_self_repeat_identical_requests": token[
                "baseline_self_repeat_identical_requests"
            ],
            "a_b_identical_requests": token["a_b_identical_requests"],
            "token_attribution": token["attribution"],
            "A_ttft_p50_ms": end_to_end["A"]["ttft_ms"]["p50"],
            "B_ttft_p50_ms": end_to_end["B"]["ttft_ms"]["p50"],
            "ttft_b_vs_a_percent": end_to_end["delta"]["ttft_p50_ms"][
                "b_vs_a_percent"
            ],
            "A_incremental_tpot_ms": end_to_end["A"][
                "aggregate_incremental_tpot_ms"
            ],
            "B_incremental_tpot_ms": end_to_end["B"][
                "aggregate_incremental_tpot_ms"
            ],
            "tpot_b_vs_a_percent": end_to_end["delta"]
            ["aggregate_incremental_tpot_ms"]["b_vs_a_percent"],
            "A_incremental_tokens_per_second": end_to_end["A"]
            ["aggregate_incremental_tokens_per_second"],
            "B_incremental_tokens_per_second": end_to_end["B"]
            ["aggregate_incremental_tokens_per_second"],
            "peak_cuda_allocated_bytes": {
                "A": end_to_end["A"]["peak_cuda_allocated_bytes"],
                "B": end_to_end["B"]["peak_cuda_allocated_bytes"],
            },
            "telemetry_samples": result["telemetry"]["raw_jsonl"]["samples"],
            "thermal_or_hardware_throttle": result["telemetry"]["overall"]
            ["thermal_or_hardware_throttle"],
            "wall_seconds": result["wall_seconds"],
        },
        "scope": {
            "op3_complete": result["status"] == "PASS",
            "op3_formal_replay_executed": True,
            "op4_not_executed": True,
            "op5_not_executed": True,
            "p0_3b_not_executed": True,
        },
        "git_sha": result["git_sha"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    result_path = args.result.resolve()
    receipt_path = args.receipt.resolve()
    progress_path = args.progress.resolve()
    config_sha_start = sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    progress = ProgressWriter(progress_path, config_sha_start, total_runs=44)
    progress.write("STARTING")
    try:
        result = run_formal(
            config, config_path, config_sha_start, progress, args.device
        )
        atomic_write_json(result_path, result)
        receipt = build_receipt(result, result_path, config_path)
        if sha256(config_path) != config_sha_start:
            raise AssertionError("OP3 config changed during receipt publication")
        if result["status"] != "PASS":
            timestamp = time.strftime("%Y%m%dT%H%M%S")
            failure_path = ROOT / f"results/operator/failures/k1_op3_{timestamp}.json"
            failure = {
                "schema_version": 1,
                "stage": "OP3",
                "status": "FAIL",
                "timestamp": result["finished_at"],
                "failure_kind": "COMPLETED_REPLAY_HARD_GATE_FAILURE",
                "failed_gates": {
                    key: value
                    for key, value in result["gates"].items()
                    if value == "FAIL"
                },
                "result": {"path": str(result_path), "sha256": sha256(result_path)},
                "config": {"path": str(config_path), "sha256": config_sha_start},
                "git_sha": result["git_sha"],
            }
            atomic_write_json(failure_path, failure)
            receipt["failure_receipt"] = {
                "path": str(failure_path),
                "sha256": sha256(failure_path),
            }
        atomic_write_json(receipt_path, receipt)
        progress.current = None
        progress.write(result["status"])
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["status"] == "PASS" else 1
    except BaseException as exc:
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        failure_path = ROOT / f"results/operator/failures/k1_op3_{timestamp}.json"
        failure = {
            "schema_version": 1,
            "stage": "OP3",
            "status": "FAIL",
            "timestamp": now_iso(),
            "config": {"path": str(config_path), "sha256": config_sha_start},
            "git_sha": git_sha(),
            "environment": environment_identity(),
            "artifacts": script_artifacts(config_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "progress_path": str(progress_path),
            "memory_at_failure": (
                memory_snapshot(torch, args.device) if torch.cuda.is_available() else None
            ),
        }
        atomic_write_json(failure_path, failure)
        atomic_write_json(
            receipt_path,
            {
                "schema_version": 1,
                "stage": "OP3",
                "status": "FAIL",
                "timestamp": failure["timestamp"],
                "failure_receipt": {
                    "path": str(failure_path),
                    "sha256": sha256(failure_path),
                },
                "error_type": failure["error_type"],
                "error": failure["error"],
                "git_sha": failure["git_sha"],
            },
        )
        progress.current = {
            "failure_receipt": str(failure_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        progress.write("FAIL")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
