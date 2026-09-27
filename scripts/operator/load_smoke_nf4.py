#!/usr/bin/env python3
"""Validate the C02 NF4 carrier on the local 8 GiB GPU.

The formal path verifies every transferred artifact hash, loads linear weights
as double-quantized NF4, pins large embeddings and unused multimodal towers to
CPU, and performs exactly 512 prefill tokens plus 32 greedy decode tokens.  A
single JSON receipt records transfer integrity, device placement, quantization,
generation, and peak VRAM; failures write the same audit envelope with a full
traceback.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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


EXPECTED_C02_MANIFEST_SHA256 = (
    "ee7d426d1e912a6c53cad1cd52739f26d7fff45b4a4f7429a4c08656f68dc5f2"
)
OFFLOADED_UNUSED_CPU_MODULES = (
    "model.vision_tower",
    "model.audio_tower",
    "model.embed_vision",
    "model.embed_audio",
)
PERSISTENT_CPU_MODULES = (
    "model.language_model.embed_tokens",
    "model.language_model.embed_tokens_per_layer",
    "lm_head",
)
CPU_MODULES = OFFLOADED_UNUSED_CPU_MODULES + PERSISTENT_CPU_MODULES
CUDA_MODULES = (
    "model.language_model.layers",
    "model.language_model.norm",
    "model.language_model.rotary_emb",
    "model.language_model.per_layer_model_projection",
    "model.language_model.per_layer_projection_norm",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_set_sha256(files: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return (completed.stdout + completed.stderr).strip()


def environment_identity() -> dict[str, Any]:
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
                "sentencepiece",
                "protobuf",
            )
        },
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,power.max_limit",
                "--format=csv,noheader",
            ]
        ),
    }


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def verify_formal_artifact(
    model_path: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    manifest_path = model_path / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"artifact manifest is missing: {manifest_path}")
    observed_manifest_sha256 = sha256(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "artifact manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, observed {observed_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate") != "C02-S.O":
        raise ValueError("artifact manifest does not identify C02-S.O")
    files = manifest.get("files_excluding_manifest")
    if not isinstance(files, list) or not files:
        raise ValueError("artifact manifest has no file inventory")

    verified: list[dict[str, Any]] = []
    for expected in files:
        relative = expected.get("path")
        if not isinstance(relative, str):
            raise ValueError("artifact inventory contains a non-path entry")
        path = model_path / relative
        if not path.is_file():
            raise FileNotFoundError(f"transferred artifact file is missing: {path}")
        observed = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if observed != expected:
            raise ValueError(
                f"transferred artifact differs from manifest: {relative}: "
                f"expected={expected}, observed={observed}"
            )
        verified.append(observed)

    observed_set_sha256 = artifact_set_sha256(verified)
    expected_set_sha256 = manifest.get("artifact_set_sha256_excluding_manifest")
    if observed_set_sha256 != expected_set_sha256:
        raise ValueError("transferred artifact-set identity mismatch")
    return {
        "mode": "formal_full_hash_verification",
        "manifest_path": str(manifest_path),
        "manifest_sha256": observed_manifest_sha256,
        "file_count": len(verified),
        "total_bytes_excluding_manifest": sum(item["bytes"] for item in verified),
        "artifact_set_sha256_excluding_manifest": observed_set_sha256,
        "status": "PASS",
    }


def bootstrap_identity(model_path: Path) -> dict[str, Any]:
    config = model_path / "config.json"
    weights = sorted(model_path.glob("*.safetensors"))
    if not config.is_file() or not weights:
        raise FileNotFoundError("bootstrap model lacks config or safetensors weights")
    return {
        "mode": "bootstrap_nonformal",
        "config_sha256": sha256(config),
        "weight_files": [
            {"path": path.name, "bytes": path.stat().st_size} for path in weights
        ],
        "formal_evidence_eligible": False,
        "status": "PASS",
    }


def get_submodule(root: Any, path: str) -> Any:
    current = root
    for component in path.split("."):
        current = getattr(current, component)
    return current


def module_devices(module: Any) -> list[str]:
    devices = {str(value.device) for value in module.parameters(recurse=True)}
    devices.update(str(value.device) for value in module.buffers(recurse=True))
    return sorted(devices)


def module_storage_snapshot(module: Any) -> dict[str, dict[str, Any]]:
    """Describe direct tensor storage without walking into unrelated children."""
    tensors = list(module.named_parameters(recurse=False))
    tensors.extend(module.named_buffers(recurse=False))
    return {
        name: {
            "device": str(value.device),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "numel": value.numel(),
            "data_ptr": value.data_ptr() if value.device.type != "meta" else None,
        }
        for name, value in tensors
    }


def hook_snapshot(module: Any) -> dict[str, Any]:
    hook = getattr(module, "_hf_hook", None)
    return {
        "present": hook is not None,
        "type": type(hook).__name__ if hook is not None else None,
        "repr": repr(hook) if hook is not None else None,
        "execution_device": str(getattr(hook, "execution_device", None)),
        "offload": getattr(hook, "offload", None),
    }


def materialize_persistent_cpu_modules(model: Any, torch: Any) -> dict[str, Any]:
    """Detach Accelerate offload hooks so embeddings can never stage on CUDA.

    Accelerate intentionally executes CPU-offloaded modules on the main CUDA
    device.  That behavior is useful for conventional layer offload, but it
    would transiently move Gemma 4's roughly 7 GiB embedding tables to the
    8 GiB GPU.  Detaching the three leaf hooks restores their CPU tensors.  The
    output head is then re-tied to the token embedding, preserving model
    semantics and avoiding a duplicate CPU allocation.
    """
    from accelerate.hooks import remove_hook_from_module

    before = {
        name: {
            "hook": hook_snapshot(get_submodule(model, name)),
            "storage": module_storage_snapshot(get_submodule(model, name)),
        }
        for name in PERSISTENT_CPU_MODULES
    }
    for name in PERSISTENT_CPU_MODULES:
        remove_hook_from_module(get_submodule(model, name), recurse=False)

    token_embedding = get_submodule(model, "model.language_model.embed_tokens")
    output_head = get_submodule(model, "lm_head")
    output_head.weight = token_embedding.weight

    after = {
        name: {
            "hook": hook_snapshot(get_submodule(model, name)),
            "storage": module_storage_snapshot(get_submodule(model, name)),
        }
        for name in PERSISTENT_CPU_MODULES
    }
    misplaced = {
        name: snapshot
        for name, snapshot in after.items()
        if not snapshot["storage"]
        or any(item["device"] != "cpu" for item in snapshot["storage"].values())
    }
    if misplaced:
        raise ValueError(f"persistent CPU modules are not resident on CPU: {misplaced}")
    if output_head.weight.data_ptr() != token_embedding.weight.data_ptr():
        raise ValueError("lm_head is not tied to the CPU token embedding")

    return {
        "strategy": "detach_leaf_offload_hooks_and_retie_lm_head",
        "before": before,
        "after": after,
        "lm_head_tied_to_token_embedding": True,
        "persistent_cpu_bytes_unique": sum(
            value.numel() * value.element_size()
            for module in (
                token_embedding,
                get_submodule(model, "model.language_model.embed_tokens_per_layer"),
            )
            for value in module.parameters(recurse=False)
        ),
    }


def memory_snapshot(torch: Any, device: int) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(device)
    mib = 1024**2
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_used_bytes": total - free,
        "device_free_bytes": free,
        "device_total_bytes": total,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / mib, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / mib, 3),
        "device_used_mib": round((total - free) / mib, 3),
    }


def token_sha256(values: list[int]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_prefill(tokenizer: Any, length: int, torch: Any) -> Any:
    seed_text = (
        "EdgeForge validates deterministic edge inference, tool use, and system "
        "measurement. "
    ) * 256
    token_ids = tokenizer.encode(seed_text, add_special_tokens=True)
    if len(token_ids) < length:
        raise ValueError(f"tokenizer produced only {len(token_ids)} prefill tokens")
    return torch.tensor([token_ids[:length]], dtype=torch.long, device="cpu")


def run_text_only_greedy(
    model: Any,
    input_ids: Any,
    decode_tokens: int,
    device: int,
    torch: Any,
) -> dict[str, Any]:
    """Run Gemma 4 while keeping both embedding tables and lm_head on CPU."""
    text_model = get_submodule(model, "model.language_model")
    token_embedding = text_model.embed_tokens
    per_layer_embedding = text_model.embed_tokens_per_layer
    output_head = model.lm_head
    cuda_device = torch.device(f"cuda:{device}")
    num_layers = text_model.config.num_hidden_layers
    per_layer_width = text_model.config.hidden_size_per_layer_input
    softcap = text_model.config.final_logit_softcapping

    attention_mask = torch.ones_like(input_ids, device="cpu")
    current_ids = input_ids
    past_key_values = None
    generated: list[int] = []
    step_seconds: list[float] = []
    embedding_cuda_allocation_deltas: list[int] = []
    generation_started = time.perf_counter()

    for token_index in range(decode_tokens):
        step_started = time.perf_counter()
        torch.cuda.synchronize(device)
        allocation_before_embedding = torch.cuda.memory_allocated(device)
        inputs_embeds_cpu = token_embedding(current_ids)
        per_layer_inputs_cpu = per_layer_embedding(current_ids).reshape(
            *current_ids.shape, num_layers, per_layer_width
        )
        torch.cuda.synchronize(device)
        allocation_after_embedding = torch.cuda.memory_allocated(device)
        allocation_delta = allocation_after_embedding - allocation_before_embedding
        embedding_cuda_allocation_deltas.append(allocation_delta)
        if allocation_delta != 0:
            raise ValueError(
                "CPU embedding lookup changed CUDA allocation: "
                f"step={token_index}, delta={allocation_delta}"
            )

        inputs_embeds = inputs_embeds_cpu.to(cuda_device)
        per_layer_inputs = per_layer_inputs_cpu.to(cuda_device)
        step_attention_mask = attention_mask.to(cuda_device)
        del inputs_embeds_cpu, per_layer_inputs_cpu

        with torch.inference_mode():
            outputs = text_model(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            last_hidden_cpu = outputs.last_hidden_state[:, -1:, :].to("cpu")
            logits = output_head(last_hidden_cpu)
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            next_token = int(logits[0, -1].argmax().item())

        generated.append(next_token)
        current_ids = torch.tensor([[next_token]], dtype=torch.long, device="cpu")
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1
        )
        del inputs_embeds, per_layer_inputs, step_attention_mask, outputs
        del last_hidden_cpu, logits
        torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - step_started)
        if token_index == 0 or (token_index + 1) % 4 == 0 or token_index + 1 == decode_tokens:
            log(
                f"greedy token {token_index + 1}/{decode_tokens}; "
                f"cache_length={past_key_values.get_seq_length()}"
            )

    return {
        "generated_token_ids": generated,
        "generated_token_ids_sha256": token_sha256(generated),
        "seconds": time.perf_counter() - generation_started,
        "prefill_and_first_token_seconds": step_seconds[0],
        "incremental_decode_seconds": sum(step_seconds[1:]),
        "incremental_decode_input_tokens": max(0, decode_tokens - 1),
        "step_seconds": step_seconds,
        "embedding_cuda_allocation_deltas_bytes": embedding_cuda_allocation_deltas,
        "embedding_cuda_allocation_delta_max_bytes": max(
            embedding_cuda_allocation_deltas, default=0
        ),
        "final_cache_length": past_key_values.get_seq_length(),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import bitsandbytes as bnb
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(f"invalid CUDA device index: {args.device}")
    if args.prefill_tokens <= 0 or args.decode_tokens <= 0:
        raise ValueError("prefill and decode token counts must be positive")
    if args.carrier == "formal" and (
        args.prefill_tokens != 512 or args.decode_tokens != 32
    ):
        raise ValueError("formal P0-3 requires exactly 512 prefill and 32 decode tokens")
    if args.carrier == "formal" and args.gpu_budget_gib > 7.5:
        raise ValueError("formal P0-3 GPU budget cannot exceed 7.5 GiB")
    model_path = args.model.resolve(strict=True)
    if args.carrier == "formal":
        log("verifying every transferred C02 artifact hash")
        integrity = verify_formal_artifact(
            model_path, args.expected_manifest_sha256
        )
    else:
        log("recording nonformal bootstrap carrier identity")
        integrity = bootstrap_identity(model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)

    device_map: dict[str, int | str] = {
        **{name: "cpu" for name in CPU_MODULES},
        **{name: args.device for name in CUDA_MODULES},
    }
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    log("loading model with NF4 and explicit CPU embedding placement")
    load_started = time.perf_counter()
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
    torch.cuda.synchronize(args.device)
    load_seconds = time.perf_counter() - load_started
    after_load_memory = memory_snapshot(torch, args.device)

    log("materializing text embeddings and tied output head permanently on CPU")
    cpu_materialization = materialize_persistent_cpu_modules(model, torch)
    persistent_cpu_before = {
        name: module_storage_snapshot(get_submodule(model, name))
        for name in PERSISTENT_CPU_MODULES
    }
    unused_offload_state = {
        name: {
            "hook": hook_snapshot(get_submodule(model, name)),
            "devices_recursive": module_devices(get_submodule(model, name)),
        }
        for name in OFFLOADED_UNUSED_CPU_MODULES
    }

    quantized_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, bnb.nn.Linear4bit)
    ]
    if not quantized_modules:
        raise ValueError("model contains no bitsandbytes Linear4bit modules")
    all_quant_types = Counter(
        str(getattr(getattr(module.weight, "quant_state", None), "quant_type", None))
        for _, module in quantized_modules
    )
    active_quantized_modules = [
        (name, module)
        for name, module in quantized_modules
        if name.startswith("model.language_model.")
    ]
    active_quant_types = Counter(
        str(getattr(getattr(module.weight, "quant_state", None), "quant_type", None))
        for _, module in active_quantized_modules
    )
    active_weight_devices = Counter(
        str(module.weight.device) for _, module in active_quantized_modules
    )
    if not active_quantized_modules or set(active_quant_types) != {"nf4"}:
        raise ValueError(
            "active language modules are not uniformly NF4: "
            f"count={len(active_quantized_modules)}, states={dict(active_quant_types)}"
        )
    if any(not device.startswith("cuda:") for device in active_weight_devices):
        raise ValueError(
            f"active NF4 language weights are not CUDA resident: {dict(active_weight_devices)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    input_ids = build_prefill(tokenizer, args.prefill_tokens, torch)
    log(
        f"running hybrid CPU-embedding greedy generation: prefill={args.prefill_tokens}, "
        f"decode={args.decode_tokens}"
    )
    generation = run_text_only_greedy(
        model, input_ids, args.decode_tokens, args.device, torch
    )
    generated = generation["generated_token_ids"]
    if len(generated) != args.decode_tokens:
        raise ValueError(
            f"decode length mismatch: expected {args.decode_tokens}, got {len(generated)}"
        )

    persistent_cpu_after = {
        name: module_storage_snapshot(get_submodule(model, name))
        for name in PERSISTENT_CPU_MODULES
    }
    if persistent_cpu_after != persistent_cpu_before:
        raise ValueError(
            "persistent CPU embedding storage changed during generation: "
            f"before={persistent_cpu_before}, after={persistent_cpu_after}"
        )
    final_memory = memory_snapshot(torch, args.device)
    budget_bytes = int(args.gpu_budget_gib * 1024**3)
    budget_checks = {
        field: final_memory[field] < budget_bytes
        for field in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "device_used_bytes",
        )
    }
    if not all(budget_checks.values()):
        raise ValueError(
            f"GPU memory exceeded {args.gpu_budget_gib} GiB: "
            f"checks={budget_checks}, memory={final_memory}"
        )

    return {
        "integrity": integrity,
        "carrier": args.carrier,
        "formal_evidence_eligible": args.carrier == "formal",
        "model_path": str(model_path),
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "torch.bfloat16",
            "linear4bit_module_count_all": len(quantized_modules),
            "linear4bit_module_count_active_language": len(active_quantized_modules),
            "observed_quant_types_all": dict(sorted(all_quant_types.items())),
            "observed_quant_types_active_language": dict(
                sorted(active_quant_types.items())
            ),
            "active_language_weight_devices": dict(
                sorted(active_weight_devices.items())
            ),
        },
        "device_map_requested": device_map,
        "hf_device_map_observed": getattr(model, "hf_device_map", None),
        "cpu_materialization": cpu_materialization,
        "persistent_cpu_storage_before_generation": persistent_cpu_before,
        "persistent_cpu_storage_after_generation": persistent_cpu_after,
        "unused_multimodal_offload_state": unused_offload_state,
        "embedding_no_remigration": True,
        "load_seconds": load_seconds,
        "generation": {
            "prefill_tokens": input_ids.shape[1],
            "decode_tokens": len(generated),
            "greedy": True,
            "seed": args.seed,
            **generation,
        },
        "memory_after_load": after_load_memory,
        "memory_final": final_memory,
        "gpu_budget_gib": args.gpu_budget_gib,
        "gpu_budget_bytes": budget_bytes,
        "gpu_budget_checks": budget_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--carrier", choices=("formal", "bootstrap"), required=True)
    parser.add_argument(
        "--expected-manifest-sha256", default=EXPECTED_C02_MANIFEST_SHA256
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--gpu-budget-gib", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt_path = args.receipt.resolve()
    started_at = now_iso()
    base: dict[str, Any] = {
        "schema_version": 1,
        "stage": "OP-P0-3",
        "started_at": started_at,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "environment": environment_identity(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    try:
        result = run_smoke(args)
        receipt = {
            **base,
            "status": "PASS",
            "finished_at": now_iso(),
            "assertions": {
                "transfer_integrity": (
                    "PASS" if args.carrier == "formal" else "N/A_BOOTSTRAP"
                ),
                "nf4_carrier": "PASS",
                "prefill_512_decode_32": (
                    "PASS"
                    if args.prefill_tokens == 512 and args.decode_tokens == 32
                    else "N/A_BOOTSTRAP"
                ),
                "gpu_memory_below_7_5_gib": "PASS",
                "embedding_no_remigration": "PASS",
            },
            **result,
        }
        atomic_write_json(receipt_path, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        failure = {
            **base,
            "status": "FAIL",
            "finished_at": now_iso(),
            "carrier": args.carrier,
            "model_path": str(args.model.resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stderr": traceback.format_exc(),
        }
        try:
            import torch

            if torch.cuda.is_available():
                failure["memory_at_failure"] = memory_snapshot(torch, args.device)
        except BaseException:
            failure["memory_capture_error"] = traceback.format_exc()
        atomic_write_json(receipt_path, failure)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
