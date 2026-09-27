#!/usr/bin/env python3
"""Merge the frozen C02 rsLoRA adapter into C01-S and audit the artifact.

The script is intentionally self-contained so the exact same file can be copied
to the training instance.  Expensive work is guarded by lineage checks, output
is staged under ``<output>.incomplete``, and a failed run still writes a receipt
with the traceback needed to reproduce the failure.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Any


EXPECTED_ADAPTER_SHA256 = (
    "e1fddbeef0cfb338c89e7f37845545a490f7b062bb1290b757ed8a0c3c34326d"
)
PROCESSOR_ASSET_NAMES = (
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
RECEIPT_SCHEMA_VERSION = 1


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


def inventory_files(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required file is missing: {path}")


def resolve_declared_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def environment_identity() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in ("torch", "transformers", "peft", "safetensors", "accelerate")
        },
    }


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def safetensors_inventory(root: Path) -> dict[str, Any]:
    from safetensors import safe_open

    dtype_numel: Counter[str] = Counter()
    tensor_count = 0
    shard_count = 0
    lora_tensor_names: list[str] = []
    for path in sorted(root.glob("*.safetensors")):
        shard_count += 1
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor_count += 1
                tensor_slice = handle.get_slice(name)
                dtype_numel[str(tensor_slice.get_dtype())] += math.prod(
                    tensor_slice.get_shape()
                )
                if "lora_" in name.lower():
                    lora_tensor_names.append(name)
    if shard_count == 0 or tensor_count == 0:
        raise ValueError("merged candidate contains no safetensors tensors")
    return {
        "shard_count": shard_count,
        "tensor_count": tensor_count,
        "dtype_numel": dict(sorted(dtype_numel.items())),
        "all_tensors_bf16": set(dtype_numel) == {"BF16"},
        "lora_tensor_count": len(lora_tensor_names),
        "lora_tensor_names_sample": lora_tensor_names[:16],
    }


def validate_sources(
    *,
    workspace: Path,
    base_model: Path,
    adapter: Path,
    source_receipt: Path,
    expected_adapter_sha256: str,
) -> dict[str, Any]:
    if not base_model.is_dir():
        raise FileNotFoundError(f"C01-S base directory is missing: {base_model}")
    if not adapter.is_dir():
        raise FileNotFoundError(f"C02 adapter directory is missing: {adapter}")

    base_manifest_path = base_model / "artifact_manifest.json"
    adapter_config_path = adapter / "adapter_config.json"
    adapter_weights_path = adapter / "adapter_model.safetensors"
    for path in (base_manifest_path, adapter_config_path, adapter_weights_path, source_receipt):
        require_file(path)

    base_manifest = load_json(base_manifest_path)
    if base_manifest.get("candidate") != "C01-S":
        raise ValueError("merge base is not the frozen C01-S candidate")
    base_tensors = base_manifest.get("tensor_inventory", {})
    if not base_tensors.get("all_tensors_bf16"):
        raise ValueError("C01-S manifest does not attest an all-BF16 base")
    if base_tensors.get("lora_tensor_count") != 0:
        raise ValueError("C01-S manifest reports residual LoRA tensors")

    adapter_config = load_json(adapter_config_path)
    declared_base = adapter_config.get("base_model_name_or_path")
    if not isinstance(declared_base, str):
        raise ValueError("adapter_config lacks base_model_name_or_path")
    if resolve_declared_path(declared_base, workspace) != base_model:
        raise ValueError(
            "adapter lineage mismatch: adapter was not trained from the selected C01-S"
        )
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("final adapter is not a LoRA/rsLoRA artifact")

    observed_adapter_sha256 = sha256(adapter_weights_path)
    if observed_adapter_sha256 != expected_adapter_sha256:
        raise ValueError(
            "adapter_model.safetensors SHA-256 mismatch: "
            f"expected {expected_adapter_sha256}, observed {observed_adapter_sha256}"
        )

    training_receipt = load_json(source_receipt)
    if training_receipt.get("status") != "PASS":
        raise ValueError("source OPD training receipt is not PASS")
    declared_adapter = training_receipt.get("final_adapter")
    if not isinstance(declared_adapter, str):
        raise ValueError("source OPD receipt lacks final_adapter")
    if Path(declared_adapter).resolve() != adapter:
        raise ValueError("source OPD receipt points at a different final adapter")

    processor_assets: list[dict[str, Any]] = []
    for name in PROCESSOR_ASSET_NAMES:
        source = base_model / name
        require_file(source)
        entry: dict[str, Any] = {
            "path": name,
            "source": "C01-S frozen base",
            "sha256": sha256(source),
        }
        adapter_copy = adapter / name
        if adapter_copy.is_file():
            entry["adapter_copy_sha256"] = sha256(adapter_copy)
            entry["adapter_copy_matches"] = (
                entry["adapter_copy_sha256"] == entry["sha256"]
            )
        processor_assets.append(entry)

    return {
        "base_model": str(base_model),
        "base_manifest": str(base_manifest_path),
        "base_manifest_sha256": sha256(base_manifest_path),
        "base_artifact_set_sha256": base_manifest.get("artifact_set_sha256"),
        "base_tensor_inventory": base_tensors,
        "adapter": str(adapter),
        "adapter_config_sha256": sha256(adapter_config_path),
        "adapter_model_bytes": adapter_weights_path.stat().st_size,
        "adapter_model_sha256": observed_adapter_sha256,
        "adapter_rank": adapter_config.get("r"),
        "adapter_alpha": adapter_config.get("lora_alpha"),
        "adapter_use_rslora": adapter_config.get("use_rslora"),
        "source_training_receipt": str(source_receipt),
        "source_training_receipt_sha256": sha256(source_receipt),
        "processor_assets": processor_assets,
    }


def copy_processor_assets(
    source_root: Path, destination_root: Path
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for name in PROCESSOR_ASSET_NAMES:
        source = source_root / name
        destination = destination_root / name
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        destination_hash = sha256(destination)
        if source_hash != destination_hash:
            raise ValueError(f"processor asset copy mismatch: {name}")
        copied.append(
            {
                "path": name,
                "source": str(source),
                "bytes": destination.stat().st_size,
                "sha256": destination_hash,
            }
        )
    return copied


def independent_reload(script: Path, model_path: Path) -> dict[str, Any]:
    command = [sys.executable, str(script), "--_reload-check", str(model_path)]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "independent reload subprocess failed\n"
            f"stdout:\n{completed.stdout[-16000:]}\n"
            f"stderr:\n{completed.stderr[-16000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("independent reload subprocess returned no result")
    result = json.loads(lines[-1])
    if result.get("status") != "PASS":
        raise RuntimeError(f"independent reload did not pass: {result}")
    result["stderr_tail"] = completed.stderr[-4000:]
    result["command"] = command
    return result


def reload_check(model_path: Path) -> int:
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText

    persisted: dict[str, dict[str, Any]] = {}
    for path in sorted(model_path.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor_slice = handle.get_slice(name)
                shape = list(tensor_slice.get_shape())
                persisted[name] = {
                    "shape": shape,
                    "dtype": str(tensor_slice.get_dtype()),
                    "numel": math.prod(shape),
                }

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    state = model.state_dict()
    meta_tensors = [name for name, value in state.items() if value.is_meta]
    if meta_tensors:
        raise ValueError(f"reload left meta tensors: {meta_tensors[:16]}")

    parameter_dtype_numel: Counter[str] = Counter()
    for value in parameters.values():
        parameter_dtype_numel[str(value.dtype)] += value.numel()
    if set(parameter_dtype_numel) != {"torch.bfloat16"}:
        raise ValueError(
            "reload produced non-BF16 parameters: "
            f"{dict(parameter_dtype_numel)}"
        )

    torch_to_safetensors_dtype = {
        "torch.bfloat16": "BF16",
        "torch.float16": "F16",
        "torch.float32": "F32",
        "torch.float64": "F64",
        "torch.int8": "I8",
        "torch.int16": "I16",
        "torch.int32": "I32",
        "torch.int64": "I64",
        "torch.uint8": "U8",
        "torch.bool": "BOOL",
    }
    missing_from_state = sorted(set(persisted) - set(state))
    if missing_from_state:
        raise ValueError(
            f"reload state lacks persisted tensors: {missing_from_state[:16]}"
        )
    mismatches: list[dict[str, Any]] = []
    for name, expected in persisted.items():
        value = state[name]
        observed_dtype = torch_to_safetensors_dtype.get(str(value.dtype), str(value.dtype))
        if list(value.shape) != expected["shape"] or observed_dtype != expected["dtype"]:
            mismatches.append(
                {
                    "name": name,
                    "expected_shape": expected["shape"],
                    "observed_shape": list(value.shape),
                    "expected_dtype": expected["dtype"],
                    "observed_dtype": observed_dtype,
                }
            )
    if mismatches:
        raise ValueError(f"reload state metadata mismatches: {mismatches[:16]}")

    # Hugging Face omits tied aliases from safetensors.  Gemma 4's lm_head is
    # one such state entry.  Accept an unpersisted state key only when it shares
    # the exact storage of a persisted state key; otherwise reload is incomplete.
    persisted_storage: dict[int, str] = {}
    for name in persisted:
        persisted_storage.setdefault(state[name].untyped_storage().data_ptr(), name)
    aliases: dict[str, str] = {}
    unmatched_state: list[str] = []
    for name in sorted(set(state) - set(persisted)):
        target = persisted_storage.get(state[name].untyped_storage().data_ptr())
        if target is None:
            unmatched_state.append(name)
        else:
            aliases[name] = target
    if unmatched_state:
        raise ValueError(
            "reload produced state tensors absent from disk and not tied to a "
            f"persisted tensor: {unmatched_state[:16]}"
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "model_class": type(model).__name__,
                "unique_parameter_tensor_count": len(parameters),
                "unique_parameter_numel": sum(parameter_dtype_numel.values()),
                "parameter_dtype_numel": dict(sorted(parameter_dtype_numel.items())),
                "buffer_tensor_count": len(buffers),
                "buffer_numel": sum(value.numel() for value in buffers.values()),
                "persisted_tensor_count": len(persisted),
                "persisted_numel": sum(item["numel"] for item in persisted.values()),
                "unpersisted_tied_state_aliases": aliases,
                "meta_tensor_count": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--source-training-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--candidate", default="C02-S.O")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--expected-adapter-sha256", default=EXPECTED_ADAPTER_SHA256
    )
    return parser.parse_args()


def execute(args: argparse.Namespace) -> dict[str, Any]:
    script = Path(__file__).resolve()
    workspace = args.workspace.resolve(strict=True)
    base_model = args.base_model.resolve(strict=True)
    adapter = args.adapter.resolve(strict=True)
    source_receipt = args.source_training_receipt.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    receipt_path = args.receipt.resolve()
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")

    if output_dir.exists():
        raise FileExistsError(f"final output already exists: {output_dir}")
    if incomplete.exists():
        raise FileExistsError(f"incomplete output already exists: {incomplete}")
    if output_dir == receipt_path or output_dir in receipt_path.parents:
        raise ValueError("receipt must be outside the generated model directory")

    log("validating frozen C01-S and C02 adapter lineage")
    sources = validate_sources(
        workspace=workspace,
        base_model=base_model,
        adapter=adapter,
        source_receipt=source_receipt,
        expected_adapter_sha256=args.expected_adapter_sha256,
    )
    incomplete.mkdir(parents=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    log("loading C01-S on CPU")
    base = AutoModelForImageTextToText.from_pretrained(
        base_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    log("attaching frozen rsLoRA adapter")
    peft_model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    log("merging adapter with safe_merge=True")
    merged = peft_model.merge_and_unload(safe_merge=True)
    if isinstance(merged, PeftModel):
        raise ValueError("merge_and_unload returned an active PeftModel")

    log("saving staged BF16 checkpoint")
    merged.save_pretrained(
        incomplete,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    copied_assets = copy_processor_assets(base_model, incomplete)

    del merged
    del peft_model
    del base
    gc.collect()

    log("auditing tensor metadata")
    tensor_inventory = safetensors_inventory(incomplete)
    if not tensor_inventory["all_tensors_bf16"]:
        raise ValueError(
            f"merged checkpoint is not uniformly BF16: {tensor_inventory['dtype_numel']}"
        )
    if tensor_inventory["lora_tensor_count"] != 0:
        raise ValueError(
            "merged checkpoint retains LoRA tensor keys: "
            f"{tensor_inventory['lora_tensor_names_sample']}"
        )
    base_tensors = sources["base_tensor_inventory"]
    if tensor_inventory["tensor_count"] != base_tensors.get("tensor_count"):
        raise ValueError("merged tensor count differs from frozen C01-S geometry")
    if tensor_inventory["dtype_numel"] != base_tensors.get("dtype_numel"):
        raise ValueError("merged tensor numel/dtype inventory differs from C01-S geometry")

    model_files = inventory_files(incomplete)
    manifest = {
        "schema_version": 1,
        "candidate": args.candidate,
        "format": "huggingface_bf16_safetensors",
        "created_at": now_iso(),
        "merge_script_sha256": sha256(script),
        "lineage": sources,
        "processor_assets": copied_assets,
        "tensor_inventory": tensor_inventory,
        "files_excluding_manifest": model_files,
        "artifact_set_sha256_excluding_manifest": artifact_set_sha256(model_files),
    }
    atomic_write_json(incomplete / "artifact_manifest.json", manifest)

    log("starting independent subprocess reload")
    reload_result = independent_reload(script, incomplete)
    expected_persisted_numel = sum(tensor_inventory["dtype_numel"].values())
    if reload_result["persisted_tensor_count"] != tensor_inventory["tensor_count"]:
        raise ValueError(
            "independent reload persisted tensor count differs from safetensors inventory"
        )
    if reload_result["persisted_numel"] != expected_persisted_numel:
        raise ValueError(
            "independent reload persisted numel differs from safetensors inventory"
        )

    output_files = inventory_files(incomplete)
    if not output_files:
        raise ValueError("output inventory is unexpectedly empty")
    output_artifact_set_sha256 = artifact_set_sha256(output_files)

    log("all gates passed; publishing final C02-S.O directory")
    incomplete.rename(output_dir)
    return {
        "sources": sources,
        "tensor_inventory": tensor_inventory,
        "processor_assets": copied_assets,
        "independent_reload": reload_result,
        "output_files": output_files,
        "output_total_bytes": sum(item["bytes"] for item in output_files),
        "output_artifact_set_sha256": output_artifact_set_sha256,
        "output_dir": str(output_dir),
        "receipt_path": str(receipt_path),
    }


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--_reload-check":
        return reload_check(Path(sys.argv[2]).resolve(strict=True))

    args = parse_args()
    started_at = now_iso()
    receipt_path = args.receipt.resolve()
    base_receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "stage": "OP-P0-1",
        "candidate": args.candidate,
        "started_at": started_at,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "environment": environment_identity(),
        "merge_script": str(Path(__file__).resolve()),
        "merge_script_sha256": sha256(Path(__file__).resolve()),
    }
    try:
        result = execute(args)
        receipt = {
            **base_receipt,
            "status": "PASS",
            "finished_at": now_iso(),
            "assertions": {
                "all_tensors_bf16": "PASS",
                "no_lora_tensor_keys": "PASS",
                "independent_subprocess_reload": "PASS",
                "output_file_inventory_and_hashes": "PASS",
            },
            **result,
        }
        atomic_write_json(receipt_path, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        failure = {
            **base_receipt,
            "status": "FAIL",
            "finished_at": now_iso(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stderr": traceback.format_exc(),
            "output_dir": str(args.output_dir.resolve()),
            "incomplete_output_dir": str(
                args.output_dir.resolve().with_name(args.output_dir.resolve().name + ".incomplete")
            ),
        }
        atomic_write_json(receipt_path, failure)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
