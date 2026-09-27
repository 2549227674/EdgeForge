#!/usr/bin/env python3
"""Merge an M1 LoRA adapter into the bf16 base and freeze an artifact manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForImageTextToText


PROCESSOR_ASSET_NAMES = (
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_inventory(root: Path) -> dict[str, Any]:
    """Inspect tensor metadata without materializing the merged weights again."""
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
                dtype = str(tensor_slice.get_dtype())
                dtype_numel[dtype] += math.prod(tensor_slice.get_shape())
                if "lora_" in name.lower():
                    lora_tensor_names.append(name)
    if shard_count == 0 or tensor_count == 0:
        raise ValueError("merged candidate contains no safetensors weights")
    if set(dtype_numel) != {"BF16"}:
        raise ValueError(f"merged candidate is not uniformly bf16: {dict(dtype_numel)}")
    if lora_tensor_names:
        raise ValueError(f"unmerged LoRA tensors remain: {lora_tensor_names[:8]}")
    return {
        "shard_count": shard_count,
        "tensor_count": tensor_count,
        "dtype_numel": dict(sorted(dtype_numel.items())),
        "all_tensors_bf16": True,
        "lora_tensor_count": 0,
    }


def artifact_set_sha256(files: list[dict[str, Any]]) -> str:
    """Return one stable identity for the complete, ordered artifact file set."""
    payload = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inventory_files(root: Path) -> list[dict[str, Any]]:
    """Freeze a complete, ordered identity for every file below one root."""
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def copy_processor_assets(base_model: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Copy the frozen processor assets without importing optional PIL backends."""
    assets = []
    for name in PROCESSOR_ASSET_NAMES:
        source = base_model / name
        if not source.is_file():
            raise FileNotFoundError(f"required processor asset is missing: {source}")
        destination = output_dir / name
        shutil.copy2(source, destination)
        source_sha = sha256(source)
        candidate_sha = sha256(destination)
        if candidate_sha != source_sha:
            raise ValueError(f"processor asset copy mismatch: {name}")
        assets.append(
            {
                "path": name,
                "bytes": destination.stat().st_size,
                "source_sha256": source_sha,
                "candidate_sha256": candidate_sha,
            }
        )
    return assets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    incomplete = args.output_dir.with_name(args.output_dir.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete output already exists: {incomplete}")
    incomplete.mkdir(parents=True)

    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    if training.get("completed_steps") != 4096 or training.get("input_tokens") != 148425311:
        raise ValueError("training report does not identify the frozen R2.2 completion")
    adapter_files = inventory_files(args.adapter)
    expected_adapter_paths = [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    if [item["path"] for item in adapter_files] != expected_adapter_paths:
        raise ValueError("adapter file set differs from the frozen PEFT artifact shape")

    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        incomplete,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    processor_assets = copy_processor_assets(args.base_model, incomplete)

    tensor_inventory = safetensors_inventory(incomplete)

    files: list[dict[str, Any]] = []
    for path in sorted(incomplete.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(incomplete)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "schema_version": 3,
        "candidate": args.candidate,
        "format": "huggingface_bf16_safetensors",
        "merge_script_sha256": sha256(Path(__file__)),
        "base_model": str(args.base_model),
        "base_model_safetensors_sha256": sha256(args.base_model / "model.safetensors"),
        "adapter": str(args.adapter),
        "adapter_files": adapter_files,
        "adapter_artifact_set_sha256": artifact_set_sha256(adapter_files),
        "training_report": str(args.training_report),
        "training_report_sha256": sha256(args.training_report),
        "training_plan_sha256": training["plan_sha256"],
        "training_completed_steps": training["completed_steps"],
        "training_input_tokens": training["input_tokens"],
        "processor_assets": processor_assets,
        "tensor_inventory": tensor_inventory,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "artifact_set_sha256": artifact_set_sha256(files),
    }
    (incomplete / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    incomplete.rename(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
