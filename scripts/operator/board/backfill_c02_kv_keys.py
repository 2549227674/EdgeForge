#!/usr/bin/env python3
"""C0-1b: transplant the official 54 shared-KV tensors into C02.

The output is the RKLLM conversion-input variant defined by the board R2.1
card: one ``model.safetensors`` plus the six C02 Hugging Face metadata files.
The old C02 ``artifact_manifest.json`` is deliberately not copied because its
four-shard inventory would be false for this derived artifact. A new external
manifest and a 54-row transplant receipt are written instead.

Only the standard library is used. Tensor payloads are streamed, so peak
process memory stays bounded by ``CHUNK`` plus the safetensors headers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


EXPECTED_C02_MANIFEST_SHA256 = (
    "ee7d426d1e912a6c53cad1cd52739f26d7fff45b4a4f7429a4c08656f68dc5f2"
)
SHARED_LAYERS = range(24, 42)
TRANSPLANT_SUFFIXES = ("k_proj.weight", "v_proj.weight", "k_norm.weight")
LM_ATTN = "model.language_model.layers.{i}.self_attn.{s}"
METADATA_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
CHUNK = 64 * 1024 * 1024


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_set_sha256(files: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_sha(repo: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def read_st_header(path: Path) -> tuple[dict[str, Any], int]:
    """Return ``(header, absolute_data_offset)`` with bounds checks."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len == 0 or header_len > size - 8:
            raise ValueError(
                f"invalid safetensors header length {header_len} for {path} ({size} B)"
            )
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return header, 8 + header_len


def tensor_keys(header: dict[str, Any]) -> set[str]:
    return {key for key in header if key != "__metadata__"}


def expected_missing_keys() -> list[str]:
    return [
        LM_ATTN.format(i=layer, s=suffix)
        for layer in SHARED_LAYERS
        for suffix in TRANSPLANT_SUFFIXES
    ]


def stream_region(src: Path, src_offset: int, nbytes: int, dst: BinaryIO) -> str:
    """Copy one tensor payload and return the SHA-256 of the bytes read."""
    digest = hashlib.sha256()
    with src.open("rb") as handle:
        handle.seek(src_offset)
        remaining = nbytes
        while remaining:
            block = handle.read(min(CHUNK, remaining))
            if not block:
                raise OSError(
                    f"unexpected EOF in {src} at {src_offset + nbytes - remaining}"
                )
            dst.write(block)
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def region_sha256(src: Path, src_offset: int, nbytes: int) -> str:
    """Hash one tensor payload without materializing it."""
    digest = hashlib.sha256()
    with src.open("rb") as handle:
        handle.seek(src_offset)
        remaining = nbytes
        while remaining:
            block = handle.read(min(CHUNK, remaining))
            if not block:
                raise OSError(
                    f"unexpected EOF in {src} at {src_offset + nbytes - remaining}"
                )
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def verify_c02_source(model_dir: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    manifest_path = model_dir / "artifact_manifest.json"
    observed_manifest_sha256 = file_sha256(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "C02 artifact manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, observed {observed_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = manifest.get("files_excluding_manifest")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("C02 artifact manifest has no file inventory")

    observed_files: list[dict[str, Any]] = []
    for expected in expected_files:
        path = model_dir / expected["path"]
        observed = {
            "path": expected["path"],
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        if observed != expected:
            raise ValueError(
                f"C02 source differs from frozen manifest at {expected['path']}: "
                f"expected={expected}, observed={observed}"
            )
        observed_files.append(observed)
    observed_set_sha256 = artifact_set_sha256(observed_files)
    if observed_set_sha256 != manifest.get("artifact_set_sha256_excluding_manifest"):
        raise ValueError("C02 artifact-set identity mismatch")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": observed_manifest_sha256,
        "artifact_set_sha256": observed_set_sha256,
        "file_count": len(observed_files),
        "status": "PASS",
    }


def verify_official_source(
    safetensors_path: Path, frozen_manifest_path: Path
) -> dict[str, Any]:
    manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {}).get(safetensors_path.name)
    if not isinstance(expected, str):
        raise ValueError(
            f"{frozen_manifest_path} does not freeze {safetensors_path.name}"
        )
    observed = file_sha256(safetensors_path)
    if observed != expected:
        raise ValueError(
            "official source SHA-256 mismatch: "
            f"expected {expected}, observed {observed}"
        )
    return {
        "manifest": str(frozen_manifest_path),
        "safetensors": str(safetensors_path),
        "safetensors_bytes": safetensors_path.stat().st_size,
        "safetensors_sha256": observed,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c02-dir", type=Path, default=Path("models/C02-S.O-bf16-r1"))
    parser.add_argument(
        "--c02-manifest-sha256", default=EXPECTED_C02_MANIFEST_SHA256
    )
    parser.add_argument(
        "--official-safetensors",
        type=Path,
        default=Path("models/gemma-4-E4B-it/model.safetensors"),
    )
    parser.add_argument(
        "--official-manifest",
        type=Path,
        default=Path("manifests/gemma4_e4b_rkllm_frozen_input_sha256.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("models/C02-S.O-rkllm-input-r1")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/operator/board/c02_rkllm_input_manifest.json"),
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("results/operator/board/c0/c0_1b_transplant_receipt.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the exact output directory only after a new artifact verifies",
    )
    args = parser.parse_args()

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "C0-1b",
        "card": "docs/operator/EdgeForge_算子面_板端_执行卡_R2.md",
        "timestamp_start": now_iso(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git_sha": git_sha(Path.cwd()),
        "status": "FAIL",
    }
    temporary_output: Path | None = None

    def bail(message: str, exc: BaseException | None = None) -> int:
        if temporary_output is not None and temporary_output.exists():
            shutil.rmtree(temporary_output)
        receipt["error"] = message
        if exc is not None:
            receipt["error_type"] = type(exc).__name__
        receipt["timestamp_end"] = now_iso()
        atomic_write_json(args.receipt, receipt)
        print(f"FAIL: {message}", file=sys.stderr)
        return 1

    try:
        if args.out_dir.exists() and not args.force:
            return bail(f"{args.out_dir} already exists (use --force to replace it)")
        if not args.c02_dir.is_dir():
            return bail(f"missing C02 directory: {args.c02_dir}")
        if not args.official_safetensors.is_file():
            return bail(f"missing official source: {args.official_safetensors}")
        if not args.official_manifest.is_file():
            return bail(f"missing official manifest: {args.official_manifest}")

        print("[1/6] verifying frozen C02 and official source identities", flush=True)
        source_identity = {
            "c02": verify_c02_source(args.c02_dir, args.c02_manifest_sha256),
            "official": verify_official_source(
                args.official_safetensors, args.official_manifest
            ),
        }
        receipt["source_identity"] = source_identity

        print("[2/6] auditing tensor key sets, shapes, and dtypes", flush=True)
        index_path = args.c02_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index["weight_map"]
        c02_keys = set(weight_map)
        expected_transplants = expected_missing_keys()
        unexpected_present = [key for key in expected_transplants if key in c02_keys]
        if unexpected_present:
            return bail(
                f"C02 already contains {len(unexpected_present)} expected-missing keys; "
                f"first={unexpected_present[0]}"
            )
        if len(expected_transplants) != 54:
            return bail(f"internal expected transplant count is {len(expected_transplants)}, not 54")

        shard_headers: dict[str, tuple[dict[str, Any], int]] = {}
        shard_tensor_keys: set[str] = set()
        for shard in sorted(set(weight_map.values())):
            shard_path = args.c02_dir / shard
            header, base = read_st_header(shard_path)
            shard_headers[shard] = (header, base)
            overlap = shard_tensor_keys & tensor_keys(header)
            if overlap:
                return bail(f"duplicate tensors across C02 shards; first={sorted(overlap)[0]}")
            shard_tensor_keys |= tensor_keys(header)
        if shard_tensor_keys != c02_keys:
            return bail(
                "C02 index/header key-set mismatch: "
                f"index_only={len(c02_keys - shard_tensor_keys)}, "
                f"header_only={len(shard_tensor_keys - c02_keys)}"
            )

        official_header, official_base = read_st_header(args.official_safetensors)
        official_keys = tensor_keys(official_header)
        output_keys = c02_keys | set(expected_transplants)
        if official_keys != output_keys:
            return bail(
                "official key set is not exactly C02 plus the 54 transplants: "
                f"official_only={len(official_keys - output_keys)}, "
                f"output_only={len(output_keys - official_keys)}"
            )
        c02_config = json.loads(
            (args.c02_dir / "config.json").read_text(encoding="utf-8")
        )
        layer_types = c02_config["text_config"]["layer_types"]
        if len(layer_types) != 42:
            return bail(f"expected 42 layer types, observed {len(layer_types)}")

        entries: list[dict[str, Any]] = []
        for key, shard in weight_map.items():
            header, base = shard_headers[shard]
            metadata = header[key]
            official_metadata = official_header[key]
            if (
                metadata["dtype"] != official_metadata["dtype"]
                or metadata["shape"] != official_metadata["shape"]
            ):
                return bail(f"common tensor geometry differs from official at {key}")
            start, end = metadata["data_offsets"]
            entries.append(
                {
                    "key": key,
                    "dtype": metadata["dtype"],
                    "shape": metadata["shape"],
                    "nbytes": end - start,
                    "src_path": args.c02_dir / shard,
                    "src_abs_offset": base + start,
                    "origin": "c02",
                }
            )
        for key in expected_transplants:
            metadata = official_header[key]
            start, end = metadata["data_offsets"]
            entries.append(
                {
                    "key": key,
                    "dtype": metadata["dtype"],
                    "shape": metadata["shape"],
                    "nbytes": end - start,
                    "src_path": args.official_safetensors,
                    "src_abs_offset": official_base + start,
                    "origin": "official_transplant",
                }
            )
        entries.sort(key=lambda item: item["key"])

        new_header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
        data_cursor = 0
        for entry in entries:
            new_header[entry["key"]] = {
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "data_offsets": [data_cursor, data_cursor + entry["nbytes"]],
            }
            data_cursor += entry["nbytes"]
        raw_header = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
        header_bytes = raw_header + b" " * ((-len(raw_header)) % 8)

        args.out_dir.parent.mkdir(parents=True, exist_ok=True)
        required = data_cursor + len(header_bytes) + 8 + (256 << 20)
        free = shutil.disk_usage(args.out_dir.parent).free
        if free < required:
            return bail(f"free space {free} B is below required {required} B")
        temporary_output = args.out_dir.with_name(
            f".{args.out_dir.name}.incomplete-{os.getpid()}"
        )
        if temporary_output.exists():
            return bail(f"temporary output unexpectedly exists: {temporary_output}")
        temporary_output.mkdir()

        print("[3/6] streaming 2,076 C02 tensors plus 54 official tensors", flush=True)
        output_safetensors = temporary_output / "model.safetensors"
        source_tensor_sha256: dict[str, str] = {}
        transplant_rows: list[dict[str, Any]] = []
        with output_safetensors.open("wb") as destination:
            destination.write(struct.pack("<Q", len(header_bytes)))
            destination.write(header_bytes)
            for entry in entries:
                tensor_sha256 = stream_region(
                    entry["src_path"],
                    entry["src_abs_offset"],
                    entry["nbytes"],
                    destination,
                )
                source_tensor_sha256[entry["key"]] = tensor_sha256
                if entry["origin"] == "official_transplant":
                    target_layer = int(entry["key"].split(".")[3])
                    suffix = entry["key"].split(".")[-2]
                    transplant_rows.append(
                        {
                            "target_key": entry["key"],
                            "source_key": entry["key"],
                            "source_file": str(args.official_safetensors),
                            "dtype": entry["dtype"],
                            "shape": entry["shape"],
                            "nbytes": entry["nbytes"],
                            "sha256": tensor_sha256,
                            "target_layer": target_layer,
                            "attention_type": layer_types[target_layer],
                            "tensor_role": suffix,
                        }
                    )

        source_hash_cache: dict[str, str] = {}
        source_copy_counts = {
            role: {"equal": 0, "different": 0}
            for role in ("k_proj", "v_proj", "k_norm")
        }
        for row in transplant_rows:
            source_layer = 22 if row["attention_type"] == "sliding_attention" else 23
            source_key = LM_ATTN.format(
                i=source_layer, s=f"{row['tensor_role']}.weight"
            )
            if source_key not in official_header:
                return bail(f"runtime source tensor is absent from official input: {source_key}")
            if source_key not in source_hash_cache:
                source_meta = official_header[source_key]
                source_start, source_end = source_meta["data_offsets"]
                source_hash_cache[source_key] = region_sha256(
                    args.official_safetensors,
                    official_base + source_start,
                    source_end - source_start,
                )
            equals_source = row["sha256"] == source_hash_cache[source_key]
            row["runtime_source_layer"] = source_layer
            row["runtime_source_key"] = source_key
            row["equals_official_runtime_source_tensor"] = equals_source
            bucket = "equal" if equals_source else "different"
            source_copy_counts[row["tensor_role"]][bucket] += 1
        source_copy_audit = {
            "mapping": "sliding_attention->22; full_attention->23",
            "counts": source_copy_counts,
            "source_layer_copy_hypothesis": "REJECTED",
            "observed_pattern": (
                "all 36 k_proj/v_proj tensors differ from their official runtime "
                "source-layer tensor; all 18 k_norm tensors are byte-identical"
            ),
        }
        expected_source_copy_counts = {
            "k_proj": {"equal": 0, "different": 18},
            "v_proj": {"equal": 0, "different": 18},
            "k_norm": {"equal": 18, "different": 0},
        }
        if source_copy_counts != expected_source_copy_counts:
            return bail(
                "official source-layer comparison no longer matches the closed audit: "
                f"{source_copy_counts}"
            )

        copied_metadata: list[dict[str, Any]] = []
        for name in METADATA_FILES:
            source = args.c02_dir / name
            if not source.is_file():
                return bail(f"missing required C02 metadata: {source}")
            destination = temporary_output / name
            shutil.copy2(source, destination)
            copied_metadata.append(
                {
                    "path": name,
                    "bytes": destination.stat().st_size,
                    "sha256": file_sha256(destination),
                }
            )

        print("[4/6] verifying every output tensor byte-for-byte", flush=True)
        check_header, check_base = read_st_header(output_safetensors)
        if tensor_keys(check_header) != output_keys:
            return bail("output safetensors key set changed after serialization")
        if output_safetensors.stat().st_size != check_base + data_cursor:
            return bail("output file length does not match its safetensors header")

        whole_digest = hashlib.sha256()
        verified_tensors = 0
        with output_safetensors.open("rb") as handle:
            prefix = handle.read(check_base)
            if len(prefix) != check_base:
                return bail("output prefix is truncated during verification")
            whole_digest.update(prefix)
            for entry in entries:
                metadata = check_header[entry["key"]]
                start, end = metadata["data_offsets"]
                if end - start != entry["nbytes"]:
                    return bail(f"output tensor length changed at {entry['key']}")
                digest = hashlib.sha256()
                remaining = entry["nbytes"]
                while remaining:
                    block = handle.read(min(CHUNK, remaining))
                    if not block:
                        return bail(f"output tensor is truncated at {entry['key']}")
                    digest.update(block)
                    whole_digest.update(block)
                    remaining -= len(block)
                if digest.hexdigest() != source_tensor_sha256[entry["key"]]:
                    return bail(f"byte-for-byte verification failed at {entry['key']}")
                verified_tensors += 1
            if handle.read(1):
                return bail("output contains trailing bytes not declared by the header")

        output_file_entry = {
            "path": "model.safetensors",
            "bytes": output_safetensors.stat().st_size,
            "sha256": whole_digest.hexdigest(),
        }
        output_files = sorted(
            [output_file_entry, *copied_metadata], key=lambda item: item["path"]
        )
        if len(transplant_rows) != 54:
            return bail(f"transplant receipt has {len(transplant_rows)} rows, not 54")

        print("[5/6] committing derived artifact and writing its manifest", flush=True)
        if args.out_dir.exists():
            shutil.rmtree(args.out_dir)
        os.replace(temporary_output, args.out_dir)
        temporary_output = None

        derived_manifest = {
            "schema_version": 1,
            "identity": "C02-S.O-rkllm-input-r1",
            "format": "huggingface_bf16_safetensors_single_file",
            "created_at": now_iso(),
            "card": "docs/operator/EdgeForge_算子面_板端_执行卡_R2.md",
            "output_dir": str(args.out_dir),
            "lineage": {
                "c02_source": source_identity["c02"],
                "official_transplant_source": source_identity["official"],
                "policy": (
                    "C02 payload plus the exact official values for the 54 keys absent "
                    "from the transformers module tree; no source-layer copying"
                ),
            },
            "tensor_inventory": {
                "c02_tensor_count": len(c02_keys),
                "official_tensor_count": len(official_keys),
                "transplant_tensor_count": len(transplant_rows),
                "output_tensor_count": len(entries),
                "output_keyset_equals_official": True,
                "common_tensor_geometry_equals_official": True,
                "all_output_tensors_byte_verified_against_selected_source": True,
                "verified_tensor_count": verified_tensors,
                "source_layer_copy_audit": source_copy_audit,
            },
            "files": output_files,
            "artifact_set_sha256": artifact_set_sha256(output_files),
            "excluded_stale_source_metadata": [
                "artifact_manifest.json",
                "model.safetensors.index.json",
            ],
        }
        atomic_write_json(args.manifest, derived_manifest)
        manifest_sha256 = file_sha256(args.manifest)

        print("[6/6] writing the 54-row transplant receipt", flush=True)
        receipt.update(
            {
                "status": "PASS",
                "timestamp_end": now_iso(),
                "audit": {
                    "c02_missing_key_count": 54,
                    "official_present_key_count": 54,
                    "output_keyset_equals_official": True,
                    "common_tensor_geometry_equals_official": True,
                    "verified_output_tensor_count": verified_tensors,
                    "source_layer_copy_audit": source_copy_audit,
                },
                "transplant_count": len(transplant_rows),
                "transplant_table": transplant_rows,
                "output": {
                    "directory": str(args.out_dir),
                    "files": output_files,
                    "artifact_set_sha256": derived_manifest["artifact_set_sha256"],
                },
                "manifest": {
                    "path": str(args.manifest),
                    "sha256": manifest_sha256,
                },
                "note": (
                    "The conversion input is C02 plus an exact transplant of the 54 "
                    "official tensors. Transformers discards those unexpected keys; "
                    "the logits-equivalence gate must pass before C0-3."
                ),
            }
        )
        atomic_write_json(args.receipt, receipt)
        print(
            f"PASS: transplanted 54 tensors; output={args.out_dir}; "
            f"model_sha256={output_file_entry['sha256']}; receipt={args.receipt}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        return bail(str(exc), exc)


if __name__ == "__main__":
    raise SystemExit(main())
