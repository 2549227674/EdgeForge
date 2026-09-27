#!/usr/bin/env python3
"""Download or offline-verify the revision-frozen Gemma-4 31B T3 teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ID = "google/gemma-4-31B-it"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
EXPECTED_REPOSITORY_BYTES = 62_578_686_256
EXPECTED_FILES = {
    ".eval_results/mmmu_pro.yaml": {"bytes": 182},
    ".gitattributes": {"bytes": 1_708},
    "README.md": {"bytes": 27_963},
    "chat_template.jinja": {"bytes": 18_683},
    "config.json": {"bytes": 4_621},
    "generation_config.json": {"bytes": 208},
    "model-00001-of-00002.safetensors": {
        "bytes": 49_784_788_364,
        "sha256": "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa",
    },
    "model-00002-of-00002.safetensors": {
        "bytes": 12_761_549_884,
        "sha256": "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10",
    },
    "model.safetensors.index.json": {"bytes": 120_246},
    "processor_config.json": {"bytes": 1_689},
    "tokenizer.json": {
        "bytes": 32_169_626,
        "sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    },
    "tokenizer_config.json": {"bytes": 3_082},
}
GENERATED_FILES = {"download_manifest.json", "READY"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_at(timestamp: float) -> dict[str, str]:
    instant = datetime.fromtimestamp(timestamp, timezone.utc)
    return {
        "utc": instant.isoformat(),
        "local": instant.astimezone().isoformat(),
    }


def _pid1_start_time() -> dict[str, Any]:
    proc_stat = Path("/proc/1/stat").read_text(encoding="utf-8")
    tail = proc_stat[proc_stat.rfind(")") + 2 :].split()
    start_ticks = int(tail[19])
    boot_time = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
        if line.startswith("btime ")
    )
    timestamp = boot_time + start_ticks / os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    return {
        "meaning": "instance/container PID 1 start time",
        "source": "/proc/1/stat starttime + /proc/stat btime",
        **_iso_at(timestamp),
    }


def _cgroup_runtime() -> dict[str, Any]:
    memory_path = Path("/sys/fs/cgroup/memory.max")
    cpu_path = Path("/sys/fs/cgroup/cpu.max")
    if not memory_path.is_file() or not cpu_path.is_file():
        return {
            "version": None,
            "available": False,
            "note": "cgroup v2 memory.max/cpu.max are not exposed in this runtime",
        }

    memory_raw = memory_path.read_text(encoding="utf-8").strip()
    cpu_raw = cpu_path.read_text(encoding="utf-8").strip()
    quota_raw, period_raw = cpu_raw.split()
    memory_bytes = None if memory_raw == "max" else int(memory_raw)
    quota = None if quota_raw == "max" else int(quota_raw)
    period = int(period_raw)
    return {
        "version": 2,
        "available": True,
        "memory": {
            "raw": memory_raw,
            "bytes": memory_bytes,
            "gib": None if memory_bytes is None else memory_bytes / 1024**3,
        },
        "cpu": {
            "raw": cpu_raw,
            "quota_us": quota,
            "period_us": period,
            "quota_cores": None if quota is None else quota / period,
        },
    }


def _gpu_runtime() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    keys = ["index", "name", "uuid", "memory_total_mib", "driver_version", "compute_capability"]
    gpus = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(keys):
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        row = dict(zip(keys, values, strict=True))
        row["index"] = int(row["index"])
        row["memory_total_mib"] = int(row["memory_total_mib"])
        gpus.append(row)
    return gpus


def capture_runtime_identity(expected_gpus: int | None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    runtime = {
        "captured_at": {"utc": now.isoformat(), "local": now.astimezone().isoformat()},
        "hostname": socket.gethostname(),
        "instance_started_at": _pid1_start_time(),
        "cgroup_v2": _cgroup_runtime(),
        "gpus": _gpu_runtime(),
    }
    if expected_gpus is not None and len(runtime["gpus"]) != expected_gpus:
        raise ValueError(f"GPU count {len(runtime['gpus'])} != expected {expected_gpus}")
    return runtime


def verify_repository(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"teacher directory does not exist: {output_dir}")

    actual_paths = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.relative_to(output_dir).as_posix() not in GENERATED_FILES
        and ".cache" not in path.relative_to(output_dir).parts
    }
    expected_paths = set(EXPECTED_FILES)
    if actual_paths != expected_paths:
        raise ValueError(
            "repository file set mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, extra={sorted(actual_paths - expected_paths)}"
        )

    inventory = []
    frozen_verified = []
    for name, identity in sorted(EXPECTED_FILES.items()):
        path = output_dir / name
        actual_size = path.stat().st_size
        if actual_size != identity["bytes"]:
            raise ValueError(f"{name} size {actual_size} != {identity['bytes']}")
        actual_sha = sha256(path)
        if "sha256" in identity and actual_sha != identity["sha256"]:
            raise ValueError(f"{name} SHA {actual_sha} != {identity['sha256']}")
        item = {
            "file": name,
            "bytes": actual_size,
            "sha256": actual_sha,
            "identity_status": "upstream_frozen_match" if "sha256" in identity else "acceptance_inventory_only",
        }
        inventory.append(item)
        if "sha256" in identity:
            frozen_verified.append(item)

    repository_bytes = sum(item["bytes"] for item in inventory)
    if repository_bytes != EXPECTED_REPOSITORY_BYTES:
        raise ValueError(f"repository bytes {repository_bytes} != {EXPECTED_REPOSITORY_BYTES}")
    return inventory, frozen_verified


def artifact_set_sha256(inventory: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        [{key: item[key] for key in ("file", "bytes", "sha256")} for item in inventory],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_acceptance_receipts(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_dir / "download_manifest.json"
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(manifest_path, manifest_payload)
    manifest_sha = sha256(manifest_path)

    ready = {
        "schema_version": 1,
        "status": "complete",
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "verified_at": manifest["verified_at"],
    }
    ready_payload = (json.dumps(ready, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _atomic_write(output_dir / "READY", ready_payload)
    return ready


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Never contact the Hub or download files; verify the existing directory and write acceptance receipts.",
    )
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=None,
        help="Fail verification unless the runtime receipt sees exactly this many GPUs.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    if args.verify_only:
        if not args.output_dir.is_dir():
            parser.error(f"--verify-only requires an existing directory: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            local_dir=args.output_dir,
            max_workers=args.max_workers,
        )

    inventory, frozen_verified = verify_repository(args.output_dir)
    runtime = capture_runtime_identity(args.expected_gpus)
    manifest = {
        "schema_version": 2,
        "status": "PASS",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "artifact_dir": str(args.output_dir),
        "verification_mode": "verify_only" if args.verify_only else "download_then_verify",
        "verified_at": runtime["captured_at"],
        "identity_authority": "upstream_huggingface_lfs_sha256_for_frozen_files",
        "expected_repository_bytes": EXPECTED_REPOSITORY_BYTES,
        "repository_bytes": sum(item["bytes"] for item in inventory),
        "repository_file_count": len(inventory),
        "repository_artifact_set_sha256": artifact_set_sha256(inventory),
        "repository_inventory": inventory,
        "verified_files": frozen_verified,
        "verified_bytes": sum(item["bytes"] for item in frozen_verified),
        "runtime": runtime,
    }
    ready = write_acceptance_receipts(args.output_dir, manifest)
    print(json.dumps({"manifest": manifest, "ready": ready}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
