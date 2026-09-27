#!/usr/bin/env python3
"""Run one audited C02 A2 -> RKLLM W8A8/RK3588/16K/1-core build.

The public entry point creates a fresh run directory, launches a worker with
TMPDIR/TMP/TEMP and cwd fixed to that directory, samples CPU/GPU resources,
and writes a final receipt even when the SDK fails.  The worker verifies the
frozen C02 conversion-input and A2 identities before importing RKLLM, runs the
39,321,600-byte scratch preflight, then performs load/build/export exactly
once.

This script contains no quality evaluation and never mutates the input model
or calibration dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "edgeforge-ope-c0-3-rkllm-build/v1"
EXPECTED = {
    "model_manifest": "39cb2a50a4e1ce6bec0b6cb056cab8b5912401494bf5a1419caa6e69282e12cf",
    "model_safetensors": "40469f7bd93b6f16dba63081ca5f430d6bdaa7f518d1ffab37f8b9a7be331d0f",
    "a2_rule": "04ba555b3983333264f5a6c28e564efe0fba14ba42db9499dc742f39ad6787a8",
    "a2_manifest": "3eafe5323bee64f72e4cffaa222dc54ca2c34014dccbfa3c4a60d36e8c1c0113",
    "a2_receipt": "7419f1d8189a99d76c029d9e122856dca5784804fee18b8eb6560c94e377e455",
    "a2_dataset": "5dc2002203096ff4a78fee176a9c963da87227cf8ba792c88ded4e6b8f5c8c98",
}
ARTIFACT_NAME = "C02-S.O-w8a8-rk3588-16k-1core-A2.rkllm"
SCRATCH_OFFSET = 13_107_200
SCRATCH_LENGTH = 26_214_400
SCRATCH_TOTAL = SCRATCH_OFFSET + SCRATCH_LENGTH
CHUNK = 64 * 1024 * 1024
GIB = 1024**3
MIN_MEMORY_BYTES = 96 * GIB
MIN_GPU_MEMORY_MIB = 48_000


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=reject_duplicates)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def verify_sha(path: Path, expected: str, label: str) -> dict[str, Any]:
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def safetensors_tensor_count(path: Path) -> int:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        header_length = int.from_bytes(prefix, "little")
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {header_length}")
        header = json.loads(handle.read(header_length))
    return sum(key != "__metadata__" for key in header)


def verify_model(model: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_identity = verify_sha(
        manifest_path, EXPECTED["model_manifest"], "C02 input manifest"
    )
    manifest = load_json(manifest_path)
    if manifest.get("identity") != "C02-S.O-rkllm-input-r1":
        raise ValueError("unexpected C02 conversion-input identity")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 7:
        raise ValueError("C02 input manifest must contain exactly seven files")
    observed_files: list[dict[str, Any]] = []
    for expected_file in files:
        relative = expected_file.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"invalid model manifest path: {relative!r}")
        target = model / relative
        observed = {
            "path": relative,
            "bytes": target.stat().st_size,
            "sha256": file_sha256(target),
        }
        if observed != expected_file:
            raise ValueError(
                f"C02 model file differs from manifest: {relative}: "
                f"expected={expected_file}, observed={observed}"
            )
        observed_files.append(observed)
    model_file = model / "model.safetensors"
    if file_sha256(model_file) != EXPECTED["model_safetensors"]:
        raise ValueError("C02 model.safetensors does not match frozen C0-1b output")
    tensor_count = safetensors_tensor_count(model_file)
    if tensor_count != 2_130:
        raise ValueError(f"expected 2,130 serialized tensors, observed {tensor_count}")
    return {
        "manifest": manifest_identity,
        "files": observed_files,
        "tensor_count": tensor_count,
        "transplant_tensor_count": manifest.get("tensor_inventory", {}).get(
            "transplant_tensor_count"
        ),
    }


def verify_a2(
    model: Path,
    dataset_path: Path,
    manifest_path: Path,
    rule_path: Path,
    selection_receipt_path: Path,
) -> dict[str, Any]:
    identities = {
        "rule": verify_sha(rule_path, EXPECTED["a2_rule"], "A2 rule"),
        "manifest": verify_sha(
            manifest_path, EXPECTED["a2_manifest"], "A2 manifest"
        ),
        "selection_receipt": verify_sha(
            selection_receipt_path, EXPECTED["a2_receipt"], "A2 receipt"
        ),
        "dataset": verify_sha(dataset_path, EXPECTED["a2_dataset"], "A2 dataset"),
    }
    rule = load_json(rule_path)
    manifest = load_json(manifest_path)
    selection_receipt = load_json(selection_receipt_path)
    dataset = load_json(dataset_path)
    summary = manifest.get("dataset")
    entries = manifest.get("entries")
    if not isinstance(summary, dict) or not isinstance(entries, list):
        raise ValueError("A2 manifest shape is invalid")
    if not isinstance(dataset, list) or len(dataset) != 128 or len(entries) != 128:
        raise ValueError("A2 must contain exactly 128 dataset rows and manifest entries")
    expected_summary = {
        "records": 128,
        "total_tokens": 105_856,
        "max_tokens": 827,
        "build_truncation_count": 0,
        "source_window_extraction_count": 128,
        "unique_uids": 128,
        "unique_source_sessions": 128,
        "zero_build_truncation": True,
    }
    for key, expected_value in expected_summary.items():
        if summary.get(key) != expected_value:
            raise ValueError(
                f"A2 summary mismatch for {key}: "
                f"expected {expected_value!r}, observed {summary.get(key)!r}"
            )
    if rule.get("status") != "FROZEN":
        raise ValueError("A2 rule is not marked FROZEN")
    if selection_receipt.get("dataset_summary") != summary:
        raise ValueError("A2 receipt and manifest dataset summaries differ")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=True
    )
    token_total = 0
    strata: dict[str, int] = {}
    for index, (item, entry) in enumerate(zip(dataset, entries), 1):
        if not isinstance(item, dict) or set(item) != {"input", "target"}:
            raise ValueError(f"A2 row {index} violates the two-field schema")
        text = item.get("input")
        if not isinstance(text, str) or item.get("target") != "":
            raise ValueError(f"A2 row {index} has invalid input/target types")
        observed_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if observed_sha != entry.get("materialized_sha256"):
            raise ValueError(f"A2 row {index} materialized SHA mismatch")
        observed_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if observed_tokens != entry.get("materialized_tokens") or observed_tokens > 1_024:
            raise ValueError(f"A2 row {index} token count/cap mismatch")
        if entry.get("selection_rank") != index:
            raise ValueError(f"A2 row {index} selection rank mismatch")
        if entry.get("build_truncated") is not False:
            raise ValueError(f"A2 row {index} records converter/build truncation")
        if entry.get("source_window_extracted") is not True:
            raise ValueError(f"A2 row {index} lacks source-window disclosure")
        token_total += observed_tokens
        stratum = entry.get("stratum")
        strata[str(stratum)] = strata.get(str(stratum), 0) + 1
    if token_total != 105_856:
        raise ValueError(f"A2 token total mismatch: {token_total}")
    quotas = {item["id"]: item["quota"] for item in rule.get("strata", [])}
    if strata != quotas:
        raise ValueError(f"A2 stratum quotas differ: expected={quotas}, observed={strata}")
    return {
        "identities": identities,
        "records": len(dataset),
        "tokens": token_total,
        "max_tokens": max(entry["materialized_tokens"] for entry in entries),
        "build_truncations": 0,
        "source_windows": 128,
        "strata": strata,
    }


def scratch_preflight(tmp_dir: Path) -> dict[str, Any]:
    if not (
        Path.cwd().resolve() == tmp_dir.parent.resolve()
        and all(
            Path(os.environ[name]).resolve() == tmp_dir.resolve()
            for name in ("TMPDIR", "TMP", "TEMP")
        )
    ):
        raise ValueError("cwd/TMPDIR/TMP/TEMP are not fixed to the run directory")
    tempfile.tempdir = None
    with tempfile.SpooledTemporaryFile(max_size=1, mode="w+b") as handle:
        handle.write(b"0\0")  # force rollover to the filesystem
        handle.flush()
        os.posix_fallocate(handle.fileno(), SCRATCH_OFFSET, SCRATCH_LENGTH)
        handle.truncate(SCRATCH_TOTAL)
        handle.flush()
        os.fsync(handle.fileno())
        observed_size = os.fstat(handle.fileno()).st_size
    if observed_size != SCRATCH_TOTAL:
        raise ValueError(
            f"scratch preflight size mismatch: expected {SCRATCH_TOTAL}, got {observed_size}"
        )
    disk = shutil.disk_usage(Path.cwd())
    if disk.free < 100_000_000_000:
        raise ValueError(
            f"run filesystem has only {disk.free} free bytes; contract requires >=100 GB"
        )
    return {
        "status": "PASS",
        "offset_bytes": SCRATCH_OFFSET,
        "length_bytes": SCRATCH_LENGTH,
        "total_bytes": observed_size,
        "run_filesystem_free_bytes": disk.free,
    }


def read_cgroup_memory_value(name: str) -> int | None:
    path = Path("/sys/fs/cgroup") / name
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def gpu_total_memory_mib() -> int | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        values = [
            int(line.strip())
            for line in completed.stdout.splitlines()
            if line.strip()
        ]
    except ValueError:
        return None
    return min(values) if values else None


def resource_preflight(
    allow_memory_variance: bool, memory_variance_note: str | None
) -> dict[str, Any]:
    memory_limit = read_cgroup_memory_value("memory.max")
    memory_high = read_cgroup_memory_value("memory.high")
    if memory_limit is None:
        raise ValueError(
            "numeric cgroup v2 memory.max is required; host-level free output is not accepted"
        )
    memory_floor_pass = memory_limit >= MIN_MEMORY_BYTES
    if not memory_floor_pass:
        if not allow_memory_variance:
            raise ValueError(
                "cgroup memory.max is below the frozen 96 GiB floor: "
                f"{memory_limit} B ({memory_limit / GIB:.2f} GiB); "
                "an explicit --allow-memory-variance plus --memory-variance-note is required"
            )
        if not memory_variance_note or not memory_variance_note.strip():
            raise ValueError(
                "--memory-variance-note is required with --allow-memory-variance"
            )
    gpu_total = gpu_total_memory_mib()
    if gpu_total is None or gpu_total < MIN_GPU_MEMORY_MIB:
        raise ValueError(
            f"GPU memory floor failed: observed {gpu_total!r} MiB, "
            f"required >= {MIN_GPU_MEMORY_MIB} MiB"
        )
    return {
        "status": "PASS" if memory_floor_pass else "PASS_WITH_OWNER_MEMORY_VARIANCE",
        "cgroup_memory_max_bytes": memory_limit,
        "cgroup_memory_max_gib": memory_limit / GIB,
        "cgroup_memory_high_bytes": memory_high,
        "frozen_memory_floor_bytes": MIN_MEMORY_BYTES,
        "frozen_memory_floor_gib": 96,
        "memory_floor_pass": memory_floor_pass,
        "memory_variance_authorized": allow_memory_variance,
        "memory_variance_note": memory_variance_note,
        "gpu_memory_total_mib": gpu_total,
        "gpu_memory_floor_mib": MIN_GPU_MEMORY_MIB,
        "gpu_memory_floor_pass": True,
    }


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for distribution in (
        "rkllm_toolkit",
        "torch",
        "transformers",
        "safetensors",
        "numpy",
        "datasets",
        "pyarrow",
        "accelerate",
    ):
        try:
            from importlib.metadata import version

            result[distribution] = version(distribution)
        except Exception:
            result[distribution] = None
    return result


def worker(args: argparse.Namespace) -> int:
    run_dir = Path.cwd().resolve()
    tmp_dir = run_dir / "tmp"
    model = require_dir(args.model, "model")
    dataset = require_file(args.dataset, "dataset")
    model_manifest = require_file(args.model_manifest, "model manifest")
    a2_manifest = require_file(args.a2_manifest, "A2 manifest")
    a2_rule = require_file(args.a2_rule, "A2 rule")
    a2_receipt = require_file(args.a2_receipt, "A2 selection receipt")
    artifact = run_dir / "artifacts" / ARTIFACT_NAME
    preflight_path = run_dir / "preflight.json"
    if artifact.exists() or preflight_path.exists():
        raise ValueError("worker refuses to overwrite an existing artifact/preflight")

    resources = resource_preflight(
        args.allow_memory_variance, args.memory_variance_note
    )
    scratch = scratch_preflight(tmp_dir)
    model_identity = verify_model(model, model_manifest)
    a2_identity = verify_a2(model, dataset, a2_manifest, a2_rule, a2_receipt)
    preflight = {
        "schema": SCHEMA,
        "stage": "preflight",
        "status": "PASS",
        "observed_at": now_iso(),
        "host": platform.node(),
        "platform": platform.platform(),
        "cwd": str(run_dir),
        "temporary_directories": {
            name: os.environ.get(name) for name in ("TMPDIR", "TMP", "TEMP")
        },
        "scratch": scratch,
        "resources": resources,
        "model": model_identity,
        "calibration": a2_identity,
        "versions": package_versions(),
        "contract": {
            "quantized_dtype": "w8a8",
            "quantized_algorithm": "normal",
            "target_platform": "RK3588",
            "optimization_level": 1,
            "num_npu_core": 1,
            "hybrid_rate": 0,
            "max_context": 16_384,
            "device": "cuda",
            "dtype": "bfloat16",
        },
    }
    atomic_json(preflight_path, preflight)
    emit("preflight", status="PASS", path=str(preflight_path))

    from rkllm.api import RKLLM

    llm = RKLLM()
    load_result = llm.load_huggingface(
        model=str(model),
        model_lora=None,
        device="cuda",
        dtype="bfloat16",
        custom_config=None,
        load_weight=True,
    )
    emit("load_huggingface", result=load_result)
    if load_result != 0:
        return 1

    build_result = llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype="w8a8",
        quantized_algorithm="normal",
        target_platform="RK3588",
        num_npu_core=1,
        extra_qparams=None,
        dataset=str(dataset),
        hybrid_rate=0,
        max_context=16_384,
    )
    emit("build", result=build_result)
    if build_result != 0:
        return 1

    artifact.parent.mkdir(parents=True, exist_ok=False)
    export_result = llm.export_rkllm(str(artifact))
    emit("export_rkllm", result=export_result)
    if export_result != 0:
        return 1
    emit(
        "artifact",
        path=str(artifact),
        bytes=artifact.stat().st_size,
        sha256=file_sha256(artifact),
    )
    return 0


def parse_proc_stat(path: Path) -> tuple[int, int] | None:
    try:
        raw = path.read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        return int(path.parent.name), int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def process_tree(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        parsed = parse_proc_stat(stat_path)
        if parsed is not None:
            parents[parsed[0]] = parsed[1]
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


def proc_memory(pid: int) -> tuple[int, int]:
    rss = hwm = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1])
    except (OSError, ValueError):
        pass
    return rss, hwm


def mem_available_kib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def cgroup_memory_sample() -> tuple[int | None, int | None]:
    return (
        read_cgroup_memory_value("memory.current"),
        read_cgroup_memory_value("memory.peak"),
    )


def cgroup_memory_events() -> dict[str, int]:
    events: dict[str, int] = {}
    try:
        lines = Path("/sys/fs/cgroup/memory.events").read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return events
    for line in lines:
        fields = line.split()
        if len(fields) == 2:
            try:
                events[fields[0]] = int(fields[1])
            except ValueError:
                continue
    return events


def gpu_memory_mib() -> int | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        return sum(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())
    except ValueError:
        return None


def parse_events(stdout_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not stdout_path.exists():
        return events
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("event"), str):
            events.append(payload)
    return events


def orchestrate(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser()
    if not run_dir.is_absolute():
        raise SystemExit("--run-dir must be absolute")
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise SystemExit(f"refusing to reuse run directory: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir()
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    resources_path = run_dir / "resource_monitor.tsv"
    receipt_path = run_dir / "receipt.json"

    forwarded = [
        "--worker",
        "--model",
        str(require_dir(args.model, "model")),
        "--model-manifest",
        str(require_file(args.model_manifest, "model manifest")),
        "--dataset",
        str(require_file(args.dataset, "dataset")),
        "--a2-manifest",
        str(require_file(args.a2_manifest, "A2 manifest")),
        "--a2-rule",
        str(require_file(args.a2_rule, "A2 rule")),
        "--a2-receipt",
        str(require_file(args.a2_receipt, "A2 selection receipt")),
    ]
    if args.allow_memory_variance:
        forwarded.append("--allow-memory-variance")
        forwarded.extend(["--memory-variance-note", args.memory_variance_note])
    command = [sys.executable, str(Path(__file__).resolve()), *forwarded]
    environment = os.environ.copy()
    for name in ("TMPDIR", "TMP", "TEMP"):
        environment[name] = str(tmp_dir)
    environment["PYTHONUNBUFFERED"] = "1"

    started_iso = now_iso()
    started_monotonic = time.monotonic()
    peak_tree_rss = 0
    peak_root_hwm = 0
    peak_gpu = 0
    peak_cgroup_current = 0
    peak_cgroup_kernel = 0
    minimum_mem_available: int | None = None
    cgroup_events_before = cgroup_memory_events()
    return_code = 127
    interrupted = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        resources_path.write_text(
            "timestamp_epoch_s\troot_pid\tprocess_count\troot_rss_kib\t"
            "root_hwm_kib\ttree_rss_kib\thost_mem_available_kib\t"
            "cgroup_memory_current_bytes\tcgroup_memory_peak_bytes\t"
            "gpu_used_mib\n",
            encoding="utf-8",
        )
        try:
            while True:
                tree = process_tree(process.pid)
                root_rss, root_hwm = proc_memory(process.pid)
                tree_rss = sum(proc_memory(pid)[0] for pid in tree)
                available = mem_available_kib()
                cgroup_current, cgroup_peak = cgroup_memory_sample()
                gpu = gpu_memory_mib()
                peak_tree_rss = max(peak_tree_rss, tree_rss)
                peak_root_hwm = max(peak_root_hwm, root_hwm)
                if gpu is not None:
                    peak_gpu = max(peak_gpu, gpu)
                if cgroup_current is not None:
                    peak_cgroup_current = max(peak_cgroup_current, cgroup_current)
                if cgroup_peak is not None:
                    peak_cgroup_kernel = max(peak_cgroup_kernel, cgroup_peak)
                if available is not None:
                    minimum_mem_available = (
                        available
                        if minimum_mem_available is None
                        else min(minimum_mem_available, available)
                    )
                with resources_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{time.time():.6f}\t{process.pid}\t{len(tree)}\t"
                        f"{root_rss}\t{root_hwm}\t{tree_rss}\t"
                        f"{available if available is not None else 'NA'}\t"
                        f"{cgroup_current if cgroup_current is not None else 'NA'}\t"
                        f"{cgroup_peak if cgroup_peak is not None else 'NA'}\t"
                        f"{gpu if gpu is not None else 'NA'}\n"
                    )
                return_code = process.poll() if process.poll() is not None else return_code
                if process.poll() is not None:
                    return_code = int(process.returncode)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            interrupted = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()

    ended_iso = now_iso()
    events = parse_events(stdout_path)
    results = {
        event["event"]: event.get("result")
        for event in events
        if event["event"] in {"load_huggingface", "build", "export_rkllm"}
    }
    artifact_path = run_dir / "artifacts" / ARTIFACT_NAME
    artifact = None
    if artifact_path.is_file():
        artifact = {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": file_sha256(artifact_path),
        }
    status = (
        "PASS"
        if not interrupted
        and return_code == 0
        and results
        == {"load_huggingface": 0, "build": 0, "export_rkllm": 0}
        and artifact is not None
        else "FAIL"
    )
    receipt = {
        "schema": SCHEMA,
        "stage": "C0-3",
        "status": status,
        "started_at": started_iso,
        "ended_at": ended_iso,
        "wall_seconds": time.monotonic() - started_monotonic,
        "host": platform.node(),
        "command": command,
        "run_directory": str(run_dir),
        "temporary_directories": {
            name: environment[name] for name in ("TMPDIR", "TMP", "TEMP")
        },
        "return_code": return_code,
        "interrupted": interrupted,
        "memory_variance_requested": {
            "allowed": args.allow_memory_variance,
            "note": args.memory_variance_note,
        },
        "sdk_results": results,
        "artifact": artifact,
        "resource_peaks": {
            "root_process_vm_hwm_kib": peak_root_hwm,
            "process_tree_rss_kib": peak_tree_rss,
            "gpu_memory_used_mib": peak_gpu,
            "cgroup_memory_current_bytes": peak_cgroup_current,
            "cgroup_memory_peak_kernel_bytes": peak_cgroup_kernel,
            "cgroup_memory_events_before": cgroup_events_before,
            "cgroup_memory_events_after": cgroup_memory_events(),
            "minimum_host_mem_available_kib_not_instance_quota": minimum_mem_available,
        },
        "evidence": {
            "preflight": str(run_dir / "preflight.json"),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "resource_monitor": str(resources_path),
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
        },
        "scope": "C02 A2 W8A8 conversion only; no quality evaluation",
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps({"status": status, "receipt": str(receipt_path), "artifact": artifact}))
    return 0 if status == "PASS" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--a2-manifest", type=Path, required=True)
    parser.add_argument("--a2-rule", type=Path, required=True)
    parser.add_argument("--a2-receipt", type=Path, required=True)
    parser.add_argument(
        "--allow-memory-variance",
        action="store_true",
        help="Proceed below the frozen 96 GiB cgroup limit only after explicit owner authorization.",
    )
    parser.add_argument(
        "--memory-variance-note",
        help="Auditable owner authorization/reason; required with --allow-memory-variance.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker and args.run_dir is None:
        parser.error("--run-dir is required")
    if args.allow_memory_variance and not args.memory_variance_note:
        parser.error("--memory-variance-note is required with --allow-memory-variance")
    return args


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)
    return orchestrate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        emit("exception", exception_type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
        raise SystemExit(1)
