#!/usr/bin/env python3
"""Capture a fail-closed identity receipt from the live R2.2 C01-S processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
from typing import Any


RUN_DIR = Path("results/t2/c01_s_family_80_20_half_r1")
EXPECTED_IDENTITIES = {
    "scripts/m1/train_recipe_race.py": None,
    "results/t1/family_80_20_half_plan_r1/family_80_20.parquet":
        "61ece8bad05f4156b4b20fa734c2b7e1f804fc879aa50d7a28cceb790042820e",
    "results/t1/family_80_20_half_plan_r1/holdout.parquet":
        "0a28a7f7b52a01f7a73ccdf927f98ac21902f8037ef236db4f86d5024a67d73b",
    "models/gemma-4-E4B-it/chat_template.jinja":
        "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
    "models/gemma-4-E4B-it/config.json": None,
}
SELECTED_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "LOCAL_RANK",
    "RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)
EXPECTED_ARGV_PAIRS = (
    ("--model", "models/gemma-4-E4B-it"),
    ("--plan", "results/t1/family_80_20_half_plan_r1/family_80_20.parquet"),
    ("--holdout", "results/t1/family_80_20_half_plan_r1/holdout.parquet"),
    ("--output-dir", str(RUN_DIR)),
    ("--recipe", "family_80_20"),
    ("--run-kind", "full"),
    ("--race-steps", "4096"),
    ("--full-steps", "4096"),
    ("--eval-every", "1024"),
    ("--checkpoint-every", "512"),
    ("--expected-world-size", "2"),
    ("--expected-plan-sha256", EXPECTED_IDENTITIES[
        "results/t1/family_80_20_half_plan_r1/family_80_20.parquet"
    ]),
    ("--expected-holdout-sha256", EXPECTED_IDENTITIES[
        "results/t1/family_80_20_half_plan_r1/holdout.parquet"
    ]),
    ("--expected-template-sha256", EXPECTED_IDENTITIES[
        "models/gemma-4-E4B-it/chat_template.jinja"
    ]),
    ("--expected-input-tokens", "148425311"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_nul_file(path: Path) -> list[str]:
    return [
        item.decode("utf-8", errors="replace")
        for item in path.read_bytes().split(b"\0")
        if item
    ]


def process_record(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    cmdline = read_nul_file(root / "cmdline")
    environment = {}
    for item in read_nul_file(root / "environ"):
        key, separator, value = item.partition("=")
        if separator and key in SELECTED_ENVIRONMENT:
            environment[key] = value
    raw_stat = (root / "stat").read_text(encoding="utf-8")
    fields = raw_stat[raw_stat.rfind(")") + 2 :].split()
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    boot_epoch = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
        if line.startswith("btime ")
    )
    start_epoch = boot_epoch + int(fields[19]) / clock_ticks
    fds = {}
    for fd in (0, 1, 2):
        try:
            fds[str(fd)] = os.readlink(root / "fd" / str(fd))
        except OSError as error:
            fds[str(fd)] = f"UNREADABLE:{type(error).__name__}"
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "process_group": int(fields[2]),
        "session": int(fields[3]),
        "tty_nr": int(fields[4]),
        "state": fields[0],
        "started_at_utc": datetime.fromtimestamp(
            start_epoch, timezone.utc
        ).isoformat(),
        "cwd": os.readlink(root / "cwd"),
        "executable": os.readlink(root / "exe"),
        "cmdline": cmdline,
        "selected_environment": environment,
        "file_descriptors": fds,
    }


def find_training_processes() -> list[dict[str, Any]]:
    records = []
    for item in sorted(Path("/proc").iterdir(), key=lambda path: path.name):
        if not item.name.isdigit():
            continue
        try:
            cmdline = read_nul_file(item / "cmdline")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = " ".join(cmdline)
        if (
            "train_recipe_race.py" not in command
            or str(RUN_DIR) not in command
            or str(item.name) == str(os.getpid())
        ):
            continue
        try:
            records.append(process_record(int(item.name)))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return records


def inspect_train_log() -> dict[str, Any]:
    path = RUN_DIR / "train_log.jsonl"
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_events = [item for item in events if item.get("event") == "train_step"]
    steps = [item.get("optimizer_step") for item in train_events]
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError("live train log is not contiguous from step 1")
    if any(
        not math.isfinite(item.get("weighted_loss", math.nan))
        for item in train_events
    ):
        raise ValueError("live train log contains a non-finite loss")
    return {
        "path": str(path),
        "sha256_at_capture": sha256(path),
        "event_count": len(events),
        "last_optimizer_step": steps[-1],
        "validation_steps": [
            item["optimizer_step"]
            for item in events
            if item.get("event") == "validation"
        ],
        "checkpoint_steps": [
            item["optimizer_step"]
            for item in events
            if item.get("event") == "checkpoint"
        ],
        "all_observed_losses_finite": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    incomplete = args.output.with_name(args.output.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(incomplete)

    processes = find_training_processes()
    ranks = sorted(
        [
            record
            for record in processes
            if record["selected_environment"].get("RANK") in {"0", "1"}
        ],
        key=lambda record: record["selected_environment"]["RANK"],
    )
    if [record["selected_environment"]["RANK"] for record in ranks] != ["0", "1"]:
        raise ValueError("live two-rank process set is incomplete")
    for record in ranks:
        environment = record["selected_environment"]
        if environment.get("WORLD_SIZE") != "2":
            raise ValueError("live rank WORLD_SIZE differs from 2")
        if environment.get("OMP_NUM_THREADS") != "4":
            raise ValueError("live rank OMP_NUM_THREADS differs from 4")
        if environment.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True":
            raise ValueError("live rank allocator configuration differs")
        cmdline = record["cmdline"]
        if not cmdline or not cmdline[0].endswith("/edgeforge-m1/venv/bin/python"):
            raise ValueError("live rank argv[0] does not use the frozen SFT venv")
        if "scripts/m1/train_recipe_race.py" not in cmdline:
            raise ValueError("live rank does not execute the frozen training entrypoint")
        for flag, expected_value in EXPECTED_ARGV_PAIRS:
            try:
                index = cmdline.index(flag)
            except ValueError as error:
                raise ValueError(f"live rank is missing {flag}") from error
            if index + 1 >= len(cmdline) or cmdline[index + 1] != expected_value:
                raise ValueError(f"live rank {flag} differs from the frozen command")
        if record["tty_nr"] != 0 or record["file_descriptors"]["0"] != "/dev/null":
            raise ValueError("live rank is not detached from the launching terminal")
        expected_log = str((Path.cwd() / (str(RUN_DIR) + ".launch.log")))
        if record["file_descriptors"]["1"] != expected_log:
            raise ValueError("live rank stdout is not the frozen launch log")
        if record["file_descriptors"]["2"] != expected_log:
            raise ValueError("live rank stderr is not the frozen launch log")

    file_identities = []
    for name, expected_sha in EXPECTED_IDENTITIES.items():
        path = Path(name)
        actual_sha = sha256(path)
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(f"frozen identity mismatch: {name}")
        file_identities.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": actual_sha}
        )

    config = json.loads(
        Path("models/gemma-4-E4B-it/config.json").read_text(encoding="utf-8")
    )
    text_config = config["text_config"]
    actual_model_shape = {
        "model_type": config["model_type"],
        "num_hidden_layers": text_config["num_hidden_layers"],
        "hidden_size": text_config["hidden_size"],
        "sliding_window": text_config["sliding_window"],
    }
    expected_model_shape = {
        "model_type": "gemma4",
        "num_hidden_layers": 42,
        "hidden_size": 2560,
        "sliding_window": 512,
    }
    if actual_model_shape != expected_model_shape:
        raise ValueError("live base model shape differs from the frozen shape")

    packages = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "pyarrow",
        "safetensors",
    ):
        packages[name] = metadata.version(name)
    gpu_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    if len(gpu_rows) != 2 or any("RTX PRO 6000" not in row for row in gpu_rows):
        raise ValueError("live GPU inventory is not two RTX PRO 6000 devices")

    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "candidate": "C01-S",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "scope": "live process, command, environment, input and runtime identity",
        "limitation": (
            "File hashes were observed while the already-running ranks were alive; "
            "this is not a cryptographic attestation of pre-exec file contents."
        ),
        "processes": processes,
        "rank_pids": [record["pid"] for record in ranks],
        "file_identities": file_identities,
        "model_shape": actual_model_shape,
        "packages": packages,
        "gpus": gpu_rows,
        "train_log": inspect_train_log(),
        "gates": {
            "two_live_ranks": "PASS",
            "rank_world_size_and_environment": "PASS",
            "detached_stdio": "PASS",
            "frozen_plan_holdout_template_identities": "PASS",
            "training_script_and_model_config_hashes": "PASS_AT_CAPTURE",
            "model_42_layers_hidden_2560_sliding_512": "PASS",
            "two_rtx_pro_6000_devices": "PASS",
            "contiguous_finite_log_at_capture": "PASS",
        },
        "generator": str(Path(__file__)),
        "generator_sha256": sha256(Path(__file__)),
    }
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    incomplete.rename(args.output)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
