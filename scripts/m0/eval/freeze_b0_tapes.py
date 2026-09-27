#!/usr/bin/env python3
"""Freeze a small, deterministic set of B0 agent trajectories as replay tapes.

The tapes are inference-load fixtures, not a second Terminal-Bench run.  Each
request preserves the recorded B0 conversation prefix before an agent response,
so every tape contains at least two progressively longer, shared-prefix prompts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "results/baseline_e4b_q4km"
TAPES_DIR = ROOT / "traces/tapes"
MANIFEST_PATH = TAPES_DIR / "manifest.json"
TRAJECTORY_HASHES_PATH = ROOT / "traces/trajectories_sha256.txt"

# Chosen before QAT inference: five different locked-task trajectories, all
# with at least two agent turns.  The short trajectories keep this M0 systems
# smoke bounded while still exercising repeated prompt prefixes.
SOURCES = (
    "regex-log__HM89SBD",
    "openssl-selfsigned-cert__oCvmg5c",
    "multi-source-data-merger__bUyWMFa",
    "constraints-scheduling__aaSRUnD",
    "sqlite-db-truncate__Z9iGbrN",
)

SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "min_p": 0.0,
    "max_tokens": 32768,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def observation_text(observation: Any) -> str:
    """Turn the ATIF tool-observation object into the next user message."""
    if isinstance(observation, str):
        return observation
    if isinstance(observation, dict):
        results = observation.get("results")
        if isinstance(results, list):
            chunks = [
                item.get("content", "")
                for item in results
                if isinstance(item, dict) and item.get("content")
            ]
            if chunks:
                return "\n\n".join(chunks)
    return json.dumps(observation, ensure_ascii=False, sort_keys=True)


def freeze_tape(source_name: str) -> tuple[dict[str, Any], Path, str]:
    source = SOURCE_ROOT / source_name / "agent/trajectory.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError(f"invalid or empty trajectory: {source}")

    history: list[dict[str, str]] = []
    requests: list[dict[str, Any]] = []
    for step in steps:
        source_role = step.get("source")
        message = step.get("message")
        if source_role == "user":
            if not isinstance(message, str) or not message:
                raise RuntimeError(f"empty initial user message in {source}")
            history.append({"role": "user", "content": message})
            continue
        if source_role != "agent":
            continue
        if not history:
            raise RuntimeError(f"agent turn has no prompt history in {source}")
        requests.append(
            {
                "request_id": f"{source_name}:{len(requests) + 1}",
                "recorded_step_id": step.get("step_id"),
                "messages": list(history),
                "reference_prompt_tokens": step.get("metrics", {}).get("prompt_tokens"),
                "reference_completion_tokens": step.get("metrics", {}).get(
                    "completion_tokens"
                ),
                "reference_cached_tokens": step.get("metrics", {}).get("cached_tokens"),
            }
        )
        if isinstance(message, str) and message:
            history.append({"role": "assistant", "content": message})
        if "observation" in step:
            history.append({"role": "user", "content": observation_text(step["observation"])})

    if len(requests) < 2:
        raise RuntimeError(f"tape lacks a repeated-prefix request pair: {source}")
    task_id = source_name.split("__", 1)[0]
    tape = {
        "schema_version": "1.0",
        "kind": "edgeforge_b0_inference_replay_tape",
        "tape_id": source_name,
        "task_id": task_id,
        "source_trajectory": source.relative_to(ROOT).as_posix(),
        "source_trajectory_sha256": sha256_file(source),
        "source_model": "B0-PTQ-Q4KM",
        "sampling": SAMPLING,
        "requests": requests,
    }
    destination = TAPES_DIR / f"{source_name}.json"
    write_json(destination, tape)
    return tape, destination, sha256_file(source)


def main() -> None:
    TAPES_DIR.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_HASHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for source_name in SOURCES:
        tape, destination, source_hash = freeze_tape(source_name)
        entries.append(
            {
                "tape_id": tape["tape_id"],
                "task_id": tape["task_id"],
                "path": destination.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(destination),
                "requests": len(tape["requests"]),
                "source_trajectory": tape["source_trajectory"],
                "source_trajectory_sha256": source_hash,
            }
        )
    # The tape manifest records its five selected source trajectories.  The
    # handoff manifest is deliberately broader: it fingerprints every B0 ATIF
    # trajectory retained in the versioned results tree, so a later clone can
    # verify the full baseline corpus without a separate artifact handoff.
    all_source_trajectories = sorted(SOURCE_ROOT.glob("*/agent/trajectory.json"))
    if not all_source_trajectories:
        raise RuntimeError(f"no B0 trajectories found under {SOURCE_ROOT}")
    source_hash_lines = [
        f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in all_source_trajectories
    ]

    manifest = {
        "schema_version": "1.0",
        "kind": "edgeforge_b0_inference_replay_tape_manifest",
        "source_model": "B0-PTQ-Q4KM",
        "source_run": "results/baseline_e4b_q4km",
        "source_trajectory_hash_manifest": TRAJECTORY_HASHES_PATH.relative_to(
            ROOT
        ).as_posix(),
        "source_trajectory_hash_entries": len(source_hash_lines),
        "selection": {
            "method": "predeclared five distinct B0 trajectories with two or more agent turns",
            "count": len(entries),
            "purpose": "fixed cache-warm systems replay load for all model variants",
        },
        "sampling": SAMPLING,
        "tapes": entries,
    }
    write_json(MANIFEST_PATH, manifest)
    TRAJECTORY_HASHES_PATH.write_text(
        "\n".join(source_hash_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
                "tapes": len(entries),
                "requests_per_pass": sum(entry["requests"] for entry in entries),
                "manifest_sha256": sha256_file(MANIFEST_PATH),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
