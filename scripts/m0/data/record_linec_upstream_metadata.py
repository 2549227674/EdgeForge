#!/usr/bin/env python3
"""Freeze the read-only HF metadata used by M0 line C gate 1."""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from huggingface_hub import HfApi


REPOSITORIES = (
    "Glint-Research/Fable-5-traces",
    "Crownelius/Complete-FABLE.5-traces-2M",
    "lambda/hermes-agent-reasoning-traces",
    "AletheiaResearch/GLM-5.2-Agent",
    "armand0e/qwen3.7-max-pi-traces",
    "WithinUsAI/claude_mythos_distilled_25k",
    "armand0e/claude-opus-4.8-pi-traces",
    "Infatoshi/kernelbench-hard-traces",
    "Infatoshi/kernelbench-mega-traces",
)


def local_readme_license(repo_id: str) -> str | None:
    path = Path("data/archive") / repo_id.replace("/", "__") / "README.md"
    if not path.exists():
        return None
    match = re.search(r"^license:\s*(.+?)\s*$", path.read_text(encoding="utf-8", errors="replace"), re.M)
    return match.group(1) if match else None


def local_snapshot_revision(repo_id: str) -> str | None:
    root = Path("data/archive") / repo_id.replace("/", "__") / ".cache/huggingface/trees"
    revisions = sorted(path.stem for path in root.glob("*.json"))
    return revisions[0] if len(revisions) == 1 else None


def main() -> None:
    api = HfApi()
    records = []
    output = Path("data/pipeline/gate1_upstream_metadata.json")
    for repo_id in REPOSITORIES:
        info_error = None
        try:
            info = api.dataset_info(repo_id)
            default_revision = info.sha
            card_license = info.card_data.get("license") if info.card_data else None
        except httpx.HTTPError as error:
            default_revision = local_snapshot_revision(repo_id)
            card_license = None
            info_error = f"{type(error).__name__}: {error}"
        viewer_data: dict[str, object]
        try:
            viewer_data = fetch_size(repo_id)
        except httpx.HTTPError as error:
            viewer_data = {"error": f"{type(error).__name__}: {error}"}
        records.append(
            {
                "repo_id": repo_id,
                "default_revision": default_revision,
                "hub_info_error": info_error,
                "license_from_hf_card": card_license,
                "license_from_local_snapshot_card": local_readme_license(repo_id),
                "dataset_viewer_size": viewer_data,
            }
        )
        write_records(output, records)
        print(f"{repo_id}: {default_revision}")


def fetch_size(repo_id: str) -> dict[str, object]:
    last_error: httpx.HTTPError | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=30) as client:
                response = client.get("https://datasets-server.huggingface.co/size", params={"dataset": repo_id})
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as error:
            last_error = error
            time.sleep(attempt + 1)
    assert last_error is not None
    raise last_error


def write_records(output: Path, records: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"schema_version": 1, "generated_at": datetime.now(UTC).isoformat(), "datasets": records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
