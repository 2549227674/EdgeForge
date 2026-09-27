#!/usr/bin/env python3
"""Classify every non-cache M0 line C file before it enters the pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def role_for(path: Path) -> tuple[str, bool]:
    value = path.as_posix()
    if "/.cache/" in value:
        raise ValueError("cache files must be filtered before classification")
    if value.endswith(".gitattributes") or value.endswith("README.md"):
        return ("repository_metadata", False)
    if "kernelbench-" in value:
        return ("kernelbench_excluded", False)
    if value.startswith("data/archive_parquet/"):
        return ("canonical_parquet", path.suffix == ".parquet")
    if "Glint-Research__Fable-5-traces/claude/history.jsonl" in value:
        return ("cli_history", False)
    if "/paste-cache/" in value or "/tool-results/" in value:
        return ("security_scan_only", False)
    if "Glint-Research__Fable-5-traces/claude/projects/" in value and path.suffix == ".jsonl":
        return ("subagent_session" if "/subagents/" in value else "session", True)
    if "lambda__hermes-agent-reasoning-traces/data/train-" in value:
        return ("duplicate_export", False)
    if "lambda__hermes-agent-reasoning-traces/data/" in value and path.suffix == ".parquet":
        return ("canonical_parquet", True)
    if "Crownelius__Complete-FABLE.5-traces-2M/data/" in value:
        return ("superseded_partial_export", False)
    if path.suffix == ".jsonl":
        if "AletheiaResearch__GLM-5.2-Agent" in value or "claude-opus-4.8-pi-traces" in value:
            return ("superseded_partial_export", False)
        return ("session", True)
    if path.name == "manifest.csv":
        return ("repository_metadata", False)
    return ("repository_metadata", False)


def dataset_for(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part in {"archive", "archive_parquet"} and index + 1 < len(parts):
            return parts[index + 1].replace("__", "/")
    raise ValueError(f"file is outside an archive root: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for root in args.root:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "/.cache/" in path.as_posix():
                continue
            role, included = role_for(path)
            rows.append(
                {
                    "path": path.as_posix(),
                    "dataset": dataset_for(path),
                    "role": role,
                    "included": str(included).lower(),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "dataset", "role", "included"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} file roles to {args.output}")


if __name__ == "__main__":
    main()
