#!/usr/bin/env python3
"""Freeze the >8,192-token source subset of the existing M1 holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-source-tokens-exclusive", type=int, default=8192)
    args = parser.parse_args()

    source = json.loads(args.holdout.read_text(encoding="utf-8"))
    records = source.get("records")
    if not isinstance(records, list):
        raise ValueError("holdout manifest has no records list")
    selected = sorted(
        (
            {
                "uid": str(row["uid"]),
                "dataset": row["dataset"],
                "family": row["family"],
                "task_type": row["task_type"],
                "source_token_count": int(row["token_count"]),
            }
            for row in records
            if int(row["token_count"]) > args.min_source_tokens_exclusive
        ),
        key=lambda row: row["uid"],
    )
    uid_sha = hashlib.sha256("\n".join(row["uid"] for row in selected).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "purpose": "fixed long-context regression probe; subset of M1 holdout source UIDs",
        "selection": f"source frozen-render length > {args.min_source_tokens_exclusive}",
        "source_holdout": str(args.holdout),
        "source_holdout_sha256": sha256(args.holdout),
        "source_holdout_records": len(records),
        "selected_records": len(selected),
        "selected_uid_sha256": uid_sha,
        "records": selected,
    }
    if len({row["uid"] for row in selected}) != len(selected):
        raise ValueError("duplicate source UID in long holdout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
