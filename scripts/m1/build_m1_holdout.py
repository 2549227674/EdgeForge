#!/usr/bin/env python3
"""Freeze the M1 family × task_type holdout from rendered length metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lengths(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def allocate(groups: dict[tuple[str, str], list[dict[str, Any]]], target: int) -> dict[tuple[str, str], int]:
    total = sum(len(rows) for rows in groups.values())
    if target < len(groups) or target > total:
        raise ValueError(f"target={target} incompatible with {len(groups)} strata and total={total}")
    counts = {key: 1 for key in groups}
    remaining = target - len(groups)
    capacities = {key: len(rows) - 1 for key, rows in groups.items()}
    capacity_total = sum(capacities.values())
    quotas = {
        key: (remaining * capacities[key] / capacity_total if capacity_total else 0.0)
        for key in groups
    }
    for key, quota in quotas.items():
        counts[key] += math.floor(quota)
    current = sum(counts.values())
    order = sorted(
        groups,
        key=lambda key: (quotas[key] - math.floor(quotas[key]), capacities[key], key),
        reverse=True,
    )
    for key in order:
        if current == target:
            break
        if counts[key] < len(groups[key]):
            counts[key] += 1
            current += 1
    if sum(counts.values()) != target:
        raise RuntimeError("failed to allocate exact holdout target")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=Path, required=True)
    parser.add_argument("--mix-manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    rows = read_lengths(args.lengths)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["task_type"])].append(row)
    counts = allocate(groups, args.target)

    selected: list[dict[str, Any]] = []
    strata: dict[str, dict[str, int]] = {}
    for key in sorted(groups):
        family, task_type = key
        candidates = sorted(groups[key], key=lambda row: row["uid"])
        group_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}\0{family}\0{task_type}".encode()).digest()[:8],
            "big",
        )
        random.Random(group_seed).shuffle(candidates)
        chosen = candidates[: counts[key]]
        selected.extend(chosen)
        strata[f"{family}|{task_type}"] = {
            "pool_records": len(candidates),
            "holdout_records": len(chosen),
        }

    selected.sort(key=lambda row: row["uid"])
    uid_digest = hashlib.sha256("\n".join(row["uid"] for row in selected).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "purpose": "M1 validation holdout; excluded from SFT training and OPD prompt pools",
        "selection": "family_task_type_stratified_largest_remainder_then_seeded_shuffle",
        "seed": args.seed,
        "pool_records": len(rows),
        "requested_records": args.target,
        "selected_records": len(selected),
        "selected_uid_sha256": uid_digest,
        "mix_records_manifest_sha256": sha256(args.mix_manifest),
        "template_sha256": sha256(args.template),
        "lengths_metadata_sha256": sha256(args.lengths),
        "strata": strata,
        "records": [
            {
                "uid": row["uid"],
                "dataset": row["dataset"],
                "family": row["family"],
                "task_type": row["task_type"],
                "token_count": row["token_count"],
            }
            for row in selected
        ],
    }
    if len({row["uid"] for row in selected}) != len(selected):
        raise ValueError("duplicate UIDs in holdout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
