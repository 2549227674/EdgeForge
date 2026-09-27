#!/usr/bin/env python3
"""Independently verify the frozen C01-S plan and holdout tensor payloads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PLAN_SHA256 = "61ece8bad05f4156b4b20fa734c2b7e1f804fc879aa50d7a28cceb790042820e"
HOLDOUT_SHA256 = "0a28a7f7b52a01f7a73ccdf927f98ac21902f8037ef236db4f86d5024a67d73b"
TEMPLATE_SHA256 = "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_lengths(array: Any) -> np.ndarray:
    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    return np.diff(offsets)


def shifted_supervised_counts(labels: Any) -> np.ndarray:
    offsets = labels.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    values = labels.values.to_numpy(zero_copy_only=False)
    prefix = np.empty(values.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(values != -100, dtype=np.int64, out=prefix[1:])
    starts = np.minimum(offsets[:-1] + 1, offsets[1:])
    return prefix[offsets[1:]] - prefix[starts]


def scan_holdout(path: Path) -> dict[str, Any]:
    rows = 0
    input_tokens = 0
    supervised_tokens = 0
    uids: set[str] = set()
    source_uids: set[str] = set()
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=512,
        columns=["uid", "source_uid", "input_ids", "labels"],
    ):
        input_ids = batch.column("input_ids")
        labels = batch.column("labels")
        lengths = list_lengths(input_ids)
        label_lengths = list_lengths(labels)
        if not np.array_equal(lengths, label_lengths):
            raise ValueError("holdout input_ids and labels lengths differ")
        supervised = shifted_supervised_counts(labels)
        batch_uids = batch.column("uid").to_pylist()
        batch_source_uids = batch.column("source_uid").to_pylist()
        rows += batch.num_rows
        input_tokens += int(lengths.sum())
        supervised_tokens += int(supervised.sum())
        uids.update(map(str, batch_uids))
        source_uids.update(map(str, batch_source_uids))
    return {
        "records": rows,
        "unique_uids": len(uids),
        "source_uids": source_uids,
        "input_tokens_from_offsets": input_tokens,
        "shifted_supervised_tokens_from_labels": supervised_tokens,
    }


def scan_plan(path: Path) -> dict[str, Any]:
    rows = 0
    input_tokens = 0
    supervised_tokens = 0
    uids: set[str] = set()
    source_uids: set[str] = set()
    family_groups: Counter[str] = Counter()
    step_records: Counter[int] = Counter()
    step_input: Counter[int] = Counter()
    step_supervised: Counter[int] = Counter()
    step_declared_supervised: dict[int, int] = {}
    rank_audits = {
        0: {"rank": 0, "records": 0, "input_tokens": 0, "supervised_tokens": 0},
        1: {"rank": 1, "records": 0, "input_tokens": 0, "supervised_tokens": 0},
    }
    expected_plan_index = 0
    previous_step = -1

    columns = [
        "uid",
        "source_uid",
        "family",
        "input_ids",
        "labels",
        "plan_index",
        "optimizer_step",
        "scheduled_tokens",
        "shifted_supervised_tokens",
        "optimizer_step_supervised_tokens",
    ]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
        input_ids = batch.column("input_ids")
        labels = batch.column("labels")
        lengths = list_lengths(input_ids)
        label_lengths = list_lengths(labels)
        if not np.array_equal(lengths, label_lengths):
            raise ValueError("plan input_ids and labels lengths differ")
        actual_supervised = shifted_supervised_counts(labels)
        scheduled = batch.column("scheduled_tokens").to_numpy(zero_copy_only=False)
        declared_supervised = batch.column("shifted_supervised_tokens").to_numpy(
            zero_copy_only=False
        )
        if not np.array_equal(lengths, scheduled):
            raise ValueError("scheduled_tokens differs from input_ids offsets")
        if not np.array_equal(actual_supervised, declared_supervised):
            raise ValueError("shifted_supervised_tokens differs from labels")

        plan_indices = batch.column("plan_index").to_numpy(zero_copy_only=False)
        steps = batch.column("optimizer_step").to_numpy(zero_copy_only=False)
        declared_step_supervision = batch.column(
            "optimizer_step_supervised_tokens"
        ).to_numpy(zero_copy_only=False)
        families = batch.column("family").to_pylist()
        batch_uids = list(map(str, batch.column("uid").to_pylist()))
        batch_source_uids = list(map(str, batch.column("source_uid").to_pylist()))

        for index in range(batch.num_rows):
            plan_index = int(plan_indices[index])
            step = int(steps[index])
            if plan_index != expected_plan_index:
                raise ValueError(
                    f"non-contiguous plan_index {plan_index} != {expected_plan_index}"
                )
            if step < previous_step:
                raise ValueError(f"optimizer steps are not sorted at {step}")
            expected_plan_index += 1
            previous_step = step

            local_index = step_records[step]
            rank = local_index % 2
            row_input = int(lengths[index])
            row_supervised = int(actual_supervised[index])
            step_records[step] += 1
            step_input[step] += row_input
            step_supervised[step] += row_supervised
            declared = int(declared_step_supervision[index])
            if step in step_declared_supervised and step_declared_supervised[step] != declared:
                raise ValueError(f"inconsistent declared supervision for step {step}")
            step_declared_supervised[step] = declared
            rank_audits[rank]["records"] += 1
            rank_audits[rank]["input_tokens"] += row_input
            rank_audits[rank]["supervised_tokens"] += row_supervised
            family_groups[
                "anthropic"
                if families[index] == "claimed_anthropic_from_source_label"
                else "other"
            ] += 1

        rows += batch.num_rows
        input_tokens += int(lengths.sum())
        supervised_tokens += int(actual_supervised.sum())
        uids.update(batch_uids)
        source_uids.update(batch_source_uids)

    steps = sorted(step_records)
    if steps != list(range(4096)):
        raise ValueError("plan optimizer step schedule is not exactly 0..4095")
    for step in steps:
        if step_supervised[step] != step_declared_supervised[step]:
            raise ValueError(f"optimizer step supervision mismatch at {step}")
    return {
        "records": rows,
        "unique_uids": len(uids),
        "source_uids": source_uids,
        "input_tokens_from_offsets": input_tokens,
        "shifted_supervised_tokens_from_labels": supervised_tokens,
        "optimizer_steps": len(steps),
        "plan_index_contiguous": expected_plan_index == rows,
        "rows_per_step_min": min(step_records.values()),
        "rows_per_step_max": max(step_records.values()),
        "step_supervised_tokens_min": min(step_supervised.values()),
        "step_supervised_tokens_max": max(step_supervised.values()),
        "declared_step_supervision_matches_labels": True,
        "family_group_records": dict(sorted(family_groups.items())),
        "rank_audits": [rank_audits[0], rank_audits[1]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    incomplete = args.output.with_name(args.output.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(incomplete)

    plan_sha = sha256(args.plan)
    holdout_sha = sha256(args.holdout)
    template_sha = sha256(args.template)
    if plan_sha != PLAN_SHA256 or holdout_sha != HOLDOUT_SHA256:
        raise ValueError("frozen parquet identity mismatch")
    if template_sha != TEMPLATE_SHA256:
        raise ValueError("frozen template identity mismatch")

    plan = scan_plan(args.plan)
    holdout = scan_holdout(args.holdout)
    manifest = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    holdout_manifest_uids = {str(item["uid"]) for item in manifest["records"]}
    gates = {
        "plan_records": plan["records"] == 81757,
        "plan_unique_uids": plan["unique_uids"] == 81757,
        "plan_input_tokens": plan["input_tokens_from_offsets"] == 148425311,
        "plan_supervised_tokens": plan["shifted_supervised_tokens_from_labels"]
        == 88486262,
        "plan_steps": plan["optimizer_steps"] == 4096,
        "family_group_records": plan["family_group_records"]
        == {"anthropic": 65362, "other": 16395},
        "holdout_records": holdout["records"] == 1948,
        "holdout_unique_uids": holdout["unique_uids"] == 1948,
        "holdout_source_uids": len(holdout["source_uids"])
        == len(holdout_manifest_uids)
        == 1500,
        "holdout_input_tokens": holdout["input_tokens_from_offsets"] == 3483799,
        "holdout_supervised_tokens": holdout[
            "shifted_supervised_tokens_from_labels"
        ]
        == 1964016,
        "holdout_manifest_identity": holdout["source_uids"] == holdout_manifest_uids,
        "training_holdout_exclusion": not (
            plan["source_uids"] & holdout_manifest_uids
        ),
        "all_numeric_totals_finite": all(
            math.isfinite(value)
            for value in (
                plan["input_tokens_from_offsets"],
                plan["shifted_supervised_tokens_from_labels"],
                holdout["input_tokens_from_offsets"],
                holdout["shifted_supervised_tokens_from_labels"],
            )
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ValueError(f"frozen input gates failed: {failed}")

    plan.pop("source_uids")
    holdout.pop("source_uids")
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "identities": {
            "plan_sha256": plan_sha,
            "holdout_sha256": holdout_sha,
            "template_sha256": template_sha,
            "holdout_manifest_sha256": sha256(args.holdout_manifest),
        },
        "plan": plan,
        "holdout": holdout,
        "holdout_manifest_source_uids": len(holdout_manifest_uids),
        "training_holdout_intersection": 0,
        "gates": {name: "PASS" for name in gates},
        "verifier": str(Path(__file__)),
        "verifier_sha256": sha256(Path(__file__)),
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
