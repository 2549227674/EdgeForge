#!/usr/bin/env python3
"""Freeze equal-token raw-uniform and family-80/20 M1 race plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ANTHROPIC_FAMILY = "claimed_anthropic_from_source_label"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holdout_uids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or payload.get("holdout") or payload.get("uids")
    if not isinstance(records, list):
        raise ValueError("holdout manifest does not contain a record list")
    result: set[str] = set()
    for item in records:
        result.add(str(item["uid"] if isinstance(item, dict) else item))
    return result


def shuffled_cycle(indices: list[int], rng: random.Random):
    while True:
        epoch = indices[:]
        rng.shuffle(epoch)
        yield from epoch


def draw_plan(
    *,
    recipe: str,
    eligible: list[int],
    anthropic: list[int],
    other: list[int],
    lengths: list[int],
    target_tokens: int,
    seed: int,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    raw_stream = shuffled_cycle(eligible, rng)
    anthropic_stream = shuffled_cycle(anthropic, random.Random(seed + 101))
    other_stream = shuffled_cycle(other, random.Random(seed + 202))

    def draw() -> int:
        if recipe == "raw_uniform":
            return next(raw_stream)
        return next(anthropic_stream) if rng.random() < 0.8 else next(other_stream)

    plan: list[tuple[int, int]] = []
    total = 0
    max_length = max(lengths[index] for index in eligible)
    while target_tokens - total > max_length:
        index = draw()
        plan.append((index, lengths[index]))
        total += lengths[index]

    remaining = target_tokens - total
    # One deterministic bridge row makes the two plans exactly equal in input
    # tokens.  At most one row per arm is prefix-shortened solely at the budget
    # boundary; this is recorded in the plan manifest.
    pool = eligible if recipe == "raw_uniform" else (anthropic if rng.random() < 0.8 else other)
    candidates = [index for index in pool if lengths[index] >= remaining]
    if not candidates:
        candidates = [index for index in eligible if lengths[index] >= remaining]
    if not candidates:
        raise RuntimeError(f"no bridge row can supply {remaining} tokens")
    index = candidates[rng.randrange(len(candidates))]
    plan.append((index, remaining))
    if sum(take for _, take in plan) != target_tokens:
        raise AssertionError("plan token total is not exact")
    return plan


def materialize(
    table: pa.Table,
    plan: list[tuple[int, int]],
    optimizer_steps: int,
    output: Path,
) -> dict[str, Any]:
    indices = pa.array([index for index, _ in plan], type=pa.int64())
    selected = table.take(indices)
    rows = selected.to_pylist()
    target_tokens = sum(take for _, take in plan)
    cumulative = 0
    step_counts = [0] * optimizer_steps
    output_rows: list[dict[str, Any]] = []
    shortened = 0
    for plan_index, (row, (_, take)) in enumerate(zip(rows, plan, strict=True)):
        if take < len(row["input_ids"]):
            row["input_ids"] = row["input_ids"][:take]
            row["labels"] = row["labels"][:take]
            row["kept_tokens"] = take
            shortened += 1
        supervised = sum(label != -100 for label in row["labels"][1:])
        step = min(optimizer_steps - 1, cumulative * optimizer_steps // target_tokens)
        step_counts[step] += supervised
        row.update(
            plan_index=plan_index,
            optimizer_step=step,
            scheduled_tokens=take,
            shifted_supervised_tokens=supervised,
        )
        output_rows.append(row)
        cumulative += take

    if any(count == 0 for count in step_counts):
        empty = [index for index, count in enumerate(step_counts) if count == 0]
        raise RuntimeError(f"optimizer steps without supervised tokens: {empty[:10]}")
    for row in output_rows:
        row["optimizer_step_supervised_tokens"] = step_counts[row["optimizer_step"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows), output, compression="zstd")
    family_counts: dict[str, int] = {}
    for row in output_rows:
        family = "anthropic" if row["family"] == ANTHROPIC_FAMILY else "other"
        family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "file": output.name,
        "sha256": sha256(output),
        "records": len(output_rows),
        "input_tokens": target_tokens,
        "shifted_supervised_tokens": sum(row["shifted_supervised_tokens"] for row in output_rows),
        "optimizer_steps": optimizer_steps,
        "budget_boundary_shortened_rows": shortened,
        "sampled_records_by_family_group": family_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendered", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--optimizer-steps", type=int, default=1024)
    parser.add_argument("--budget-divisor", type=int, default=8)
    parser.add_argument(
        "--recipes",
        nargs="+",
        choices=("raw_uniform", "family_80_20"),
        default=("raw_uniform", "family_80_20"),
    )
    args = parser.parse_args()

    files = sorted(args.rendered.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no rendered shards under {args.rendered}")
    table = ds.dataset([str(path) for path in files], format="parquet").to_table()
    held_out = holdout_uids(args.holdout)
    uids = table.column("uid").to_pylist()
    source_uids = (
        table.column("source_uid").to_pylist()
        if "source_uid" in table.column_names
        else uids
    )
    families = table.column("family").to_pylist()
    lengths = [int(value) for value in table.column("kept_tokens").to_pylist()]
    eligible = [index for index, uid in enumerate(source_uids) if uid not in held_out]
    anthropic = [index for index in eligible if families[index] == ANTHROPIC_FAMILY]
    other = [index for index in eligible if families[index] != ANTHROPIC_FAMILY]
    train_tokens = sum(lengths[index] for index in eligible)
    if args.budget_divisor <= 0:
        raise ValueError("--budget-divisor must be positive")
    target_tokens = train_tokens // args.budget_divisor

    plans = {}
    recipe_seed_offsets = {"raw_uniform": 0, "family_80_20": 10000}
    for recipe in args.recipes:
        plan = draw_plan(
            recipe=recipe,
            eligible=eligible,
            anthropic=anthropic,
            other=other,
            lengths=lengths,
            target_tokens=target_tokens,
            seed=args.seed + recipe_seed_offsets[recipe],
        )
        plans[recipe] = materialize(
            table,
            plan,
            args.optimizer_steps,
            args.output_dir / f"{recipe}.parquet",
        )

    holdout_indices = [index for index, uid in enumerate(source_uids) if uid in held_out]
    holdout_table = table.take(pa.array(holdout_indices, type=pa.int64()))
    holdout_path = args.output_dir / "holdout.parquet"
    pq.write_table(holdout_table, holdout_path, compression="zstd")
    report = {
        "seed": args.seed,
        "rendered_report_sha256": sha256(args.rendered / "render_report.json"),
        "holdout_manifest_sha256": sha256(args.holdout),
        "rendered_records": table.num_rows,
        "holdout_manifest_records": len(held_out),
        "holdout_rendered_windows": len(holdout_indices),
        "holdout_rendered_source_records": len({source_uids[index] for index in holdout_indices}),
        "training_windows": len(eligible),
        "training_source_records": len({source_uids[index] for index in eligible}),
        "training_windows_anthropic": len(anthropic),
        "training_windows_other": len(other),
        "full_post_crop_training_tokens": train_tokens,
        "budget_divisor": args.budget_divisor,
        "target_token_budget": target_tokens,
        "plans": plans,
        "holdout_file": {
            "file": holdout_path.name,
            "sha256": sha256(holdout_path),
            "records": holdout_table.num_rows,
            "input_tokens": sum(int(value) for value in holdout_table.column("kept_tokens").to_pylist()),
        },
    }
    if args.budget_divisor == 8:
        report["one_eighth_token_budget"] = target_tokens
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
