#!/usr/bin/env python3
"""Freeze the unique-source T3 OPD prompt pool and its exact token budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ANTHROPIC_FAMILY = "claimed_anthropic_from_source_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=9600)
    parser.add_argument("--anthropic-fraction", type=float, default=0.8)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--optimizer-batch-size", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--total-length-cap", type=int, default=4096)
    parser.add_argument("--prompt-length-cap", type=int, default=3072)
    parser.add_argument("--completion-length-cap", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_int64(values: list[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode()).hexdigest()


def largest_remainder(capacities: dict[str, int], requested: int) -> dict[str, int]:
    total_capacity = sum(capacities.values())
    if requested > total_capacity:
        raise ValueError(f"requested {requested} rows from capacity {total_capacity}")
    quotas = {key: requested * capacity / total_capacity for key, capacity in capacities.items()}
    allocation = {key: min(capacities[key], math.floor(quota)) for key, quota in quotas.items()}
    remaining = requested - sum(allocation.values())
    order = sorted(
        capacities,
        key=lambda key: (-(quotas[key] - math.floor(quotas[key])), key),
    )
    while remaining:
        advanced = False
        for key in order:
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
                advanced = True
                if not remaining:
                    break
        if not advanced:
            raise RuntimeError("largest-remainder allocation could not satisfy request")
    return allocation


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.prompts % args.generation_batch_size:
        raise ValueError("prompt count must be divisible by generation_batch_size")
    if args.prompts % args.optimizer_batch_size:
        raise ValueError("prompt count must be divisible by optimizer_batch_size")
    if not 0 < args.anthropic_fraction < 1:
        raise ValueError("anthropic_fraction must be strictly between zero and one")
    if args.prompt_length_cap + 1 > args.total_length_cap:
        raise ValueError("prompt cap must leave at least one completion token")

    holdout_payload = json.loads(args.holdout_manifest.read_text(encoding="utf-8"))
    holdout_source_uids = {row["uid"] for row in holdout_payload["records"]}
    plan = pq.ParquetFile(args.plan)
    columns = ["uid", "source_uid", "dataset", "family", "task_type", "input_ids", "labels", "plan_index"]
    candidate_by_source: dict[str, dict[str, Any]] = {}
    plan_holdout_intersection: set[str] = set()
    rejected_no_supervision = 0
    rejected_prompt_too_long = 0
    for batch in plan.iter_batches(batch_size=512, columns=columns):
        data = batch.to_pydict()
        for index, source_uid in enumerate(data["source_uid"]):
            if source_uid in holdout_source_uids:
                plan_holdout_intersection.add(source_uid)
                continue
            labels = data["labels"][index]
            prompt_length = next((i for i, token in enumerate(labels) if token != -100), None)
            if prompt_length is None or prompt_length <= 0:
                rejected_no_supervision += 1
                continue
            if prompt_length > args.prompt_length_cap:
                rejected_prompt_too_long += 1
                continue
            uid = data["uid"][index]
            prompt_ids = data["input_ids"][index][:prompt_length]
            reference_completion_length = sum(token != -100 for token in labels)
            candidate = {
                "plan_uid": uid,
                "source_uid": source_uid,
                "dataset": data["dataset"][index],
                "family": data["family"][index],
                "task_type": data["task_type"][index],
                "plan_index": int(data["plan_index"][index]),
                "prompt_ids": prompt_ids,
                "prompt_length": prompt_length,
                "reference_completion_length": reference_completion_length,
                "prompt_ids_sha256": sha256_int64(prompt_ids),
                "prefix_selection_score": stable_hash(args.seed, "prefix", uid),
                "source_selection_score": stable_hash(args.seed, "source", source_uid),
            }
            previous = candidate_by_source.get(source_uid)
            if previous is None or candidate["prefix_selection_score"] < previous["prefix_selection_score"]:
                candidate_by_source[source_uid] = candidate

    if plan_holdout_intersection:
        raise RuntimeError(
            f"source plan unexpectedly intersects frozen holdout: {len(plan_holdout_intersection)} sources"
        )

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "anthropic": defaultdict(list),
        "other": defaultdict(list),
    }
    for candidate in candidate_by_source.values():
        broad_group = "anthropic" if candidate["family"] == ANTHROPIC_FAMILY else "other"
        stratum = f"{candidate['family']}|{candidate['task_type']}"
        grouped[broad_group][stratum].append(candidate)
    for strata in grouped.values():
        for candidates in strata.values():
            candidates.sort(key=lambda row: row["source_selection_score"])

    anthropic_prompts = round(args.prompts * args.anthropic_fraction)
    other_prompts = args.prompts - anthropic_prompts
    requested_by_group = {"anthropic": anthropic_prompts, "other": other_prompts}
    selected: list[dict[str, Any]] = []
    allocation: dict[str, dict[str, int]] = {}
    capacity: dict[str, dict[str, int]] = {}
    for broad_group, requested in requested_by_group.items():
        strata = grouped[broad_group]
        capacities = {stratum: len(rows) for stratum, rows in strata.items()}
        selected_counts = largest_remainder(capacities, requested)
        allocation[broad_group] = selected_counts
        capacity[broad_group] = capacities
        for stratum, count in selected_counts.items():
            selected.extend(strata[stratum][:count])

    if len(selected) != args.prompts or len({row["source_uid"] for row in selected}) != args.prompts:
        raise RuntimeError("selected prompt pool is not complete and source-unique")
    # Pack similarly sized prompts into each generation batch. This prevents
    # one long prompt from unnecessarily reducing the completion cap of three
    # short prompts, while seeded batch shuffling avoids a length curriculum.
    selected.sort(
        key=lambda row: (
            row["prompt_length"],
            stable_hash(args.seed, "within_length", row["source_uid"]),
        )
    )
    packed_batches = [
        selected[index : index + args.generation_batch_size]
        for index in range(0, len(selected), args.generation_batch_size)
    ]
    if any(len(batch) != args.generation_batch_size for batch in packed_batches):
        raise RuntimeError("length packing produced an incomplete generation batch")
    packed_batches.sort(
        key=lambda batch: stable_hash(
            args.seed,
            "batch_order",
            *sorted(row["source_uid"] for row in batch),
        )
    )
    selected = [row for batch in packed_batches for row in batch]
    for order_index, row in enumerate(selected):
        individual_completion_cap = min(
            args.completion_length_cap,
            args.total_length_cap - int(row["prompt_length"]),
        )
        if individual_completion_cap <= 0:
            raise RuntimeError(f"no completion room for {row['source_uid']}")
        row["order_index"] = order_index
        row["individual_completion_cap"] = individual_completion_cap
        row["retained_reference_completion_length"] = min(
            int(row["reference_completion_length"]),
            individual_completion_cap,
        )

    for batch_index, batch_start in enumerate(range(0, len(selected), args.generation_batch_size)):
        batch = selected[batch_start : batch_start + args.generation_batch_size]
        batch_prompt_width = max(int(row["prompt_length"]) for row in batch)
        batch_completion_cap = min(
            args.completion_length_cap,
            args.total_length_cap - batch_prompt_width,
        )
        for row in batch:
            row["generation_batch_index"] = batch_index
            row["batch_prompt_width"] = batch_prompt_width
            row["batch_completion_cap"] = batch_completion_cap

    parquet_rows = [
        {
            key: row[key]
            for key in (
                "order_index",
                "plan_index",
                "plan_uid",
                "source_uid",
                "dataset",
                "family",
                "task_type",
                "prompt_ids",
                "prompt_length",
                "reference_completion_length",
                "retained_reference_completion_length",
                "individual_completion_cap",
                "generation_batch_index",
                "batch_prompt_width",
                "batch_completion_cap",
                "prompt_ids_sha256",
            )
        }
        for row in selected
    ]
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = args.output_parquet.with_suffix(args.output_parquet.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(parquet_rows), temporary_parquet, compression="zstd")
    os.replace(temporary_parquet, args.output_parquet)
    parquet_sha256 = sha256_file(args.output_parquet)

    rollout_windows = args.prompts // args.generation_batch_size
    optimizer_steps = args.prompts // args.optimizer_batch_size
    warmup_steps = round(optimizer_steps * args.warmup_ratio)
    selected_source_uids = [row["source_uid"] for row in selected]
    row_identity_lines = [
        "\t".join(
            [
                str(row["order_index"]),
                row["source_uid"],
                row["plan_uid"],
                row["prompt_ids_sha256"],
                str(row["prompt_length"]),
                str(row["batch_completion_cap"]),
            ]
        )
        for row in selected
    ]
    prompt_lengths = [int(row["prompt_length"]) for row in selected]
    reference_completion_lengths = [int(row["reference_completion_length"]) for row in selected]
    retained_reference_completion_lengths = [
        int(row["retained_reference_completion_length"]) for row in selected
    ]
    individual_completion_caps = [int(row["individual_completion_cap"]) for row in selected]
    batch_completion_caps = [int(row["batch_completion_cap"]) for row in selected]
    group_counts = Counter(
        "anthropic" if row["family"] == ANTHROPIC_FAMILY else "other" for row in selected
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "M1",
        "stage": "T3_OPD_prompt_pool",
        "selection": {
            "unit": "one_conversation_prefix_per_unique_source_uid",
            "prefix_choice": "minimum_seeded_sha256_over_eligible_plan_windows",
            "source_choice": "family_task_type_largest_remainder_then_seeded_sha256_without_replacement",
            "training_order": "prompt_length_packed_by_generation_batch_then_seeded_batch_order",
            "seed": args.seed,
            "requested_prompts": args.prompts,
            "selected_prompts": len(selected),
            "selected_unique_sources": len(set(selected_source_uids)),
            "family_mix": {
                "anthropic_family_label": ANTHROPIC_FAMILY,
                "anthropic": group_counts["anthropic"],
                "other": group_counts["other"],
                "anthropic_fraction": group_counts["anthropic"] / len(selected),
                "other_fraction": group_counts["other"] / len(selected),
            },
            "stratum_capacity": capacity,
            "stratum_allocation": allocation,
        },
        "exclusion": {
            "holdout_manifest": str(args.holdout_manifest),
            "holdout_manifest_sha256": sha256_file(args.holdout_manifest),
            "holdout_sources": len(holdout_source_uids),
            "source_plan_holdout_intersection": len(plan_holdout_intersection),
            "selected_holdout_intersection": len(set(selected_source_uids) & holdout_source_uids),
        },
        "source": {
            "plan": str(args.plan),
            "plan_sha256": sha256_file(args.plan),
            "plan_rows": plan.metadata.num_rows,
            "eligible_unique_sources": len(candidate_by_source),
            "rejected_rows_no_supervision": rejected_no_supervision,
            "rejected_rows_prompt_too_long": rejected_prompt_too_long,
        },
        "length_policy": {
            "total_length_cap": args.total_length_cap,
            "prompt_length_cap": args.prompt_length_cap,
            "completion_length_cap": args.completion_length_cap,
            "rule": "min(completion_length_cap,total_length_cap-real_prompt_length)",
            "prompt_tokens": {
                "sum": sum(prompt_lengths),
                "min": min(prompt_lengths),
                "p50": float(np.percentile(prompt_lengths, 50)),
                "p90": float(np.percentile(prompt_lengths, 90)),
                "p99": float(np.percentile(prompt_lengths, 99)),
                "max": max(prompt_lengths),
            },
            "reference_completion_tokens": {
                "sum": sum(reference_completion_lengths),
                "min": min(reference_completion_lengths),
                "p50": float(np.percentile(reference_completion_lengths, 50)),
                "p90": float(np.percentile(reference_completion_lengths, 90)),
                "p99": float(np.percentile(reference_completion_lengths, 99)),
                "max": max(reference_completion_lengths),
            },
            "retained_reference_completion_tokens": {
                "sum": sum(retained_reference_completion_lengths),
                "retention_fraction": sum(retained_reference_completion_lengths)
                / sum(reference_completion_lengths),
                "fully_retained_prompts": sum(
                    retained == original
                    for retained, original in zip(
                        retained_reference_completion_lengths,
                        reference_completion_lengths,
                        strict=True,
                    )
                ),
                "fully_retained_fraction": sum(
                    retained == original
                    for retained, original in zip(
                        retained_reference_completion_lengths,
                        reference_completion_lengths,
                        strict=True,
                    )
                )
                / len(reference_completion_lengths),
            },
            "individual_completion_cap": {
                "sum": sum(individual_completion_caps),
                "min": min(individual_completion_caps),
                "p50": float(np.percentile(individual_completion_caps, 50)),
                "p90": float(np.percentile(individual_completion_caps, 90)),
                "max": max(individual_completion_caps),
            },
            "length_packed_batch_completion_cap": {
                "sum_over_prompts": sum(batch_completion_caps),
                "min": min(batch_completion_caps),
                "p50": float(np.percentile(batch_completion_caps, 50)),
                "p90": float(np.percentile(batch_completion_caps, 90)),
                "max": max(batch_completion_caps),
            },
        },
        "training_budget": {
            "generation_batch_size": args.generation_batch_size,
            "rollout_windows": rollout_windows,
            "optimizer_batch_size": args.optimizer_batch_size,
            "num_generations": 1,
            "optimizer_steps": optimizer_steps,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "maximum_rollout_tokens_under_length_packed_dynamic_caps": sum(batch_completion_caps),
            "absolute_uniform_cap_upper_bound": args.prompts * args.completion_length_cap,
            "actual_rollout_tokens": "record_at_run_time_after_EOS",
        },
        "artifact": {
            "parquet": str(args.output_parquet),
            "parquet_rows": len(parquet_rows),
            "parquet_sha256": parquet_sha256,
            "selected_source_uid_set_sha256": hashlib.sha256(
                "\n".join(sorted(selected_source_uids)).encode()
            ).hexdigest(),
            "ordered_row_identity_sha256": hashlib.sha256("\n".join(row_identity_lines).encode()).hexdigest(),
        },
        "rows": [
            {
                key: row[key]
                for key in (
                    "order_index",
                    "plan_index",
                    "plan_uid",
                    "source_uid",
                    "dataset",
                    "family",
                    "task_type",
                    "prompt_length",
                    "reference_completion_length",
                    "retained_reference_completion_length",
                    "individual_completion_cap",
                    "generation_batch_index",
                    "batch_prompt_width",
                    "batch_completion_cap",
                    "prompt_ids_sha256",
                )
            }
            for row in selected
        ],
    }
    atomic_json(args.output_manifest, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "prompts": len(selected),
                "unique_sources": len(set(selected_source_uids)),
                "family_mix": manifest["selection"]["family_mix"],
                "holdout_intersection": manifest["exclusion"]["selected_holdout_intersection"],
                "optimizer_steps": optimizer_steps,
                "rollout_windows": rollout_windows,
                "warmup_steps": warmup_steps,
                "max_rollout_tokens": sum(batch_completion_caps),
                "parquet_sha256": parquet_sha256,
                "manifest": str(args.output_manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
