#!/usr/bin/env python3
"""Summarize prompt/completion trade-offs for a tokenized T2 plan.

The plan may repeat a source across epochs.  This audit keeps the first planned
segment for each source_uid so a repeated source cannot distort the T3 prompt
length decision.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="PROMPT,COMPLETION",
        help="A 4K split to audit; repeat for multiple candidates.",
    )
    parser.add_argument("--dynamic-total-cap", type=int, default=4096)
    parser.add_argument("--dynamic-prompt-cap", type=int, default=3072)
    parser.add_argument("--dynamic-completion-cap", type=int, default=3072)
    return parser.parse_args()


def parse_splits(raw_splits: list[str]) -> list[tuple[int, int]]:
    if not raw_splits:
        return [(3072, 1024), (2560, 1536), (2048, 2048), (1536, 2560)]
    splits: list[tuple[int, int]] = []
    for raw in raw_splits:
        prompt, completion = (int(value) for value in raw.split(",", maxsplit=1))
        splits.append((prompt, completion))
    return splits


def percentile(values: list[int], quantile: int) -> float:
    return float(np.percentile(np.asarray(values), quantile, method="linear"))


def summarize(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"name": name, "sources": len(rows)}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [int(row[key]) for row in rows]
        summary[key] = {
            "mean": round(float(np.mean(values)), 2),
            "p50": percentile(values, 50),
            "p75": percentile(values, 75),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "max": max(values),
            **{f"gt_{limit}": sum(value > limit for value in values) for limit in (1024, 1536, 2048, 2560, 3072)},
        }
    return summary


def main() -> None:
    args = parse_args()
    columns = ["source_uid", "dataset", "family", "task_type", "input_ids", "labels", "plan_index"]
    sources: dict[str, dict[str, object]] = {}
    parquet = pq.ParquetFile(args.plan)
    for batch in parquet.iter_batches(batch_size=512, columns=columns):
        data = batch.to_pydict()
        for row_index, source_uid in enumerate(data["source_uid"]):
            labels = data["labels"][row_index]
            first_supervised = next((i for i, token in enumerate(labels) if token != -100), None)
            if first_supervised is None:
                continue
            row = {
                "source_uid": source_uid,
                "dataset": data["dataset"][row_index],
                "family": data["family"][row_index],
                "task_type": data["task_type"][row_index],
                "prompt_tokens": first_supervised,
                "completion_tokens": sum(token != -100 for token in labels),
                "total_tokens": len(labels),
                "plan_index": data["plan_index"][row_index],
            }
            previous = sources.get(source_uid)
            if previous is None or (row["plan_index"], row["total_tokens"]) < (
                previous["plan_index"],
                previous["total_tokens"],
            ):
                sources[source_uid] = row

    rows = list(sources.values())
    groups = {
        "all": rows,
        "anthropic": [row for row in rows if row["family"] == "claimed_anthropic_from_source_label"],
        "other": [row for row in rows if row["family"] != "claimed_anthropic_from_source_label"],
        "tool_agent": [row for row in rows if row["task_type"] == "tool_agent"],
        "chat_or_reasoning": [row for row in rows if row["task_type"] == "chat_or_reasoning"],
    }
    report: dict[str, object] = {
        "plan": args.plan,
        "plan_rows": parquet.metadata.num_rows,
        "source_representative_rule": "minimum (plan_index,total_tokens) per source_uid",
        "summaries": [summarize(name, subset) for name, subset in groups.items()],
        "splits": [],
    }
    prompt_total = sum(int(row["prompt_tokens"]) for row in rows)
    completion_total = sum(int(row["completion_tokens"]) for row in rows)
    for max_prompt, max_completion in parse_splits(args.split):
        prompt_untruncated = sum(int(row["prompt_tokens"]) <= max_prompt for row in rows)
        completion_untruncated = sum(int(row["completion_tokens"]) <= max_completion for row in rows)
        both_untruncated = sum(
            int(row["prompt_tokens"]) <= max_prompt and int(row["completion_tokens"]) <= max_completion
            for row in rows
        )
        report["splits"].append(
            {
                "max_prompt_length": max_prompt,
                "max_completion_length": max_completion,
                "source_prompt_untruncated": prompt_untruncated,
                "source_prompt_untruncated_fraction": prompt_untruncated / len(rows),
                "reference_completion_untruncated": completion_untruncated,
                "reference_completion_untruncated_fraction": completion_untruncated / len(rows),
                "both_untruncated": both_untruncated,
                "both_untruncated_fraction": both_untruncated / len(rows),
                "prompt_token_retention": sum(
                    min(int(row["prompt_tokens"]), max_prompt) for row in rows
                )
                / prompt_total,
                "reference_completion_token_retention": sum(
                    min(int(row["completion_tokens"]), max_completion) for row in rows
                )
                / completion_total,
            }
        )

    dynamic_rows = []
    for row in rows:
        original_prompt = int(row["prompt_tokens"])
        original_completion = int(row["completion_tokens"])
        retained_prompt = min(original_prompt, args.dynamic_prompt_cap)
        completion_cap = min(
            args.dynamic_completion_cap,
            args.dynamic_total_cap - retained_prompt,
        )
        if completion_cap <= 0:
            raise ValueError("dynamic caps leave no completion room")
        dynamic_rows.append(
            {
                "original_prompt": original_prompt,
                "original_completion": original_completion,
                "retained_prompt": retained_prompt,
                "completion_cap": completion_cap,
                "retained_completion": min(original_completion, completion_cap),
            }
        )
    report["dynamic_policy"] = {
        "total_length_cap": args.dynamic_total_cap,
        "prompt_length_cap": args.dynamic_prompt_cap,
        "completion_length_cap": args.dynamic_completion_cap,
        "rule": "completion_cap=min(dynamic_completion_cap,dynamic_total_cap-retained_prompt)",
        "source_prompt_untruncated": sum(
            item["retained_prompt"] == item["original_prompt"] for item in dynamic_rows
        ),
        "source_prompt_untruncated_fraction": sum(
            item["retained_prompt"] == item["original_prompt"] for item in dynamic_rows
        )
        / len(dynamic_rows),
        "reference_completion_untruncated": sum(
            item["retained_completion"] == item["original_completion"] for item in dynamic_rows
        ),
        "reference_completion_untruncated_fraction": sum(
            item["retained_completion"] == item["original_completion"] for item in dynamic_rows
        )
        / len(dynamic_rows),
        "both_untruncated": sum(
            item["retained_prompt"] == item["original_prompt"]
            and item["retained_completion"] == item["original_completion"]
            for item in dynamic_rows
        ),
        "both_untruncated_fraction": sum(
            item["retained_prompt"] == item["original_prompt"]
            and item["retained_completion"] == item["original_completion"]
            for item in dynamic_rows
        )
        / len(dynamic_rows),
        "prompt_token_retention": sum(item["retained_prompt"] for item in dynamic_rows) / prompt_total,
        "reference_completion_token_retention": sum(
            item["retained_completion"] for item in dynamic_rows
        )
        / completion_total,
        "completion_cap_distribution": {
            "min": min(item["completion_cap"] for item in dynamic_rows),
            "p50": percentile([item["completion_cap"] for item in dynamic_rows], 50),
            "p90": percentile([item["completion_cap"] for item in dynamic_rows], 90),
            "max": max(item["completion_cap"] for item in dynamic_rows),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
