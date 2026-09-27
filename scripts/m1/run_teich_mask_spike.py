#!/usr/bin/env python3
"""Compare Teich 0.3.x masking with the frozen M0 Gemma 4 mask semantics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from datasets import Dataset
from teich import mask_data, prepare_data
from teich.audit import audit_sft_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]


def load_renderer() -> Any:
    path = ROOT / "scripts/m0/data/render_linec_samples.py"
    spec = importlib.util.spec_from_file_location("edgeforge_m0_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review_uids(path: Path) -> list[str]:
    pattern = re.compile(r"^\|\s*\d+\s*\|\s*`([^`]+)`")
    uids = [match.group(1) for line in path.read_text(encoding="utf-8").splitlines() if (match := pattern.match(line))]
    if len(uids) != 20:
        raise ValueError(f"expected 20 frozen review UIDs, found {len(uids)}")
    return uids


def load_records(directory: Path, uids: list[str]) -> list[dict[str, Any]]:
    wanted = set(uids)
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                uid = record.get("uid")
                if uid in wanted:
                    found[uid] = record
        if len(found) == len(wanted):
            break
    missing = wanted.difference(found)
    if missing:
        raise ValueError(f"missing frozen review records: {sorted(missing)}")
    return [found[uid] for uid in uids]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--review", type=Path, default=ROOT / "data/pipeline/gate7_mask_review.md")
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--samples-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()

    uids = review_uids(args.review)
    records = load_records(args.records, uids)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    renderer = load_renderer()
    legacy = [renderer.render_one(tokenizer, record) for record in records]

    source = Dataset.from_list(records)
    prepared, prep_report = prepare_data(
        source,
        tokenizer,
        max_length=args.max_length,
        oversized_policy="trim_followups",
        tokenize=True,
        preserve_columns=["uid"],
        return_report=True,
        strict=True,
        chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
        verbose=False,
    )
    prepared_uids = list(prepared["uid"])
    trainer = SimpleNamespace(
        train_dataset=prepared,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=None,
        args=SimpleNamespace(
            dataset_text_field="text",
            max_length=args.max_length,
            packing=False,
            truncation_mode=None,
        ),
    )
    mask_data(
        trainer,
        tokenizer=tokenizer,
        train_on_reasoning=True,
        train_on_final_answers=True,
        train_on_tools=True,
        train_on_user=False,
        train_on_system=False,
        train_on_developer=False,
        train_on_tool_responses=False,
        # Run the audit explicitly below so an incompatibility still produces a
        # machine-readable spike report and the frozen-renderer fallback samples.
        audit=False,
        verbose=False,
    )

    audit_report = audit_sft_dataset(
        trainer.train_dataset,
        tokenizer,
        sample_size=len(trainer.train_dataset),
    )

    comparisons: list[dict[str, Any]] = []
    legacy_by_uid = dict(zip(uids, legacy, strict=True))
    for uid, actual in zip(prepared_uids, trainer.train_dataset, strict=False):
        expected = legacy_by_uid[uid]
        expected_ids = [row["id"] for row in expected["tokens"]][: args.max_length]
        expected_targets = [bool(row["mask"]) for row in expected["tokens"]][: args.max_length]
        actual_ids = list(actual["input_ids"])
        actual_labels = list(actual["labels"])
        actual_targets = [label != -100 for label in actual_labels]
        ids_equal = expected_ids == actual_ids
        mask_equal = expected_targets == actual_targets
        comparisons.append(
            {
                "uid": uid,
                "ids_equal": ids_equal,
                "mask_equal": mask_equal,
                "tokens": len(actual_ids),
                "supervised_tokens": sum(actual_targets),
                "mask_disagreements": sum(a != b for a, b in zip(expected_targets, actual_targets, strict=False))
                + abs(len(expected_targets) - len(actual_targets)),
            }
        )

    # The spike contract says to fall back to the frozen M0 renderer when Teich
    # semantics differ. Emit those samples even when the Teich comparison fails,
    # so the subsequent QLoRA smoke consumes the already-reviewed mask contract.
    fallback_samples: list[dict[str, Any]] = []
    for uid, rendered_sample in zip(uids, legacy, strict=True):
        tokens = rendered_sample["tokens"][: args.max_length]
        fallback_samples.append(
            {
                "uid": uid,
                "input_ids": [row["id"] for row in tokens],
                "labels": [row["id"] if bool(row["mask"]) else -100 for row in tokens],
            }
        )

    dropped_rows = list(getattr(prep_report, "dropped_rows", []))
    teich_exact = (
        len(comparisons) == 20
        and audit_report.ok
        and all(item["ids_equal"] and item["mask_equal"] for item in comparisons)
    )

    report = {
        "teich_version": __import__("teich").__version__,
        "max_length": args.max_length,
        "requested_records": len(records),
        "prepared_records": len(prepared),
        "masked_records": len(trainer.train_dataset),
        "trimmed_records": len(getattr(prep_report, "trimmed_rows", [])),
        "dropped_records": len(dropped_rows),
        "dropped_rows": dropped_rows,
        "audit_ok": audit_report.ok,
        "audit_errors": audit_report.errors,
        "audit_warnings": audit_report.warnings,
        "exact_token_rows": sum(item["ids_equal"] for item in comparisons),
        "exact_mask_rows": sum(item["mask_equal"] for item in comparisons),
        "teich_semantics_match": teich_exact,
        "decision": "teich" if teich_exact else "fallback_m0_renderer",
        "fallback_samples": len(fallback_samples),
        "pass": teich_exact,
        "comparisons": comparisons,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(rendered, encoding="utf-8")
    if args.samples_out:
        args.samples_out.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_out.open("w", encoding="utf-8") as handle:
            for sample in fallback_samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
