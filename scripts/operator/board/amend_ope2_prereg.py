#!/usr/bin/env python3
"""Create an owner-authorized OPE2 sampling amendment without replacing the parent preregistration."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amendment-id", required=True)
    parser.add_argument("--effective-after-terminal-count", type=int, required=True)
    parser.add_argument("--last-terminal-cell-id", required=True)
    parser.add_argument("--interrupted-cell-id", required=True)
    parser.add_argument("--authorized-at", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--min-reps", type=int, required=True)
    parser.add_argument("--min-timed-ms", type=float, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite protocol amendment: {args.output}")
    if min(args.warmup, args.min_reps, args.min_timed_ms) <= 0:
        raise SystemExit("amended sampling values must all be positive")

    amended = copy.deepcopy(read_json(args.parent))
    protocol = amended["measurement_protocol"]
    original = {
        "warmup_repetitions": protocol["warmup_repetitions"],
        "timed_repetitions_minimum": protocol["timed_repetitions_minimum"],
        "timed_window_minimum_ms": protocol["timed_window_minimum_ms"],
    }
    replacement = {
        "warmup_repetitions": args.warmup,
        "timed_repetitions_minimum": args.min_reps,
        "timed_window_minimum_ms": args.min_timed_ms,
    }
    protocol.update(replacement)
    amended["status"] = "PREREGISTERED_PROTOCOL_AMENDMENT_OWNER_AUTHORIZED"
    amended["generated_at"] = dt.datetime.now().astimezone().isoformat()
    amended["source"]["parent_preregistration"] = str(args.parent)
    amended["protocol_amendment"] = {
        "id": args.amendment_id,
        "authorized_by": "owner",
        "authorized_at": args.authorized_at,
        "reason": (
            "Uniform 50+200 sampling materially over-samples multi-second cells. "
            "Retain the complete matrix and correctness/telemetry gates while using "
            "a conservative duration-aware sample budget for unfinished cells."
        ),
        "scope": "Only terminals appended after the durable cutoff; no completed terminal is replaced or rerun.",
        "effective_after_terminal_count": args.effective_after_terminal_count,
        "last_terminal_cell_id": args.last_terminal_cell_id,
        "interrupted_uncommitted_cell_id": args.interrupted_cell_id,
        "interrupted_cell_disposition": "No terminal was committed; rerun once under the amended protocol.",
        "original_sampling": original,
        "amended_sampling": replacement,
        "unchanged": [
            "all shapes and axes",
            "correctness oracle",
            "IQR/p50 > 0.30 one-retry rule",
            "governor and telemetry gates",
            "single logical matrix with no replication claim",
        ],
        "evidence_label": "PROTOCOL_AMENDED_DURING_RUN",
    }
    amended["claim_limits"].append(
        "Sampling protocol A1 was owner-authorized after the durable cutoff; results are an engineering matrix with a disclosed mid-run protocol amendment, not an unchanged-preregistration claim."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(amended, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": amended["status"],
                "amendment": amended["protocol_amendment"],
                "expected_terminal_counts": amended["expected_terminal_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
