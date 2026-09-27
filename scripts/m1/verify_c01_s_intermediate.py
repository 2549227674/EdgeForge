#!/usr/bin/env python3
"""Freeze a machine-readable audit for one C01-S intermediate checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket

from verify_c01_s_completion import (
    EXPECTED_EVALUATION_STEPS,
    EXPECTED_STEPS,
    verify_checkpoint,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.step <= 0 or args.step >= EXPECTED_STEPS or args.step % 512:
        raise ValueError("intermediate step must be a 512 multiple in [512, 3584]")
    if args.output.exists():
        raise FileExistsError(f"audit output already exists: {args.output}")
    incomplete = args.output.with_name(args.output.name + ".incomplete")
    if incomplete.exists():
        raise FileExistsError(f"incomplete audit output already exists: {incomplete}")

    checkpoint_dir = args.run_dir / "checkpoints" / f"step-{args.step:05d}"
    state, evidence = verify_checkpoint(
        checkpoint_dir, args.step, inspect_numeric_path=True
    )
    evaluations = state.get("evaluations") or []
    expected_evaluations = [
        step for step in EXPECTED_EVALUATION_STEPS if step <= args.step
    ]
    if [item.get("optimizer_step") for item in evaluations] != expected_evaluations:
        raise ValueError("checkpoint holdout evaluation schedule mismatch")
    for item in evaluations:
        if (
            item.get("records") != 1948
            or item.get("input_tokens") != 3483799
            or item.get("supervised_tokens") != 1964016
            or item.get("world_size") != 2
        ):
            raise ValueError(
                f"checkpoint holdout identity mismatch at {item.get('optimizer_step')}"
            )
        for key in (
            "sample_mean_loss",
            "sample_loss_standard_error",
            "token_weighted_loss",
            "elapsed_seconds",
        ):
            if not math.isfinite(item.get(key, math.nan)):
                raise ValueError(
                    f"checkpoint non-finite holdout {key} at {item.get('optimizer_step')}"
                )

    verifier_path = Path(__file__).with_name("verify_c01_s_completion.py")
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "candidate": "C01-S",
        "optimizer_step": args.step,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "checkpoint": evidence,
        "holdout_evaluations": evaluations,
        "numeric_path": {
            "base_compute_dtype": "bfloat16",
            "adapter_trainable_dtype": "float32",
            "optimizer_moment_dtype": "float32",
            "final_merge_required_dtype": "bfloat16",
            "owner_decision_date": "2026-08-12",
        },
        "gates": {
            "checkpoint_identity": "PASS",
            "checkpoint_file_hashes": "PASS",
            "scheduler_and_two_rank_rng": "PASS",
            "fp32_adapter_and_optimizer_finite": "PASS_OWNER_ACCEPTED",
            "holdout_identity_and_finite_losses": "PASS",
            "end_to_end_resume_smoke": "WAIVED_BY_OWNER",
        },
        "verifier": str(Path(__file__)),
        "verifier_sha256": sha256(Path(__file__)),
        "completion_verifier_sha256": sha256(verifier_path),
    }
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    incomplete.rename(args.output)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
