#!/usr/bin/env python3
"""Validate the OPE5-05 read-only topology and compile/disassembly audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


BASE_SOURCE_SHA256 = "7843a9f797bcf566eae34e9982b6ecab832a53060aaa8287403e42c10b132c9e"
SCOPE_SHA256 = "8f3a2c6949980d04096a1e995ff2aafae653aed031fdf0960b1261c86059d1f4"
PREREG_SHA256 = "98a3fe20f7982f7fb58fca01f1fe23fa483f4ca11f387d24767805f1e7fb43ce"
EVENTS = {"cycles:u", "instructions:u", "l2d_cache_refill:u", "bus_access:u"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_perf(path: Path) -> dict[str, int]:
    observed: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#") or len(row) < 3:
                continue
            event = row[2].strip()
            if event not in EVENTS:
                continue
            value = row[0].strip()
            if value in {"", "<not counted>", "<not supported>"}:
                raise AssertionError((event, value))
            observed[event] = int(value)
    return observed


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    audit = (repo / args.audit_dir).resolve()
    output = (repo / args.output).resolve()
    base = repo / "scripts/operator/board/ope5_fp16_gemv_bench.cpp"
    candidate = repo / "scripts/operator/board/ope5_a55_fp16_gemv_bench.cpp"
    scope = repo / "configs/operator/board/ope5_a55_fp16_gemv_scope.json"
    prereg = repo / "configs/operator/board/ope5_a55_fp16_gemv_prereg.json"
    meta = read_env(audit / "audit_meta.env")

    base_text = base.read_text(encoding="utf-8")
    expected = base_text.replace(
        "constexpr int kFirstCpu = 4;", "constexpr int kFirstCpu = 0;"
    ).replace(
        "failed to pin main worker to CPU4", "failed to pin main worker to CPU0"
    ).replace("OPE5_FP16_GEMV", "OPE5_A55_FP16_GEMV")
    candidate_text = candidate.read_text(encoding="utf-8")
    perf_counts = parse_perf(audit / "perf_true.stat.csv")
    affinity_rows = list(
        csv.DictReader(
            (audit / "affinity.tsv").open(encoding="utf-8", newline=""),
            delimiter="\t",
        )
    )
    scalar = (audit / "disassembly_scalar_fp32.txt").read_text(encoding="utf-8").lower()
    autovec = (audit / "disassembly_autovec_o3.txt").read_text(encoding="utf-8").lower()
    candidate_disassembly = (
        audit / "disassembly_neon_fp16_f32acc.txt"
    ).read_text(encoding="utf-8").lower()
    codegen = {
        "scalar_vector_fmla_matches": len(re.findall(r"\bfmla\s+v", scalar)),
        "autovec_vector_fmla_matches": len(re.findall(r"\bfmla\s+v", autovec)),
        "candidate_fcvtl_matches": len(
            re.findall(r"\bfcvtl2?\s+v", candidate_disassembly)
        ),
        "candidate_vector_fmla_matches": len(
            re.findall(r"\bfmla\s+v", candidate_disassembly)
        ),
    }
    checks = {
        "frozen_scope_and_prereg_identity": sha256(scope) == SCOPE_SHA256
        and sha256(prereg) == PREREG_SHA256,
        "base_source_identity": sha256(base) == BASE_SOURCE_SHA256,
        "source_exact_three_category_derivative": candidate_text == expected,
        "board_source_hash_matches": meta["source_sha256"] == sha256(candidate),
        "policy0_topology": meta["policy0_related_cpus"] == "0 1 2 3",
        "policy0_max_frequency": meta["policy0_max_khz"] == "1800000",
        "first_core_features": all(
            feature in meta["first_core_features"].split()
            for feature in ("asimd", "fphp", "asimdhp")
        ),
        "four_single_cpu_affinities": len(affinity_rows) == 4
        and {int(row["cpu"]) for row in affinity_rows} == {0, 1, 2, 3}
        and all(row["observed_allowed_list"] == row["cpu"] for row in affinity_rows)
        and all(row["pass"] == "yes" for row in affinity_rows),
        "perf_true_all_events_numeric": set(perf_counts) == EVENTS
        and (audit / "perf_true.exit_code.txt").read_text(encoding="utf-8").strip()
        == "0",
        "compiler_target": meta["compiler_target"] == "-mcpu=cortex-a55",
        "compile_once_no_benchmark": meta["compile_invocations"] == "1"
        and meta["benchmark_execution_count"] == "0",
        "codegen_gates": codegen["scalar_vector_fmla_matches"] == 0
        and codegen["candidate_fcvtl_matches"] > 0
        and codegen["candidate_vector_fmla_matches"] > 0
        and (audit / "vectorization.log").is_file(),
        "temporary_binary_removed": not (
            audit / "ope5_a55_fp16_gemv_audit.bin"
        ).exists(),
    }
    passed = sum(checks.values())
    status = (
        "PASS_A55_PRELOCK_NO_BENCHMARK_EXECUTION"
        if passed == len(checks)
        else "FAIL_A55_PRELOCK"
    )
    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(audit.iterdir())
        if path.is_file()
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "OPE5-05-A55-FP16-GEMV-PRELOCK-AUDIT",
        "status": status,
        "recorded_at": meta["recorded_at"],
        "board": "orangepi@192.168.137.66",
        "execution_scope": "read-only topology/perf true probe plus compile/disassembly-only",
        "benchmark_execution_count": 0,
        "formal_segment_count": 0,
        "check_summary": {
            "passed": passed,
            "total": len(checks),
            "failed": [name for name, value in checks.items() if not value],
        },
        "checks": checks,
        "source_derivation": {
            "base_path": str(base.relative_to(repo)),
            "base_sha256": sha256(base),
            "candidate_path": str(candidate.relative_to(repo)),
            "candidate_sha256": sha256(candidate),
            "allowed_change_categories": [
                "kFirstCpu 4 to 0",
                "main-worker affinity error CPU4 to CPU0",
                "OPE5_FP16_GEMV diagnostic tag to OPE5_A55_FP16_GEMV",
            ],
            "exact_allowlist_match": candidate_text == expected,
        },
        "topology": {
            "policy0_related_cpus": meta["policy0_related_cpus"],
            "policy0_max_khz": int(meta["policy0_max_khz"]),
            "affinity": affinity_rows,
            "required_features": ["asimd", "fphp", "asimdhp"],
        },
        "perf_true_counts": perf_counts,
        "codegen": codegen,
        "raw_audit_directory": str(audit.relative_to(repo)),
        "raw_artifact_sha256": artifact_hashes,
        "claim_limits": [
            "The benchmark binary was compiled and disassembled but not executed.",
            "The perf probe used true and is a capability check, not a performance smoke.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "checks": f"{passed}/{len(checks)}",
                "benchmark_execution_count": 0,
                "candidate_source_sha256": sha256(candidate),
            },
            ensure_ascii=False,
        )
    )
    return 0 if status.startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
