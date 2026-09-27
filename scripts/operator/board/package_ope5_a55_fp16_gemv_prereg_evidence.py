#!/usr/bin/env python3
"""Build and verify the deterministic OPE5-05 A55 preregistration archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


SCOPE_STATUS = "PASS_OPE5_05_FOUR_QUESTIONS_PREREG_AUTHORIZED"
PREREG_STATUS = "PASS_FROZEN_BEFORE_IMPLEMENTATION_NO_BOARD_EXECUTION"
MANIFEST = Path(
    "results/operator/board/ope5/"
    "ope5_a55_fp16_gemv_prereg_evidence_manifest_20260921.json"
)
ARCHIVE = Path(
    "exports/operator/board/"
    "EdgeForge_OPE5_A55_FP16_GEMV_prereg_evidence_20260921.zip"
)
FIXED_ZIP_TIME = (2026, 9, 21, 17, 10, 0)


FILES: tuple[tuple[str, str], ...] = (
    (
        "configs/operator/board/ope5_a55_fp16_gemv_scope.json",
        "frozen OPE5-05 four-question scope",
    ),
    (
        "configs/operator/board/ope5_a55_fp16_gemv_prereg.json",
        "frozen OPE5-05 preregistration",
    ),
    (
        "scripts/operator/board/verify_ope5_a55_fp16_gemv_prereg.py",
        "independent scope and preregistration verifier",
    ),
    (
        "scripts/operator/board/package_ope5_a55_fp16_gemv_prereg_evidence.py",
        "deterministic preregistration evidence packager",
    ),
    (
        "results/operator/board/ope5/"
        "ope5_a55_fp16_gemv_scope_receipt_20260921.json",
        "24-of-24 scope verification receipt",
    ),
    (
        "results/operator/board/ope5/"
        "ope5_a55_fp16_gemv_prereg_receipt_20260921.json",
        "31-of-31 preregistration verification receipt",
    ),
    (
        "docs/operator/"
        "EdgeForge_算子面_板端_OPE5_A55_FP16_GEMV立项与测前冻结_2026-09-21.md",
        "human-readable OPE5-05 freeze report",
    ),
    (
        "configs/operator/board/ope5_backlog_triage.json",
        "frozen OPE5 backlog ordering and disposition",
    ),
    (
        "results/operator/board/board_env_identity_20260917T2022+0800.json",
        "captured board ISA identity",
    ),
    (
        "configs/operator/board/ope1_shape_manifest.json",
        "authentic shape and invocation census",
    ),
    (
        "results/operator/board/bandwidth_ceiling.json",
        "shared OPE0 platform ceiling with A76 lineage limit",
    ),
    (
        "results/operator/board/ope3/ope3_receipt_20260919.json",
        "separate same-shape int8 context",
    ),
    (
        "results/operator/board/ope5/ope5_receipt_20260921.json",
        "separate RMSNorm primary disposition context",
    ),
    (
        "configs/operator/board/ope5_fp16_gemv_prereg.json",
        "immutable OPE5-02 A76 protocol",
    ),
    (
        "scripts/operator/board/ope5_fp16_gemv_bench.cpp",
        "locked OPE5-02 source lineage parent",
    ),
    (
        "results/operator/board/ope5/"
        "ope5_fp16_gemv_source_lock_20260921.json",
        "immutable OPE5-02 source lock",
    ),
    (
        "results/operator/board/ope5/ope5_fp16_gemv_receipt_20260921.json",
        "immutable OPE5-02 A76 primary receipt",
    ),
    (
        "results/operator/board/ope5/"
        "ope5_fp16_gemv_segment1_20260921T145232+0800/telemetry_1hz.csv",
        "OPE5-02 policy0 and A76 frequency identity evidence",
    ),
    (
        "results/operator/board/ope4/ope4_receipt_20260921.json",
        "OPE4 attribution boundary anchor",
    ),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    return parser.parse_args()


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    repo = parse_args().repo.resolve()
    scope_receipt = load_json(
        repo
        / "results/operator/board/ope5/"
        "ope5_a55_fp16_gemv_scope_receipt_20260921.json"
    )
    prereg_receipt = load_json(
        repo
        / "results/operator/board/ope5/"
        "ope5_a55_fp16_gemv_prereg_receipt_20260921.json"
    )
    if scope_receipt["status"] != SCOPE_STATUS:
        raise RuntimeError("refusing to package a failed OPE5-05 scope")
    if scope_receipt["check_summary"] != {"passed": 24, "total": 24}:
        raise RuntimeError("unexpected OPE5-05 scope check summary")
    if prereg_receipt["status"] != PREREG_STATUS:
        raise RuntimeError("refusing to package a failed OPE5-05 preregistration")
    if prereg_receipt["check_summary"] != {"passed": 31, "total": 31}:
        raise RuntimeError("unexpected OPE5-05 preregistration check summary")
    if scope_receipt["board_connection_performed"] is not False:
        raise RuntimeError("scope unexpectedly connected to the board")
    if prereg_receipt["board_connection_performed"] is not False:
        raise RuntimeError("preregistration unexpectedly connected to the board")
    if prereg_receipt["implementation"] != {
        "implemented": False,
        "source_locked": False,
        "formal_execution_started": False,
    }:
        raise RuntimeError("implementation state is not a clean preimplementation freeze")

    future_files = (
        "scripts/operator/board/ope5_a55_fp16_gemv_bench.cpp",
        "scripts/operator/board/run_ope5_a55_fp16_gemv.sh",
        "scripts/operator/board/ope5_a55_fp16_gemv_measure_payload.sh",
    )
    present = [relative for relative in future_files if (repo / relative).exists()]
    if present:
        raise RuntimeError(f"future implementation files exist at freeze: {present}")

    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for relative, role in FILES:
        if relative in seen:
            raise RuntimeError(f"duplicate manifest member: {relative}")
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(relative)
        entries.append({"path": relative, "sha256": sha256(path), "role": role})

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "OPE5-05-A55-FP16-GEMV-PREREG-EVIDENCE",
        "status": PREREG_STATUS,
        "recorded_at": prereg_receipt["recorded_at"],
        "scope_status": SCOPE_STATUS,
        "scope_checks": "24/24",
        "prereg_checks": "31/31",
        "board_connection_performed": False,
        "candidate_implemented": False,
        "source_locked": False,
        "formal_execution_started": False,
        "selected_primary": "OPE5-02 four-A76 FP16 GEMV",
        "future_a55_formal_segments": 1,
        "a76_rerun_authorized": False,
        "performance_smoke_authorized": False,
        "same_day_a55_retest_authorized": False,
        "instability_retry_authorized": False,
        "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
        "payload_file_count": len(entries),
        "archive_member_count_including_manifest": len(entries) + 1,
        "files": entries,
        "claim_limits": [
            "This archive records scope and preregistration only; it contains no A55 implementation, source-lock, board connection, benchmark execution, or A55 performance result.",
            "The immutable A76 primary comes from one owner-waived nonconcurrent formal segment and is not paired replication.",
            "The future A55/A76 result is a system-level cluster comparison, not a single-variable microarchitecture causal estimate or RKLLM placement claim.",
            "Execution card, facts, handoff, this manifest, archive checksum, and the archive itself are mutable/index or self-referential artifacts and are not hashed as payload members.",
        ],
    }
    manifest_path = repo / MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    archive_path = repo / ARCHIVE
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    members = [entry["path"] for entry in entries] + [MANIFEST.as_posix()]
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for relative in members:
            bundle.writestr(zip_info(relative), (repo / relative).read_bytes())

    with zipfile.ZipFile(archive_path) as bundle:
        if bundle.namelist() != members:
            raise RuntimeError("archive member order or identity mismatch")
        for entry in entries:
            if sha256_bytes(bundle.read(entry["path"])) != entry["sha256"]:
                raise RuntimeError(f"archive hash mismatch: {entry['path']}")
        if bundle.read(MANIFEST.as_posix()) != manifest_path.read_bytes():
            raise RuntimeError("archived manifest differs from local manifest")

    archive_hash = sha256(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(
        f"{archive_hash}  {archive_path.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": PREREG_STATUS,
                "payload_files": len(entries),
                "archive_members": len(members),
                "manifest_sha256": sha256(manifest_path),
                "archive_sha256": archive_hash,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
