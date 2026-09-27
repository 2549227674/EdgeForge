#!/usr/bin/env python3
"""Build and verify the deterministic OPE5-02 formal evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


STATUS = "VALID_OPE5_02_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_4XA76_FP16_GEMV"
SEGMENT = Path(
    "results/operator/board/ope5/"
    "ope5_fp16_gemv_segment1_20260921T145232+0800"
)
MANIFEST = Path(
    "results/operator/board/ope5/ope5_fp16_gemv_evidence_manifest_20260921.json"
)
ARCHIVE = Path(
    "exports/operator/board/EdgeForge_OPE5_FP16_GEMV_board_evidence_20260921.zip"
)
FIXED_ZIP_TIME = (2026, 9, 21, 14, 53, 16)


STATIC_FILES: tuple[tuple[str, str], ...] = (
    ("configs/operator/board/ope5_backlog_triage.json", "frozen backlog adjudication"),
    ("configs/operator/board/ope5_fp16_gemv_scope.json", "frozen four-question scope"),
    ("configs/operator/board/ope5_fp16_gemv_prereg.json", "frozen FP16 GEMV preregistration"),
    (
        "results/operator/board/ope5/ope5_fp16_gemv_scope_receipt_20260921.json",
        "scope verification receipt",
    ),
    (
        "results/operator/board/ope5/ope5_fp16_gemv_prereg_receipt_20260921.json",
        "preregistration verification receipt",
    ),
    (
        "results/operator/board/ope5/ope5_fp16_gemv_source_lock_20260921.json",
        "pre-execution source lock",
    ),
    ("scripts/operator/board/ope5_fp16_gemv_bench.cpp", "locked benchmark source"),
    ("scripts/operator/board/run_ope5_fp16_gemv.sh", "locked formal runner"),
    (
        "scripts/operator/board/ope5_fp16_gemv_measure_payload.sh",
        "locked board payload",
    ),
    ("scripts/operator/board/telemetry_sampler.sh", "locked telemetry dependency"),
    (
        "scripts/operator/board/summarize_ope5_fp16_gemv.py",
        "independent post-run validator",
    ),
    (
        "scripts/operator/board/package_ope5_fp16_gemv_evidence.py",
        "deterministic evidence packager",
    ),
    (
        "results/operator/board/ope5/ope5_fp16_gemv_receipt_20260921.json",
        "28-of-28 machine receipt",
    ),
    (
        "docs/operator/EdgeForge_算子面_板端_OPE5_FP16_GEMV正式段报告_2026-09-21.md",
        "human-readable formal report",
    ),
    (
        "results/operator/board/bandwidth_ceiling.json",
        "OPE0 Tier-A denominator",
    ),
    (
        "results/operator/board/ope3/ope3_receipt_20260919.json",
        "OPE3 same-geometry int8 anchor",
    ),
    (
        "results/operator/board/ope3/ope3_three_horizontal_20260919.json",
        "OPE3 lineage-separated comparison context",
    ),
    (
        "results/operator/board/ope4/ope4_receipt_20260921.json",
        "OPE4 model-stage anchor",
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


def main() -> int:
    repo = parse_args().repo.resolve()
    receipt_path = repo / "results/operator/board/ope5/ope5_fp16_gemv_receipt_20260921.json"
    receipt: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != STATUS or receipt["measurement_valid"] is not True:
        raise RuntimeError("refusing to package an invalid OPE5-02 receipt")
    if receipt["check_summary"] != {"failed": [], "passed": 28, "total": 28}:
        raise RuntimeError("unexpected OPE5-02 check summary")

    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for relative, role in STATIC_FILES:
        if relative in seen:
            raise RuntimeError(f"duplicate manifest member: {relative}")
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(relative)
        entries.append({"path": relative, "sha256": sha256(path), "role": role})
    segment_dir = repo / SEGMENT
    for path in sorted(segment_dir.iterdir()):
        if not path.is_file():
            continue
        relative = path.relative_to(repo).as_posix()
        if relative in seen:
            raise RuntimeError(f"duplicate manifest member: {relative}")
        seen.add(relative)
        entries.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "role": "raw single formal segment artifact",
            }
        )
    if len(entries) != len(STATIC_FILES) + 34:
        raise RuntimeError(f"unexpected payload count: {len(entries)}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "OPE5-02-FP16-GEMV-FORMAL-EVIDENCE",
        "status": STATUS,
        "recorded_at": receipt["recorded_at"],
        "formal_segments": 1,
        "same_day_retest_performed": False,
        "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
        "payload_file_count": len(entries),
        "archive_member_count_including_manifest": len(entries) + 1,
        "files": entries,
        "claim_limits": [
            "The measured FP16 GEMV is an isolated four-A76 comparator, not proof of RKLLM CPU dispatch or a deployment speedup.",
            "The 42-call result is a p50 multiplication, not a measured population or subtractable RKLLM TPOT component.",
            "bus_access count*8 is a preregistered hypothesis diagnostic only; count*64 remains rejected and neither is an acceptance gate.",
            "Execution card, facts, handoff, this manifest, archive checksum, and the archive itself are mutable/index or self-referential artifacts and are not hashed as payload members.",
        ],
    }
    manifest_path = repo / MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
                "status": STATUS,
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
