#!/usr/bin/env python3
"""Build and verify the deterministic OPE5-05 A55 formal evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


STATUS = "VALID_OPE5_05_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_4XA55_FP16_GEMV_COMPARATOR"
PRELOCK = Path(
    "results/operator/board/ope5/"
    "ope5_a55_fp16_gemv_prelock_20260921T173741+0800"
)
SEGMENT = Path(
    "results/operator/board/ope5/"
    "ope5_a55_fp16_gemv_segment1_20260921T174008+0800"
)
MANIFEST = Path(
    "results/operator/board/ope5/ope5_a55_fp16_gemv_evidence_manifest_20260921.json"
)
ARCHIVE = Path(
    "exports/operator/board/EdgeForge_OPE5_A55_FP16_GEMV_board_evidence_20260921.zip"
)
FIXED_ZIP_TIME = (2026, 9, 21, 17, 42, 0)


STATIC_FILES: tuple[tuple[str, str], ...] = (
    ("configs/operator/board/ope5_backlog_triage.json", "frozen backlog ordering"),
    (
        "configs/operator/board/ope5_a55_fp16_gemv_scope.json",
        "frozen OPE5-05 four-question scope",
    ),
    (
        "configs/operator/board/ope5_a55_fp16_gemv_prereg.json",
        "frozen OPE5-05 preregistration",
    ),
    (
        "results/operator/board/ope5/ope5_a55_fp16_gemv_scope_receipt_20260921.json",
        "24-of-24 scope receipt",
    ),
    (
        "results/operator/board/ope5/ope5_a55_fp16_gemv_prereg_receipt_20260921.json",
        "31-of-31 preregistration receipt",
    ),
    (
        "scripts/operator/board/verify_ope5_a55_fp16_gemv_prereg.py",
        "scope and preregistration verifier",
    ),
    (
        "scripts/operator/board/ope5_a55_fp16_gemv_bench.cpp",
        "locked A55 benchmark source",
    ),
    (
        "scripts/operator/board/run_ope5_a55_fp16_gemv.sh",
        "locked formal runner",
    ),
    (
        "scripts/operator/board/ope5_a55_fp16_gemv_measure_payload.sh",
        "locked formal payload",
    ),
    (
        "scripts/operator/board/audit_ope5_a55_fp16_gemv_prelock.sh",
        "prelock topology and codegen audit driver",
    ),
    (
        "scripts/operator/board/verify_ope5_a55_fp16_gemv_prelock.py",
        "prelock receipt verifier",
    ),
    (
        "results/operator/board/ope5/ope5_a55_fp16_gemv_prelock_receipt_20260921.json",
        "13-of-13 prelock receipt with zero benchmark execution",
    ),
    (
        "results/operator/board/ope5/ope5_a55_fp16_gemv_source_lock_20260921.json",
        "source lock preceding formal execution",
    ),
    (
        "scripts/operator/board/summarize_ope5_a55_fp16_gemv.py",
        "independent formal segment summarizer",
    ),
    (
        "scripts/operator/board/telemetry_sampler.sh",
        "locked telemetry dependency",
    ),
    (
        "results/operator/board/ope5/ope5_a55_fp16_gemv_receipt_20260921.json",
        "32-of-32 final machine receipt",
    ),
    (
        "docs/operator/EdgeForge_算子面_板端_OPE5_A55_FP16_GEMV正式段报告_2026-09-21.md",
        "human-readable formal report",
    ),
    (
        "scripts/operator/board/ope5_fp16_gemv_bench.cpp",
        "immutable A76 source lineage parent",
    ),
    (
        "results/operator/board/ope5/ope5_fp16_gemv_source_lock_20260921.json",
        "immutable A76 source lock",
    ),
    (
        "results/operator/board/ope5/ope5_fp16_gemv_receipt_20260921.json",
        "immutable A76 primary receipt",
    ),
    (
        "results/operator/board/bandwidth_ceiling.json",
        "shared A76-derived OPE0 platform ceiling",
    ),
    (
        "configs/operator/board/ope1_shape_manifest.json",
        "authentic shape and invocation census",
    ),
    (
        "results/operator/board/ope4/ope4_receipt_20260921.json",
        "OPE4 attribution boundary anchor",
    ),
    (
        "scripts/operator/board/package_ope5_a55_fp16_gemv_evidence.py",
        "deterministic formal evidence packager",
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
    receipt_path = (
        repo
        / "results/operator/board/ope5/ope5_a55_fp16_gemv_receipt_20260921.json"
    )
    receipt: dict[str, Any] = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != STATUS or receipt["measurement_valid"] is not True:
        raise RuntimeError("refusing to package an invalid OPE5-05 receipt")
    if receipt["check_summary"] != {"failed": [], "passed": 32, "total": 32}:
        raise RuntimeError("unexpected OPE5-05 check summary")
    strength = receipt["evidence_strength"]
    if strength != {
        "formal_segments": 1,
        "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
        "same_day_retest_performed": False,
        "same_day_retest_scheduled": False,
        "automatic_retry_performed": False,
        "a76_rerun_performed": False,
        "performance_smoke_performed": False,
    }:
        raise RuntimeError("unexpected OPE5-05 execution strength")

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

    for directory, expected_count, role in (
        (PRELOCK, 12, "raw prelock audit artifact"),
        (SEGMENT, 36, "raw single formal segment artifact"),
    ):
        files = [path for path in sorted((repo / directory).iterdir()) if path.is_file()]
        if len(files) != expected_count:
            raise RuntimeError(
                f"unexpected raw file count for {directory}: {len(files)}"
            )
        for path in files:
            relative = path.relative_to(repo).as_posix()
            if relative in seen:
                raise RuntimeError(f"duplicate manifest member: {relative}")
            seen.add(relative)
            entries.append(
                {"path": relative, "sha256": sha256(path), "role": role}
            )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "OPE5-05-A55-FP16-GEMV-FORMAL-EVIDENCE",
        "status": STATUS,
        "recorded_at": receipt["recorded_at"],
        "formal_segments": 1,
        "same_day_retest_performed": False,
        "automatic_retry_performed": False,
        "a76_rerun_performed": False,
        "performance_smoke_performed": False,
        "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
        "payload_file_count": len(entries),
        "archive_member_count_including_manifest": len(entries) + 1,
        "files": entries,
        "claim_limits": [
            "The A55 result is a synthetic isolated comparator, not proof of RKLLM placement or deployment speedup.",
            "The A55/A76 result is a nonconcurrent system-level cluster comparison, not a frequency-normalized single-variable causal estimate.",
            "A55 bus_access remains a raw count only; no x8/x64 byte conversion or cross-cluster L2D/bus ratio is claimed.",
            "Execution card, facts, handoff, this manifest, archive checksum, and the archive itself are mutable/index or self-referential artifacts and are not payload members.",
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
