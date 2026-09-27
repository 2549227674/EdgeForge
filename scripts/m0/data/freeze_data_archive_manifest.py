#!/usr/bin/env python3
"""Create the M0 line C gate-1 checksum manifest and completeness report."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq


PARQUET_EXPORTS = {
    "Glint-Research/Fable-5-traces",
    "Crownelius/Complete-FABLE.5-traces-2M",
    "AletheiaResearch/GLM-5.2-Agent",
    "armand0e/claude-opus-4.8-pi-traces",
    "Infatoshi/kernelbench-hard-traces",
    "Infatoshi/kernelbench-mega-traces",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_roles(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def viewer_rows(metadata: dict[str, object]) -> int:
    try:
        return int(metadata["dataset_viewer_size"]["size"]["dataset"]["num_rows"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing Dataset Viewer row count for {metadata['repo_id']}") from error


def parquet_rows(paths: list[Path]) -> tuple[int, dict[str, object]]:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths), {"source_unit": "rows"}


def json_line_rows(paths: list[Path]) -> tuple[int, dict[str, object]]:
    count = 0
    for path in paths:
        with path.open("rb") as handle:
            count += sum(1 for _ in handle)
    return count, {"source_unit": "records"}


def local_rows(repo_id: str) -> tuple[int, dict[str, object]]:
    archive = Path("data/archive") / repo_id.replace("/", "__")
    export = Path("data/archive_parquet") / repo_id.replace("/", "__")
    if repo_id in PARQUET_EXPORTS:
        paths = sorted(export.rglob("*.parquet"))
        if not paths:
            raise ValueError(f"canonical parquet export missing for {repo_id}")
        return parquet_rows(paths)
    if repo_id == "lambda/hermes-agent-reasoning-traces":
        paths = [archive / "data/kimi/train.parquet", archive / "data/glm-5.1/train.parquet"]
        ids = set()
        for path in paths:
            ids.update(pq.read_table(path, columns=["id"])["id"].to_pylist())
        top_level = sorted((archive / "data").glob("train-*.parquet"))
        top_ids = set()
        for path in top_level:
            top_ids.update(pq.read_table(path, columns=["id"])["id"].to_pylist())
        return len(ids), {
            "source_unit": "records",
            "config_files": [path.as_posix() for path in paths],
            "config_unique_ids": len(ids),
            "top_level_export_unique_ids": len(top_ids),
            "top_level_overlap_with_configs": len(top_ids & ids),
        }
    if repo_id == "armand0e/qwen3.7-max-pi-traces":
        paths = sorted(archive.glob("*.jsonl"))
        return len(paths), {"source_unit": "source_sessions", "source_files": len(paths)}
    if repo_id == "WithinUsAI/claude_mythos_distilled_25k":
        return json_line_rows(sorted(archive.glob("*.jsonl")))
    raise ValueError(f"no canonical source rule for {repo_id}")


def revision_for(repo_id: str, downloads: dict[str, dict[str, object]], metadata: dict[str, object]) -> tuple[str | None, str]:
    if repo_id in downloads:
        return str(downloads[repo_id]["resolved_revision"]), "refs/convert/parquet"
    return metadata.get("default_revision"), "default"


def write_report(path: Path, datasets: list[dict[str, object]]) -> None:
    lines = [
        "# M0 §5 门①完整性对账",
        "",
        "上游行数来自执行时冻结的 Hugging Face Dataset Viewer `/size` 响应；下载集以记录的 `refs/convert/parquet` commit 为准。",
        "",
        "| 数据集 | 上游行数 | 本地行数 | 完整度 | revision | 结论 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in datasets:
        status = "通过" if item["local_rows"] == item["upstream_rows"] else "未通过"
        lines.append(
            "| {repo_id} | {upstream_rows:,} | {local_rows:,} | {completeness:.1%} | `{revision}` | {status} |".format(
                status=status, **item
            )
        )
    crown = next(item for item in datasets if item["repo_id"] == "Crownelius/Complete-FABLE.5-traces-2M")
    hermes = next(item for item in datasets if item["repo_id"] == "lambda/hermes-agent-reasoning-traces")
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            f"- Crownelius 原 `data/archive/` 单分片为 50,651 行；已补齐并冻结 `refs/convert/parquet` 的 4 个分片，共 {crown['local_rows']:,} 行。",
            f"- Hermes 以 `kimi` 与 `glm-5.1` 两 config 的唯一 `id` 计数。顶层二分片的 {hermes['row_count_detail']['top_level_export_unique_ids']:,} 个 id 与 config 集的交集为 {hermes['row_count_detail']['top_level_overlap_with_configs']:,}，故它们不另计入分母。",
            "- Glint 的 4,665 行是前缀展开行；门④解析并标注源会话标识，实际折叠发生在门②，不在门①行数口径中处理。",
            "- KernelBench 两集在 manifest 中登记并保留 checksum，但文件角色为 `kernelbench_excluded`，不进入训练。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--exclude-glob", action="append", default=[])
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("data/pipeline/gate1_completeness.md"))
    parser.add_argument("--upstream-metadata", type=Path, default=Path("data/pipeline/gate1_upstream_metadata.json"))
    parser.add_argument("--downloads", type=Path, default=Path("data/pipeline/gate1_downloads.json"))
    args = parser.parse_args()

    role_rows = read_roles(args.roles)
    roles = {row["path"]: row for row in role_rows}
    for root in args.root:
        for path in root.rglob("*"):
            if not path.is_file() or ".cache" in path.parts:
                continue
            text = path.as_posix()
            if any(fnmatch.fnmatch(text, pattern) for pattern in args.exclude_glob):
                continue
            if text not in roles:
                raise ValueError(f"file is not classified in {args.roles}: {text}")

    upstream = json.loads(args.upstream_metadata.read_text(encoding="utf-8"))
    upstream_by_repo = {item["repo_id"]: item for item in upstream["datasets"]}
    download_data = json.loads(args.downloads.read_text(encoding="utf-8"))
    downloads = {item["repo_id"]: item for item in download_data["downloads"]}

    grouped_files: dict[str, list[dict[str, object]]] = defaultdict(list)
    for relative, role in sorted(roles.items()):
        path = Path(relative)
        grouped_files[role["dataset"]].append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": role["role"],
                "included": role["included"] == "true",
            }
        )

    datasets = []
    for repo_id in sorted(upstream_by_repo):
        source = upstream_by_repo[repo_id]
        local_count, detail = local_rows(repo_id)
        upstream_count = viewer_rows(source)
        revision, revision_source = revision_for(repo_id, downloads, source)
        license_value = source.get("license_from_hf_card") or source.get("license_from_local_snapshot_card") or "unlabeled"
        datasets.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "revision_source": revision_source,
                "hub_default_revision": source.get("default_revision"),
                "upstream_rows": upstream_count,
                "local_rows": local_count,
                "completeness": local_count / upstream_count if upstream_count else None,
                "license": license_value,
                "row_count_detail": detail,
                "files": grouped_files[repo_id],
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.report, datasets)
    failed = [item["repo_id"] for item in datasets if item["local_rows"] != item["upstream_rows"]]
    if failed:
        raise SystemExit(f"incomplete datasets: {', '.join(failed)}")
    print(f"wrote {args.output} and {args.report} for {len(datasets)} datasets")


if __name__ == "__main__":
    main()
