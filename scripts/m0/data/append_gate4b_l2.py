#!/usr/bin/env python3
"""Append the gate-2 verified near-repeat rates to the gate-4b report."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    l1_counts = {}
    for line in Path("data/pipeline/gate2_dedup.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and "|" in line and "门④输入" not in line and "---" not in line:
            fields = [field.strip() for field in line.strip("|").split("|")]
            if len(fields) == 4 and fields[1].replace(",", "").isdigit():
                l1_counts[fields[0]] = int(fields[3].replace(",", ""))
    summary = json.loads(Path("data/pipeline/gate2_l2_summary.json").read_text(encoding="utf-8"))
    removed = summary["removed_by_dataset"]
    lines = [
        "",
        "## 经验证的集内近重复率（门② L2）",
        "",
        "L2 使用 128-entry bottom-k 5-gram MinHash 生成候选、完整 5-gram Jaccard ≥0.8 验证。分母为 L1 后保留数。",
        "",
        "| 数据集 | L1 后 | L2 删除 | 经验证近重复率 |",
        "|---|---:|---:|---:|",
    ]
    for dataset in sorted(l1_counts):
        removed_count = int(removed.get(dataset, 0))
        rate = f"{removed_count / l1_counts[dataset]:.2%}" if l1_counts[dataset] else "—"
        lines.append(f"| {dataset} | {l1_counts[dataset]:,} | {removed_count:,} | {rate} |")
    lines.append(f"候选生成阶段跳过 {summary['skipped_large_bands']} 个过大 band；该项已在门②报告留档。")
    report = Path("data/pipeline/gate4b_degeneracy.md")
    report.write_text(report.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
