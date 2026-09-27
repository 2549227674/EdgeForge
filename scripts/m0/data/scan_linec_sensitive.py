#!/usr/bin/env python3
"""Summarize TruffleHog safely and prepare gate-3 redaction evidence."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RULES = {
    "openai_style_key": (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "<REDACTED_OPENAI_STYLE_KEY>"),
    "groq_style_key": (re.compile(r"\bgsk_[A-Za-z0-9_-]{20,}\b"), "<REDACTED_GROQ_STYLE_KEY>"),
    "postgres_uri": (re.compile(r"\bpostgres(?:ql)?://[^\s/@:]+:[^\s/@]+@[^\s/]+", re.I), "<REDACTED_POSTGRES_URI>"),
    "mongodb_uri": (re.compile(r"\bmongodb(?:\+srv)?://[^\s/@:]+:[^\s/@]+@[^\s/]+", re.I), "<REDACTED_MONGODB_URI>"),
    "credential_assignment": (re.compile(r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}", re.I), "<REDACTED_CREDENTIAL_ASSIGNMENT>"),
}


def values(record: dict):
    for message in record["messages"]:
        yield str(message.get("content", ""))
        yield str(message.get("reasoning_content", ""))
        for call in message.get("tool_calls", []):
            yield json.dumps(call, ensure_ascii=False)


def main() -> None:
    detector_counts: Counter[str] = Counter()
    files_by_detector: dict[str, set[str]] = defaultdict(set)
    for line in Path("logs/m0/m0_gate3_trufflehog.log").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "DetectorName" not in item:
            continue
        detector = str(item["DetectorName"])
        detector_counts[detector] += 1
        file_name = item.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {}).get("file")
        if file_name:
            files_by_detector[detector].add(str(file_name))

    hits = []
    sample_rows = []
    fallback_rows = []
    sampled_by_dataset: Counter[str] = Counter()
    for source in sorted(Path("data/pipeline/ir_l2").glob("*.jsonl")):
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                matched = sorted({name for value in values(record) for name, (regex, _) in RULES.items() if regex.search(value)})
                if matched:
                    hits.append({"uid": record["uid"], "dataset": record["dataset"], "rules": matched})
                summary = {
                    "uid": record["uid"],
                    "dataset": record["dataset"],
                    "message_roles": [message["role"] for message in record["messages"]],
                    "matched_sensitive_rules": matched,
                }
                if sampled_by_dataset[record["dataset"]] < 8:
                    sample_rows.append(summary)
                    sampled_by_dataset[record["dataset"]] += 1
                elif len(fallback_rows) < 50:
                    fallback_rows.append(summary)
    sample_rows.extend(fallback_rows[: max(0, 50 - len(sample_rows))])

    Path("data/pipeline/gate3_sensitive_hits.jsonl").write_text(
        "\n".join(json.dumps(hit, ensure_ascii=False) for hit in hits) + "\n", encoding="utf-8"
    )
    Path("data/pipeline/redaction_rules.tsv").write_text(
        "rule\tpattern_class\treplacement\tapply_stage\n"
        + "\n".join(f"{name}\tcredential\t{replacement}\tafter_gate5_before_gate7" for name, (_, replacement) in RULES.items())
        + "\n",
        encoding="utf-8",
    )
    Path("data/pipeline/gate3_manual_50.md").write_text(
        "# 门③抽检样本（待人工复核）\n\n"
        "以下为按数据集均衡抽取的结构和敏感规则命中摘要；不包含原文或密钥材料。\n\n"
        "| 数据集 | uid | 角色序列 | 敏感规则 |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {row['dataset']} | `{row['uid']}` | {' → '.join(row['message_roles'])} | {', '.join(row['matched_sensitive_rules']) or '—'} |"
            for row in sample_rows[:50]
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# M0 §5 门③安全清扫",
        "",
        "TruffleHog 完整扫描原始 archive 与 parquet export（缓存排除）；下表仅保留检测器与文件计数，绝不复述候选凭据。所有结果均为未验证或验证超时，仍按高风险训练内容处理。",
        "",
        "| 检测器 | 命中数 | 涉及文件数 |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {detector} | {detector_counts[detector]:,} | {len(files_by_detector[detector]):,} |" for detector in sorted(detector_counts))
    lines.extend(
        [
            "",
            f"结构化 IR 中命中可复用脱敏规则的样本：{len(hits):,}。具体 uid/rule 映射在 `gate3_sensitive_hits.jsonl`，不含匹配文本。",
            "脱敏写回严格顺延至门⑤之后，保证去污染 n-gram 扫描仍以原文进行。",
        ]
    )
    Path("data/pipeline/gate3_security.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
