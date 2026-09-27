#!/usr/bin/env python3
"""Gate 4b: detect templating, exact repeats, and identity/harness noise in IR."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PREFIX_LIMIT = 200
COMMON_PREFIX_LIMIT = 2000
PATTERNS = {
    "claude_control": re.compile(r"<local-command-(?:caveat|stdout)>|<command-(?:name|message|args)>", re.I),
    "ansi": re.compile(r"\x1b\[[0-9;]*m"),
    "model_switch": re.compile(r"Set model to .*(?:Fable|Opus|Mythos|Sonnet)", re.I),
    "identity": re.compile(r"\b(?:I am|I'm) (?:Claude|Fable|Mythos|Opus)\b", re.I),
    "stub": re.compile(r"Full implementation would be \d+\+? LOC|\.\.\.\s*// omitted", re.I),
    "invented_eval": re.compile(r"internal Anthropic eval", re.I),
}
STRUCTURAL_DUPLICATE_DATASETS = {"Glint-Research__Fable-5-traces"}


def assistant_body(record: dict) -> str:
    parts = []
    for message in record["messages"]:
        if message["role"] != "assistant":
            continue
        parts.append(str(message.get("content", "")))
        parts.append(str(message.get("reasoning_content", "")))
        if message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right), COMMON_PREFIX_LIMIT)
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def main() -> None:
    ir_root = Path("data/pipeline/ir")
    report_rows = []
    boilerplate = []
    review_samples = []
    for path in sorted(ir_root.glob("*.jsonl")):
        prefixes: Counter[str] = Counter()
        exact: Counter[str] = Counter()
        bodies_by_prefix: dict[str, str] = {}
        flags: Counter[str] = Counter()
        records = 0
        sampled = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                body = assistant_body(record)
                records += 1
                prefix = body[:PREFIX_LIMIT]
                prefixes[prefix] += 1
                exact[hashlib.sha256(body.encode()).hexdigest()] += 1
                if prefix not in bodies_by_prefix:
                    bodies_by_prefix[prefix] = body[:COMMON_PREFIX_LIMIT]
                else:
                    bodies_by_prefix[prefix] = common_prefix(bodies_by_prefix[prefix], body)
                matched = [name for name, regex in PATTERNS.items() if regex.search(body)]
                flags.update(matched)
                if sampled < 20:
                    review_samples.append(
                        {
                            "dataset": record["dataset"],
                            "uid": record["uid"],
                            "matched_patterns": matched,
                            "assistant_excerpt": body[:1000],
                        }
                    )
                    sampled += 1
        prefix, prefix_count = prefixes.most_common(1)[0]
        exact_unique = len(exact)
        common = bodies_by_prefix[prefix]
        if (
            path.stem not in STRUCTURAL_DUPLICATE_DATASETS
            and prefix_count >= 20
            and prefix_count / records > 0.20
            and len(common.strip()) >= 20
        ):
            boilerplate.append({"dataset": path.stem, "string": common})
        report_rows.append(
            {
                "dataset": path.stem,
                "records": records,
                "top_prefix_count": prefix_count,
                "top_prefix_rate": prefix_count / records if records else 0,
                "exact_unique": exact_unique,
                "exact_duplicate_rate": 1 - exact_unique / records if records else 0,
                "flags": dict(flags),
            }
        )

    Path("data/pipeline/boilerplate_strings.txt").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in boilerplate) + "\n", encoding="utf-8"
    )
    Path("data/pipeline/gate4b_review_samples.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in review_samples) + "\n", encoding="utf-8"
    )
    lines = [
        "# M0 §5 门④b：退化、模板与身份噪声",
        "",
        "近重率将在门②使用剥样板后的 5-gram 会话聚类计算，并回填本表。",
        "",
        "| 数据集 | 合格 IR | 最高频前缀覆盖率 | 集内精确重复率 | 身份/桩代码正则命中 |",
        "|---|---:|---:|---:|---|",
    ]
    for item in report_rows:
        flag_text = ", ".join(f"{key}={value}" for key, value in sorted(item["flags"].items())) or "—"
        lines.append(
            f"| {item['dataset']} | {item['records']:,} | {item['top_prefix_rate']:.2%} | {item['exact_duplicate_rate']:.2%} | {flag_text} |"
        )
    lines.extend(
        [
            "",
            "`boilerplate_strings.txt` 保存覆盖率超过 20% 的最长共同前缀；后续去重和去污染在剥除它们后运行。",
            "Glint 的高重复来自已核实的前缀展开，故明确不写入剥样板列表；它由门②按源会话折叠。",
            "`gate4b_review_samples.jsonl` 为每集抽样记录，供人工核验身份泄漏与桩代码。",
        ]
    )
    Path("data/pipeline/gate4b_degeneracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
