#!/usr/bin/env python3
"""Run gate 2 L0/L1 deduplication on normalized source-session records."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


CONTROL_RE = re.compile(r"<local-command-(?:caveat|stdout)>.*?</local-command-(?:caveat|stdout)>|<command-(?:name|message|args)>.*?</command-(?:name|message|args)>", re.S | re.I)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SPACE_RE = re.compile(r"\s+")


def normalized(text: str) -> str:
    text = CONTROL_RE.sub(" ", text)
    text = ANSI_RE.sub("", text)
    return SPACE_RE.sub(" ", text).strip().lower()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def assistant_body(record: dict, boilerplate: list[str]) -> str:
    parts = []
    for message in record["messages"]:
        if message["role"] != "assistant":
            continue
        parts.append(str(message.get("content", "")))
        parts.append(str(message.get("reasoning_content", "")))
        if message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
    body = "\n".join(parts)
    for prefix in boilerplate:
        if body.startswith(prefix):
            body = body[len(prefix) :]
    return body.strip()


def first_user(record: dict) -> str:
    return next((str(message.get("content", "")) for message in record["messages"] if message["role"] == "user"), "")


def quality(record: dict) -> int:
    provenance = record.get("provenance", {})
    seen_count = int(provenance.get("seen_count", 1) or 1)
    chars = sum(len(str(message.get("content", ""))) + len(str(message.get("reasoning_content", ""))) for message in record["messages"])
    return seen_count * 10**12 + len(record["messages"]) * 10**7 + chars


def load_boilerplate() -> list[str]:
    path = Path("data/pipeline/boilerplate_strings.txt")
    return [json.loads(line)["string"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def create_db(path: Path, boilerplate: list[str]) -> tuple[sqlite3.Connection, Counter[str]]:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute(
        """CREATE TABLE records (
            uid TEXT PRIMARY KEY, dataset TEXT NOT NULL, source_session TEXT NOT NULL,
            first_user_hash TEXT NOT NULL, assistant_hash TEXT NOT NULL, score INTEGER NOT NULL,
            l0_seen_duplicate INTEGER NOT NULL, keep INTEGER NOT NULL DEFAULT 1
        )"""
    )
    input_counts: Counter[str] = Counter()
    for source in sorted(Path("data/pipeline/ir").glob("*.jsonl")):
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                input_counts[record["dataset"]] += 1
                provenance = record.get("provenance", {})
                con.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                    (
                        record["uid"],
                        record["dataset"],
                        record["source_session"],
                        digest(normalized(first_user(record))),
                        digest(normalized(assistant_body(record, boilerplate))),
                        quality(record),
                        int(int(provenance.get("seen_count", 1) or 1) > 1),
                    ),
                )
    con.commit()
    return con, input_counts


def cluster(con: sqlite3.Connection, columns: str, reason: str, output) -> int:
    removed = 0
    query = f"""
        SELECT {columns}, GROUP_CONCAT(uid || char(30) || score, char(31)), COUNT(*)
        FROM records WHERE keep = 1 GROUP BY {columns} HAVING COUNT(*) > 1
    """
    for row in con.execute(query):
        *key, members_text, count = row
        members = [item.split(chr(30), 1) for item in members_text.split(chr(31))]
        keeper = max(members, key=lambda item: (int(item[1]), item[0]))[0]
        discarded = [uid for uid, _ in members if uid != keeper]
        con.executemany("UPDATE records SET keep = 0 WHERE uid = ?", [(uid,) for uid in discarded])
        output.write(json.dumps({"reason": reason, "key": key, "keeper": keeper, "discarded": discarded}, ensure_ascii=False) + "\n")
        removed += len(discarded)
    con.commit()
    return removed


def materialize(con: sqlite3.Connection) -> Counter[str]:
    output_root = Path("data/pipeline/ir_dedup")
    output_root.mkdir(parents=True, exist_ok=True)
    handles = {}
    counts: Counter[str] = Counter()
    try:
        for source in sorted(Path("data/pipeline/ir").glob("*.jsonl")):
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    keep = con.execute("SELECT keep FROM records WHERE uid = ?", (record["uid"],)).fetchone()[0]
                    if not keep:
                        continue
                    dataset = record["dataset"]
                    if dataset not in handles:
                        handles[dataset] = (output_root / f"{dataset}.jsonl").open("w", encoding="utf-8")
                    handles[dataset].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[dataset] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def main() -> None:
    boilerplate = load_boilerplate()
    con, before = create_db(Path("data/pipeline/gate2.sqlite"), boilerplate)
    with Path("data/pipeline/gate2_dupe_clusters.jsonl").open("w", encoding="utf-8") as clusters:
        session_removed = cluster(con, "dataset, source_session", "source_session", clusters)
        user_removed = cluster(con, "first_user_hash", "first_user_exact", clusters)
        assistant_removed = cluster(con, "dataset, assistant_hash", "assistant_body_exact_after_boilerplate", clusters)
    after = materialize(con)
    l0 = {dataset: count for dataset, count in con.execute("SELECT dataset, COUNT(*) FROM records WHERE l0_seen_duplicate = 1 GROUP BY dataset")}
    con.close()
    lines = [
        "# M0 §5 门②：L0/L1 去重",
        "",
        "单位为结构化后的源会话/训练样本。L0 仅记录 Crownelius 上游 `seen_count>1` 信号；L1 依次折叠同源会话、首轮 user 精确重叠、以及剥样板后的 assistant 正文精确重复。",
        "",
        "| 数据集 | 门④输入 | L0 上游已知重复 | L1后保留 |",
        "|---|---:|---:|---:|",
    ]
    for dataset in sorted(before):
        lines.append(f"| {dataset} | {before[dataset]:,} | {l0.get(dataset, 0):,} | {after.get(dataset, 0):,} |")
    lines.extend(
        [
            "",
            f"- 同源会话折叠删除：{session_removed:,}",
            f"- 首轮 user 精确重叠删除：{user_removed:,}",
            f"- 剥样板后 assistant 正文精确重复删除：{assistant_removed:,}",
            "- L2 近重 MinHash 将在本结果上运行；簇明细在 `gate2_dupe_clusters.jsonl`。",
        ]
    )
    Path("data/pipeline/gate2_dedup.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
