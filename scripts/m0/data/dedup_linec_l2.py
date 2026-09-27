#!/usr/bin/env python3
"""Gate-2 L2 near-dedup: 128-entry 5-gram MinHash candidates, full verification."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path


WORD_RE = re.compile(r"\w+", re.UNICODE)


def boilerplate() -> list[str]:
    return [json.loads(line)["string"] for line in Path("data/pipeline/boilerplate_strings.txt").read_text(encoding="utf-8").splitlines() if line]


def assistant(record: dict, prefixes: list[str]) -> str:
    parts = []
    for message in record["messages"]:
        if message["role"] != "assistant":
            continue
        parts.extend([str(message.get("content", "")), str(message.get("reasoning_content", ""))])
        if message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
    value = "\n".join(parts)
    for prefix in prefixes:
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def text(record: dict, prefixes: list[str]) -> str:
    users = "\n".join(str(message.get("content", "")) for message in record["messages"] if message["role"] == "user")
    return f"{users}\n{assistant(record, prefixes)}"


def words(value: str) -> list[str]:
    return WORD_RE.findall(value.lower())


def sampled_signature(tokens: list[str]) -> tuple[int, ...]:
    shingles = len(tokens) - 4
    if shingles <= 0:
        return ()
    step = max(1, math.ceil(shingles / 512))
    hashes = {
        zlib.crc32("\x1f".join(tokens[index : index + 5]).encode())
        for index in range(0, shingles, step)
    }
    return tuple(sorted(hashes)[:128])


def full_shingles(value: str) -> set[int]:
    items = words(value)
    return {zlib.crc32("\x1f".join(items[index : index + 5]).encode()) for index in range(max(0, len(items) - 4))}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def join(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def main() -> None:
    prefixes = boilerplate()
    index: dict[tuple[int, ...], list[str]] = defaultdict(list)
    metadata: dict[str, tuple[str, int]] = {}
    candidate_ids: set[str] = set()
    skipped_large_bands = 0
    for source in sorted(Path("data/pipeline/ir_dedup").glob("*.jsonl")):
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                uid = record["uid"]
                score = int(record.get("provenance", {}).get("seen_count", 1) or 1) * 10**12 + len(text(record, prefixes))
                metadata[uid] = (record["dataset"], score)
                signature = sampled_signature(words(text(record, prefixes)))
                for band_start in range(0, min(16, len(signature) - 1), 2):
                    index[signature[band_start : band_start + 2]].append(uid)
    candidates: set[tuple[str, str]] = set()
    for ids in index.values():
        if len(ids) < 2:
            continue
        if len(ids) > 200:
            skipped_large_bands += 1
            continue
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                candidates.add((left, right) if left < right else (right, left))
                candidate_ids.update((left, right))

    content: dict[str, str] = {}
    for source in sorted(Path("data/pipeline/ir_dedup").glob("*.jsonl")):
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["uid"] in candidate_ids:
                    content[record["uid"]] = text(record, prefixes)
    shingle_cache: dict[str, set[int]] = {}
    unions = UnionFind()
    verified = 0
    for left, right in candidates:
        first = shingle_cache.setdefault(left, full_shingles(content[left]))
        second = shingle_cache.setdefault(right, full_shingles(content[right]))
        if not first or not second:
            continue
        jaccard = len(first & second) / len(first | second)
        if jaccard >= 0.8:
            unions.join(left, right)
            verified += 1
    groups: dict[str, list[str]] = defaultdict(list)
    for uid in unions.parent:
        groups[unions.find(uid)].append(uid)
    dropped = set()
    per_dataset: Counter[str] = Counter()
    with Path("data/pipeline/gate2_l2_clusters.jsonl").open("w", encoding="utf-8") as handle:
        for members in groups.values():
            keeper = max(members, key=lambda uid: (metadata[uid][1], uid))
            discarded = [uid for uid in members if uid != keeper]
            dropped.update(discarded)
            per_dataset.update(metadata[uid][0] for uid in discarded)
            handle.write(json.dumps({"reason": "5gram_jaccard_ge_0.8", "keeper": keeper, "discarded": discarded}, ensure_ascii=False) + "\n")
    output_root = Path("data/pipeline/ir_l2")
    output_root.mkdir(parents=True, exist_ok=True)
    retained: Counter[str] = Counter()
    handles = {}
    try:
        for source in sorted(Path("data/pipeline/ir_dedup").glob("*.jsonl")):
            with source.open(encoding="utf-8") as input_handle:
                for line in input_handle:
                    record = json.loads(line)
                    if record["uid"] in dropped:
                        continue
                    dataset = record["dataset"]
                    if dataset not in handles:
                        handles[dataset] = (output_root / f"{dataset}.jsonl").open("w", encoding="utf-8")
                    handles[dataset].write(json.dumps(record, ensure_ascii=False) + "\n")
                    retained[dataset] += 1
    finally:
        for handle in handles.values():
            handle.close()
    result = {
        "num_perm": 128,
        "shingle": "normalized word 5-gram",
        "candidate_signature": "128-entry bottom-k MinHash; eight 2-hash LSH bands",
        "threshold": 0.8,
        "candidate_pairs": len(candidates),
        "verified_pairs": verified,
        "skipped_large_bands": skipped_large_bands,
        "removed_by_dataset": per_dataset,
        "retained_by_dataset": retained,
    }
    Path("data/pipeline/gate2_l2_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    with Path("data/pipeline/gate2_dedup.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## L2 近重\n\n"
            f"使用 128-entry bottom-k 5-gram MinHash 生成候选，后以完整 5-gram Jaccard ≥0.8 验证。候选对 {len(candidates):,}，验证阳性对 {verified:,}，删除 {len(dropped):,} 个样本；跳过过大 band {skipped_large_bands} 个。\n"
        )
        for dataset in sorted(retained):
            handle.write(f"- {dataset}: L2 后保留 {retained[dataset]:,}，本级删除 {per_dataset[dataset]:,}\n")


if __name__ == "__main__":
    main()
