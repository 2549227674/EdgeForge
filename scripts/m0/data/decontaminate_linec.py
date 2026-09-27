#!/usr/bin/env python3
"""Gate 5 canary and 13-gram decontamination; only removes training records."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import yaml


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
CANARY_RE = re.compile(r"terminal-bench-canary GUID [0-9a-f-]{36}", re.I)
N = 13


def normalized(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


def ngrams(value: str) -> set[str]:
    tokens = normalized(value)
    return {"\x1f".join(tokens[index : index + N]) for index in range(max(0, len(tokens) - N + 1))}


def test_rows() -> Iterable[tuple[str, str, str, bool]]:
    mmlu_selected = {(item["subject"], item["doc_id"]) for item in json.load(open("manifests/mmlu_fast_500.json"))["selected"]}
    for path in sorted(Path("data/eval/mmlu/cais_mmlu").glob("*/test-*.parquet")):
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            subject = str(row["subject"])
            value = "\n".join([str(row["question"]), json.dumps(row["choices"], ensure_ascii=False), str(row["answer"])])
            yield "MMLU", f"mmlu:{subject}:{index}", value, (subject, index) in mmlu_selected
    gsm_selected = {item["doc_id"] for item in json.load(open("manifests/gsm8k_fast_200.json"))["selected"]}
    gsm = Path("data/eval/gsm8k/main/test-00000-of-00001.parquet")
    for index, row in enumerate(pq.read_table(gsm).to_pylist()):
        yield "GSM8K", f"gsm8k:{index}", f"{row['question']}\n{row['answer']}", index in gsm_selected
    human = Path("data/eval/humaneval/openai_humaneval/test-00000-of-00001.parquet")
    for row in pq.read_table(human).to_pylist():
        yield "HumanEval", str(row["task_id"]), f"{row['prompt']}\n{row['canonical_solution']}", False
    baseline = yaml.safe_load(Path("configs/m0/m0_baseline_job.yaml").read_text(encoding="utf-8"))
    names = [item.split("/", 1)[1] for item in baseline["datasets"][0]["task_names"]]
    package = Path(
        os.environ.get(
            "EDGEFORGE_TB_PACKAGE_ROOT",
            Path.home() / ".cache/harbor/tasks/packages/terminal-bench",
        )
    )
    for name in names:
        task_root = next((path for path in (package / name).glob("*") if path.is_dir()), None)
        if task_root is None:
            raise FileNotFoundError(f"missing terminal-bench source for {name}")
        files = [task_root / "instruction.md", task_root / "solution/solve.sh"] + sorted((task_root / "tests").rglob("*"))
        parts = [path.read_text(encoding="utf-8", errors="replace") for path in files if path.is_file()]
        yield "TB2.1", f"tb:{name}", "\n".join(parts), True


def train_value(record: dict) -> str:
    parts = []
    for message in record["messages"]:
        parts.extend([str(message.get("content", "")), str(message.get("reasoning_content", ""))])
        if message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def main() -> None:
    index: dict[str, set[str]] = defaultdict(set)
    test_type: dict[str, str] = {}
    test_ngrams: dict[str, set[str]] = {}
    frozen: set[str] = set()
    canaries: set[str] = set()
    test_counts: Counter[str] = Counter()
    for kind, test_id, value, is_frozen in test_rows():
        grams = ngrams(value)
        test_counts[kind] += 1
        test_type[test_id] = kind
        test_ngrams[test_id] = grams
        if is_frozen:
            frozen.add(test_id)
        canaries.update(CANARY_RE.findall(value))
        for gram in grams:
            index[gram].add(test_id)
    expected = {"MMLU": 14042, "GSM8K": 1319, "HumanEval": 164, "TB2.1": 20}
    if {kind: test_counts[kind] for kind in expected} != expected:
        raise ValueError(f"frozen test counts differ: {dict(test_counts)}")

    deletions: set[str] = set()
    per_test_hits: dict[str, set[str]] = defaultdict(set)
    per_kind_train_hits: dict[str, set[str]] = defaultdict(set)
    canary_hits: dict[str, set[str]] = defaultdict(set)
    hit_rows = []
    input_count = 0
    for source in sorted(Path("data/pipeline/ir_l2").glob("*.jsonl")):
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                input_count += 1
                value = train_value(record)
                uid = record["uid"]
                found_canaries = sorted(canary for canary in canaries if canary in value)
                matched = defaultdict(set)
                for gram in ngrams(value):
                    for test_id in index.get(gram, ()):
                        matched[test_id].add(gram)
                if found_canaries or matched:
                    deletions.add(uid)
                    for canary in found_canaries:
                        canary_hits["TB2.1"].add(uid)
                    for test_id, hit_grams in matched.items():
                        kind = test_type[test_id]
                        per_kind_train_hits[kind].add(uid)
                        per_test_hits[test_id].update(hit_grams)
                    hit_rows.append(
                        {
                            "uid": uid,
                            "dataset": record["dataset"],
                            "canary_test_types": ["TB2.1"] if found_canaries else [],
                            "ngram_test_ids": sorted(matched),
                        }
                    )
    output_root = Path("data/pipeline/ir_gate5")
    output_root.mkdir(parents=True, exist_ok=True)
    handles = {}
    retained_by_dataset: Counter[str] = Counter()
    try:
        for source in sorted(Path("data/pipeline/ir_l2").glob("*.jsonl")):
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record["uid"] in deletions:
                        continue
                    dataset = record["dataset"]
                    if dataset not in handles:
                        handles[dataset] = (output_root / f"{dataset}.jsonl").open("w", encoding="utf-8")
                    handles[dataset].write(json.dumps(record, ensure_ascii=False) + "\n")
                    retained_by_dataset[dataset] += 1
    finally:
        for handle in handles.values():
            handle.close()
    hits_path = Path("data/pipeline/gate5_hits_raw/hits.jsonl")
    hits_path.parent.mkdir(parents=True, exist_ok=True)
    hits_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in hit_rows) + "\n", encoding="utf-8")
    lines = [
        "# M0 §5 门⑤去污染声明",
        "",
        "训练侧使用未脱敏文本扫描。命中仅删除训练样本；所有冻结测试题、manifest 和 baseline 配置均未修改。",
        "",
        "| 测试集 | 分母 | canary命中 | n-gram标阳训练样本 | 被删训练样本 | 剩余训练样本 | 冻结子集命中 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in ["MMLU", "GSM8K", "HumanEval", "TB2.1"]:
        kind_deleted = {row["uid"] for row in hit_rows if any(test_type[test_id] == kind for test_id in row["ngram_test_ids"]) or kind in row["canary_test_types"]}
        frozen_hits = sum(1 for test_id in frozen if test_type[test_id] == kind and per_test_hits[test_id])
        lines.append(
            f"| {kind} | {test_counts[kind]:,} | {len(canary_hits[kind]):,} | {len(per_kind_train_hits[kind]):,} | {len(kind_deleted):,} | {input_count - len(deletions):,} | {frozen_hits:,} |"
        )
    lines.extend(
        [
            "",
            "边界声明：本表证明训练池中未检出与冻结测试集的字面 canary/13-gram 重叠；不证明语义层面无污染，也不追溯上游模型的预训练污染。",
            f"总训练样本：扫描前 {input_count:,}，删除 {len(deletions):,}，保留 {input_count - len(deletions):,}。",
        ]
    )
    Path("data/pipeline/gate5_decontamination.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
