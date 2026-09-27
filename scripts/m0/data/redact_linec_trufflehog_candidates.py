#!/usr/bin/env python3
"""Redact exact TruffleHog values from final training payload fields only.

Scan JSON and projected payload files are ignored because scan JSON can contain
candidate values.  Provenance fields are intentionally never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def scan_candidates(path: Path, payload_root: Path, input_root: Path) -> tuple[dict[Path, dict[str, set[str]]], Counter[str], Counter[str]]:
    candidates: dict[Path, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    detectors: Counter[str] = Counter()
    verification: Counter[str] = Counter()
    root = payload_root.resolve()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        raw = item.get("Raw")
        detector = str(item.get("DetectorName", "unknown"))
        file_name = item.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {}).get("file")
        if not isinstance(raw, str) or not raw or not file_name:
            raise ValueError("TruffleHog finding lacks an exact raw value or payload location")
        payload = Path(str(file_name)).resolve()
        if payload.parent != root:
            raise ValueError(f"finding outside projected payload: {payload}")
        candidates[(input_root / payload.name).resolve()][raw].add(detector)
        detectors[detector] += 1
        verification[str(bool(item.get("Verified", False))).lower()] += 1
    return candidates, detectors, verification


def redact_any(value: Any, replacements: dict[str, set[str]], applied: Counter[str], matched: set[str]) -> Any:
    if isinstance(value, str):
        for raw, detectors in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            count = value.count(raw)
            if count:
                value = value.replace(raw, "<REDACTED_TRUFFLEHOG_CANDIDATE>")
                matched.add(raw)
                for detector in detectors:
                    applied[detector] += count
        return value
    if isinstance(value, list):
        return [redact_any(item, replacements, applied, matched) for item in value]
    if isinstance(value, dict):
        return {key: redact_any(item, replacements, applied, matched) for key, item in value.items()}
    return value


def redact_payload_fields(record: dict[str, Any], replacements: dict[str, set[str]], applied: Counter[str], matched: set[str]) -> None:
    for message in record["messages"]:
        for key in ("content", "reasoning_content"):
            if key in message:
                message[key] = redact_any(message[key], replacements, applied, matched)
        if "tool_calls" in message:
            message["tool_calls"] = redact_any(message["tool_calls"], replacements, applied, matched)
    if "tools" in record:
        record["tools"] = redact_any(record["tools"], replacements, applied, matched)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, required=True, help="ignored TruffleHog JSONL from projected payloads")
    parser.add_argument("--payload-root", type=Path, default=ROOT / "data/pipeline/gate3_scan_payloads")
    parser.add_argument("--input", type=Path, default=ROOT / "data/pipeline/mix_records")
    args = parser.parse_args()
    input_root = args.input.resolve()
    candidates, detector_hits, verification = scan_candidates(args.scan, args.payload_root, input_root)

    records_redacted = 0
    replacement_hits: Counter[str] = Counter()
    matched_values: set[str] = set()
    for source in sorted(candidates):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=source.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    applied: Counter[str] = Counter()
                    redact_payload_fields(record, candidates[source], applied, matched_values)
                    if applied:
                        records_redacted += 1
                        provenance = record.setdefault("provenance", {})
                        provenance["redaction_rules"] = sorted(set(provenance.get("redaction_rules", [])) | {"trufflehog_unverified_candidate"})
                        provenance["trufflehog_candidate_detectors"] = sorted(applied)
                        replacement_hits.update(applied)
                    temporary.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, source)

    unresolved = sum(1 for values in candidates.values() for raw in values if raw not in matched_values)
    if unresolved:
        raise RuntimeError(f"{unresolved} projected-payload candidate values were not redacted")

    mix_path = ROOT / "data/mix.yaml"
    mix = yaml.safe_load(mix_path.read_text(encoding="utf-8"))
    mix["post_gate6_trufflehog_redaction"] = {
        "scan_scope": "trainer payload fields only; UID/provenance excluded",
        "scan_candidate_count": sum(detector_hits.values()),
        "verified_candidate_count": verification.get("true", 0),
        "redacted_records": records_redacted,
        "replacement_occurrences": sum(replacement_hits.values()),
        "detector_candidate_counts": dict(sorted(detector_hits.items())),
    }
    mix.setdefault("redaction_rule_record_counts", {})["trufflehog_unverified_candidate"] = records_redacted
    mix_path.write_text(yaml.safe_dump(mix, allow_unicode=True, sort_keys=False), encoding="utf-8")

    report = ROOT / "data/pipeline/gate3_final_mix_trufflehog.md"
    lines = [
        "# 门③最终 mix TruffleHog 复核",
        "",
        "扫描范围为实际送入训练模板的 messages 与 tools 载荷字段；UID、provenance、source hash 和文件路径均不是训练 token，明确排除。扫描 JSON 与投影载荷保留在忽略路径，本报告只保留检测器和计数。",
        "",
        f"- 扫描候选：{sum(detector_hits.values()):,}；验证为真：{verification.get('true', 0):,}；均按高风险候选处理。",
        f"- 精确候选值已替换：{sum(replacement_hits.values()):,} 处，覆盖 {records_redacted:,} 条规范池记录。",
        "- 替换占位符：`<REDACTED_TRUFFLEHOG_CANDIDATE>`；原始候选值未写入版本化产物。",
        "- 必须重建投影并复扫；只有复扫零候选才可签署门③最终输入安全检查。",
        "",
        "| 检测器 | 扫描候选 | 实际替换 |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {detector_hits[name]:,} | {replacement_hits[name]:,} |" for name in sorted(detector_hits))
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": sum(detector_hits.values()), "records_redacted": records_redacted, "replacements": sum(replacement_hits.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
