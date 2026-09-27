#!/usr/bin/env python3
"""Gate 6: build the full post-gate pool and configurable sampling recipes."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


REDACTIONS = {
    "openai_style_key": (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "<REDACTED_OPENAI_STYLE_KEY>"),
    "groq_style_key": (re.compile(r"\bgsk_[A-Za-z0-9_-]{20,}\b"), "<REDACTED_GROQ_STYLE_KEY>"),
    "postgres_uri": (re.compile(r"\bpostgres(?:ql)?://[^\s/@:]+:[^\s/@]+@[^\s/]+", re.I), "<REDACTED_POSTGRES_URI>"),
    "mongodb_uri": (re.compile(r"\bmongodb(?:\+srv)?://[^\s/@:]+:[^\s/@]+@[^\s/]+", re.I), "<REDACTED_MONGODB_URI>"),
    "credential_assignment": (re.compile(r"\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}", re.I), "<REDACTED_CREDENTIAL_ASSIGNMENT>"),
    "home_path": (re.compile(r"/home/[A-Za-z0-9_.-]+/"), "/home/<USER>/"),
    "ansi": (re.compile(r"\x1b\[[0-9;]*m"), ""),
}


def family(record: dict) -> tuple[str, str]:
    if record["dataset"] == "Glint-Research__Fable-5-traces":
        return "fable-5", "root dataset provenance"
    if record["dataset"] == "armand0e__claude-opus-4.8-pi-traces":
        return "opus-4.8", "root dataset provenance"
    if record["dataset"] == "WithinUsAI__claude_mythos_distilled_25k":
        return "mythos_synthetic", "root dataset provenance"
    if record["dataset"] == "lambda__hermes-agent-reasoning-traces":
        return record["source_family"], "config provenance"
    if record["dataset"] == "AletheiaResearch__GLM-5.2-Agent":
        return "glm-5.2", "root dataset provenance"
    if record["dataset"] == "armand0e__qwen3.7-max-pi-traces":
        return "qwen3.7-max", "root dataset provenance"
    label = str(record.get("first_source_dataset", "")).lower()
    if any(token in label for token in ("claude", "opus", "sonnet", "fable", "mythos")):
        return "claimed_anthropic_from_source_label", "first_source_dataset label only"
    return "unresolved_crownelius_source", "first_source_dataset has no model-family claim"


def task_type(record: dict) -> str:
    provenance = record.get("provenance", {})
    if provenance.get("category"):
        return f"hermes:{provenance['category']}/{provenance.get('subcategory', 'unspecified')}"
    if record.get("tools") or any(message.get("tool_calls") for message in record["messages"]):
        return "tool_agent"
    return "chat_or_reasoning"


def redact_value(value: str, applied: set[str]) -> str:
    for name, (regex, replacement) in REDACTIONS.items():
        value, count = regex.subn(replacement, value)
        if count:
            applied.add(name)
    return value


def redact_any(value, applied: set[str]):
    if isinstance(value, str):
        return redact_value(value, applied)
    if isinstance(value, list):
        return [redact_any(item, applied) for item in value]
    if isinstance(value, dict):
        return {key: redact_any(item, applied) for key, item in value.items()}
    return value


def redact(record: dict) -> dict:
    applied: set[str] = set()
    for message in record["messages"]:
        for field in ("content", "reasoning_content"):
            if field in message:
                message[field] = redact_value(str(message[field]), applied)
        for call in message.get("tool_calls", []):
            arguments = call.get("function", {}).get("arguments")
            if isinstance(arguments, dict):
                call["function"]["arguments"] = redact_any(arguments, applied)
    if applied:
        record.setdefault("provenance", {})["redaction_rules"] = sorted(applied)
    return record


def is_anthropic_style(value: str) -> bool:
    return value in {"fable-5", "opus-4.8", "mythos_synthetic", "claimed_anthropic_from_source_label"}


def main() -> None:
    manifest = json.loads(Path("manifests/data_archive_sha256.json").read_text(encoding="utf-8"))
    metadata = {item["repo_id"]: item for item in manifest["datasets"]}
    exclusions_path = Path("data/pipeline/gate3_security_exclusions.json")
    exclusions = {
        item["uid"]
        for item in json.loads(exclusions_path.read_text(encoding="utf-8")).get("records", [])
    } if exclusions_path.exists() else set()
    records = []
    mythos_dropped = 0
    security_dropped = 0
    for source in sorted(Path("data/pipeline/ir_gate5").glob("*.jsonl")):
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["dataset"] == "WithinUsAI__claude_mythos_distilled_25k":
                    mythos_dropped += 1
                    continue
                if record["uid"] in exclusions:
                    security_dropped += 1
                    continue
                label, evidence = family(record)
                record["_family"] = label
                record["_family_evidence"] = evidence
                record["_task_type"] = task_type(record)
                records.append(record)
    other = [record for record in records if not is_anthropic_style(record["_family"])]
    anthro = [record for record in records if is_anthropic_style(record["_family"])]
    # Gate 6 does not physically discard valid records. Family proportions are
    # training-time recipes so M1 can compare them at an equal token/step budget.
    kept = records
    output_root = Path("data/pipeline/mix_records")
    output_root.mkdir(parents=True, exist_ok=True)
    for stale_output in output_root.glob("*.jsonl"):
        stale_output.unlink()
    output_files = {}
    grouped: Counter[tuple[str, str, str, str, bool]] = Counter()
    used_files = set()
    redactions: Counter[str] = Counter()
    try:
        for record in kept:
            record = redact(record)
            for rule in record.get("provenance", {}).get("redaction_rules", []):
                redactions[rule] += 1
            dataset = record["dataset"]
            if dataset not in output_files:
                output_files[dataset] = (output_root / f"{dataset}.jsonl").open("w", encoding="utf-8")
            output_files[dataset].write(json.dumps(record, ensure_ascii=False) + "\n")
            repo_id = record["first_source_dataset"] if dataset == "Crownelius__Complete-FABLE.5-traces-2M" else record["first_source_dataset"]
            canonical_repo = repo_id if repo_id in metadata else (
                "Crownelius/Complete-FABLE.5-traces-2M" if dataset == "Crownelius__Complete-FABLE.5-traces-2M" else record["first_source_dataset"]
            )
            license_value = metadata.get(canonical_repo, {}).get("license", "unresolved_downstream_source")
            grouped[(dataset, str(record["first_source_dataset"]), record["_family"], record["_task_type"], bool(record["is_subagent"]))] += 1
            used_files.add(record.get("provenance", {}).get("file"))
    finally:
        for handle in output_files.values():
            handle.close()
    file_hashes = {file["path"]: file["sha256"] for item in manifest["datasets"] for file in item["files"]}
    kernel_hashes = {
        file["sha256"]
        for item in manifest["datasets"]
        if item["repo_id"] in {"Infatoshi/kernelbench-hard-traces", "Infatoshi/kernelbench-mega-traces"}
        for file in item["files"]
    }
    used_hashes = {file_hashes[path] for path in used_files if path in file_hashes}
    kernel_assertion = {
        "schema_version": 1,
        "kernelbench_file_sha256": sorted(kernel_hashes),
        "mix_provenance_file_sha256_overlap": sorted(kernel_hashes & used_hashes),
        "assertion_passed": not bool(kernel_hashes & used_hashes),
    }
    Path("data/pipeline/kernelbench_exclusion.json").write_text(json.dumps(kernel_assertion, indent=2) + "\n", encoding="utf-8")
    source_entries = []
    for (dataset, first_source, label, type_label, is_subagent), count in sorted(grouped.items()):
        root_repo = "Crownelius/Complete-FABLE.5-traces-2M" if dataset == "Crownelius__Complete-FABLE.5-traces-2M" else first_source
        root = metadata.get(root_repo, {})
        source_entries.append(
            {
                "dataset": dataset,
                "repo_id": root_repo,
                "revision": root.get("revision"),
                "first_source_dataset": first_source,
                "family": label,
                "family_evidence": "source metadata label" if dataset == "Crownelius__Complete-FABLE.5-traces-2M" else "root/config provenance",
                "license": root.get("license", "unlabeled"),
                "is_subagent": is_subagent,
                "task_type": type_label,
                "records": count,
                "gate_passed": ["gate1", "gate4", "gate4b", "gate2", "gate3_redaction_applied", "gate5", "gate6"],
            }
        )
    mix = {
        "schema_version": 2,
        "purpose": "M0 line-C full post-gate canonical pool; renderer input is data/pipeline/mix_records/",
        "canonical_pool": {
            "records": len(kept),
            "physical_family_cap_applied": False,
            "anthropic_style_records": len(anthro),
            "other_family_records": len(other),
        },
        "sampling_policy": {
            "selection": "compare recipes at equal optimizer-step and token budgets before choosing the M1 default",
            "within_group": "uniform_record",
            "recipes": {
                "raw_uniform": {
                    "anthropic_style_probability": len(anthro) / len(kept),
                    "other_family_probability": len(other) / len(kept),
                },
                "family_80_20": {
                    "anthropic_style_probability": 0.80,
                    "other_family_probability": 0.20,
                },
                "family_60_40": {
                    "anthropic_style_probability": 0.60,
                    "other_family_probability": 0.40,
                },
            },
        },
        "mythos_disposition": {"kept_after_gate2": mythos_dropped, "action": "excluded", "reason": "below 2% of original 25,000 after boilerplate stripping and dedup"},
        "redaction_rule_record_counts": dict(sorted(redactions.items())),
        "sources": source_entries,
    }
    Path("data/mix.yaml").write_text(yaml.safe_dump(mix, allow_unicode=True, sort_keys=False), encoding="utf-8")
    family_counts = Counter(record["_family"] for record in kept)
    task_counts = Counter(record["_task_type"] for record in kept)
    lines = [
        "# M0 §5 门⑥配平",
        "",
        f"Mythos 在剥样板和去重后仅余 {mythos_dropped:,} 条（低于原始25,000条的2%），按预声明方案整集剔除。另有 {security_dropped:,} 条记录因最终 mix TruffleHog 候选无法安全精确替换而按门③高风险规则排除。其余通过硬门的 {len(kept):,} 条记录全部保留为规范数据池，不再物理执行家族 cap。Crownelius 当前完整 export 的 `first_source_dataset` 远超原卡所列三家；因此仅将其数据源标签中的 Claude/Fable/Opus/Sonnet/Mythos 声明归为 `claimed_anthropic_from_source_label`，不把名称推定为已核实 teacher 身份。",
        "",
        "| 家族/来源标签 | 记录数 |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {count:,} |" for label, count in sorted(family_counts.items()))
    lines.extend(["", "| 任务类型 | 记录数 |", "|---|---:|"])
    lines.extend(f"| {label} | {count:,} |" for label, count in sorted(task_counts.items()))
    lines.extend(
        [
            "",
            f"- 门⑤后输入：{len(records) + mythos_dropped + security_dropped:,}；Mythos 规则删除 {mythos_dropped:,}；门③安全排除 {security_dropped:,}；规范池保留 {len(kept):,}。",
            f"- 原始均匀采样的 Anthropic-style 占比：{len(anthro) / len(kept):.2%}（{len(anthro):,}/{len(kept):,}）；非 Anthropic 分组 {len(other):,}。",
            "- 不物理删除家族样本；`data/mix.yaml` 同时记录 raw-uniform、80/20、60/40 三种训练时采样配方，须在相同 optimizer-step 与 token 预算下对照后定稿。",
            f"- 脱敏规则命中记录数：{sum(redactions.values()):,}；脱敏在门⑤后、门⑦前落地。",
            f"- KernelBench 排除断言：{'通过' if kernel_assertion['assertion_passed'] else '失败'}。",
        ]
    )
    Path("data/pipeline/gate6_balance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
