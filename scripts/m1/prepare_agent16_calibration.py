#!/usr/bin/env python3
"""Prepare and independently verify the frozen EdgeForge M1 agent-16 dataset.

This program is deliberately CPU-only and content-silent.  It uses the
segmented M1 source audit as a candidate index, excludes the complete frozen
holdout by source UID, recovers only the selected source records, renders with
the frozen Gemma deployment template, and writes private text only to the
RKLLM ``data_quant.json`` file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_RECORDS = 154_097
ANTHROPIC_LABEL = "claimed_anthropic_from_source_label"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def family_group(family: str) -> str:
    return "anthropic" if family == ANTHROPIC_LABEL else "other"


def seeded_hash(seed: int, *parts: str) -> str:
    payload = "\0".join([str(seed), *parts]).encode("utf-8")
    return sha256_bytes(payload)


def tokenizer_identity(model_dir: Path) -> dict[str, Any]:
    names = ("tokenizer.json", "tokenizer_config.json")
    files = {
        name: {"bytes": (model_dir / name).stat().st_size, "sha256": sha256_file(model_dir / name)}
        for name in names
    }
    return {
        "files": files,
        "identity_sha256": sha256_bytes(canonical_bytes(files)),
    }


def verify_source_pool(workspace: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("record_count") != EXPECTED_RECORDS:
        raise RuntimeError("canonical source manifest record count drift")
    actual_files: list[dict[str, Any]] = []
    total_records = 0
    for expected in manifest["files"]:
        path = workspace / expected["path"]
        if not path.is_file():
            raise RuntimeError(f"canonical source file missing: {expected['path']}")
        with path.open("rb") as source:
            records = sum(bool(line.strip()) for line in source)
        actual = {
            "path": expected["path"],
            "bytes": path.stat().st_size,
            "records": records,
            "sha256": sha256_file(path),
        }
        if actual != {key: expected[key] for key in actual}:
            raise RuntimeError(f"canonical source identity drift: {expected['path']}")
        actual_files.append(actual)
        total_records += records
    if total_records != EXPECTED_RECORDS:
        raise RuntimeError("canonical source record total drift")
    identity_payload = {"record_count": total_records, "files": actual_files}
    return {
        "manifest": str(manifest_path.relative_to(workspace)),
        "manifest_sha256": sha256_file(manifest_path),
        "record_count": total_records,
        "files": actual_files,
        "identity_sha256": sha256_bytes(canonical_bytes(identity_payload)),
        "verified": True,
    }


def load_holdout(path: Path) -> tuple[dict[str, Any], set[str]]:
    payload = read_json(path)
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != 1500:
        raise RuntimeError("frozen M1 holdout must contain exactly 1500 records")
    uids = {str(row["uid"]) for row in rows}
    if len(uids) != 1500:
        raise RuntimeError("frozen M1 holdout contains duplicate UIDs")
    return payload, uids


def iter_audit(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def load_candidates(
    audit_path: Path, holdout_uids: set[str]
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    holdout_seen: set[str] = set()
    for row in iter_audit(audit_path):
        uid = str(row["uid"])
        if uid in seen:
            raise RuntimeError("duplicate source UID in candidate audit")
        seen.add(uid)
        if uid in holdout_uids:
            holdout_seen.add(uid)
            continue
        family = str(row["family"])
        task_type = str(row["task_type"])
        group = family_group(family)
        groups[(group, family, task_type)].append(
            {
                "uid": uid,
                "dataset": str(row["dataset"]),
                "family_group": group,
                "family": family,
                "task_type": task_type,
                "original_tokens": int(row["original_tokens"]),
            }
        )
    if len(seen) != EXPECTED_RECORDS:
        raise RuntimeError(f"candidate audit row count drift: {len(seen)}")
    if holdout_seen != holdout_uids:
        raise RuntimeError("candidate audit does not contain every holdout UID")
    return groups, {
        "audit_records": len(seen),
        "audit_unique_uids": len(seen),
        "holdout_uids_found": len(holdout_seen),
        "eligible_records": len(seen) - len(holdout_seen),
        "eligible_strata": len(groups),
    }


def hamilton_group_slots(rule: dict[str, Any]) -> dict[str, int]:
    total = int(rule["sample_count"])
    weights = rule["family_group_apportionment"]["weights"]
    quotas = {key: total * float(value) for key, value in weights.items()}
    counts = {key: math.floor(value) for key, value in quotas.items()}
    remaining = total - sum(counts.values())
    order = sorted(weights, key=lambda key: (-(quotas[key] - counts[key]), key))
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def allocate_strata(
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    group_slots: dict[str, int],
    seed: int,
) -> dict[tuple[str, str, str], int]:
    result: dict[tuple[str, str, str], int] = {}
    for group_name, slots in sorted(group_slots.items()):
        strata = {key: rows for key, rows in groups.items() if key[0] == group_name}
        if not strata or slots <= 0:
            raise RuntimeError(f"invalid empty allocation group: {group_name}")
        if slots < len(strata):
            ranked = sorted(
                strata,
                key=lambda key: (
                    -len(strata[key]),
                    seeded_hash(seed, *key),
                    key[1],
                    key[2],
                ),
            )
            for key in ranked[:slots]:
                result[key] = 1
            continue

        counts = {key: 1 for key in strata}
        remaining = slots - len(strata)
        capacities = {key: len(rows) - 1 for key, rows in strata.items()}
        capacity_total = sum(capacities.values())
        quotas = {
            key: remaining * capacities[key] / capacity_total if capacity_total else 0.0
            for key in strata
        }
        for key, quota in quotas.items():
            counts[key] += math.floor(quota)
        left = slots - sum(counts.values())
        ranked = sorted(
            strata,
            key=lambda key: (
                -(quotas[key] - math.floor(quotas[key])),
                -capacities[key],
                seeded_hash(seed, *key),
                key[1],
                key[2],
            ),
        )
        for key in ranked:
            if left == 0:
                break
            if counts[key] < len(strata[key]):
                counts[key] += 1
                left -= 1
        if left:
            raise RuntimeError(f"could not allocate all slots for {group_name}")
        result.update(counts)
    return result


def select_rows(
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    allocation: dict[tuple[str, str, str], int],
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    seed = int(rule["seed"])
    count = int(rule["sample_count"])
    target = int(rule["target_total_tokens"])
    cap = int(rule["max_context_tokens"])
    selected: list[dict[str, Any]] = []
    for key in sorted(allocation):
        ranked = sorted(
            groups[key],
            key=lambda row: (
                abs(min(int(row["original_tokens"]), cap) * count - target),
                seeded_hash(seed, *key, row["uid"]),
                row["uid"],
            ),
        )
        selected.extend(ranked[: allocation[key]])
    if len(selected) != count or len({row["uid"] for row in selected}) != count:
        raise RuntimeError("selection did not produce exactly 16 unique UIDs")
    return selected


def expected_allocation(rule: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        rule["undercoverage_policy"]["expected_allocations"],
        key=lambda row: (row["family_group"], row["family"], row["task_type"]),
    )


def actual_allocation(
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    allocation: dict[tuple[str, str, str], int],
) -> list[dict[str, Any]]:
    return [
        {
            "family_group": key[0],
            "family": key[1],
            "task_type": key[2],
            "eligible_records": len(groups[key]),
            "slots": allocation[key],
        }
        for key in sorted(allocation)
    ]


def recover_records(
    workspace: Path,
    source_identity: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    wanted = {row["uid"] for row in selected}
    recovered: dict[str, dict[str, Any]] = {}
    for source_item in source_identity["files"]:
        path = workspace / source_item["path"]
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                uid = str(record.get("uid"))
                if uid not in wanted:
                    continue
                if uid in recovered:
                    raise RuntimeError("selected UID appeared more than once in source pool")
                recovered[uid] = {
                    "source_file": path.name,
                    "source_line": line_number,
                    "record": record,
                }
                if len(recovered) == len(wanted):
                    break
        if len(recovered) == len(wanted):
            break
    if set(recovered) != wanted:
        raise RuntimeError("failed to recover every selected UID")
    return recovered


def render_selected(
    tokenizer: Any,
    selected: list[dict[str, Any]],
    recovered: dict[str, dict[str, Any]],
    max_context: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    dataset: list[dict[str, str]] = []
    entries: list[dict[str, Any]] = []
    for rank, selected_row in enumerate(selected, 1):
        source = recovered[selected_row["uid"]]
        record = source["record"]
        if str(record.get("uid")) != selected_row["uid"]:
            raise RuntimeError("recovered source UID drift")
        if str(record.get("_family") or "unknown") != selected_row["family"]:
            raise RuntimeError("recovered source family drift")
        if str(record.get("_task_type") or "unknown") != selected_row["task_type"]:
            raise RuntimeError("recovered source task_type drift")
        rendered = tokenizer.apply_chat_template(
            record["messages"],
            tools=record.get("tools") or None,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
            preserve_thinking=True,
        )
        original_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if len(original_ids) != int(selected_row["original_tokens"]):
            raise RuntimeError("candidate-index rendered token count drift")
        truncated = len(original_ids) > max_context
        if truncated:
            calibration_text = tokenizer.decode(
                original_ids[:max_context],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        else:
            calibration_text = rendered
        calibration_ids = tokenizer(calibration_text, add_special_tokens=False)["input_ids"]
        if len(calibration_ids) > max_context:
            raise RuntimeError("decode/re-encode calibration exceeds 16384 tokens")
        dataset.append({"input": calibration_text, "target": ""})
        entries.append(
            {
                "selection_rank": rank,
                "uid": selected_row["uid"],
                "dataset": selected_row["dataset"],
                "source_file": source["source_file"],
                "source_line": source["source_line"],
                "family": selected_row["family"],
                "family_group": selected_row["family_group"],
                "task_type": selected_row["task_type"],
                "original_rendered_tokens": len(original_ids),
                "calibration_tokens": len(calibration_ids),
                "rendered_sha256": sha256_bytes(rendered.encode("utf-8")),
                "calibration_sha256": sha256_bytes(calibration_text.encode("utf-8")),
                "truncated_to_16384": truncated,
            }
        )
    return dataset, entries


def dataset_bytes(dataset: list[dict[str, str]]) -> bytes:
    return json.dumps(dataset, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def build_manifest(
    workspace: Path,
    args: argparse.Namespace,
    rule: dict[str, Any],
    source_identity: dict[str, Any],
    tokenizer_info: dict[str, Any],
    holdout_payload: dict[str, Any],
    holdout_uids: set[str],
    candidate_summary: dict[str, Any],
    allocation_rows: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    dataset_sha256: str,
) -> dict[str, Any]:
    selected_uids = {row["uid"] for row in entries}
    intersection = sorted(selected_uids & holdout_uids)
    selected_uid_sha = sha256_bytes("\n".join(sorted(selected_uids)).encode("utf-8"))
    holdout_uid_sha = sha256_bytes("\n".join(sorted(holdout_uids)).encode("utf-8"))
    proof_payload = {
        "selected_uid_set_sha256": selected_uid_sha,
        "holdout_uid_set_sha256": holdout_uid_sha,
        "intersection": intersection,
    }
    template_path = args.model / "chat_template.jinja"
    token_total = sum(row["calibration_tokens"] for row in entries)
    return {
        "schema": "edgeforge-m1-agent16-calibration/v1",
        "frozen_date": rule["frozen_date"],
        "purpose": "M1 agent-domain calibration dataset identity; no W8A8 build or evaluation performed",
        "selection_rule": {
            "path": str(args.rule.relative_to(workspace)),
            "sha256": sha256_file(args.rule),
            "seed": rule["seed"],
            "sample_count": rule["sample_count"],
            "target_total_tokens": rule["target_total_tokens"],
            "accepted_total_token_range": rule["accepted_total_token_range"],
        },
        "source_pool": source_identity,
        "candidate_index": {
            "path": str(args.audit_index.relative_to(workspace)),
            "sha256": sha256_file(args.audit_index),
            **candidate_summary,
        },
        "render_contract": {
            "model_dir": str(args.model.relative_to(workspace)),
            "template_sha256": sha256_file(template_path),
            "tokenizer": tokenizer_info,
            "options": {
                "add_generation_prompt": False,
                "enable_thinking": True,
                "preserve_thinking": True,
            },
            "max_context_tokens": rule["max_context_tokens"],
            "truncate_then_roundtrip": "first 16384 tokenizer IDs; decode retaining special tokens with cleanup disabled; re-encode <=16384",
        },
        "holdout_exclusion": {
            "manifest": str(args.holdout.relative_to(workspace)),
            "manifest_sha256": sha256_file(args.holdout),
            "manifest_declared_selected_uid_sha256": holdout_payload["selected_uid_sha256"],
            "holdout_records": len(holdout_uids),
            "holdout_uid_set_sha256": holdout_uid_sha,
            "selected_uid_set_sha256": selected_uid_sha,
            "intersection_count": len(intersection),
            "intersection_uids": intersection,
            "exclusion_proof_sha256": sha256_bytes(canonical_bytes(proof_payload)),
            "pass": not intersection,
        },
        "apportionment": {
            "family_group_slots": hamilton_group_slots(rule),
            "allocated_strata": allocation_rows,
            "complete_stratum_coverage_claimed": False,
            "eligible_strata": candidate_summary["eligible_strata"],
            "covered_strata": len(allocation_rows),
        },
        "dataset": {
            "path": str(args.output.relative_to(workspace)),
            "sha256": dataset_sha256,
            "schema": "list[{input: rendered_calibration_text, target: empty_string}]",
            "records": len(entries),
            "unique_uids": len({row["uid"] for row in entries}),
            "total_calibration_tokens": token_total,
            "truncated_records": sum(row["truncated_to_16384"] for row in entries),
            "all_records_at_most_16384": all(row["calibration_tokens"] <= 16384 for row in entries),
        },
        "entries": entries,
        "identity_caveat": "family_group=anthropic is derived only from the source family label; it is not a verified teacher identity",
        "privacy": "No source message, tool schema, rendered text, or calibration body is present in this manifest.",
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(workspace)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def validate_result(rule: dict[str, Any], manifest: dict[str, Any], dataset: list[dict[str, str]]) -> None:
    entries = manifest["entries"]
    low, high = map(int, rule["accepted_total_token_range"])
    total = sum(row["calibration_tokens"] for row in entries)
    if len(dataset) != 16 or len(entries) != 16:
        raise RuntimeError("agent-16 output count gate failed")
    if len({row["uid"] for row in entries}) != 16:
        raise RuntimeError("agent-16 UID uniqueness gate failed")
    if manifest["holdout_exclusion"]["intersection_count"] != 0:
        raise RuntimeError("agent-16 holdout exclusion gate failed")
    if any(row["calibration_tokens"] > 16384 for row in entries):
        raise RuntimeError("agent-16 per-record token cap failed")
    if not low <= total <= high:
        raise RuntimeError(f"agent-16 total token gate failed: {total} not in [{low}, {high}]")
    if any(set(row) != {"input", "target"} or row["target"] != "" for row in dataset):
        raise RuntimeError("agent-16 dataset schema gate failed")


def audit_markdown(
    workspace: Path,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> str:
    dataset = manifest["dataset"]
    lines = [
        "# EdgeForge M1 agent-16 calibration audit",
        "",
        "## Verdict",
        "",
        "PASS. The calibration asset contains exactly 16 unique source UIDs, excludes all 1,500 frozen holdout UIDs, respects the 16,384-token per-record cap after decode/re-encode, and falls inside the frozen total-token gate. No W8A8 build or evaluation was run.",
        "",
        "## Frozen identities",
        "",
        f"- Selection rule: `{manifest['selection_rule']['path']}` / `{manifest['selection_rule']['sha256']}`",
        f"- Source pool manifest: `{manifest['source_pool']['manifest']}` / `{manifest['source_pool']['manifest_sha256']}`; composite pool identity `{manifest['source_pool']['identity_sha256']}`",
        f"- Candidate index: `{manifest['candidate_index']['path']}` / `{manifest['candidate_index']['sha256']}`",
        f"- Holdout: `{manifest['holdout_exclusion']['manifest']}` / `{manifest['holdout_exclusion']['manifest_sha256']}`",
        f"- Chat template: `{manifest['render_contract']['template_sha256']}`",
        f"- Tokenizer identity: `{manifest['render_contract']['tokenizer']['identity_sha256']}`",
        f"- Dataset: `{manifest['dataset']['path']}` / `{manifest['dataset']['sha256']}`",
        "",
        "## Selection and undercoverage",
        "",
        f"After holdout exclusion, the candidate index contains {manifest['candidate_index']['eligible_records']:,} eligible records in {manifest['candidate_index']['eligible_strata']} nonempty family×task_type strata. Since 49 > 16, this asset does not claim complete stratum coverage. The frozen rule first apportions 13 anthropic-label / 3 other records by Hamilton 80/20, then applies its declared over-capacity/under-capacity stratum rule and deterministic seed hash ordering.",
        "",
        "| family_group | family | task_type | eligible | selected |",
        "|---|---|---|---:|---:|",
    ]
    for row in manifest["apportionment"]["allocated_strata"]:
        lines.append(
            f"| `{row['family_group']}` | `{row['family']}` | `{row['task_type']}` | {row['eligible_records']} | {row['slots']} |"
        )
    lines.extend(
        [
            "",
            "The `anthropic` bucket is only a source-label sampling group. It is not evidence of a verified teacher identity.",
            "",
            "## Acceptance gates",
            "",
            f"- Records / unique UIDs: {dataset['records']} / {dataset['unique_uids']}.",
            f"- Holdout intersection: {manifest['holdout_exclusion']['intersection_count']}; exclusion proof SHA-256 `{manifest['holdout_exclusion']['exclusion_proof_sha256']}`.",
            f"- Total effective tokens: {dataset['total_calibration_tokens']:,}; allowed range {manifest['selection_rule']['accepted_total_token_range'][0]:,}–{manifest['selection_rule']['accepted_total_token_range'][1]:,}.",
            f"- Per-record cap: all <=16,384 = `{str(dataset['all_records_at_most_16384']).lower()}`; truncated records = {dataset['truncated_records']}.",
            "- Render contract: `add_generation_prompt=false`, `enable_thinking=true`, `preserve_thinking=true`.",
            "- Dataset schema: `list[{input: rendered_calibration_text, target: \"\"}]`.",
            "- Determinism: a second in-process re-render matched every rendered/calibration SHA and the complete dataset bytes; the external verify-only rerun is recorded in the receipt after execution.",
            "- Privacy: source messages and rendered bodies appear only in the private `data_quant.json`; logs, receipt, manifest, and this audit contain identity/count/hash metadata only.",
            "",
            "## Runtime isolation",
            "",
            f"- Host: `{receipt['runtime']['hostname']}`; host logical CPUs: {receipt['runtime']['host_logical_cpus']}; cgroup CPU quota: {receipt['runtime']['cgroup_cpu_quota_cores']} cores; cgroup memory limit: {receipt['runtime']['cgroup_memory_limit_bytes']} bytes.",
            f"- CUDA visibility: `{receipt['runtime']['cuda_visible_devices']}`; workers: 1; `OMP_NUM_THREADS={receipt['runtime']['omp_num_threads']}`; niceness: {receipt['runtime']['niceness']}.",
            f"- Generation wall clock: {receipt['runtime']['wall_seconds']:.3f} s; process max RSS: {receipt['runtime']['max_rss_kib']:,} KiB.",
            "- C01-S training directories, model output, merge watcher, GPUs, W8A8 build, and B0-before evaluation were not accessed or invoked by this script.",
            "",
            "## Absolute cloud paths",
            "",
            f"- Dataset: `{args.output}`",
            f"- Identity manifest: `{args.manifest}`",
            f"- Receipt: `{args.receipt}`",
            f"- Audit: `{args.audit}`",
            f"- Generator: `{Path(__file__).resolve()}`",
            f"- Selection rule: `{args.rule}`",
            "",
        ]
    )
    return "\n".join(lines)


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    workspace = args.workspace.resolve()
    rule = read_json(args.rule)
    if int(rule["seed"]) != 20260811 or int(rule["sample_count"]) != 16:
        raise RuntimeError("selection rule identity drift")
    source_identity = verify_source_pool(workspace, args.source_manifest)
    holdout_payload, holdout_uids = load_holdout(args.holdout)
    groups, candidate_summary = load_candidates(args.audit_index, holdout_uids)
    if candidate_summary["eligible_strata"] != rule["undercoverage_policy"]["observed_eligible_strata"]:
        raise RuntimeError("eligible stratum count drift from frozen rule")
    slots = hamilton_group_slots(rule)
    if slots != rule["family_group_apportionment"]["slots"]:
        raise RuntimeError("family-group Hamilton allocation drift")
    allocation = allocate_strata(groups, slots, int(rule["seed"]))
    allocation_rows = actual_allocation(groups, allocation)
    if allocation_rows != expected_allocation(rule):
        raise RuntimeError("family-by-task_type allocation drift from frozen rule")
    selected = select_rows(groups, allocation, rule)
    recovered = recover_records(workspace, source_identity, selected)

    from transformers import AutoTokenizer, __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=True
    )
    tokenizer_info = tokenizer_identity(args.model)
    tokenizer_info.update(
        {
            "class": type(tokenizer).__name__,
            "transformers_version": transformers_version,
        }
    )
    dataset, entries = render_selected(
        tokenizer, selected, recovered, int(rule["max_context_tokens"])
    )
    rendered_again, entries_again = render_selected(
        tokenizer, selected, recovered, int(rule["max_context_tokens"])
    )
    if dataset_bytes(dataset) != dataset_bytes(rendered_again) or entries != entries_again:
        raise RuntimeError("second-pass re-render was not byte stable")
    data_sha = sha256_bytes(dataset_bytes(dataset))
    manifest = build_manifest(
        workspace,
        args,
        rule,
        source_identity,
        tokenizer_info,
        holdout_payload,
        holdout_uids,
        candidate_summary,
        allocation_rows,
        entries,
        data_sha,
    )
    validate_result(rule, manifest, dataset)
    context = {
        "rule": rule,
        "second_pass_rerender_match": True,
        "selection_uids_sha256": manifest["holdout_exclusion"]["selected_uid_set_sha256"],
    }
    return manifest, dataset, context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("generate", "verify-only"), required=True)
    parser.add_argument("--rule", type=Path, default=Path("manifests/m1_calib_agent16_selection_rule.json"))
    parser.add_argument("--source-manifest", type=Path, default=Path("manifests/mix_records_sha256.json"))
    parser.add_argument("--holdout", type=Path, default=Path("manifests/m1_holdout.json"))
    parser.add_argument("--audit-index", type=Path, default=Path("results/t1/segmented_4096_r6/segment_audit.jsonl.gz"))
    parser.add_argument("--model", type=Path, default=Path("models/gemma-4-E4B-it"))
    parser.add_argument("--output", type=Path, default=Path("data/m1/calibration/agent16/data_quant.json"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/m1_calib_agent16_sha256.json"))
    parser.add_argument("--receipt", type=Path, default=Path("results/m1/agent16_calibration_receipt.json"))
    parser.add_argument("--audit", type=Path, default=Path("results/m1/agent16_calibration_audit.md"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    for name in ("rule", "source_manifest", "holdout", "audit_index", "model", "output", "manifest", "receipt", "audit"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, workspace / path)
    return args


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    started = time.perf_counter()
    manifest, dataset, context = prepare(args)
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    data_bytes = dataset_bytes(dataset)

    if args.mode == "verify-only":
        for path in (args.output, args.manifest, args.receipt, args.audit):
            if not path.is_file():
                raise SystemExit(f"verification target missing: {path}")
        if args.output.read_bytes() != data_bytes:
            raise RuntimeError("verify-only dataset byte drift")
        if args.manifest.read_bytes() != manifest_bytes:
            raise RuntimeError("verify-only manifest byte drift")
        stored_manifest = read_json(args.manifest)
        declared = stored_manifest["dataset"]["sha256"]
        if sha256_file(args.output) != declared:
            raise RuntimeError("declared dataset SHA does not match file")
        receipt = read_json(args.receipt)
        declared_files = receipt["declared_files"]
        for path_text, expected_sha in declared_files.items():
            if sha256_file(Path(path_text)) != expected_sha:
                raise RuntimeError(f"declared file hash drift: {path_text}")
        result = {
            "status": "PASS",
            "mode": "verify-only",
            "records": len(dataset),
            "total_tokens": manifest["dataset"]["total_calibration_tokens"],
            "holdout_intersection": manifest["holdout_exclusion"]["intersection_count"],
            "dataset_sha256": sha256_bytes(data_bytes),
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "second_pass_rerender_match": context["second_pass_rerender_match"],
            "wall_seconds": round(time.perf_counter() - started, 3),
        }
        print(json.dumps(result, sort_keys=True))
        return 0

    outputs = (args.output, args.manifest, args.receipt, args.audit)
    conflicts = [str(path) for path in outputs if path.exists()]
    if conflicts:
        raise SystemExit("refusing to overwrite existing agent-16 outputs: " + ", ".join(conflicts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data_bytes)
    args.manifest.write_bytes(manifest_bytes)

    memory_limit_path = Path("/sys/fs/cgroup/memory.max")
    memory_limit = memory_limit_path.read_text().strip() if memory_limit_path.exists() else "unknown"
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    cpu_max = cpu_max_path.read_text().strip() if cpu_max_path.exists() else "unknown"
    cpu_quota_cores: float | str = "unknown"
    if cpu_max != "unknown":
        quota, period = cpu_max.split()
        if quota == "max":
            cpu_quota_cores = "unlimited"
        else:
            cpu_quota_cores = int(quota) / int(period)
    receipt = {
        "schema": "edgeforge-m1-agent16-calibration-receipt/v1",
        "status": "PASS",
        "scope": "calibration preparation/freeze/audit only; no W8A8 build and no evaluation",
        "record_count": len(dataset),
        "total_calibration_tokens": manifest["dataset"]["total_calibration_tokens"],
        "holdout_intersection_count": manifest["holdout_exclusion"]["intersection_count"],
        "second_pass_rerender_match": context["second_pass_rerender_match"],
        "declared_files": {
            str(args.output): sha256_file(args.output),
            str(args.manifest): sha256_file(args.manifest),
            str(args.rule): sha256_file(args.rule),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
        "runtime": {
            "hostname": platform.node(),
            "host_logical_cpus": os.cpu_count(),
            "cgroup_cpu_max": cpu_max,
            "cgroup_cpu_quota_cores": cpu_quota_cores,
            "cgroup_memory_limit_bytes": memory_limit,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "niceness": os.nice(0),
            "workers": 1,
            "wall_seconds": round(time.perf_counter() - started, 3),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "privacy": "No private source or rendered body is present in this receipt or in process output.",
    }
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.audit.write_text(audit_markdown(args.workspace, args, manifest, receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": len(dataset),
                "total_tokens": manifest["dataset"]["total_calibration_tokens"],
                "holdout_intersection": manifest["holdout_exclusion"]["intersection_count"],
                "dataset_sha256": manifest["dataset"]["sha256"],
                "manifest_sha256": sha256_file(args.manifest),
                "receipt_sha256": sha256_file(args.receipt),
                "audit_sha256": sha256_file(args.audit),
                "wall_seconds": receipt["runtime"]["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
