#!/usr/bin/env python3
"""Freeze or verify the OPE A2 source-diverse RKLLM calibration dataset.

The public manifest and receipt contain identities, counts, lengths, and hashes
only.  The materialized calibration text is written exclusively to the private
``data_quant.json`` output.  Source-window extraction is explicitly recorded;
``zero build truncation`` means the resulting inputs are already below the
converter cap before RKLLM sees them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO


EXPECTED_RULE_SCHEMA = "edgeforge-ope-a2-calibration-selection-rule/v1"
EXPECTED_OUTPUT_COUNT = 128
KERNELBENCH_NEEDLES = ("kernelbench-hard-traces", "kernelbench-mega-traces")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def dataset_bytes(value: list[dict[str, str]]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


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


def seeded_hash(seed: str, *parts: str) -> str:
    return sha256_bytes("\0".join((seed, *parts)).encode("utf-8"))


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires data")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path.resolve())


def group_for_record(
    source_file: str, record: dict[str, Any], strata: list[dict[str, Any]]
) -> str | None:
    family = str(record.get("_family") or record.get("source_family") or "unknown")
    task_type = str(record.get("_task_type") or "unknown")
    provenance = record.get("provenance") or {}
    category = provenance.get("category") if isinstance(provenance, dict) else None
    matches: list[str] = []
    for spec in strata:
        if source_file != spec["source_file"] or family != spec["family"]:
            continue
        if "task_type" in spec and task_type != spec["task_type"]:
            continue
        if "category" in spec and category != spec["category"]:
            continue
        matches.append(str(spec["id"]))
    if len(matches) > 1:
        raise RuntimeError(f"record matched multiple frozen strata: {record.get('uid')}")
    return matches[0] if matches else None


def kernelbench_hit(record: dict[str, Any]) -> bool:
    fields = [
        record.get("dataset"),
        record.get("first_source_dataset"),
        record.get("source_family"),
        record.get("_family"),
    ]
    provenance = record.get("provenance")
    if isinstance(provenance, dict):
        fields.extend(provenance.values())
    text = json.dumps(fields, ensure_ascii=False, sort_keys=True).lower()
    return any(needle in text for needle in KERNELBENCH_NEEDLES)


def scan_source_pool(
    workspace: Path,
    source_manifest_path: Path,
    holdout_uids: set[str],
    rule: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    source_manifest = read_json(source_manifest_path)
    if int(source_manifest.get("record_count", -1)) != int(rule["source_pool"]["expected_records"]):
        raise RuntimeError("canonical source-manifest record count drift")
    strata = rule["strata"]
    byte_floor = int(rule["sample_contract"]["source_json_bytes"]["min_inclusive"])
    byte_ceiling = int(rule["sample_contract"]["source_json_bytes"]["max_inclusive"])
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actual_files: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    observed_groups: Counter[str] = Counter()
    exclusion_counts: Counter[str] = Counter()
    kernelbench_hits = 0
    holdout_seen: set[str] = set()
    unassigned_eligible: Counter[str] = Counter()

    for expected in source_manifest["files"]:
        path = workspace / expected["path"]
        if not path.is_file():
            raise RuntimeError(f"canonical source file missing: {expected['path']}")
        digest = hashlib.sha256()
        records = 0
        with path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                digest.update(line)
                if not line.strip():
                    continue
                records += 1
                record = json.loads(line)
                uid = str(record["uid"])
                if uid in seen_uids:
                    raise RuntimeError(f"duplicate canonical source UID: {uid}")
                seen_uids.add(uid)
                if kernelbench_hit(record):
                    kernelbench_hits += 1
                if uid in holdout_uids:
                    holdout_seen.add(uid)
                    exclusion_counts["frozen_holdout"] += 1
                    continue
                group = group_for_record(path.name, record, strata)
                if group is None:
                    unassigned_eligible[path.name] += 1
                    continue
                observed_groups[group] += 1
                if len(line) < byte_floor:
                    exclusion_counts["source_json_below_floor"] += 1
                    continue
                if len(line) > byte_ceiling:
                    exclusion_counts["source_json_above_ceiling"] += 1
                    continue
                candidates[group].append(
                    {
                        "uid": uid,
                        "dataset": str(record.get("dataset") or path.stem),
                        "family": str(record.get("_family") or record.get("source_family") or "unknown"),
                        "task_type": str(record.get("_task_type") or "unknown"),
                        "category": (record.get("provenance") or {}).get("category"),
                        "source_session": str(record.get("source_session") or uid),
                        "source_file": path.name,
                        "source_path": str(path.resolve()),
                        "source_line": records,
                        "source_offset": offset,
                        "source_json_bytes": len(line),
                        "stratum": group,
                    }
                )
        actual = {
            "path": expected["path"],
            "bytes": path.stat().st_size,
            "records": records,
            "sha256": digest.hexdigest(),
        }
        expected_projection = {key: expected[key] for key in actual}
        if actual != expected_projection:
            raise RuntimeError(f"canonical source identity drift: {expected['path']}")
        actual_files.append(actual)

    if len(seen_uids) != int(rule["source_pool"]["expected_records"]):
        raise RuntimeError("canonical source UID count drift")
    if holdout_seen != holdout_uids:
        raise RuntimeError("not every frozen holdout UID was found in the canonical pool")
    if kernelbench_hits:
        raise RuntimeError(f"kernelbench provenance leaked into canonical pool: {kernelbench_hits}")

    seed = str(rule["seed"])
    for spec in strata:
        group = str(spec["id"])
        quota = int(spec["quota"])
        if len(candidates[group]) < quota:
            raise RuntimeError(
                f"frozen stratum {group} has {len(candidates[group])} bounded candidates for quota {quota}"
            )
        candidates[group].sort(
            key=lambda row: (seeded_hash(seed, group, row["uid"]), row["uid"])
        )

    identity_payload = {
        "record_count": len(seen_uids),
        "files": actual_files,
    }
    summary = {
        "manifest": relative(source_manifest_path, workspace),
        "manifest_sha256": sha256_file(source_manifest_path),
        "record_count": len(seen_uids),
        "files": actual_files,
        "identity_sha256": sha256_bytes(canonical_bytes(identity_payload)),
        "holdout_records_found_and_excluded": len(holdout_seen),
        "kernelbench_provenance_hits": kernelbench_hits,
        "observed_records_by_frozen_stratum_before_byte_bounds": dict(sorted(observed_groups.items())),
        "bounded_candidates_by_stratum": {
            spec["id"]: len(candidates[spec["id"]]) for spec in strata
        },
        "unassigned_non_holdout_records_by_file": dict(sorted(unassigned_eligible.items())),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "verified": True,
    }
    return candidates, summary


def read_record(candidate: dict[str, Any]) -> dict[str, Any]:
    path = Path(candidate["source_path"])
    with path.open("rb") as source:
        source.seek(int(candidate["source_offset"]))
        line = source.readline()
    if len(line) != int(candidate["source_json_bytes"]):
        raise RuntimeError(f"source-line byte drift: {candidate['uid']}")
    record = json.loads(line)
    if str(record.get("uid")) != candidate["uid"]:
        raise RuntimeError(f"source-offset UID drift: {candidate['uid']}")
    return record


def render_record(tokenizer: Any, record: dict[str, Any], options: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        record["messages"],
        tools=record.get("tools") or None,
        tokenize=False,
        add_generation_prompt=bool(options["add_generation_prompt"]),
        enable_thinking=bool(options["enable_thinking"]),
        preserve_thinking=bool(options["preserve_thinking"]),
    )


def materialize_candidate(
    tokenizer: Any,
    candidate: dict[str, Any],
    record: dict[str, Any],
    rule: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]] | None:
    contract = rule["sample_contract"]
    target = int(contract["window_source_tokens"])
    minimum = int(contract["min_materialized_tokens"])
    maximum = int(contract["max_sample_tokens"])
    rendered = render_record(tokenizer, record, contract["render_options"])
    source_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if len(source_ids) < target:
        return None
    span = len(source_ids) - target + 1
    offset_hash = seeded_hash(str(rule["seed"]), candidate["uid"], "window_offset")
    start = int(offset_hash, 16) % span
    end = start + target
    selected_ids = list(source_ids[start:end])
    tail_shrinks = 0
    while True:
        text = tokenizer.decode(
            selected_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        materialized_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(materialized_ids) <= target:
            break
        selected_ids.pop()
        tail_shrinks += 1
        if len(selected_ids) < minimum:
            return None
    if not minimum <= len(materialized_ids) <= maximum:
        return None
    dataset_row = {"input": text, "target": ""}
    entry = {
        "uid": candidate["uid"],
        "dataset": candidate["dataset"],
        "family": candidate["family"],
        "task_type": candidate["task_type"],
        "category": candidate["category"],
        "source_session_sha256": sha256_bytes(candidate["source_session"].encode("utf-8")),
        "source_file": candidate["source_file"],
        "source_line": candidate["source_line"],
        "source_json_bytes": candidate["source_json_bytes"],
        "stratum": candidate["stratum"],
        "full_rendered_tokens": len(source_ids),
        "full_rendered_sha256": sha256_bytes(rendered.encode("utf-8")),
        "window_source_token_start": start,
        "window_source_token_end_exclusive": start + len(selected_ids),
        "window_source_tokens": len(selected_ids),
        "roundtrip_tail_shrinks": tail_shrinks,
        "materialized_tokens": len(materialized_ids),
        "materialized_sha256": sha256_bytes(text.encode("utf-8")),
        "source_window_extracted": len(source_ids) > len(selected_ids),
        "build_truncated": False,
    }
    return dataset_row, entry


def select_and_materialize(
    tokenizer: Any,
    candidates: dict[str, list[dict[str, Any]]],
    rule: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, int]]:
    dataset: list[dict[str, str]] = []
    entries: list[dict[str, Any]] = []
    used_uids: set[str] = set()
    used_sessions: set[str] = set()
    rejected: Counter[str] = Counter()

    for spec in rule["strata"]:
        group = str(spec["id"])
        quota = int(spec["quota"])
        selected_in_group = 0
        for candidate in candidates[group]:
            if candidate["uid"] in used_uids:
                rejected["duplicate_uid"] += 1
                continue
            if candidate["source_session"] in used_sessions:
                rejected["duplicate_source_session"] += 1
                continue
            record = read_record(candidate)
            materialized = materialize_candidate(tokenizer, candidate, record, rule)
            if materialized is None:
                rejected["materialization_length_gate"] += 1
                continue
            row, entry = materialized
            entry["selection_rank"] = len(entries) + 1
            dataset.append(row)
            entries.append(entry)
            used_uids.add(candidate["uid"])
            used_sessions.add(candidate["source_session"])
            selected_in_group += 1
            if selected_in_group == quota:
                break
        if selected_in_group != quota:
            raise RuntimeError(
                f"frozen stratum {group} underfilled after materialization: {selected_in_group}/{quota}"
            )
    return dataset, entries, dict(sorted(rejected.items()))


def tokenizer_identity(model_dir: Path, tokenizer: Any, transformers_version: str) -> dict[str, Any]:
    names = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")
    files = {
        name: {
            "bytes": (model_dir / name).stat().st_size,
            "sha256": sha256_file(model_dir / name),
        }
        for name in names
    }
    return {
        "model_dir": str(model_dir),
        "class": type(tokenizer).__name__,
        "transformers_version": transformers_version,
        "files": files,
        "identity_sha256": sha256_bytes(canonical_bytes(files)),
    }


def build_manifest(
    workspace: Path,
    args: argparse.Namespace,
    rule: dict[str, Any],
    source_summary: dict[str, Any],
    tokenizer_info: dict[str, Any],
    holdout_info: dict[str, Any],
    kernelbench_info: dict[str, Any],
    dataset: list[dict[str, str]],
    entries: list[dict[str, Any]],
    rejected: dict[str, int],
) -> dict[str, Any]:
    lengths = [int(row["materialized_tokens"]) for row in entries]
    quota_actual = Counter(row["stratum"] for row in entries)
    quota_expected = {row["id"]: int(row["quota"]) for row in rule["strata"]}
    selected_uid_sha = sha256_bytes("\n".join(sorted(row["uid"] for row in entries)).encode("utf-8"))
    data = dataset_bytes(dataset)
    return {
        "schema": "edgeforge-ope-a2-calibration-manifest/v1",
        "status": "PASS",
        "frozen_at": rule["frozen_at"],
        "purpose": rule["purpose"],
        "selection_rule": {
            "path": relative(args.rule, workspace),
            "sha256": sha256_file(args.rule),
            "seed": rule["seed"],
            "algorithm": rule["selection_algorithm"],
        },
        "source_pool": source_summary,
        "holdout_exclusion": holdout_info,
        "kernelbench_exclusion": kernelbench_info,
        "render_contract": tokenizer_info | {"options": rule["sample_contract"]["render_options"]},
        "quota": {
            "expected": quota_expected,
            "actual": dict(sorted(quota_actual.items())),
            "summary": rule["quota_summary"],
            "rationale": rule["quota_rationale"],
            "pass": dict(sorted(quota_actual.items())) == quota_expected,
        },
        "dataset": {
            "path": relative(args.output, workspace),
            "schema": rule["sample_contract"]["dataset_schema"],
            "records": len(dataset),
            "unique_uids": len({row["uid"] for row in entries}),
            "unique_source_sessions": len({row["source_session_sha256"] for row in entries}),
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "total_tokens": sum(lengths),
            "max_total_tokens": int(rule["sample_contract"]["max_total_tokens"]),
            "min_tokens": min(lengths),
            "p50_tokens": percentile(lengths, 0.50),
            "p90_tokens": percentile(lengths, 0.90),
            "p95_tokens": percentile(lengths, 0.95),
            "max_tokens": max(lengths),
            "max_sample_tokens": int(rule["sample_contract"]["max_sample_tokens"]),
            "source_window_extraction_count": sum(bool(row["source_window_extracted"]) for row in entries),
            "build_truncation_count": sum(bool(row["build_truncated"]) for row in entries),
            "zero_build_truncation": not any(row["build_truncated"] for row in entries),
            "selected_uid_set_sha256": selected_uid_sha,
        },
        "materialization_rejections_before_quota_fill": rejected,
        "entries": entries,
        "disclosure": "The 1024-token gate was infeasible for source-diverse whole-record selection: outside Crown, only two GLM-5.2 full records were <=1024 and Hermes/Qwen/Glint had none. A2 therefore uses deterministic windows from complete frozen-template renders. Every emitted input is <=1024 before RKLLM, so converter/build truncation is zero; source-window extraction is separately explicit.",
        "privacy": "No source message, tool schema, rendered body, or materialized calibration text is present in this manifest.",
        "generator": {
            "path": relative(Path(__file__).resolve(), workspace),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def validate(
    rule: dict[str, Any],
    manifest: dict[str, Any],
    dataset: list[dict[str, str]],
) -> None:
    contract = rule["sample_contract"]
    info = manifest["dataset"]
    if len(dataset) != EXPECTED_OUTPUT_COUNT or info["records"] != EXPECTED_OUTPUT_COUNT:
        raise RuntimeError("A2 N=128 gate failed")
    if info["unique_uids"] != EXPECTED_OUTPUT_COUNT:
        raise RuntimeError("A2 unique-UID gate failed")
    if info["unique_source_sessions"] != EXPECTED_OUTPUT_COUNT:
        raise RuntimeError("A2 unique-source-session gate failed")
    if info["total_tokens"] > int(contract["max_total_tokens"]):
        raise RuntimeError("A2 total-token cap failed")
    if info["max_tokens"] > int(contract["max_sample_tokens"]):
        raise RuntimeError("A2 per-sample token cap failed")
    if info["build_truncation_count"] != 0:
        raise RuntimeError("A2 build-truncation gate failed")
    if not manifest["quota"]["pass"]:
        raise RuntimeError("A2 frozen quota gate failed")
    if manifest["holdout_exclusion"]["intersection_count"] != 0:
        raise RuntimeError("A2 holdout exclusion gate failed")
    if manifest["kernelbench_exclusion"]["canonical_pool_hits"] != 0:
        raise RuntimeError("A2 KernelBench exclusion gate failed")
    if any(set(row) != {"input", "target"} or row["target"] != "" for row in dataset):
        raise RuntimeError("A2 RKLLM dataset schema gate failed")


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]]]:
    workspace = args.workspace.resolve()
    rule = read_json(args.rule)
    if rule.get("schema") != EXPECTED_RULE_SCHEMA or rule.get("status") != "FROZEN":
        raise RuntimeError("A2 selection rule is not the expected frozen schema")
    if int(rule["sample_contract"]["sample_count"]) != EXPECTED_OUTPUT_COUNT:
        raise RuntimeError("A2 frozen sample count drift")
    if sum(int(row["quota"]) for row in rule["strata"]) != EXPECTED_OUTPUT_COUNT:
        raise RuntimeError("A2 frozen quota sum drift")

    holdout_payload = read_json(args.holdout)
    holdout_uids = {str(row["uid"]) for row in holdout_payload["records"]}
    if len(holdout_uids) != int(holdout_payload["selected_records"]):
        raise RuntimeError("frozen holdout UID count drift")
    holdout_uid_sha = sha256_bytes("\n".join(sorted(holdout_uids)).encode("utf-8"))
    if holdout_uid_sha != holdout_payload["selected_uid_sha256"]:
        raise RuntimeError("frozen holdout UID identity drift")

    kernelbench_payload = read_json(args.kernelbench_assertion)
    if kernelbench_payload.get("assertion_passed") is not True:
        raise RuntimeError("canonical KernelBench exclusion assertion is not PASS")
    if kernelbench_payload.get("mix_provenance_file_sha256_overlap") != []:
        raise RuntimeError("canonical KernelBench exclusion overlap is nonempty")

    candidates, source_summary = scan_source_pool(
        workspace, args.source_manifest, holdout_uids, rule
    )

    from transformers import AutoTokenizer, __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=True
    )
    tokenizer_info = tokenizer_identity(args.model, tokenizer, transformers_version)
    if tokenizer_info["files"]["chat_template.jinja"]["sha256"] != read_json(args.source_manifest)["chat_template_sha256"]:
        raise RuntimeError("C02 chat template does not match canonical source manifest")

    dataset, entries, rejected = select_and_materialize(tokenizer, candidates, rule)
    selected_uids = {row["uid"] for row in entries}
    intersection = sorted(selected_uids & holdout_uids)
    holdout_info = {
        "manifest": relative(args.holdout, workspace),
        "manifest_sha256": sha256_file(args.holdout),
        "holdout_records": len(holdout_uids),
        "holdout_uid_set_sha256": holdout_uid_sha,
        "intersection_count": len(intersection),
        "intersection_uids": intersection,
        "pass": not intersection,
    }
    kernelbench_info = {
        "assertion": relative(args.kernelbench_assertion, workspace),
        "assertion_sha256": sha256_file(args.kernelbench_assertion),
        "upstream_assertion_passed": True,
        "upstream_sha_overlap": [],
        "canonical_pool_hits": source_summary["kernelbench_provenance_hits"],
        "selected_hits": 0,
        "pass": source_summary["kernelbench_provenance_hits"] == 0,
    }
    manifest = build_manifest(
        workspace,
        args,
        rule,
        source_summary,
        tokenizer_info,
        holdout_info,
        kernelbench_info,
        dataset,
        entries,
        rejected,
    )
    validate(rule, manifest, dataset)
    return manifest, dataset


def runtime_receipt(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    memory_limit_path = Path("/sys/fs/cgroup/memory.max")
    cpu_max_path = Path("/sys/fs/cgroup/cpu.max")
    return {
        "schema": "edgeforge-ope-c0-2-a2-calibration-receipt/v1",
        "status": "PASS",
        "scope": "selection, source-window materialization, and freeze only; no RKLLM build or quality evaluation",
        "gates": {
            "records_128": manifest["dataset"]["records"] == 128,
            "total_tokens_within_105865": manifest["dataset"]["total_tokens"] <= 105865,
            "max_sample_tokens_within_1024": manifest["dataset"]["max_tokens"] <= 1024,
            "zero_build_truncation": manifest["dataset"]["build_truncation_count"] == 0,
            "quota_exact": manifest["quota"]["pass"],
            "holdout_intersection_zero": manifest["holdout_exclusion"]["intersection_count"] == 0,
            "kernelbench_hits_zero": manifest["kernelbench_exclusion"]["canonical_pool_hits"] == 0,
        },
        "dataset_summary": manifest["dataset"],
        "quota_summary": manifest["quota"]["summary"],
        "disclosure": manifest["disclosure"],
        "declared_files": {
            str(args.output.resolve()): sha256_file(args.output),
            str(args.manifest.resolve()): sha256_file(args.manifest),
            str(args.rule.resolve()): sha256_file(args.rule),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
        "runtime": {
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "logical_cpus": os.cpu_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "cgroup_memory_max": memory_limit_path.read_text().strip() if memory_limit_path.exists() else "unknown",
            "cgroup_cpu_max": cpu_max_path.read_text().strip() if cpu_max_path.exists() else "unknown",
            "wall_seconds": round(time.perf_counter() - started, 3),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "privacy": "No private source or calibration body is present in this receipt or in process output.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "verify-only"), required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--rule", type=Path, default=Path("configs/operator/board/a2_calibration_rule.json"))
    parser.add_argument("--source-manifest", type=Path, default=Path("manifests/mix_records_sha256.json"))
    parser.add_argument("--holdout", type=Path, default=Path("manifests/m1_holdout.json"))
    parser.add_argument("--kernelbench-assertion", type=Path, default=Path("data/pipeline/kernelbench_exclusion.json"))
    parser.add_argument("--model", type=Path, default=Path("models/C02-S.O-rkllm-input-r1"))
    parser.add_argument("--output", type=Path, default=Path("data/operator/board/a2_calibration/data_quant.json"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/operator/board/a2_calibration_manifest.json"))
    parser.add_argument("--receipt", type=Path, default=Path("results/operator/board/c0/c0_2_a2_calibration_receipt.json"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    for name in (
        "rule",
        "source_manifest",
        "holdout",
        "kernelbench_assertion",
        "model",
        "output",
        "manifest",
        "receipt",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, workspace / value)
    return args


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    started = time.perf_counter()
    manifest, dataset = prepare(args)
    manifest_blob = pretty_bytes(manifest)
    dataset_blob = dataset_bytes(dataset)

    if args.mode == "verify-only":
        for path in (args.output, args.manifest, args.receipt):
            if not path.is_file():
                raise SystemExit(f"verification target missing: {path}")
        if args.output.read_bytes() != dataset_blob:
            raise RuntimeError("A2 verify-only dataset byte drift")
        if args.manifest.read_bytes() != manifest_blob:
            raise RuntimeError("A2 verify-only manifest byte drift")
        receipt = read_json(args.receipt)
        for path_text, expected in receipt["declared_files"].items():
            if sha256_file(Path(path_text)) != expected:
                raise RuntimeError(f"A2 declared-file hash drift: {path_text}")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "verify-only",
                    "records": manifest["dataset"]["records"],
                    "total_tokens": manifest["dataset"]["total_tokens"],
                    "dataset_sha256": manifest["dataset"]["sha256"],
                    "manifest_sha256": sha256_bytes(manifest_blob),
                },
                sort_keys=True,
            )
        )
        return 0

    conflicts = [str(path) for path in (args.output, args.manifest, args.receipt) if path.exists()]
    if conflicts:
        raise SystemExit("refusing to overwrite frozen A2 outputs: " + ", ".join(conflicts))
    for path in (args.output, args.manifest, args.receipt):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(dataset_blob)
    args.manifest.write_bytes(manifest_blob)
    receipt = runtime_receipt(args, manifest, started)
    args.receipt.write_bytes(pretty_bytes(receipt))
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": manifest["dataset"]["records"],
                "total_tokens": manifest["dataset"]["total_tokens"],
                "max_sample_tokens": manifest["dataset"]["max_tokens"],
                "source_windows": manifest["dataset"]["source_window_extraction_count"],
                "build_truncations": manifest["dataset"]["build_truncation_count"],
                "dataset_sha256": sha256_file(args.output),
                "manifest_sha256": sha256_file(args.manifest),
                "receipt_sha256": sha256_file(args.receipt),
                "wall_seconds": receipt["runtime"]["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
