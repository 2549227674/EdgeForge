#!/usr/bin/env python3
"""Stream M1 records through the frozen M0 renderer contract.

The stats pass records untrimmed lengths.  The render pass applies the M1
deterministic trimming policy and writes compact Parquet shards suitable for
training without materializing the full corpus in memory.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
HISTOGRAM_BOUNDS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
_WORKER_TOKENIZER: Any = None
_WORKER_RENDERER: Any = None


def load_renderer() -> Any:
    path = ROOT / "scripts/m0/data/render_linec_samples.py"
    spec = importlib.util.spec_from_file_location("edgeforge_m0_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_records(directory: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield path.name, line_number, json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_identity(
    manifest_path: Path, records_directory: Path, model_directory: Path
) -> dict[str, Any]:
    """Fail closed unless corpus files and the frozen template match the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    total_records = 0
    for expected in manifest["files"]:
        path = ROOT / expected["path"]
        if path.parent.resolve() != records_directory.resolve():
            raise ValueError(f"identity file is outside --records: {path}")
        actual_records = 0
        with path.open("rb") as handle:
            for line in handle:
                actual_records += bool(line.strip())
        actual = {
            "path": expected["path"],
            "bytes": path.stat().st_size,
            "records": actual_records,
            "sha256": sha256(path),
        }
        checks.append(actual)
        for field in ("bytes", "records", "sha256"):
            if actual[field] != expected[field]:
                raise ValueError(
                    f"input identity mismatch for {expected['path']} {field}: "
                    f"expected={expected[field]} actual={actual[field]}"
                )
        total_records += actual_records
    if total_records != manifest["record_count"]:
        raise ValueError(
            f"input identity record count mismatch: {total_records} != {manifest['record_count']}"
        )
    template_path = model_directory / "chat_template.jinja"
    template_sha = sha256(template_path)
    if template_sha != manifest["chat_template_sha256"]:
        raise ValueError(
            f"chat template mismatch: {template_sha} != {manifest['chat_template_sha256']}"
        )
    mix_path = ROOT / "data/mix.yaml"
    mix_sha = sha256(mix_path)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "record_count": total_records,
        "files": checks,
        "mix_yaml": {
            "actual_sha256": mix_sha,
            "manifest_sha256": manifest["mix_yaml_sha256"],
            "match": mix_sha == manifest["mix_yaml_sha256"],
            "gate": "warning_only_not_a_renderer_input",
        },
        "chat_template_sha256": template_sha,
        "pass": True,
    }


def init_worker(model: str, need_renderer: bool) -> None:
    global _WORKER_TOKENIZER, _WORKER_RENDERER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True)
    _WORKER_RENDERER = load_renderer() if need_renderer else None


def render_text(tokenizer: Any, record: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        record["messages"],
        tools=record.get("tools") or None,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        preserve_thinking=True,
    )


def compact_render(tokenizer: Any, renderer: Any, record: dict[str, Any]) -> tuple[list[int], list[int]]:
    text = render_text(tokenizer, record)
    tagged_messages, marker_pairs = renderer.add_content_markers(record["messages"])
    marked = tokenizer.apply_chat_template(
        tagged_messages,
        tools=record.get("tools") or None,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        preserve_thinking=True,
    )
    # A tool-call-only assistant turn has no text field to mark.  The frozen
    # renderer still supervises its <|tool_call> span, so an empty marker set
    # is valid and must not be passed to re.compile("").
    if marker_pairs:
        stripped, content_ranges = renderer.remove_markers_and_ranges(marked, marker_pairs)
    else:
        stripped, content_ranges = marked, []
    if stripped != text:
        raise AssertionError("marker removal did not reproduce native rendered text")
    ranges = renderer.ranges_for_rendered_text(text, content_ranges)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    labels: list[int] = []
    range_index = 0
    for token_id, (start, end) in zip(input_ids, offsets, strict=True):
        while range_index < len(ranges) and ranges[range_index][1] <= start:
            range_index += 1
        supervised = (
            end > start
            and range_index < len(ranges)
            and start < ranges[range_index][1]
            and end > ranges[range_index][0]
        )
        labels.append(token_id if supervised else -100)
    return input_ids, labels


def compact_render_selected(
    tokenizer: Any,
    renderer: Any,
    record: dict[str, Any],
    target_message_indices: set[int],
) -> tuple[list[int], list[int]]:
    """Render a window while supervising only selected assistant messages.

    The frozen M0 mask marks every assistant span.  Segmentation needs the
    same span semantics but must zero labels belonging to context rounds.
    Content/reasoning spans are located with removable markers; tool calls are
    mapped to their owning assistant message in deterministic render order.
    """
    messages = record["messages"]
    text = render_text(tokenizer, record)
    tagged = copy.deepcopy(messages)
    marker_pairs: list[tuple[str, str]] = []
    marker_index = 0
    for message_index, message in enumerate(tagged):
        if message_index not in target_message_indices or message.get("role") != "assistant":
            continue
        for key in ("reasoning", "reasoning_content", "content"):
            value = message.get(key)
            if not isinstance(value, str) or not value:
                continue
            start = renderer.marker("SEGMENT", marker_index, "START")
            end = renderer.marker("SEGMENT", marker_index, "END")
            message[key] = start + value + end
            marker_pairs.append((start, end))
            marker_index += 1

    marked_record = {**record, "messages": tagged}
    marked = render_text(tokenizer, marked_record)
    if marker_pairs:
        stripped, content_ranges = renderer.remove_markers_and_ranges(marked, marker_pairs)
    else:
        stripped, content_ranges = marked, []
    if stripped != text:
        raise AssertionError("selected marker removal did not reproduce native rendered text")

    ranges = content_ranges[:]
    # Expand marked reasoning bodies to the complete frozen thought-channel
    # span, matching M0's mask convention including the channel delimiters.
    thought_matches = list(renderer.THOUGHT_RE.finditer(text))
    for content_start, content_end in content_ranges:
        for match in thought_matches:
            if match.start() <= content_start and content_end <= match.end():
                ranges.append((match.start(), match.end()))
                break

    # Literal <|tool_call|> text can appear inside assistant content, so regex
    # match order alone cannot identify structured calls.  Mark one field in
    # each target structured call, remove the markers, then expand that field
    # range to its enclosing rendered tool-call span.
    tagged_tools = copy.deepcopy(messages)
    tool_marker_pairs: list[tuple[str, str]] = []
    tool_marker_index = 0
    for message_index, message in enumerate(tagged_tools):
        if message_index not in target_message_indices or message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            marked = False
            for key in ("name", "arguments"):
                value = function.get(key)
                if isinstance(value, str):
                    start = renderer.marker("SEGMENT_TOOL", tool_marker_index, "START")
                    end = renderer.marker("SEGMENT_TOOL", tool_marker_index, "END")
                    function[key] = start + value + end
                    tool_marker_pairs.append((start, end))
                    tool_marker_index += 1
                    marked = True
                    break
            if not marked:
                raise AssertionError("structured target tool call has no markable string field")
    if tool_marker_pairs:
        tool_marked_record = {**record, "messages": tagged_tools}
        tool_marked = render_text(tokenizer, tool_marked_record)
        tool_stripped, tool_field_ranges = renderer.remove_markers_and_ranges(
            tool_marked, tool_marker_pairs
        )
        if tool_stripped != text:
            raise AssertionError("tool marker removal did not reproduce native rendered text")
        tool_matches = list(renderer.TOOL_CALL_RE.finditer(text))
        structured_ranges: set[tuple[int, int]] = set()
        for field_start, field_end in tool_field_ranges:
            enclosing = next(
                (
                    match
                    for match in tool_matches
                    if match.start() <= field_start and field_end <= match.end()
                ),
                None,
            )
            # The template repeats a call's marked name in its tool-response
            # header.  Only the occurrence enclosed by TOOL_CALL_RE is loss.
            if enclosing is not None:
                structured_ranges.add((enclosing.start(), enclosing.end()))
        if len(structured_ranges) != len(tool_marker_pairs):
            raise AssertionError(
                f"target structured tool spans {len(structured_ranges)} "
                f"!= source calls {len(tool_marker_pairs)}"
            )
        ranges.extend(structured_ranges)
    ranges = renderer.merge_ranges(ranges)

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    labels: list[int] = []
    range_index = 0
    for token_id, (start, end) in zip(input_ids, offsets, strict=True):
        while range_index < len(ranges) and ranges[range_index][1] <= start:
            range_index += 1
        supervised = (
            end > start
            and range_index < len(ranges)
            and start < ranges[range_index][1]
            and end > ranges[range_index][0]
        )
        labels.append(token_id if supervised else -100)
    return input_ids, labels


def message_rounds(messages: list[dict[str, Any]]) -> tuple[list[int], list[list[int]]]:
    """Return the fixed preamble and user-started message rounds."""
    user_indices = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    if not user_indices:
        return [], [list(range(len(messages)))]
    preamble = list(range(user_indices[0]))
    rounds: list[list[int]] = []
    for position, start in enumerate(user_indices):
        stop = user_indices[position + 1] if position + 1 < len(user_indices) else len(messages)
        rounds.append(list(range(start, stop)))
    return preamble, rounds


def suppress_full_render_omissions(
    messages: list[dict[str, Any]], original_indices: list[int], original_last_user: int
) -> list[dict[str, Any]]:
    """Keep reasoning visibility identical to rendering the unsliced trace."""
    working = copy.deepcopy(messages)
    for message, original_index in zip(working, original_indices, strict=True):
        if (
            message.get("role") == "assistant"
            and original_index <= original_last_user
            and not message.get("tool_calls")
        ):
            message.pop("reasoning", None)
            message.pop("reasoning_content", None)
    return working


def target_supervision_slices(
    input_ids: list[int], labels: list[int], max_length: int
) -> list[tuple[list[int], list[int], int, int]]:
    """Cover an oversized target using overlapping inputs and disjoint labels."""
    supervised_positions = [index for index, label in enumerate(labels) if label != -100]
    if not supervised_positions:
        return []
    context_tokens = max_length // 4
    parts: list[tuple[list[int], list[int], int, int]] = []
    cursor = 0
    while cursor < len(supervised_positions):
        first_target = supervised_positions[cursor]
        start = max(0, first_target - context_tokens)
        stop = min(len(input_ids), start + max_length)
        next_cursor = cursor
        assigned: set[int] = set()
        while (
            next_cursor < len(supervised_positions)
            and supervised_positions[next_cursor] < stop
        ):
            assigned.add(supervised_positions[next_cursor])
            next_cursor += 1
        part_labels = [
            label if position in assigned else -100
            for position, label in enumerate(labels[start:stop], start)
        ]
        if not assigned:
            raise AssertionError("oversized target slice made no supervision progress")
        parts.append((input_ids[start:stop], part_labels, start, stop))
        cursor = next_cursor
    if sum(sum(label != -100 for label in part[1]) for part in parts) != len(
        supervised_positions
    ):
        raise AssertionError("oversized target supervision was lost or duplicated")
    return parts


def segment_record(
    tokenizer: Any, renderer: Any, record: dict[str, Any], max_length: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split one source trace into deterministic, target-disjoint round windows."""
    full_input_ids, full_labels = compact_render(tokenizer, renderer, record)
    original_supervised = sum(label != -100 for label in full_labels)
    messages = record["messages"]
    preamble, rounds = message_rounds(messages)
    original_last_user = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=-1,
    )
    windows: list[dict[str, Any]] = []
    round_audit: list[dict[str, Any]] = []

    def render_candidate(round_start: int, target_round: int) -> tuple[list[int], list[int], list[int]]:
        original_indices = preamble[:]
        for round_index in range(round_start, target_round + 1):
            original_indices.extend(rounds[round_index])
        window_messages = suppress_full_render_omissions(
            [messages[index] for index in original_indices], original_indices, original_last_user
        )
        original_to_local = {original: local for local, original in enumerate(original_indices)}
        target_local = {
            original_to_local[index]
            for index in rounds[target_round]
            if index in original_to_local and messages[index].get("role") == "assistant"
        }
        window_record = {**record, "messages": window_messages}
        input_ids, labels = compact_render_selected(tokenizer, renderer, window_record, target_local)
        return input_ids, labels, original_indices

    for target_round, round_indices in enumerate(rounds):
        assistant_indices = [
            index for index in round_indices if messages[index].get("role") == "assistant"
        ]
        if not assistant_indices:
            round_audit.append(
                {"round_index": target_round, "status": "skipped", "reason": "no_assistant_message"}
            )
            continue

        target_ids, target_labels, target_original_indices = render_candidate(target_round, target_round)
        original_target_supervised = sum(label != -100 for label in target_labels)
        if not original_target_supervised:
            round_audit.append(
                {
                    "round_index": target_round,
                    "status": "skipped",
                    "reason": "no_supervised_tokens",
                    "target_only_tokens": len(target_ids),
                }
            )
            continue
        if len(target_ids) > max_length:
            full_target_tokens = len(target_ids)
            parts = target_supervision_slices(target_ids, target_labels, max_length)
            part_uids: list[str] = []
            retained_in_parts = 0
            for part_index, (part_ids, part_labels, crop_start, crop_stop) in enumerate(parts):
                supervised = sum(label != -100 for label in part_labels)
                retained_in_parts += supervised
                window_uid = (
                    f"{record['uid']}::segment:{target_round:04d}:part:{part_index:04d}"
                )
                part_uids.append(window_uid)
                windows.append(
                    {
                        "uid": window_uid,
                        "source_uid": record["uid"],
                        "dataset": record["dataset"],
                        "family": str(record.get("_family") or "unknown"),
                        "task_type": str(record.get("_task_type") or "unknown"),
                        "is_subagent": bool(record.get("is_subagent", False)),
                        "input_ids": part_ids,
                        "labels": part_labels,
                        "source_original_tokens": len(full_input_ids),
                        "source_original_supervised_tokens": original_supervised,
                        "kept_tokens": len(part_ids),
                        "target_supervised_tokens": supervised,
                        "target_round_index": target_round,
                        "target_part_index": part_index,
                        "target_part_count": len(parts),
                        "context_start_round": target_round,
                        "context_rounds": 0,
                        "target_crop_mode": "target_supervision_sliding_slices",
                        "window_original_message_start": min(target_original_indices),
                        "window_original_message_end": max(target_original_indices) + 1,
                        "segmentation_mode": "maximal_complete_round_suffix_target_only_loss",
                    }
                )
            if retained_in_parts != original_target_supervised:
                raise AssertionError(
                    f"target supervision lost for {record['uid']} round {target_round}: "
                    f"{original_target_supervised} -> {retained_in_parts}"
                )
            round_audit.append({
                "round_index": target_round,
                "status": "written",
                "window_uids": part_uids,
                "windows_written": len(parts),
                "target_only_tokens_before_crop": full_target_tokens,
                "target_supervised_tokens_before_crop": original_target_supervised,
                "target_supervised_tokens_after_crop": retained_in_parts,
                "target_crop_mode": "target_supervision_sliding_slices",
                "input_overlap_tokens": max_length // 4,
            })
            continue
        else:
            target_supervised = original_target_supervised
            crop_mode = "none"
            crop_audit = {}

            # Find the earliest complete context-round suffix that still fits.
            low = 0
            high = target_round
            best_start = target_round
            best = (target_ids, target_labels, preamble + rounds[target_round])
            while low <= high:
                midpoint = (low + high) // 2
                candidate = render_candidate(midpoint, target_round)
                if len(candidate[0]) <= max_length:
                    best_start = midpoint
                    best = candidate
                    high = midpoint - 1
                else:
                    low = midpoint + 1
            input_ids, labels, original_indices = best
            supervised = sum(label != -100 for label in labels)
            if supervised != target_supervised:
                raise AssertionError(
                    f"target supervision changed with context for {record['uid']} round {target_round}: "
                    f"{target_supervised} -> {supervised}"
                )
        supervised = sum(label != -100 for label in labels)
        window_uid = f"{record['uid']}::segment:{target_round:04d}"
        windows.append(
            {
                "uid": window_uid,
                "source_uid": record["uid"],
                "dataset": record["dataset"],
                "family": str(record.get("_family") or "unknown"),
                "task_type": str(record.get("_task_type") or "unknown"),
                "is_subagent": bool(record.get("is_subagent", False)),
                "input_ids": input_ids,
                "labels": labels,
                "source_original_tokens": len(full_input_ids),
                "source_original_supervised_tokens": original_supervised,
                "kept_tokens": len(input_ids),
                "target_supervised_tokens": supervised,
                "target_round_index": target_round,
                "target_part_index": 0,
                "target_part_count": 1,
                "context_start_round": best_start,
                "context_rounds": target_round - best_start,
                "target_crop_mode": crop_mode,
                "window_original_message_start": min(original_indices) if original_indices else 0,
                "window_original_message_end": max(original_indices) + 1 if original_indices else 0,
                "segmentation_mode": "maximal_complete_round_suffix_target_only_loss",
            }
        )
        round_audit.append(
            {
                "round_index": target_round,
                "status": "written",
                "window_uid": window_uid,
                "window_tokens": len(input_ids),
                "target_supervised_tokens": supervised,
                "context_start_round": best_start,
                "context_rounds": target_round - best_start,
                "target_crop_mode": crop_mode,
                **crop_audit,
            }
        )

    retained_supervised = sum(window["target_supervised_tokens"] for window in windows)
    audit = {
        "uid": record["uid"],
        "dataset": record["dataset"],
        "family": str(record.get("_family") or "unknown"),
        "task_type": str(record.get("_task_type") or "unknown"),
        "original_tokens": len(full_input_ids),
        "original_supervised_tokens": original_supervised,
        "rounds": len(rounds),
        "windows_written": len(windows),
        "retained_supervised_tokens": retained_supervised,
        "supervised_token_retention_rate": (
            retained_supervised / original_supervised if original_supervised else None
        ),
        "drop_reason": None if windows else "no_trainable_segment_window",
        "round_audit": round_audit,
    }
    return windows, audit


def trim_record(
    tokenizer: Any,
    renderer: Any,
    record: dict[str, Any],
    max_length: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    working = copy.deepcopy(record)
    input_ids, labels = compact_render(tokenizer, renderer, working)
    original_tokens = len(input_ids)
    deleted_messages = 0
    modes: list[str] = []

    while len(input_ids) > max_length:
        user_indices = [
            index for index, message in enumerate(working["messages"])
            if message.get("role") == "user"
        ]
        if len(user_indices) < 2:
            break
        cut_at = user_indices[-1]
        prefix = working["messages"][:cut_at]
        if not any(message.get("role") == "assistant" for message in prefix):
            break
        deleted_messages += len(working["messages"]) - len(prefix)
        working["messages"] = prefix
        input_ids, labels = compact_render(tokenizer, renderer, working)
        if "drop_tail_turns" not in modes:
            modes.append("drop_tail_turns")

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        modes.append("prefix_tokens")

    supervised_tokens = sum(label != -100 for label in labels)
    dropped = supervised_tokens == 0
    audit = {
        "uid": record["uid"],
        "original_tokens": original_tokens,
        "kept_tokens": len(input_ids),
        "supervised_tokens": supervised_tokens,
        "deleted_messages": deleted_messages,
        "trim_mode": "+".join(modes) if modes else "none",
        "drop_reason": "no_supervised_tokens_after_trim" if dropped else None,
    }
    if dropped:
        return None, audit
    return {
        "uid": record["uid"],
        "dataset": record["dataset"],
        "family": record.get("_family"),
        "task_type": record.get("_task_type"),
        "is_subagent": bool(record.get("is_subagent", False)),
        "input_ids": input_ids,
        "labels": labels,
        "original_tokens": original_tokens,
        "kept_tokens": len(input_ids),
        "deleted_messages": deleted_messages,
        "trim_mode": audit["trim_mode"],
    }, audit


def stat_one(tokenizer: Any, task: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
    source_file, line_number, record = task
    try:
        text = render_text(tokenizer, record)
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as error:
        return {
            "failure": {
                "source_file": source_file,
                "line_number": line_number,
                "uid": record.get("uid"),
                "error": f"{type(error).__name__}: {error}",
            }
        }
    return {
        "row": {
            "uid": record["uid"],
            "dataset": record["dataset"],
            "family": str(record.get("_family") or "unknown"),
            "task_type": str(record.get("_task_type") or "unknown"),
            "is_subagent": bool(record.get("is_subagent", False)),
            "token_count": token_count,
        }
    }


def stats_worker(task: tuple[str, int, dict[str, Any]]) -> dict[str, Any]:
    return stat_one(_WORKER_TOKENIZER, task)


def render_one_worker(task: tuple[str, int, dict[str, Any]], max_length: int) -> dict[str, Any]:
    source_file, line_number, record = task
    try:
        rendered, audit = trim_record(_WORKER_TOKENIZER, _WORKER_RENDERER, record, max_length)
    except Exception as error:
        return {
            "failure": {
                "source_file": source_file,
                "line_number": line_number,
                "uid": record.get("uid"),
                "error": f"{type(error).__name__}: {error}",
            }
        }
    return {"rendered": rendered, "audit": audit}


def render_worker(task_and_length: tuple[tuple[str, int, dict[str, Any]], int]) -> dict[str, Any]:
    task, max_length = task_and_length
    return render_one_worker(task, max_length)


def segment_one_worker(task: tuple[str, int, dict[str, Any]], max_length: int) -> dict[str, Any]:
    source_file, line_number, record = task
    try:
        windows, audit = segment_record(_WORKER_TOKENIZER, _WORKER_RENDERER, record, max_length)
    except Exception as error:
        return {
            "failure": {
                "source_file": source_file,
                "line_number": line_number,
                "uid": record.get("uid"),
                "error": f"{type(error).__name__}: {error}",
            }
        }
    return {"windows": windows, "audit": audit}


def segment_worker(task_and_length: tuple[tuple[str, int, dict[str, Any]], int]) -> dict[str, Any]:
    task, max_length = task_and_length
    return segment_one_worker(task, max_length)


def percentile(values: list[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def histogram(values: list[int]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        lower = 0
        placed = False
        for bound in HISTOGRAM_BOUNDS:
            if value <= bound:
                counts[f"{lower + 1}-{bound}"] += 1
                placed = True
                break
            lower = bound
        if not placed:
            counts[f">{HISTOGRAM_BOUNDS[-1]}"] += 1
    return dict(counts)


def length_summary(values: list[int]) -> dict[str, Any]:
    return {
        "records": len(values),
        "tokens": sum(values),
        "min": min(values) if values else 0,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0,
        "over_8192": sum(value > 8192 for value in values),
        "over_16384": sum(value > 16384 for value in values),
        "histogram": histogram(values),
    }


def run_stats(args: argparse.Namespace, tokenizer: Any) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lengths: list[int] = []
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    metadata_path = args.output_dir / "lengths.jsonl.gz"
    started = time.perf_counter()
    pool = None
    try:
        if args.workers > 1:
            pool = mp.get_context("spawn").Pool(
                args.workers,
                initializer=init_worker,
                initargs=(str(args.model), False),
            )
            results = pool.imap(stats_worker, iter_records(args.records), chunksize=args.worker_chunksize)
        else:
            results = (stat_one(tokenizer, task) for task in iter_records(args.records))
        with gzip.open(metadata_path, "wt", encoding="utf-8") as output:
            for index, result in enumerate(results, 1):
                if "failure" in result:
                    failures.append(result["failure"])
                    continue
                row = result["row"]
                token_count = row["token_count"]
                family = row["family"]
                task_type = row["task_type"]
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                lengths.append(token_count)
                groups[(family, task_type)].append(token_count)
                source_counts[row["dataset"]] += 1
                if index % args.progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(f"stats rows={index} rendered={len(lengths)} failures={len(failures)} elapsed={elapsed:.1f}s", file=sys.stderr)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    report = {
        "mode": "stats",
        "renderer": "scripts/m0/data/render_linec_samples.py",
        "template_sha256": hashlib.sha256((args.model / "chat_template.jinja").read_bytes()).hexdigest(),
        "records_seen": len(lengths) + len(failures),
        "records_rendered": len(lengths),
        "render_failures": failures,
        "overall": length_summary(lengths),
        "by_family_task_type": {
            f"{family}|{task_type}": length_summary(values)
            for (family, task_type), values in sorted(groups.items())
        },
        "by_dataset_records": dict(sorted(source_counts.items())),
        "elapsed_seconds": time.perf_counter() - started,
        "lengths_file": metadata_path.name,
    }
    report_path = args.output_dir / "stats.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures and report["records_rendered"] == args.expected_records else 1


def run_render(args: argparse.Namespace, tokenizer: Any, renderer: Any) -> int:
    if not args.max_length or args.max_length <= 0:
        raise ValueError("--max-length must be positive in render mode")
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("uid", pa.string()),
            ("dataset", pa.string()),
            ("family", pa.string()),
            ("task_type", pa.string()),
            ("is_subagent", pa.bool_()),
            ("input_ids", pa.list_(pa.int32())),
            ("labels", pa.list_(pa.int32())),
            ("original_tokens", pa.int32()),
            ("kept_tokens", pa.int32()),
            ("deleted_messages", pa.int32()),
            ("trim_mode", pa.string()),
        ]
    )
    batch: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    failures: list[dict[str, Any]] = []
    trim_counts: Counter[str] = Counter()
    written = 0
    seen = 0
    total_kept_tokens = 0
    total_supervised_tokens = 0
    started = time.perf_counter()
    audit_path = args.output_dir / "trim_audit.jsonl.gz"

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        path = args.output_dir / f"train-{len(shard_paths):05d}.parquet"
        pq.write_table(pa.Table.from_pylist(batch, schema=schema), path, compression="zstd")
        shard_paths.append(path)
        batch = []

    pool = None
    try:
        if args.workers > 1:
            pool = mp.get_context("spawn").Pool(
                args.workers,
                initializer=init_worker,
                initargs=(str(args.model), True),
            )
            tasks = ((task, args.max_length) for task in iter_records(args.records))
            results = pool.imap(render_worker, tasks, chunksize=args.worker_chunksize)
        else:
            def single_results() -> Iterator[dict[str, Any]]:
                global _WORKER_TOKENIZER, _WORKER_RENDERER
                _WORKER_TOKENIZER = tokenizer
                _WORKER_RENDERER = renderer
                for task in iter_records(args.records):
                    yield render_one_worker(task, args.max_length)
            results = single_results()
        with gzip.open(audit_path, "wt", encoding="utf-8") as audit_output:
            for result in results:
                seen += 1
                if "failure" in result:
                    failures.append(result["failure"])
                    continue
                rendered = result["rendered"]
                audit = result["audit"]
                audit_output.write(json.dumps(audit, ensure_ascii=False, separators=(",", ":")) + "\n")
                trim_counts[audit["trim_mode"]] += 1
                if rendered is not None:
                    total_kept_tokens += rendered["kept_tokens"]
                    total_supervised_tokens += audit["supervised_tokens"]
                    batch.append(rendered)
                    written += 1
                    if len(batch) >= args.shard_rows:
                        flush()
                if seen % args.progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(f"render rows={seen} written={written} failures={len(failures)} elapsed={elapsed:.1f}s", file=sys.stderr)
            flush()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    shards = []
    for path in shard_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        shards.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
    report = {
        "mode": "render",
        "renderer": "scripts/m0/data/render_linec_samples.py",
        "max_length": args.max_length,
        "records_seen": seen,
        "records_written": written,
        "records_dropped": seen - written - len(failures),
        "render_failures": failures,
        "trim_counts": dict(sorted(trim_counts.items())),
        "kept_tokens": total_kept_tokens,
        "supervised_tokens": total_supervised_tokens,
        "elapsed_seconds": time.perf_counter() - started,
        "audit_file": audit_path.name,
        "shards": shards,
    }
    report_path = args.output_dir / "render_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures and seen == args.expected_records else 1


def retention_summary(rows: list[dict[str, Any]], record_gate: float, token_gate: float) -> dict[str, Any]:
    pool_records = len(rows)
    retained_records = sum(row["windows_written"] > 0 for row in rows)
    original_supervised = sum(row["original_supervised_tokens"] for row in rows)
    retained_supervised = sum(row["retained_supervised_tokens"] for row in rows)
    record_rate = retained_records / pool_records if pool_records else 0.0
    token_rate = retained_supervised / original_supervised if original_supervised else 0.0
    return {
        "pool_records": pool_records,
        "retained_records": retained_records,
        "record_retention_rate": record_rate,
        "original_supervised_tokens": original_supervised,
        "retained_supervised_tokens": retained_supervised,
        "supervised_token_retention_rate": token_rate,
        "record_gate": record_gate,
        "supervised_token_gate": token_gate,
        "pass": record_rate >= record_gate and token_rate >= token_gate,
    }


def run_segment(args: argparse.Namespace, tokenizer: Any, renderer: Any) -> int:
    """Render branch-2 turn-level windows and evaluate the D-9 source-record gate."""
    if not args.max_length or args.max_length <= 0:
        raise ValueError("--max-length must be positive in segment mode")
    if args.identity_manifest is None:
        raise ValueError("--identity-manifest is required in segment mode")
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_identity = verify_input_identity(args.identity_manifest, args.records, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("uid", pa.string()),
            ("source_uid", pa.string()),
            ("dataset", pa.string()),
            ("family", pa.string()),
            ("task_type", pa.string()),
            ("is_subagent", pa.bool_()),
            ("input_ids", pa.list_(pa.int32())),
            ("labels", pa.list_(pa.int32())),
            ("source_original_tokens", pa.int32()),
            ("source_original_supervised_tokens", pa.int32()),
            ("kept_tokens", pa.int32()),
            ("target_supervised_tokens", pa.int32()),
            ("target_round_index", pa.int32()),
            ("target_part_index", pa.int32()),
            ("target_part_count", pa.int32()),
            ("context_start_round", pa.int32()),
            ("context_rounds", pa.int32()),
            ("target_crop_mode", pa.string()),
            ("window_original_message_start", pa.int32()),
            ("window_original_message_end", pa.int32()),
            ("segmentation_mode", pa.string()),
        ]
    )
    batch: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    failures: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    seen = 0
    windows_written = 0
    total_window_tokens = 0
    total_target_supervised = 0
    window_uids: set[str] = set()
    started = time.perf_counter()
    audit_path = args.output_dir / "segment_audit.jsonl.gz"

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        path = args.output_dir / f"train-{len(shard_paths):05d}.parquet"
        pq.write_table(pa.Table.from_pylist(batch, schema=schema), path, compression="zstd")
        shard_paths.append(path)
        batch = []

    pool = None
    try:
        source_tasks = iter_records(args.records)
        if args.max_records is not None:
            source_tasks = islice(source_tasks, args.max_records)
        if args.workers > 1:
            pool = mp.get_context("spawn").Pool(
                args.workers,
                initializer=init_worker,
                initargs=(str(args.model), True),
            )
            tasks = ((task, args.max_length) for task in source_tasks)
            results = pool.imap(segment_worker, tasks, chunksize=args.worker_chunksize)
        else:
            def single_results() -> Iterator[dict[str, Any]]:
                global _WORKER_TOKENIZER, _WORKER_RENDERER
                _WORKER_TOKENIZER = tokenizer
                _WORKER_RENDERER = renderer
                for task in source_tasks:
                    yield segment_one_worker(task, args.max_length)
            results = single_results()
        with gzip.open(audit_path, "wt", encoding="utf-8") as audit_output:
            for result in results:
                seen += 1
                if "failure" in result:
                    failures.append(result["failure"])
                    continue
                audit = result["audit"]
                windows = result["windows"]
                audits.append(audit)
                audit_output.write(json.dumps(audit, ensure_ascii=False, separators=(",", ":")) + "\n")
                for window in windows:
                    if window["uid"] in window_uids:
                        raise AssertionError(f"duplicate window uid: {window['uid']}")
                    window_uids.add(window["uid"])
                    if len(window["input_ids"]) > args.max_length:
                        raise AssertionError(f"window exceeds max length: {window['uid']}")
                    if not any(label != -100 for label in window["labels"]):
                        raise AssertionError(f"window has no supervision: {window['uid']}")
                    batch.append(window)
                    windows_written += 1
                    total_window_tokens += window["kept_tokens"]
                    total_target_supervised += window["target_supervised_tokens"]
                    if len(batch) >= args.shard_rows:
                        flush()
                if seen % args.progress_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"segment sources={seen} windows={windows_written} "
                        f"failures={len(failures)} elapsed={elapsed:.1f}s",
                        file=sys.stderr,
                    )
            flush()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    by_family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for audit in audits:
        by_family_rows[audit["family"]].append(audit)
        by_group_rows[(audit["family"], audit["task_type"])].append(audit)
    by_family = {
        family: retention_summary(rows, args.record_retention_gate, args.supervised_retention_gate)
        for family, rows in sorted(by_family_rows.items())
    }
    by_family_task_type = {
        f"{family}|{task_type}": retention_summary(
            rows, args.record_retention_gate, args.supervised_retention_gate
        )
        for (family, task_type), rows in sorted(by_group_rows.items())
    }
    d9_pass = bool(by_family) and all(summary["pass"] for summary in by_family.values())
    shards = [
        {"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in shard_paths
    ]
    source_retained = sum(audit["windows_written"] > 0 for audit in audits)
    report = {
        "schema_version": 2,
        "mode": "segment",
        "branch": "alpha_fail_branch_2",
        "renderer": "scripts/m0/data/render_linec_samples.py",
        "segmentation_contract": {
            "unit": "user-started complete message round",
            "context": "maximal suffix of complete preceding rounds that fits with target",
            "loss": "assistant spans in target round only; all context labels are -100",
            "overlap": "context input may repeat; each target token is supervised exactly once",
            "deduplication": "window uid = source uid + target round index + optional part index",
            "oversized_target_round": "1/4-window input overlap; disjoint target labels cover all supervised tokens",
            "reasoning_visibility": "suppress reasoning omitted by the full frozen-template render",
        },
        "template_sha256": hashlib.sha256((args.model / "chat_template.jinja").read_bytes()).hexdigest(),
        "input_identity": input_identity,
        "max_length": args.max_length,
        "source_records_seen": seen,
        "source_records_rendered": len(audits),
        "source_records_retained": source_retained,
        "source_records_dropped": len(audits) - source_retained,
        "windows_written": windows_written,
        "render_failures": failures,
        "total_window_input_tokens": total_window_tokens,
        "total_target_supervised_tokens": total_target_supervised,
        "total_original_supervised_tokens": sum(audit["original_supervised_tokens"] for audit in audits),
        "d9": {
            "gate_scope": "per source family; family_task_type rows are diagnostic",
            "record_retention_gate": args.record_retention_gate,
            "supervised_token_retention_gate": args.supervised_retention_gate,
            "by_family": by_family,
            "by_family_task_type": by_family_task_type,
            "pass": d9_pass,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "audit_file": audit_path.name,
        "shards": shards,
    }
    report_path = args.output_dir / "render_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    mechanically_complete = not failures and seen == args.expected_records
    return 0 if mechanically_complete and d9_pass else (2 if mechanically_complete else 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stats", "render", "segment"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--shard-rows", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=8)
    parser.add_argument("--expected-records", type=int, default=154097)
    parser.add_argument("--record-retention-gate", type=float, default=0.90)
    parser.add_argument("--supervised-retention-gate", type=float, default=0.70)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    renderer = load_renderer()
    if args.mode == "stats":
        return run_stats(args, tokenizer)
    if args.mode == "segment":
        return run_segment(args, tokenizer, renderer)
    return run_render(args, tokenizer, renderer)


if __name__ == "__main__":
    raise SystemExit(main())
