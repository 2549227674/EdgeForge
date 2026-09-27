#!/usr/bin/env python3
"""Normalize the seven trainable M0 line-C sources into a common JSONL IR."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


TRAINABLE_REPOS = (
    "Glint-Research/Fable-5-traces",
    "Crownelius/Complete-FABLE.5-traces-2M",
    "lambda/hermes-agent-reasoning-traces",
    "AletheiaResearch/GLM-5.2-Agent",
    "armand0e/qwen3.7-max-pi-traces",
    "WithinUsAI/claude_mythos_distilled_25k",
    "armand0e/claude-opus-4.8-pi-traces",
)

ROLE_MAP = {"human": "user", "gpt": "assistant", "assistant": "assistant", "user": "user", "system": "system", "tool": "tool"}
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.S | re.I)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S | re.I)
CHATML_RE = re.compile(r"<\|im_start\|>([a-zA-Z_]+)\n(.*?)(?=<\|im_end\|>|\Z)", re.S)


@dataclass
class ParseStats:
    input_units: int = 0
    output_records: int = 0
    failures: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = Counter()


def decode(value: Any) -> Any:
    """Decode JSON strings while preserving ordinary natural-language strings."""
    if not isinstance(value, str):
        return value
    current: Any = value
    for _ in range(2):
        if not isinstance(current, str):
            break
        stripped = current.strip()
        if not stripped or stripped[0] not in "[{\"":
            break
        try:
            current = json.loads(stripped)
        except json.JSONDecodeError:
            break
    return current


def text_and_parts(value: Any) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    """Return visible text, thought text, function calls, and tool-result text."""
    value = decode(value)
    if value is None:
        return "", "", [], []
    if isinstance(value, str):
        return value, "", [], []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return str(value), "", [], []
    text, reasoning, calls, results = [], [], [], []
    for part in value:
        if isinstance(part, str):
            text.append(part)
            continue
        if not isinstance(part, dict):
            text.append(str(part))
            continue
        part_type = str(part.get("type", "")).lower()
        if part_type in {"text", "input_text"}:
            text.append(str(part.get("text", part.get("content", ""))))
        elif part_type in {"thinking", "reasoning", "thought"}:
            reasoning.append(str(part.get("thinking", part.get("text", part.get("content", "")))))
        elif part_type in {"toolcall", "tool_call", "tool_use", "function"}:
            calls.append(normalize_call(part))
        elif part_type in {"toolresult", "tool_result", "function_result"}:
            results.append(str(part.get("content", part.get("text", ""))))
        elif "text" in part:
            text.append(str(part["text"]))
        elif "content" in part:
            text.append(str(part["content"]))
    return "\n".join(item for item in text if item), "\n".join(item for item in reasoning if item), calls, results


def mapping(value: Any) -> dict[str, Any]:
    value = decode(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass
    return {"_unparsed": str(value)}


def normalize_call(call: Any, fallback_id: str = "") -> dict[str, Any]:
    call = decode(call)
    if not isinstance(call, dict):
        return {"id": fallback_id, "type": "function", "function": {"name": "unknown", "arguments": mapping(call)}}
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = function.get("name", call.get("name", "unknown"))
    arguments = function.get("arguments", function.get("input", call.get("input", call.get("arguments", {}))))
    return {
        "id": str(call.get("id", call.get("tool_use_id", fallback_id))),
        "type": "function",
        "function": {"name": str(name), "arguments": mapping(arguments)},
    }


def normalize_schema(value: Any) -> dict[str, Any]:
    """Fill the fields the frozen Gemma Jinja template reads directly."""
    schema = decode(value)
    if not isinstance(schema, dict):
        schema = {}
    normalized = dict(schema)
    normalized["description"] = str(normalized.get("description") or "")
    normalized["type"] = str(normalized.get("type") or "string")
    properties = normalized.get("properties")
    normalized["properties"] = {
        str(key): normalize_schema(item) for key, item in properties.items()
    } if isinstance(properties, dict) else {}
    required = normalized.get("required")
    normalized["required"] = [str(item) for item in required] if isinstance(required, list) else []
    normalized["nullable"] = bool(normalized.get("nullable", False))
    normalized["enum"] = normalized.get("enum") if isinstance(normalized.get("enum"), list) else []
    normalized["items"] = normalize_schema(normalized["items"]) if isinstance(normalized.get("items"), dict) else {}
    return normalized


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert both OpenAI and flat tool schemas to the native template shape."""
    tools = decode(tools)
    if not isinstance(tools, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw in tools:
        tool = decode(raw)
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not name:
            continue
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": str(function.get("description") or ""),
                    "parameters": normalize_schema(function.get("parameters")),
                },
            }
        )
    return normalized


def extract_inline_markup(content: str, fallback_prefix: str) -> tuple[str, str, list[dict[str, Any]]]:
    thoughts = THINK_RE.findall(content)
    content = THINK_RE.sub("", content)
    calls = []
    for index, body in enumerate(TOOL_CALL_RE.findall(content)):
        decoded = decode(body)
        if isinstance(decoded, dict):
            calls.append(normalize_call(decoded, f"{fallback_prefix}-tool-{index}"))
    content = TOOL_CALL_RE.sub("", content)
    return content.strip(), "\n".join(thought.strip() for thought in thoughts if thought.strip()), calls


def messages_from_sequence(items: Any, uid_prefix: str) -> list[dict[str, Any]]:
    items = decode(items)
    if not isinstance(items, list):
        return []
    messages = []
    for index, raw in enumerate(items):
        item = decode(raw)
        if not isinstance(item, dict):
            continue
        role = ROLE_MAP.get(str(item.get("role", item.get("from", ""))).lower())
        if not role:
            continue
        content, reasoning, calls, results = text_and_parts(item.get("content", item.get("value", "")))
        if item.get("reasoning_content"):
            reasoning = "\n".join(part for part in [reasoning, str(item["reasoning_content"])] if part)
        if item.get("thinking"):
            reasoning = "\n".join(part for part in [reasoning, str(item["thinking"])] if part)
        if isinstance(content, str):
            content, markup_reasoning, markup_calls = extract_inline_markup(content, f"{uid_prefix}-{index}")
            reasoning = "\n".join(part for part in [reasoning, markup_reasoning] if part)
            calls.extend(markup_calls)
        for call_index, call in enumerate(item.get("tool_calls", []) or []):
            calls.append(normalize_call(call, f"{uid_prefix}-{index}-tool-{call_index}"))
        if role == "user" and results and not content:
            messages.append({"role": "tool", "name": str(item.get("name", "tool")), "content": "\n".join(results)})
            continue
        message: dict[str, Any] = {"role": role, "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        if role == "tool":
            message["name"] = str(item.get("name", "tool"))
            if item.get("tool_call_id"):
                message["tool_call_id"] = str(item["tool_call_id"])
        messages.append(message)
    # Several event exporters split one model turn into consecutive assistant
    # fragments.  Keeping them separate makes the canonical Gemma template's
    # thinking_gate treat later fragments as independent non-tool turns and
    # silently drop their reasoning.  Coalescing only adjacent assistant
    # fragments preserves order and all content/calls without inventing turns.
    coalesced: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] != "assistant" or not coalesced or coalesced[-1]["role"] != "assistant":
            coalesced.append(message)
            continue
        previous = coalesced[-1]
        for field in ("content", "reasoning_content"):
            value = str(message.get(field, ""))
            if not value:
                continue
            if previous.get(field):
                previous[field] = f"{previous[field]}\n{value}"
            else:
                previous[field] = value
        if message.get("tool_calls"):
            previous.setdefault("tool_calls", []).extend(message["tool_calls"])
    return coalesced


def events_to_messages(events: Iterable[dict[str, Any]], uid_prefix: str) -> list[dict[str, Any]]:
    sequence = []
    for event in sorted(events, key=lambda item: str(item.get("timestamp", ""))):
        if event.get("type") != "message":
            continue
        message = event.get("message")
        if isinstance(message, dict):
            sequence.append(message)
    return messages_from_sequence(sequence, uid_prefix)


def validate(messages: list[dict[str, Any]]) -> str | None:
    if not messages:
        return "no_messages"
    roles = {message.get("role") for message in messages}
    if not roles <= {"system", "user", "assistant", "tool"}:
        return "invalid_role"
    if "user" not in roles or "assistant" not in roles:
        return "missing_user_or_assistant"
    for message in messages:
        if message["role"] == "assistant" and not (
            message.get("content") or message.get("reasoning_content") or message.get("tool_calls")
        ):
            return "empty_assistant"
        for call in message.get("tool_calls", []):
            if not isinstance(call.get("function", {}).get("arguments"), dict):
                return "non_mapping_arguments"
    return None


def source_file_session(metadata: dict[str, Any], fallback: str) -> str:
    source_file = str(metadata.get("source_file", ""))
    match = re.match(r"\d+-([0-9a-f-]{36})-", source_file)
    return match.group(1) if match else str(metadata.get("session_id", fallback))


def teich_records(repo_id: str, path: Path, family: str) -> Iterable[dict[str, Any]]:
    table = pq.read_table(path)
    for row_index, row in enumerate(table.to_pylist()):
        metadata = decode(row.get("metadata"))
        metadata = metadata if isinstance(metadata, dict) else {}
        trace = decode(row.get("trace"))
        if isinstance(trace, str):
            events = [decode(line) for line in trace.splitlines() if line.strip()]
        elif isinstance(trace, list):
            events = [decode(event) for event in trace]
        else:
            events = []
        events = [event for event in events if isinstance(event, dict)]
        source_session = source_file_session(metadata, str(row.get("session_id", row_index)))
        yield {
            "uid": f"{repo_id.replace('/', '__')}:{source_session}:{row_index}",
            "dataset": repo_id.replace("/", "__"),
            "source_session": source_session,
            "source_family": family,
            "first_source_dataset": repo_id,
            "is_subagent": False,
            "harness": str(row.get("harness", "pi")),
            "tools": normalize_tools(row.get("tools")),
            "messages": events_to_messages(events, f"{repo_id}-{source_session}"),
            "provenance": {"file": path.as_posix(), "row_index": row_index, "source_file": metadata.get("source_file")},
        }


def hermes_records(path: Path, family: str) -> Iterable[dict[str, Any]]:
    table = pq.read_table(path)
    for row_index, row in enumerate(table.to_pylist()):
        source_session = str(row["id"])
        tools = decode(row.get("tools"))
        yield {
            "uid": f"lambda__hermes-agent-reasoning-traces:{source_session}:0",
            "dataset": "lambda__hermes-agent-reasoning-traces",
            "source_session": source_session,
            "source_family": family,
            "first_source_dataset": "lambda/hermes-agent-reasoning-traces",
            "is_subagent": False,
            "harness": "sharegpt",
            "tools": normalize_tools(tools),
            "messages": messages_from_sequence(row.get("conversations"), f"hermes-{source_session}"),
            "provenance": {
                "file": path.as_posix(),
                "row_index": row_index,
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "task": row.get("task"),
            },
        }


def mythos_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            source_session = str(row.get("id", row_index))
            yield {
                "uid": f"WithinUsAI__claude_mythos_distilled_25k:{source_session}:0",
                "dataset": "WithinUsAI__claude_mythos_distilled_25k",
                "source_session": source_session,
                "source_family": "mythos_synthetic",
                "first_source_dataset": "WithinUsAI/claude_mythos_distilled_25k",
                "is_subagent": False,
                "harness": "chat",
                "tools": [],
                "messages": messages_from_sequence(row.get("messages"), f"mythos-{source_session}"),
                "provenance": {"file": path.as_posix(), "row_index": row_index, "category": row.get("category"), "source": row.get("source")},
            }


def qwen_records(root: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(root.glob("*.jsonl")):
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        session = next((str(event["id"]) for event in events if event.get("type") == "session"), path.stem)
        yield {
            "uid": f"armand0e__qwen3.7-max-pi-traces:{session}:0",
            "dataset": "armand0e__qwen3.7-max-pi-traces",
            "source_session": session,
            "source_family": "qwen3.7-max",
            "first_source_dataset": "armand0e/qwen3.7-max-pi-traces",
            "is_subagent": False,
            "harness": "pi",
            "tools": [],
            "messages": events_to_messages(events, f"qwen-{session}"),
            "provenance": {"file": path.as_posix()},
        }


def crown_row_messages(row: dict[str, Any], uid_prefix: str) -> list[dict[str, Any]] | None:
    payload = decode(row["row_json"])
    if not isinstance(payload, dict):
        return None
    if "messages" in payload:
        return messages_from_sequence(payload["messages"], uid_prefix)
    if "conversations" in payload:
        return messages_from_sequence(payload["conversations"], uid_prefix)
    if payload.get("completion") is not None:
        prompt = payload.get("context", payload.get("prompt", payload.get("instruction", payload.get("input", ""))))
        completion = str(payload["completion"])
        completion, reasoning, calls = extract_inline_markup(completion, uid_prefix)
        assistant: dict[str, Any] = {"role": "assistant", "content": completion}
        if payload.get("cot"):
            reasoning = "\n".join(part for part in [str(payload["cot"]), reasoning] if part)
        if reasoning:
            assistant["reasoning_content"] = reasoning
        if calls:
            assistant["tool_calls"] = calls
        return [{"role": "user", "content": str(prompt)}, assistant]
    if payload.get("prompt") is not None and payload.get("response") is not None:
        return messages_from_sequence(
            [{"role": "user", "content": payload["prompt"]}, {"role": "assistant", "content": payload["response"]}], uid_prefix
        )
    if isinstance(payload.get("text"), str):
        turns = [{"role": role, "content": text} for role, text in CHATML_RE.findall(payload["text"])]
        return messages_from_sequence(turns, uid_prefix)
    return None


def crown_records(root: Path) -> Iterable[dict[str, Any]]:
    event_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for offset, row in enumerate(batch.to_pylist()):
                payload = decode(row["row_json"])
                row_number = int(row.get("first_source_row_index", offset))
                if isinstance(payload, dict) and payload.get("type") in {"user", "assistant"} and payload.get("sessionId"):
                    key = (str(row["first_source_dataset"]), str(row["first_source_config"]), str(payload["sessionId"]))
                    event_groups[key].append({"payload": payload, "row": row, "path": path.as_posix(), "row_number": row_number})
                    continue
                source_session = str(row["row_hash"])
                yield {
                    "uid": f"Crownelius__Complete-FABLE.5-traces-2M:{source_session}:0",
                    "dataset": "Crownelius__Complete-FABLE.5-traces-2M",
                    "source_session": source_session,
                    "source_family": "unresolved_crownelius_source",
                    "first_source_dataset": row["first_source_dataset"],
                    "is_subagent": "/subagents/" in str(row["first_source_split"]),
                    "harness": "crownelius_row_json",
                    "tools": [],
                    "messages": crown_row_messages(row, f"crown-{source_session}") or [],
                    "provenance": {
                        "file": path.as_posix(),
                        "row_index": row_number,
                        "row_hash": row["row_hash"],
                        "seen_count": row["seen_count"],
                        "first_source_config": row["first_source_config"],
                        "first_source_split": row["first_source_split"],
                    },
                }
    for (first_source, config, session), grouped in event_groups.items():
        first = grouped[0]
        lineage = hashlib.sha256(f"{first_source}\0{config}".encode()).hexdigest()[:12]
        source_session = f"{session}@{lineage}"
        yield {
            "uid": f"Crownelius__Complete-FABLE.5-traces-2M:{source_session}:0",
            "dataset": "Crownelius__Complete-FABLE.5-traces-2M",
            "source_session": source_session,
            "source_family": "unresolved_crownelius_source",
            "first_source_dataset": first_source,
            "is_subagent": "/subagents/" in config,
            "harness": "claude_code_events",
            "tools": [],
            "messages": events_to_messages(
                [
                    {
                        "type": "message",
                        "timestamp": item["payload"].get("timestamp"),
                        "message": item["payload"].get("message"),
                    }
                    for item in grouped
                ],
                f"crown-{session}",
            ),
            "provenance": {
                "file": first["path"],
                "row_index": first["row_number"],
                "row_hashes": [item["row"]["row_hash"] for item in grouped],
                "seen_count": max(int(item["row"]["seen_count"]) for item in grouped),
                "first_source_config": config,
                "upstream_session_id": session,
            },
        }


def generators() -> dict[str, Iterable[dict[str, Any]]]:
    return {
        "Glint-Research/Fable-5-traces": teich_records(
            "Glint-Research/Fable-5-traces", Path("data/archive_parquet/Glint-Research__Fable-5-traces/pi_agent/train/0000.parquet"), "fable-5"
        ),
        "AletheiaResearch/GLM-5.2-Agent": teich_records(
            "AletheiaResearch/GLM-5.2-Agent", Path("data/archive_parquet/AletheiaResearch__GLM-5.2-Agent/default/train/0000.parquet"), "glm-5.2"
        ),
        "armand0e/claude-opus-4.8-pi-traces": teich_records(
            "armand0e/claude-opus-4.8-pi-traces", Path("data/archive_parquet/armand0e__claude-opus-4.8-pi-traces/default/train/0000.parquet"), "opus-4.8"
        ),
        "lambda/hermes-agent-reasoning-traces": (
            record
            for path, family in [
                (Path("data/archive/lambda__hermes-agent-reasoning-traces/data/kimi/train.parquet"), "kimi-k2.5"),
                (Path("data/archive/lambda__hermes-agent-reasoning-traces/data/glm-5.1/train.parquet"), "glm-5.1"),
            ]
            for record in hermes_records(path, family)
        ),
        "armand0e/qwen3.7-max-pi-traces": qwen_records(Path("data/archive/armand0e__qwen3.7-max-pi-traces")),
        "WithinUsAI/claude_mythos_distilled_25k": mythos_records(
            Path("data/archive/WithinUsAI__claude_mythos_distilled_25k/claude_mythos_distilled_25k.jsonl")
        ),
        "Crownelius/Complete-FABLE.5-traces-2M": crown_records(Path("data/archive_parquet/Crownelius__Complete-FABLE.5-traces-2M")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/pipeline/ir"))
    parser.add_argument("--report", type=Path, default=Path("data/pipeline/gate4_parse_rates.md"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for repo_id, records in generators().items():
        stats = ParseStats()
        output = args.output_root / f"{repo_id.replace('/', '__')}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                stats.input_units += 1
                reason = validate(record["messages"])
                if reason:
                    stats.failures[reason] += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.output_records += 1
        summaries.append((repo_id, stats, output))
        print(f"{repo_id}: {stats.output_records}/{stats.input_units}")
    lines = ["# M0 §5 门④结构解析率", "", "| 数据集 | 输入单位 | 合格 IR | 失败率 | 失败原因 |", "|---|---:|---:|---:|---|"]
    for repo_id, stats, _ in summaries:
        rate = 1 - stats.output_records / stats.input_units if stats.input_units else 1
        reasons = ", ".join(f"{reason}={count}" for reason, count in stats.failures.most_common()) or "—"
        lines.append(f"| {repo_id} | {stats.input_units:,} | {stats.output_records:,} | {rate:.2%} | {reasons} |")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
