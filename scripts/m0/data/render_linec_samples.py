#!/usr/bin/env python3
"""Render deterministic Line C Gate 7 samples with the native Gemma 4 template.

This is deliberately a sample-level M0 check.  The M1 trainer will render the
full selected mix using the same message representation and template options.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MARKER_PREFIX = "__EDGEFORGE_MASK_"
MARKER_SUFFIX = "__"
TOOL_CALL_RE = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL)
THOUGHT_RE = re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL)


def read_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def assistant_metrics(record: dict[str, Any]) -> tuple[int, int, int]:
    messages = record["messages"]
    assistants = [m for m in messages if m.get("role") == "assistant"]
    tool_calls = sum(len(m.get("tool_calls", [])) for m in assistants)
    reasoning = sum(bool(m.get("reasoning") or m.get("reasoning_content")) for m in assistants)
    return len(assistants), tool_calls, reasoning


def template_thinking_safe(record: dict[str, Any]) -> bool:
    """Whether every source reasoning field is renderable by the frozen template."""
    messages = record["messages"]
    last_user = max((index for index, message in enumerate(messages) if message.get("role") == "user"), default=-1)
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not (message.get("reasoning") or message.get("reasoning_content")):
            continue
        if index <= last_user and not message.get("tool_calls"):
            return False
    return True


def choose_samples(records: list[dict[str, Any]], count: int, seed: int, max_chars: int) -> list[dict[str, Any]]:
    """Select a deterministic source-stratified M0 review sample.

    The stored review artifact includes one JSON row per token, so ultra-long
    trajectories are unsuitable for a human boundary review.  They remain in
    the training mix and will be rendered in M1; this only bounds the M0 sample.
    """
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        assistants, tool_calls, _ = assistant_metrics(record)
        if (
            len(json.dumps(record["messages"], ensure_ascii=False)) <= max_chars
            and assistants > 1
            and tool_calls > 0
            and template_thinking_safe(record)
        ):
            by_source[record["dataset"]].append(record)

    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()

    # One record per source where possible gives the small check useful coverage.
    for dataset in sorted(by_source):
        pool = by_source[dataset][:]
        rng.shuffle(pool)
        pool.sort(
            key=lambda r: (
                assistant_metrics(r)[1] > 0,
                assistant_metrics(r)[0] > 1,
                assistant_metrics(r)[2] > 0,
                assistant_metrics(r),
            ),
            reverse=True,
        )
        candidate = pool[0]
        chosen.append(candidate)
        used.add(candidate["uid"])
        if len(chosen) == count:
            return chosen

    # Cycle sources to avoid a populous source consuming the compact sample.
    ranked: dict[str, list[dict[str, Any]]] = {}
    for dataset, pool in by_source.items():
        shuffled = [r for r in pool if r["uid"] not in used]
        rng.shuffle(shuffled)
        shuffled.sort(
            key=lambda r: (
                assistant_metrics(r)[1] > 0,
                assistant_metrics(r)[0] > 1,
                assistant_metrics(r)[2] > 0,
                assistant_metrics(r),
            ),
            reverse=True,
        )
        ranked[dataset] = shuffled
    while len(chosen) < count and any(ranked.values()):
        for dataset in sorted(ranked):
            if ranked[dataset] and len(chosen) < count:
                chosen.append(ranked[dataset].pop(0))
    return chosen[:count]


def marker(label: str, index: int, side: str) -> str:
    return f"{MARKER_PREFIX}{label}_{index}_{side}{MARKER_SUFFIX}"


def add_content_markers(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Tag assistant text fields without changing the post-removal rendered text."""
    tagged = copy.deepcopy(messages)
    pairs: list[tuple[str, str]] = []
    index = 0
    for message in tagged:
        if message.get("role") != "assistant":
            continue
        for key in ("reasoning", "reasoning_content", "content"):
            value = message.get(key)
            if not isinstance(value, str) or not value:
                continue
            start = marker("TEXT", index, "START")
            end = marker("TEXT", index, "END")
            message[key] = start + value + end
            pairs.append((start, end))
            index += 1
    return tagged, pairs


def remove_markers_and_ranges(marked: str, pairs: list[tuple[str, str]]) -> tuple[str, list[tuple[int, int]]]:
    boundaries: dict[str, str] = {}
    for start, end in pairs:
        boundaries[start] = "start"
        boundaries[end] = "end"

    output: list[str] = []
    ranges: list[tuple[int, int]] = []
    open_starts: list[int] = []
    pattern = re.compile("|".join(re.escape(item) for item in boundaries))
    pos = 0
    output_length = 0
    for match in pattern.finditer(marked):
        chunk = marked[pos : match.start()]
        output.append(chunk)
        output_length += len(chunk)
        found = match.group(0)
        kind = boundaries[found]
        current = output_length
        if kind == "start":
            open_starts.append(current)
        else:
            if not open_starts:
                raise ValueError("encountered an unmatched mask end marker")
            ranges.append((open_starts.pop(), current))
        pos = match.end()
    output.append(marked[pos:])
    if open_starts:
        raise ValueError("encountered an unmatched mask start marker")
    return "".join(output), ranges


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted((s, e) for s, e in ranges if e > s):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def ranges_for_rendered_text(text: str, content_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # Tool calls and thinking channels are model outputs.  Tool responses are
    # intentionally not covered by any range and therefore have loss mask 0.
    ranges = content_ranges[:]
    ranges.extend((match.start(), match.end()) for match in TOOL_CALL_RE.finditer(text))
    ranges.extend((match.start(), match.end()) for match in THOUGHT_RE.finditer(text))
    return merge_ranges(ranges)


def token_rows(tokenizer: Any, text: str, ranges: list[tuple[int, int]]) -> tuple[list[dict[str, Any]], int]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    rows: list[dict[str, Any]] = []
    masked = 0
    for token_id, (start, end) in zip(input_ids, offsets):
        # A token is a target token whenever it overlaps an assistant span.
        mask = int(any(start < right and end > left for left, right in ranges))
        masked += mask
        rows.append(
            {
                "token": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "id": int(token_id),
                "mask": mask,
                "char_start": int(start),
                "char_end": int(end),
            }
        )
    return rows, masked


def roundtrip(tokenizer: Any, text: str) -> bool:
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return decoded == text


def thinking_present(messages: list[dict[str, Any]], rendered: str) -> tuple[int, int]:
    expected = [
        m.get("reasoning") or m.get("reasoning_content")
        for m in messages
        if m.get("role") == "assistant" and (m.get("reasoning") or m.get("reasoning_content"))
    ]
    present = sum(str(value) in rendered for value in expected)
    return present, len(expected)


def render_one(tokenizer: Any, record: dict[str, Any]) -> dict[str, Any]:
    messages = record["messages"]
    kwargs = {
        "tools": record.get("tools") or None,
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    tagged_messages, pairs = add_content_markers(messages)
    marked = tokenizer.apply_chat_template(tagged_messages, **kwargs)
    stripped, content_ranges = remove_markers_and_ranges(marked, pairs)
    if stripped != rendered:
        raise AssertionError("marker removal did not reproduce the native rendered text")

    ranges = ranges_for_rendered_text(rendered, content_ranges)
    rows, masked = token_rows(tokenizer, rendered, ranges)
    thought_present, thought_total = thinking_present(messages, rendered)
    assistants, tool_calls, reasoning = assistant_metrics(record)
    return {
        "uid": record["uid"],
        "dataset": record["dataset"],
        "source_session": record.get("source_session"),
        "message_count": len(messages),
        "assistant_messages": assistants,
        "tool_calls": tool_calls,
        "reasoning_messages": reasoning,
        "rendered": rendered,
        "mask_ranges": [{"char_start": s, "char_end": e} for s, e in ranges],
        "tokens": rows,
        "token_count": len(rows),
        "masked_token_count": masked,
        "roundtrip_ok": roundtrip(tokenizer, rendered),
        "thinking_present": thought_present,
        "thinking_expected": thought_total,
        "masking_convention": "assistant text, <|channel>thought...<channel|>, and <|tool_call>...<tool_call|> are 1; system/user/tool_response and turn markers are 0 unless a token overlaps a target span.",
    }


def write_reports(samples: list[dict[str, Any]], failures: list[dict[str, str]], output: Path, template_sha: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for stale_sample in output.glob("*.json"):
        stale_sample.unlink()
    for index, sample in enumerate(samples, 1):
        (output / f"{index:02d}_{sample['uid']}.json").write_text(
            json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    report = ROOT / "data/pipeline/gate7_render_validation.md"
    total_tokens = sum(sample["token_count"] for sample in samples)
    masked_tokens = sum(sample["masked_token_count"] for sample in samples)
    roundtrip_pass = sum(sample["roundtrip_ok"] for sample in samples)
    thinking_pass = sum(sample["thinking_present"] == sample["thinking_expected"] for sample in samples)
    tool_samples = sum(sample["tool_calls"] > 0 for sample in samples)
    multi_turn = sum(sample["assistant_messages"] > 1 for sample in samples)
    datasets = Counter(sample["dataset"] for sample in samples)
    lines = [
        "# Gate 7 — Gemma 4 native rendering validation",
        "",
        "- Scope: deterministic M0 sample only; full mix rendering is deferred to M1.",
        "- Template: `models/gemma-4-E4B-it/chat_template.jinja`",
        f"- Template SHA-256: `{template_sha}`",
        "- Rendering options: `enable_thinking=true`, `preserve_thinking=true`, `add_generation_prompt=false`; tool schemas are passed through `tools`.",
        "- Selection: source-stratified multi-assistant tool trajectories whose source reasoning fields satisfy the frozen template's `thinking_gate`; each is bounded to 32,000 source-message characters so per-token review remains human-readable.",
        f"- Samples requested/rendered: 20/{len(samples)}; rendering failures: {len(failures)}.",
        f"- Tokenizer byte-exact round trips: {roundtrip_pass}/{len(samples)}.",
        f"- Thinking integrity (all source reasoning fields present): {thinking_pass}/{len(samples)}.",
        f"- Tool-call samples: {tool_samples}/{len(samples)}; multi-assistant-turn samples: {multi_turn}/{len(samples)}.",
        f"- Tokens / masked target tokens: {total_tokens} / {masked_tokens}.",
        "",
        "## Per-source coverage",
        "",
        "| Dataset | Samples |",
        "|---|---:|",
    ]
    lines.extend(f"| `{dataset}` | {count} |" for dataset, count in sorted(datasets.items()))
    lines.extend([
        "",
        "## Loss-mask convention",
        "",
        "Mask 1 is assigned to assistant content, `<|channel>thought…<channel|>`, and `<|tool_call>…<tool_call|>` spans. System/tool declarations, user content, tool responses, and turn markers are excluded. A token that straddles a target boundary is conservatively assigned mask 1.",
        "",
        "## Failures",
        "",
    ])
    if failures:
        lines.append("| UID | Error |")
        lines.append("|---|---|")
        lines.extend(f"| `{item['uid']}` | {item['error']} |" for item in failures)
    else:
        lines.append("None.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    review = ROOT / "data/pipeline/gate7_mask_review.md"
    review_lines = [
        "# Gate 7 mask review worksheet",
        "",
        "Twenty deterministic rendered samples are stored under the ignored `data/pipeline/gate7_render_samples/` directory. Each JSON file prints `token`, `id`, `mask`, and character offsets for boundary inspection.",
        "",
        "This automated worksheet verifies all 20 renderings and identifies the four required boundary types. Human visual sign-off remains pending; this file does not claim that a human has reviewed the samples.",
        "",
        "| # | UID | Dataset | thought spans | tool-call spans | tool-response spans | round trip | thinking | Reviewer |",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for index, sample in enumerate(samples, 1):
        text = sample["rendered"]
        review_lines.append(
            "| {idx} | `{uid}` | `{dataset}` | {thought} | {calls} | {responses} | {roundtrip} | {thinking}/{expected} | pending |".format(
                idx=index,
                uid=sample["uid"],
                dataset=sample["dataset"],
                thought=len(list(THOUGHT_RE.finditer(text))),
                calls=len(list(TOOL_CALL_RE.finditer(text))),
                responses=text.count("<|tool_response>"),
                roundtrip="pass" if sample["roundtrip_ok"] else "fail",
                thinking=sample["thinking_present"],
                expected=sample["thinking_expected"],
            )
        )
    review.write_text("\n".join(review_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/pipeline/mix_records")
    parser.add_argument("--output", type=Path, default=ROOT / "data/pipeline/gate7_render_samples")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/gemma-4-E4B-it")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--max-chars", type=int, default=32000)
    args = parser.parse_args()

    records = read_records(args.input)
    selected = choose_samples(records, args.count, args.seed, args.max_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    template_path = args.tokenizer / "chat_template.jinja"
    template_sha = hashlib.sha256(template_path.read_bytes()).hexdigest()

    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in selected:
        try:
            samples.append(render_one(tokenizer, record))
        except Exception as exc:  # Keep the supplied deterministic sample visible.
            failures.append({"uid": record["uid"], "error": f"{type(exc).__name__}: {exc}"})

    write_reports(samples, failures, args.output, template_sha)
    print(json.dumps({"selected": len(selected), "rendered": len(samples), "failures": failures}, ensure_ascii=False))
    return 1 if failures or len(samples) != args.count else 0


if __name__ == "__main__":
    raise SystemExit(main())
