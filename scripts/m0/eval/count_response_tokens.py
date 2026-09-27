#!/usr/bin/env python3
"""Stage 1 of EdgeForge's agent token split: emit a per-response token-count
sidecar from B0 trajectories, using a llama.cpp tokenizer over the SAME GGUF as
the frozen baseline endpoint.

The sidecar stores counts only, never response text, so the trajectory content
is not duplicated into a second artifact.  ``metrics.py`` (stage 2) reads the
sidecar offline, which keeps the frozen agent-metrics reproducible without a
running server.

Segments counted per agent response (schema branch B, confirmed 2026-08-05):
* ``thinking``          -> ``steps[].reasoning_content`` (present on 711/836)
* ``message``           -> ``steps[].message`` (the rendered Analysis/Plan prose)
* ``command_content``   -> concatenated ``keystrokes`` across ``tool_calls[]``

The original model JSON body is NOT persisted, so there is no ``prose`` vs
``command_payload`` byte-accurate split; ``message`` and ``command_content`` are
harbor-normalized views and are labelled as such.  Because the exact JSON
envelope is unavailable, the residual check in stage 2 is a stability check, not
a near-constant check.

Tokenizer modes:
* ``llama-tokenize``   -> subprocess per segment against the baseline GGUF.
* ``endpoint``         -> POST /tokenize on a running llama-server.
The GGUF path or endpoint must be the baseline B0 artifact; the mode and a
fingerprint are written into the sidecar for audit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen


AGENT_STEP_SOURCE = "agent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Root holding <trial>/agent/trajectory.json files.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Sidecar JSON path (counts only, no response text).")
    parser.add_argument("--tokenizer-mode", required=True,
                        choices=["llama-tokenize", "endpoint"])
    parser.add_argument("--llama-tokenize-bin", type=Path,
                        default=Path("third_party/llama.cpp/build/bin/llama-tokenize"))
    parser.add_argument("--gguf", type=Path,
                        default=Path("models/gguf/gemma4-e4b-it-Q4_K_M.gguf"),
                        help="Baseline B0 GGUF; must match eval_config model.path.")
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--schema-branch", default="B", choices=["A", "B", "C"])
    return parser.parse_args()


def trajectory_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*/agent/trajectory.json"))
    if not files:
        raise FileNotFoundError(f"no trajectory.json under {root}")
    return files


def make_llama_tokenize_counter(binary: Path, gguf: Path) -> Callable[[str], int]:
    if not binary.exists():
        raise FileNotFoundError(f"llama-tokenize not found: {binary}")
    if not gguf.exists():
        raise FileNotFoundError(f"GGUF not found: {gguf}")

    def count(text: str) -> int:
        if not text:
            return 0
        proc = subprocess.run(
            [str(binary), "-m", str(gguf), "-p", text, "--no-bos"],
            capture_output=True, text=True, check=True,
        )
        # llama-tokenize prints one "id -> 'piece'" line per token.
        return sum(1 for line in proc.stdout.splitlines() if "->" in line)

    return count


def make_endpoint_counter(base_url: str) -> Callable[[str], int]:
    url = base_url.rstrip("/") + "/tokenize"

    def count(text: str) -> int:
        if not text:
            return 0
        body = json.dumps({"content": text, "add_special": False}).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError(f"unexpected /tokenize response for {text[:40]!r}")
        return len(tokens)

    return count


def command_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    chunks: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, dict):
            keystrokes = arguments.get("keystrokes")
            if isinstance(keystrokes, str):
                chunks.append(keystrokes)
    return "".join(chunks)


def has_keystrokes(tool_calls: Any) -> bool:
    """Whether Harbor retained a keystrokes field, even when it is empty."""
    if not isinstance(tool_calls, list):
        return False
    return any(
        isinstance(call, dict)
        and isinstance(call.get("arguments"), dict)
        and isinstance(call["arguments"].get("keystrokes"), str)
        for call in tool_calls
    )


def tokenizer_fingerprint(mode: str, args: argparse.Namespace,
                          counter: Callable[[str], int]) -> dict[str, Any]:
    probe = counter("hello world")
    fingerprint: dict[str, Any] = {"mode": mode, "probe_hello_world_tokens": probe}
    if mode == "llama-tokenize":
        fingerprint["gguf"] = str(args.gguf)
        fingerprint["binary"] = str(args.llama_tokenize_bin)
    else:
        fingerprint["endpoint"] = args.endpoint
    return fingerprint


def build_sidecar(files: list[Path], counter: Callable[[str], int],
                  fingerprint: dict[str, Any], branch: str) -> dict[str, Any]:
    responses: list[dict[str, Any]] = []
    for path in files:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        trial = path.parts[-3]
        for step in trajectory.get("steps", []):
            if not isinstance(step, dict) or step.get("source") != AGENT_STEP_SOURCE:
                continue
            reasoning = step.get("reasoning_content")
            message = step.get("message")
            record = {
                "trial": trial,
                "step_id": step.get("step_id"),
                "has_reasoning": bool(isinstance(reasoning, str) and reasoning.strip()),
                "thinking_tokens": counter(reasoning) if isinstance(reasoning, str) else 0,
                "message_tokens": counter(message) if isinstance(message, str) else 0,
                "command_content_tokens": counter(command_text(step.get("tool_calls"))),
                "tool_calls": (len(step["tool_calls"])
                               if isinstance(step.get("tool_calls"), list) else 0),
                "has_keystrokes": has_keystrokes(step.get("tool_calls")),
                "completion_tokens": (step.get("metrics") or {}).get("completion_tokens"),
            }
            responses.append(record)
    return {
        "schema_version": "edgeforge-token-sidecar/v1",
        "schema_branch": branch,
        "tokenizer": fingerprint,
        "note": ("Counts only; no response text is stored. message and "
                 "command_content are harbor-normalized views, not the original "
                 "model JSON body (branch B)."),
        "agent_responses": len(responses),
        "responses": responses,
    }


def main() -> int:
    args = parse_args()
    try:
        files = trajectory_files(args.input)
        if args.tokenizer_mode == "llama-tokenize":
            counter = make_llama_tokenize_counter(args.llama_tokenize_bin, args.gguf)
        else:
            counter = make_endpoint_counter(args.endpoint)
        fingerprint = tokenizer_fingerprint(args.tokenizer_mode, args, counter)
        sidecar = build_sidecar(files, counter, fingerprint, args.schema_branch)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"responses": sidecar["agent_responses"],
                          "output": str(args.output),
                          "tokenizer": fingerprint}, ensure_ascii=False))
    except (FileNotFoundError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"count_response_tokens.py: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
