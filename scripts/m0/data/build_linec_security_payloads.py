#!/usr/bin/env python3
"""Project final-mix fields that are actually consumed by the trainer.

The security scan must not inspect UID, provenance, source hashes, or file
paths: they are metadata, not training tokens, and high-entropy IDs create
detector false positives.  Output line numbers intentionally mirror the source
mix JSONL files so a finding can be applied back without storing a UID here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def payload(record: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for message in record["messages"]:
        item = {key: message[key] for key in ("content", "reasoning_content", "tool_calls") if key in message}
        messages.append(item)
    return {"messages": messages, "tools": record.get("tools", [])}


def main() -> None:
    source_root = ROOT / "data/pipeline/mix_records"
    output_root = ROOT / "data/pipeline/gate3_scan_payloads"
    output_root.mkdir(parents=True, exist_ok=True)
    for stale in output_root.glob("*.jsonl"):
        stale.unlink()
    count = 0
    for source in sorted(source_root.glob("*.jsonl")):
        with source.open(encoding="utf-8") as reader, (output_root / source.name).open("w", encoding="utf-8") as writer:
            for line in reader:
                writer.write(json.dumps(payload(json.loads(line)), ensure_ascii=False) + "\n")
                count += 1
    print(json.dumps({"payload_records": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
