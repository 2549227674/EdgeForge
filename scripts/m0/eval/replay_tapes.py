#!/usr/bin/env python3
"""Replay frozen EdgeForge inference tapes and collect llama.cpp timings.

Each tape is replayed twice without interleaving another tape.  Only the second
pass is summarized, which gives every model the same cache-warm measurement
convention while retaining first-pass records for audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"connection error: {exc}") from exc


def timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("ok")]
    fields = {
        "prompt_ms": [record["timings"].get("prompt_ms") for record in successful],
        "predicted_per_second": [
            record["timings"].get("predicted_per_second") for record in successful
        ],
        "tpot_ms": [record.get("tpot_ms") for record in successful],
        "cache_n": [record["timings"].get("cache_n") for record in successful],
        "prompt_n": [record["timings"].get("prompt_n") for record in successful],
        "predicted_n": [record["timings"].get("predicted_n") for record in successful],
    }
    summary: dict[str, Any] = {"requests": len(records), "successful": len(successful)}
    for name, raw_values in fields.items():
        values = [float(value) for value in raw_values if isinstance(value, (int, float))]
        summary[name] = {
            "mean": sum(values) / len(values) if values else None,
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tapes-manifest", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    manifest_path = args.tapes_manifest
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sampling = manifest["sampling"]
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    all_records: list[dict[str, Any]] = []

    with records_path.open("w", encoding="utf-8") as handle:
        for tape_entry in manifest["tapes"]:
            tape_path = ROOT / tape_entry["path"]
            tape = json.loads(tape_path.read_text(encoding="utf-8"))
            if len(tape["requests"]) < 2:
                raise RuntimeError(f"tape has no repeated prefix: {tape_path}")
            for pass_number in (1, 2):
                for request_index, tape_request in enumerate(tape["requests"], start=1):
                    payload = {
                        "model": args.model,
                        "messages": tape_request["messages"],
                        "temperature": sampling["temperature"],
                        "top_p": sampling["top_p"],
                        "top_k": sampling["top_k"],
                        "min_p": sampling["min_p"],
                        "max_tokens": sampling["max_tokens"],
                        "stream": False,
                    }
                    record: dict[str, Any] = {
                        "tape_id": tape["tape_id"],
                        "task_id": tape["task_id"],
                        "tape_sha256": tape_entry["sha256"],
                        "pass": pass_number,
                        "measurement_convention": "cache_warm" if pass_number == 2 else "cache_prime",
                        "request_index": request_index,
                        "request_id": tape_request["request_id"],
                    }
                    try:
                        status, response = post_json(args.base_url, payload, args.timeout)
                        timings = response.get("timings")
                        if not isinstance(timings, dict):
                            raise RuntimeError("response omitted llama.cpp timings")
                        predicted_per_second = timings.get("predicted_per_second")
                        record.update(
                            {
                                "ok": True,
                                "http_status": status,
                                "timings": timings,
                                "usage": response.get("usage"),
                                "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                                "content_chars": len(
                                    response.get("choices", [{}])[0]
                                    .get("message", {})
                                    .get("content", "")
                                    or ""
                                ),
                                "tpot_ms": (
                                    1000.0 / predicted_per_second
                                    if isinstance(predicted_per_second, (int, float))
                                    and predicted_per_second > 0
                                    else None
                                ),
                            }
                        )
                    except Exception as exc:  # keep the full tape audit trail
                        record.update({"ok": False, "error": str(exc)})
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    all_records.append(record)
                    print(
                        f"tape={record['tape_id']} pass={pass_number} "
                        f"request={request_index}/{len(tape['requests'])} ok={record['ok']}",
                        file=sys.stderr,
                        flush=True,
                    )

    warm_records = [record for record in all_records if record["pass"] == 2]
    summary = {
        "schema_version": "1.0",
        "model": args.model,
        "base_url": args.base_url,
        "tapes_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "measurement_convention": "each tape replayed consecutively twice; metrics summarize pass 2 only",
        "all_passes": timing_summary(all_records),
        "cache_warm_pass_2": timing_summary(warm_records),
        "records": records_path.relative_to(ROOT).as_posix(),
        "completed_unix_s": time.time(),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
