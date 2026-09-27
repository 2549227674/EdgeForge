#!/usr/bin/env python3
"""Shared, dependency-light audit utilities for the OP0 measurement base."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any, Iterable


GPU_QUERY_FIELDS = (
    "timestamp",
    "pstate",
    "clocks.current.sm",
    "clocks.current.memory",
    "temperature.gpu",
    "power.draw",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def command_result(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float], digits: int = 6) -> dict[str, float | int]:
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("cannot summarize an empty sequence")
    q1 = percentile(samples, 0.25)
    p50 = percentile(samples, 0.50)
    q3 = percentile(samples, 0.75)
    return {
        "count": len(samples),
        "min": round(min(samples), digits),
        "q1": round(q1, digits),
        "p50": round(p50, digits),
        "q3": round(q3, digits),
        "max": round(max(samples), digits),
        "mean": round(statistics.fmean(samples), digits),
        "iqr": round(q3 - q1, digits),
    }


def parse_optional_float(value: str) -> float | None:
    normalized = value.strip()
    if not normalized or "N/A" in normalized or "Not Supported" in normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


class NvidiaSmiMonitor:
    """Sample laptop clocks, temperature, power, and throttle flags at 1 Hz."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.phase = "unassigned"
        self._stop = threading.Event()
        self._sample_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def _sample_once(self) -> None:
        with self._sample_lock:
            command = [
                "nvidia-smi",
                "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
                "--format=csv,noheader,nounits",
            ]
            completed = subprocess.run(command, check=False, capture_output=True)
            if completed.returncode != 0:
                message = completed.stderr.decode("utf-8", errors="replace").strip()
                self.errors.append(message or f"nvidia-smi exited {completed.returncode}")
                return
            raw = completed.stdout.decode("utf-8", errors="replace").strip()
            if not raw:
                self.errors.append("nvidia-smi returned an empty sample")
                return
            row = next(csv.reader([raw]))
            if len(row) != len(GPU_QUERY_FIELDS):
                self.errors.append(f"unexpected nvidia-smi field count: {raw}")
                return
            record = dict(zip(GPU_QUERY_FIELDS, (value.strip() for value in row)))
            record["captured_at"] = now_iso()
            record["phase"] = self.phase
            self.samples.append(record)

    def sample_now(self) -> None:
        """Take a synchronous witness sample for sub-second measurement phases."""
        self._sample_once()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._sample_once()
            except BaseException as exc:  # telemetry must not kill a benchmark
                self.errors.append(f"{type(exc).__name__}: {exc}")
            remaining = self.interval_seconds - (time.monotonic() - started)
            self._stop.wait(max(0.0, remaining))

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU monitor has already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 5.0)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return self._summarize_samples(self.samples)

    def summary_for_phase(self, phase: str) -> dict[str, Any]:
        return self._summarize_samples(
            [sample for sample in self.samples if sample["phase"] == phase]
        )

    def _summarize_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        numeric_fields = {
            "sm_mhz": "clocks.current.sm",
            "memory_mhz": "clocks.current.memory",
            "temperature_c": "temperature.gpu",
            "power_w": "power.draw",
        }
        numeric: dict[str, Any] = {}
        for label, field in numeric_fields.items():
            values = [
                parsed
                for sample in samples
                if (parsed := parse_optional_float(sample[field])) is not None
            ]
            numeric[label] = summarize(values, digits=3) if values else None

        throttle_fields = GPU_QUERY_FIELDS[6:]
        throttle_flags = {
            field: any(sample[field].strip().lower() == "active" for sample in samples)
            for field in throttle_fields
        }
        return {
            "interval_seconds": self.interval_seconds,
            "sample_count": len(samples),
            "errors": self.errors,
            "numeric": numeric,
            "pstates": dict(sorted(Counter(s["pstate"] for s in samples).items())),
            "throttle_flags": throttle_flags,
            "thermal_or_hardware_throttle": any(
                throttle_flags[field]
                for field in (
                    "clocks_event_reasons.sw_thermal_slowdown",
                    "clocks_event_reasons.hw_thermal_slowdown",
                    "clocks_event_reasons.hw_power_brake_slowdown",
                )
            ),
            "samples": samples,
        }
