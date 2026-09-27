#!/usr/bin/env python3
"""Sample the RKNN debugfs utilization file on a monotonic-raw timeline."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("phase_file", type=Path)
    parser.add_argument("interval_seconds", type=float, nargs="?", default=0.1)
    args = parser.parse_args()
    if not 0.05 <= args.interval_seconds <= 1.0:
        parser.error("interval must be between 0.05 and 1.0 seconds")

    clock_id = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
    load_path = Path("/sys/kernel/debug/rknpu/load")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.clock_gettime(clock_id)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "board_wall_time",
                "monotonic_raw_ns",
                "phase",
                "npu_load_core0_pct",
                "npu_load_core1_pct",
                "npu_load_core2_pct",
            ]
        )
        while True:
            now_ns = time.clock_gettime_ns(clock_id)
            try:
                phase = args.phase_file.read_text(encoding="utf-8").strip()
            except OSError:
                phase = "NA"
            phase = phase.replace(",", ";")
            loads: list[str] = []
            try:
                loads = re.findall(r"([0-9]+)%", load_path.read_text())
            except OSError:
                pass
            loads = (loads + ["NA", "NA", "NA"])[:3]
            writer.writerow(
                [
                    dt.datetime.now().astimezone().isoformat(timespec="microseconds"),
                    now_ns,
                    phase,
                    *loads,
                ]
            )
            handle.flush()
            deadline += args.interval_seconds
            delay = deadline - time.clock_gettime(clock_id)
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.clock_gettime(clock_id)


if __name__ == "__main__":
    raise SystemExit(main())
