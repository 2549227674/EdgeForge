#!/usr/bin/env python3
"""Plot OPE4 10 Hz RKNPU load aligned to the five prompt windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


CARRIERS = (
    ("c02_a2", "C02 A2 primary"),
    ("official_a1", "Official A1 read-only"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_rows = read_csv(args.board_run / "npu_load_10hz.csv")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.8), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    linestyles = ("-", "--", ":")

    for axis, (carrier, title) in zip(axes, CARRIERS):
        prompts = read_csv(args.board_run / carrier / "prompt_summary.csv")
        windows = [
            (int(row["run_start_ns"]), int(row["run_return_ns"]), int(row["prompt_id"]))
            for row in prompts
        ]
        origin = windows[0][0]
        end = windows[-1][1]
        selected = [
            row
            for row in load_rows
            if origin <= int(row["monotonic_raw_ns"]) <= end
        ]
        x = [(int(row["monotonic_raw_ns"]) - origin) / 1e9 for row in selected]
        core_values: list[list[float]] = []
        for core, (color, linestyle) in enumerate(zip(colors, linestyles)):
            values = [float(row[f"npu_load_core{core}_pct"]) for row in selected]
            core_values.append(values)
            axis.plot(
                x,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=1.35 if core == 0 else 1.0,
                label=f"core{core}",
            )
        for start_ns, end_ns, prompt_id in windows:
            start = (start_ns - origin) / 1e9
            stop = (end_ns - origin) / 1e9
            if prompt_id % 2:
                axis.axvspan(start, stop, color="#7F7F7F", alpha=0.075, linewidth=0)
            axis.axvline(start, color="#666666", alpha=0.28, linewidth=0.6)
            axis.text(
                (start + stop) / 2,
                101.5,
                f"P{prompt_id}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
            )
        core0 = core_values[0]
        ordered = sorted(core0)
        p50 = ordered[len(ordered) // 2]
        mean = sum(core0) / len(core0)
        axis.text(
            0.99,
            0.08,
            f"core0 mean {mean:.2f}% · p50 {p50:.0f}%",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.5},
        )
        axis.set_title(title, loc="left", fontweight="normal")
        axis.set_ylabel("RKNPU load (%)")
        axis.set_xlabel("Seconds since first prompt run start")
        axis.set_ylim(-3, 106)
        axis.set_xlim(0, (end - origin) / 1e9)
        axis.set_yticks((0, 25, 50, 75, 100))
        axis.grid(axis="y", color="#B0B0B0", linewidth=0.6, alpha=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower left", frameon=False, ncol=3)

    figure.suptitle(
        "OPE4 10 Hz RKNPU load across the fixed five prompts — board-diagnostic-1core"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, metadata={"Title": "OPE4 RKNPU load curve"})
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
