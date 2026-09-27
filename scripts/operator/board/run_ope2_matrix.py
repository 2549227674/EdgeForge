#!/usr/bin/env python3
"""Execute the preregistered OPE2 matrix on the RK3588S board."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


RESULT_MARKER = re.compile(r"OPE2_RESULT_JSON=(\{.*\})")
IOVA_FAILURE = re.compile(
    r"failed to allocate IOVA|dma[ _-]?map.*(?:fail|error)|iommu.*(?:fail|error)|iova.*(?:fail|error)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--mode", choices=["preflight", "full"], default="full")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse an interrupted matrix directory and skip cells with durable terminals",
    )
    return parser.parse_args()


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat()


def uptime_s() -> float:
    return float(Path("/proc/uptime").read_text().split()[0])


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def dmesg_failure_tail() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["dmesg", "--color=never"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        lines = result.stdout.splitlines()[-300:]
        matches = [line for line in lines if IOVA_FAILURE.search(line)]
        return {"returncode": result.returncode, "tail": lines, "iova_failure_matches": matches}
    except OSError as exc:
        return {"error": str(exc), "tail": [], "iova_failure_matches": []}


def parse_result(text: str) -> dict[str, Any] | None:
    matches = RESULT_MARKER.findall(text)
    if not matches:
        return None
    return json.loads(matches[-1])


def invoke(
    bench: Path,
    phase_file: Path,
    raw_attempts: Path,
    cell_id: str,
    type_cli: str,
    mkn: list[int],
    core_cli: str,
    layout_cli: str,
    pattern: str,
    seed: int,
    warmup: int,
    min_reps: int,
    min_timed_ms: float,
    attempt_index: int,
) -> dict[str, Any]:
    phase_file.write_text(cell_id + "\n", encoding="utf-8")
    command = [
        "taskset",
        "-c",
        "7",
        str(bench),
        "--type",
        type_cli,
        "--m",
        str(mkn[0]),
        "--k",
        str(mkn[1]),
        "--n",
        str(mkn[2]),
        "--core-mask",
        core_cli,
        "--b-layout",
        layout_cli,
        "--warmup",
        str(warmup),
        "--min-reps",
        str(min_reps),
        "--min-timed-ms",
        str(min_timed_ms),
        "--pattern",
        pattern,
        "--seed",
        str(seed),
    ]
    start_wall = now_iso()
    start_uptime = uptime_s()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    end_uptime = uptime_s()
    combined = completed.stdout + "\n" + completed.stderr
    parsed = parse_result(combined)
    attempt = {
        "cell_id": cell_id,
        "attempt_index": attempt_index,
        "command": command,
        "controller": {
            "board_wall_start": start_wall,
            "board_wall_end": now_iso(),
            "board_uptime_start_s": start_uptime,
            "board_uptime_end_s": end_uptime,
        },
        "process_returncode": completed.returncode,
        "result": parsed,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-40:],
    }
    if completed.returncode != 0:
        attempt["failure_dmesg"] = dmesg_failure_tail()
    append_jsonl(raw_attempts, attempt)
    return attempt


def na_terminal(cell_id: str, scope: str, reason: str, axes: dict[str, Any], shape: dict[str, Any]) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "scope": scope,
        "terminal_status": "N/A",
        "terminal_reason": reason,
        "axes": axes,
        "shape": shape,
        "attempts": [],
    }


def runnable_axis(type_id: str, core_id: str, layout_id: str) -> tuple[bool, str | None]:
    if core_id in {"core0_1", "core0_1_2"}:
        return False, "N/A_CORE_MASK_RUNTIME_2_3_2_REJECTED"
    if type_id == "int4xint4_to_int16" and layout_id == "normal":
        return False, "N/A_INT4_NORMAL_RUNTIME_2_3_2_ABORTS_IN_SET_IO"
    return True, None


def terminal_from_attempts(
    cell_id: str,
    scope: str,
    axes: dict[str, Any],
    shape: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    final = attempts[-1]
    result = final.get("result")
    if final["process_returncode"] == 0 and result and result.get("status") == "PASS":
        ratio = float(result["run_only_us"]["iqr_over_p50"])
        status = "PASS" if ratio <= 0.30 else "DISCARDED_UNSTABLE_AFTER_RETRY"
        reason = None if status == "PASS" else "IQR_OVER_P50_GT_0.30_ON_FINAL_ATTEMPT"
    elif result and result.get("status") == "CORRECTNESS_FAIL":
        status = "FAIL_CORRECTNESS"
        reason = "ORACLE_MISMATCH"
    elif shape.get("scope") == "boundary_probe":
        status = "N/A"
        reason = "N/A_API_BOUNDARY_REJECTION"
    else:
        status = "FAIL_API"
        reason = result.get("error_stage") if result else "PROCESS_FAILED_WITHOUT_RESULT_MARKER"
    return {
        "cell_id": cell_id,
        "scope": scope,
        "terminal_status": status,
        "terminal_reason": reason,
        "axes": axes,
        "shape": shape,
        "attempts": attempts,
    }


def axis_maps(prereg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    types = {item["id"]: item for item in prereg["axes"]["types"]}
    cores = {item["id"]: item for item in prereg["axes"]["core_masks"]}
    layouts = {item["id"]: item for item in prereg["axes"]["b_layouts"]}
    return types, cores, layouts


def cross_axes(prereg: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    for type_axis in prereg["axes"]["types"]:
        for core_axis in prereg["axes"]["core_masks"]:
            for layout_axis in prereg["axes"]["b_layouts"]:
                yield type_axis, core_axis, layout_axis


def main() -> int:
    args = parse_args()
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    terminals_path = args.output_dir / "matrix_terminals.jsonl"
    raw_attempts_path = args.output_dir / "raw_attempts.jsonl"
    progress_path = args.output_dir / "progress.json"
    capability_path = args.output_dir / "capability_probe.json"
    if args.resume:
        if not args.output_dir.is_dir():
            raise FileNotFoundError(f"resume matrix directory is missing: {args.output_dir}")
        existing_terminals = [
            json.loads(line)
            for line in terminals_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        existing_ids = {item["cell_id"] for item in existing_terminals}
        if len(existing_ids) != len(existing_terminals):
            raise ValueError("cannot resume a matrix containing duplicate terminal cell IDs")
        if not capability_path.is_file():
            raise FileNotFoundError(f"resume capability record is missing: {capability_path}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        existing_terminals = []
        existing_ids: set[str] = set()
    protocol = prereg["measurement_protocol"]
    warmup = int(protocol["warmup_repetitions"])
    min_reps = int(protocol["timed_repetitions_minimum"])
    min_timed_ms = float(protocol["timed_window_minimum_ms"])

    # Exact execution-time capability probes. These are deliberately small and
    # precede the formal matrix; failed axes receive explicit N/A terminals.
    capability_cases = [
        ("probe_core01", "int8", [4, 64, 64], "core01", "normal"),
        ("probe_core012", "int8", [4, 64, 64], "core012", "normal"),
        ("probe_int4_normal", "int4", [4, 64, 64], "core0", "normal"),
        ("probe_int4_native", "int4", [4, 64, 64], "core0", "native"),
    ]
    if not args.resume:
        capability_attempts = []
        for name, type_cli, mkn, core_cli, layout_cli in capability_cases:
            capability_attempts.append(
                invoke(
                    args.bench,
                    args.phase_file,
                    raw_attempts_path,
                    name,
                    type_cli,
                    mkn,
                    core_cli,
                    layout_cli,
                    "random",
                    1,
                    1,
                    1,
                    0.0,
                    1,
                )
            )
        capability = {
            "status": "CAPABILITY_AXES_ADJUDICATED",
            "runtime": "librknnrt 2.3.2 (429f97ae6b@2025-04-09T09:09:27)",
            "decisions": {
                "core0": "RUNNABLE",
                "core0_1": "N/A: rknn_matmul_set_core_mask returns -1; runtime logs fallback, which is not accepted as two-core evidence",
                "core0_1_2": "N/A: rknn_matmul_set_core_mask returns -1; runtime logs fallback, which is not accepted as three-core evidence",
                "int4_normal": "N/A: vendor and project harness both abort in runtime 2.3.2 before run; no fallback result is accepted",
                "int4_native": "RUNNABLE after passing unpacked signed int4 values to the vendor normal-to-native conversion API; three-seed preflight follows",
            },
            "attempts": capability_attempts,
        }
        write_json(capability_path, capability)

    initial_terminal_count = len(existing_terminals)
    terminal_count = initial_terminal_count
    runnable_count = sum(bool(item.get("attempts")) for item in existing_terminals)

    def record(terminal: dict[str, Any]) -> None:
        nonlocal terminal_count
        if terminal["cell_id"] in existing_ids:
            raise ValueError(f"duplicate terminal during execution: {terminal['cell_id']}")
        append_jsonl(terminals_path, terminal)
        existing_ids.add(terminal["cell_id"])
        terminal_count += 1
        write_json(
            progress_path,
            {
                "updated_at": now_iso(),
                "mode": args.mode,
                "terminal_count": terminal_count,
                "runnable_completed": runnable_count,
                "last_cell_id": terminal["cell_id"],
                "last_status": terminal["terminal_status"],
            },
        )

    # Randomized correctness preflight, with explicit N/A rows for rejected axes.
    for type_axis, core_axis, layout_axis in cross_axes(prereg):
        axes = {
            "type": type_axis["id"],
            "core_mask": core_axis["id"],
            "b_layout": layout_axis["id"],
        }
        runnable, reason = runnable_axis(**{
            "type_id": type_axis["id"],
            "core_id": core_axis["id"],
            "layout_id": layout_axis["id"],
        })
        for seed in (1, 2, 3):
            cell_id = safe_id(
                f"preflight__{type_axis['id']}__{core_axis['id']}__{layout_axis['id']}__seed{seed}"
            )
            if cell_id in existing_ids:
                continue
            shape = {"shape_id": "randomized_selfcal_m4_k64_n64", "logical_mkn": [4, 64, 64]}
            if not runnable:
                record(na_terminal(cell_id, "randomized_preflight", reason or "N/A", axes, shape))
                continue
            attempts = [
                invoke(
                    args.bench,
                    args.phase_file,
                    raw_attempts_path,
                    cell_id,
                    type_axis["cli"],
                    [4, 64, 64],
                    core_axis["cli"],
                    layout_axis["cli"],
                    "random",
                    seed,
                    1,
                    1,
                    0.0,
                    1,
                )
            ]
            runnable_count += 1
            record(terminal_from_attempts(cell_id, "randomized_preflight", axes, shape, attempts))

    if args.mode == "preflight":
        args.phase_file.write_text("preflight_complete\n", encoding="utf-8")
        write_json(
            args.output_dir / "runner_summary.json",
            {
                "status": "PREFLIGHT_COMPLETE",
                "terminal_count": terminal_count,
                "runnable_completed": runnable_count,
            },
        )
        return 0

    for shape in prereg["formal_shapes"]:
        for type_axis, core_axis, layout_axis in cross_axes(prereg):
            axes = {
                "type": type_axis["id"],
                "core_mask": core_axis["id"],
                "b_layout": layout_axis["id"],
            }
            cell_id = safe_id(
                f"formal__{shape['shape_id']}__{type_axis['id']}__{core_axis['id']}__{layout_axis['id']}"
            )
            if cell_id in existing_ids:
                continue
            runnable, reason = runnable_axis(type_axis["id"], core_axis["id"], layout_axis["id"])
            if not runnable:
                record(na_terminal(cell_id, "formal", reason or "N/A", axes, shape))
                continue
            attempts = [
                invoke(
                    args.bench,
                    args.phase_file,
                    raw_attempts_path,
                    cell_id,
                    type_axis["cli"],
                    shape["analytics_by_type"][type_axis["id"]]["alignment"]["api_mkn"],
                    core_axis["cli"],
                    layout_axis["cli"],
                    "constant",
                    1,
                    warmup,
                    min_reps,
                    min_timed_ms,
                    1,
                )
            ]
            first_result = attempts[0].get("result")
            if (
                attempts[0]["process_returncode"] == 0
                and first_result
                and first_result.get("status") == "PASS"
                and float(first_result["run_only_us"]["iqr_over_p50"]) > 0.30
            ):
                attempts.append(
                    invoke(
                        args.bench,
                        args.phase_file,
                        raw_attempts_path,
                        cell_id,
                        type_axis["cli"],
                        shape["analytics_by_type"][type_axis["id"]]["alignment"]["api_mkn"],
                        core_axis["cli"],
                        layout_axis["cli"],
                        "constant",
                        1,
                        warmup,
                        min_reps,
                        min_timed_ms,
                        2,
                    )
                )
            runnable_count += 1
            terminal = terminal_from_attempts(cell_id, "formal", axes, shape, attempts)
            record(terminal)
            if runnable_count % 10 == 0:
                print(
                    json.dumps(
                        {
                            "progress": terminal_count,
                            "runnable_completed": runnable_count,
                            "last": cell_id,
                            "status": terminal["terminal_status"],
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    # OPE1 attention shapes have an explicit OPE5-standby disposition.
    for shape in prereg["attention_standby_shapes"]:
        for type_axis, core_axis, layout_axis in cross_axes(prereg):
            axes = {
                "type": type_axis["id"],
                "core_mask": core_axis["id"],
                "b_layout": layout_axis["id"],
            }
            cell_id = safe_id(
                f"attention_na__{shape['shape_id']}__{type_axis['id']}__{core_axis['id']}__{layout_axis['id']}"
            )
            if cell_id in existing_ids:
                continue
            record(na_terminal(cell_id, "attention_standby", "N/A_OPE5_STANDBY", axes, shape))

    # Synthetic M-sweep: all preregistered core rows receive terminals, but the
    # two rejected multi-core masks remain N/A rather than silently falling back.
    types, cores, layouts = axis_maps(prereg)
    sweep = prereg["m_sweep"]
    for m_value in sweep["m_values"]:
        for core_id in sweep["core_masks"]:
            type_axis = types[sweep["type_axis"]]
            core_axis = cores[core_id]
            layout_axis = layouts[sweep["b_layout"]]
            axes = {
                "type": type_axis["id"],
                "core_mask": core_axis["id"],
                "b_layout": layout_axis["id"],
            }
            shape = {
                "shape_id": f"synthetic_m{m_value}_mlp_down",
                "scope": "synthetic_m_sweep",
                "logical_mkn": [m_value, sweep["k"], sweep["n"]],
                "source_shape_id": sweep["source_shape_id"],
                "claim_limit": sweep["claim_limit"],
            }
            cell_id = safe_id(f"m_sweep__m{m_value}__{core_id}")
            if cell_id in existing_ids:
                continue
            runnable, reason = runnable_axis(type_axis["id"], core_id, layout_axis["id"])
            if not runnable:
                record(na_terminal(cell_id, "synthetic_m_sweep", reason or "N/A", axes, shape))
                continue
            attempts = [
                invoke(
                    args.bench,
                    args.phase_file,
                    raw_attempts_path,
                    cell_id,
                    type_axis["cli"],
                    shape["logical_mkn"],
                    core_axis["cli"],
                    layout_axis["cli"],
                    "constant",
                    1,
                    warmup,
                    min_reps,
                    min_timed_ms,
                    1,
                )
            ]
            first_result = attempts[0].get("result")
            if (
                attempts[0]["process_returncode"] == 0
                and first_result
                and first_result.get("status") == "PASS"
                and float(first_result["run_only_us"]["iqr_over_p50"]) > 0.30
            ):
                attempts.append(
                    invoke(
                        args.bench,
                        args.phase_file,
                        raw_attempts_path,
                        cell_id,
                        type_axis["cli"],
                        shape["logical_mkn"],
                        core_axis["cli"],
                        layout_axis["cli"],
                        "constant",
                        1,
                        warmup,
                        min_reps,
                        min_timed_ms,
                        2,
                    )
                )
            runnable_count += 1
            record(terminal_from_attempts(cell_id, "synthetic_m_sweep", axes, shape, attempts))

    expected = prereg["expected_terminal_counts"]["all_matrix_terminals"]
    status = "PASS_TERMINAL_MATRIX_COMPLETE" if terminal_count == expected else "FAIL_TERMINAL_COUNT"
    summary = {
        "status": status,
        "terminal_count": terminal_count,
        "expected_terminal_count": expected,
        "runnable_completed": runnable_count,
        "resumed": args.resume,
        "resume_initial_terminal_count": initial_terminal_count,
        "finished_at": now_iso(),
    }
    write_json(args.output_dir / "runner_summary.json", summary)
    args.phase_file.write_text("matrix_complete\n", encoding="utf-8")
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 0 if status == "PASS_TERMINAL_MATRIX_COMPLETE" else 90


if __name__ == "__main__":
    raise SystemExit(main())
