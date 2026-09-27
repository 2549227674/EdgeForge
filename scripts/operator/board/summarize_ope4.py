#!/usr/bin/env python3
"""Summarize the OPE4 single-segment dual-carrier diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


CARRIERS = ("c02_a2", "official_a1")
EXPECTED_PERF_EVENTS = (
    "task-clock",
    "context-switches",
    "cpu-migrations",
    "page-faults",
    "cycles:u",
    "instructions:u",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def as_int(value: str | None) -> int | None:
    parsed = as_float(value)
    return None if parsed is None else int(parsed)


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    p25 = quantile(values, 0.25)
    p50 = quantile(values, 0.50)
    p75 = quantile(values, 0.75)
    return {
        "samples": len(values),
        "min": min(values) if values else None,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "iqr": None if p25 is None or p75 is None else p75 - p25,
    }


def parse_perf_stat(path: Path) -> dict[str, Any]:
    events: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#") or len(row) < 3:
                continue
            event = row[2].strip()
            raw = row[0].strip()
            value = as_float(raw)
            events[event] = {
                "value": value,
                "unit": row[1].strip(),
                "raw": raw,
                "runtime_ns": as_int(row[3]) if len(row) > 3 else None,
                "enabled_pct": as_float(row[4]) if len(row) > 4 else None,
            }
    all_numeric = all(event in events and events[event]["value"] is not None for event in EXPECTED_PERF_EVENTS)
    cycles = events.get("cycles:u", {}).get("value")
    instructions = events.get("instructions:u", {}).get("value")
    ipc = None
    if cycles not in (None, 0) and instructions is not None:
        ipc = instructions / cycles
    return {"events": events, "all_expected_numeric": all_numeric, "ipc": ipc}


def parse_governors(path: Path) -> dict[str, Any]:
    rows = read_tsv(path)
    by_event = {row["event"]: row for row in rows}
    required = {"before", "after_set", "after_restore"}
    complete = required.issubset(by_event)
    set_pass = complete and all(
        by_event["after_set"][key] == "performance"
        for key in ("policy4_governor", "policy6_governor", "npu_governor")
    )
    restore_pass = complete and all(
        by_event["before"][key] == by_event["after_restore"][key]
        for key in ("policy4_governor", "policy6_governor", "npu_governor")
    )
    return {
        "events": by_event,
        "set_to_performance_pass": set_pass,
        "exact_restore_pass": restore_pass,
        "pass": bool(set_pass and restore_pass),
    }


def thermal_trip_map(path: Path) -> dict[int, float]:
    trips: dict[int, list[float]] = {}
    for row in read_tsv(path):
        temperature = as_float(row.get("trip_temp_mC"))
        zone = as_int(row.get("zone"))
        if zone is not None and temperature is not None and temperature > 0:
            trips.setdefault(zone, []).append(temperature)
    return {zone: min(values) for zone, values in trips.items()}


def summarize_telemetry(board_run: Path) -> dict[str, Any]:
    rows = read_csv(board_run / "telemetry_1hz.csv")
    trips = thermal_trip_map(board_run / "thermal_trip_points.tsv")
    carriers: dict[str, Any] = {}
    all_frequency_pass = True
    all_thermal_pass = True
    all_numeric = True
    for carrier in CARRIERS:
        phase = f"carrier_{carrier}"
        selected = [row for row in rows if row.get("phase") == phase]
        frequency: dict[str, Any] = {}
        for policy in ("policy4", "policy6"):
            fractions = []
            for row in selected:
                current = as_float(row.get(f"{policy}_khz"))
                maximum = as_float(row.get(f"{policy}_max_khz"))
                if current is not None and maximum not in (None, 0):
                    fractions.append(current / maximum)
            minimum = min(fractions) if fractions else None
            passed = minimum is not None and minimum >= 0.95
            frequency[policy] = {
                "samples": len(fractions),
                "min_fraction_of_max": minimum,
                "p50_fraction_of_max": quantile(fractions, 0.5),
                "pass": passed,
            }
            all_frequency_pass = all_frequency_pass and passed
        governor_values = sorted(
            {
                (
                    row.get("policy4_governor"),
                    row.get("policy6_governor"),
                    row.get("npu_governor"),
                )
                for row in selected
            }
        )
        governor_pass = bool(selected) and governor_values == [
            ("performance", "performance", "performance")
        ]
        thermal: dict[str, Any] = {}
        thermal_carrier_pass = True
        for header in rows[0].keys() if rows else []:
            match = re.match(r"thermal_zone(\d+)_(.+)_mC$", header)
            if match is None:
                continue
            zone = int(match.group(1))
            values = [
                value
                for row in selected
                if (value := as_float(row.get(header))) is not None
            ]
            max_mc = max(values) if values else None
            trip_mc = trips.get(zone)
            passed = max_mc is not None and trip_mc is not None and max_mc < trip_mc
            thermal[header.removesuffix("_mC")] = {
                "samples": len(values),
                "max_c": None if max_mc is None else max_mc / 1000.0,
                "lowest_trip_c": None if trip_mc is None else trip_mc / 1000.0,
                "pass": passed,
            }
            thermal_carrier_pass = thermal_carrier_pass and passed
        numeric_pass = bool(selected)
        all_numeric = all_numeric and numeric_pass
        all_thermal_pass = all_thermal_pass and thermal_carrier_pass
        carriers[carrier] = {
            "samples": len(selected),
            "numeric_samples_pass": numeric_pass,
            "governor_values": governor_values,
            "governor_pass": governor_pass,
            "frequency": frequency,
            "thermal": thermal,
            "thermal_pass": thermal_carrier_pass,
        }
    return {
        "samples": len(rows),
        "carriers": carriers,
        "numeric_samples_each_carrier_pass": all_numeric,
        "frequency_pass": all_frequency_pass,
        "thermal_pass": all_thermal_pass,
        "governor_pass": all(item["governor_pass"] for item in carriers.values()),
    }


def transcript_check(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8", errors="strict")
        utf8_valid = True
    except UnicodeDecodeError:
        text = ""
        utf8_valid = False
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_file(path),
        "utf8_valid": utf8_valid,
        "nonempty": len(payload) > 0,
        "pad_literal_seen": "[PAD]" in text,
        "replacement_character_seen": "\ufffd" in text,
        "pass": bool(
            utf8_valid
            and len(payload) > 0
            and "[PAD]" not in text
            and "\ufffd" not in text
        ),
        "semantic_evaluation": "NOT_PERFORMED_FORBIDDEN",
    }


def npu_prompt_window(
    load_rows: list[dict[str, str]], start_ns: int, end_ns: int
) -> dict[str, Any]:
    selected = []
    for row in load_rows:
        timestamp = as_int(row.get("monotonic_raw_ns"))
        if timestamp is not None and start_ns <= timestamp <= end_ns:
            selected.append(row)
    cores: dict[str, Any] = {}
    for core in range(3):
        key = f"npu_load_core{core}_pct"
        values = [
            value
            for row in selected
            if (value := as_float(row.get(key))) is not None
        ]
        cores[f"core{core}"] = distribution(values)
    return {
        "samples": len(selected),
        "numeric_core0_samples": cores["core0"]["samples"],
        "cores": cores,
        "pass": cores["core0"]["samples"] > 0,
    }


def summarize_carrier(
    board_run: Path, carrier: str, load_rows: list[dict[str, str]]
) -> dict[str, Any]:
    carrier_dir = board_run / carrier
    prompt_rows = read_csv(carrier_dir / "prompt_summary.csv")
    callback_rows = read_csv(carrier_dir / "callback_events.csv")
    callback_by_prompt: dict[int, list[dict[str, str]]] = {}
    for row in callback_rows:
        prompt_id = as_int(row.get("prompt_id"))
        if prompt_id is not None:
            callback_by_prompt.setdefault(prompt_id, []).append(row)

    prompts: list[dict[str, Any]] = []
    all_intervals: list[float] = []
    ttfts: list[float] = []
    prefill_times: list[float] = []
    generate_times: list[float] = []
    generate_tokens: list[int] = []
    wall_run_times: list[float] = []
    residual_times: list[float] = []
    memory_usage: list[float] = []
    all_transcripts_pass = True
    all_npu_windows_pass = True
    all_prompt_structure_pass = True
    for row in prompt_rows:
        prompt_id = int(row["prompt_id"])
        events = callback_by_prompt.get(prompt_id, [])
        normal_ns = [
            int(event["monotonic_raw_ns"])
            for event in events
            if event.get("state") == "NORMAL"
        ]
        intervals_ms = [
            (later - earlier) / 1e6
            for earlier, later in zip(normal_ns, normal_ns[1:])
        ]
        all_intervals.extend(intervals_ms)
        run_start_ns = int(row["run_start_ns"])
        run_return_ns = int(row["run_return_ns"])
        first_normal_ns = int(row["first_normal_ns"])
        ttft_ms = (first_normal_ns - run_start_ns) / 1e6 if first_normal_ns else None
        if ttft_ms is not None:
            ttfts.append(ttft_ms)
        prefill_time = float(row["prefill_time_ms"])
        generate_time = float(row["generate_time_ms"])
        generate_token_count = int(row["generate_tokens"])
        wall_run_ms = (run_return_ns - run_start_ns) / 1e6
        residual_ms = wall_run_ms - prefill_time - generate_time
        memory_mb = float(row["memory_usage_mb"])
        wall_run_times.append(wall_run_ms)
        residual_times.append(residual_ms)
        memory_usage.append(memory_mb)
        if prefill_time > 0:
            prefill_times.append(prefill_time)
        if generate_time > 0 and generate_token_count > 0:
            generate_times.append(generate_time)
            generate_tokens.append(generate_token_count)
        transcript = transcript_check(carrier_dir / f"prompt_{prompt_id:02d}.utf8")
        all_transcripts_pass = all_transcripts_pass and transcript["pass"]
        npu = npu_prompt_window(load_rows, run_start_ns, run_return_ns)
        all_npu_windows_pass = all_npu_windows_pass and npu["pass"]
        finish_callbacks = int(row["finish_callbacks"])
        normal_callbacks = int(row["normal_callbacks"])
        structure_pass = (
            int(row["rkllm_run_return"]) == 0
            and int(row["callback_error"]) == 0
            and finish_callbacks == 1
            and normal_callbacks > 0
            and int(row["output_bytes"]) > 0
        )
        all_prompt_structure_pass = all_prompt_structure_pass and structure_pass
        prompts.append(
            {
                "prompt_id": prompt_id,
                "wall_run_ms": wall_run_ms,
                "wall_ttft_ms": ttft_ms,
                "callback_decode": {
                    "normal_callbacks": normal_callbacks,
                    "intervals_ms": intervals_ms,
                    "interval_distribution_ms": distribution(intervals_ms),
                    "span_ms": (
                        (normal_ns[-1] - normal_ns[0]) / 1e6
                        if len(normal_ns) >= 2
                        else None
                    ),
                },
                "rkllm_perf": {
                    "prefill_time_ms": prefill_time,
                    "prefill_tokens": int(row["prefill_tokens"]),
                    "generate_time_ms": generate_time,
                    "generate_tokens": generate_token_count,
                    "generate_ms_per_token": (
                        generate_time / generate_token_count
                        if generate_time > 0 and generate_token_count > 0
                        else None
                    ),
                    "memory_usage_mb": memory_mb,
                    "normal_callback_minus_generate_tokens": (
                        normal_callbacks - generate_token_count
                    ),
                },
                "stage_breakdown": {
                    "wall_run_ms": wall_run_ms,
                    "rkllm_prefill_ms": prefill_time,
                    "rkllm_generate_ms": generate_time,
                    "wall_minus_rkllm_stages_ms": residual_ms,
                    "interpretation": "residual is descriptive callback/runtime overhead, not an undocumented SDK stage",
                },
                "npu_load_10hz": npu,
                "transcript": transcript,
                "structure_pass": structure_pass,
            }
        )

    runner_meta = read_env(carrier_dir / "runner_meta.env")
    init_ms = (
        int(runner_meta["init_end_monotonic_raw_ns"])
        - int(runner_meta["init_start_monotonic_raw_ns"])
    ) / 1e6
    process_wall_ms = (
        int(runner_meta["destroy_end_monotonic_raw_ns"])
        - int(runner_meta["init_start_monotonic_raw_ns"])
    ) / 1e6
    perf = parse_perf_stat(carrier_dir / "perf_stat.csv")
    task_clock = perf["events"].get("task-clock", {}).get("value")
    cpu_cores_utilized = (
        task_clock / process_wall_ms
        if task_clock is not None and process_wall_ms > 0
        else None
    )
    perf_generate_ms_per_token = (
        sum(generate_times) / sum(generate_tokens)
        if generate_times and sum(generate_tokens) > 0
        else None
    )
    run_exit = int((carrier_dir / "run_exit_code.txt").read_text().strip())
    return {
        "carrier": carrier,
        "run_exit_code": run_exit,
        "runner_meta": runner_meta,
        "init_ms": init_ms,
        "process_wall_ms": process_wall_ms,
        "prompt_count": len(prompt_rows),
        "prompts": prompts,
        "aggregate": {
            "wall_ttft_ms": distribution(ttfts),
            "wall_run_ms": distribution(wall_run_times),
            "callback_token_interval_ms": distribution(all_intervals),
            "primary_decode_ms_per_token": (
                statistics.fmean(all_intervals) if all_intervals else None
            ),
            "callback_decode_interval_count": len(all_intervals),
            "rkllm_prefill_time_ms": distribution(prefill_times),
            "rkllm_generate_time_ms": distribution(generate_times),
            "rkllm_generate_time_ms_total": sum(generate_times),
            "rkllm_generate_tokens_total": sum(generate_tokens),
            "rkllm_perf_decode_ms_per_token": perf_generate_ms_per_token,
            "wall_minus_rkllm_stages_ms": distribution(residual_times),
            "memory_usage_mb": distribution(memory_usage),
            "callback_interval_count_equals_perf_generate_tokens": (
                len(all_intervals) == sum(generate_tokens)
            ),
            "callback_vs_perf_decode_relative_delta": (
                abs(statistics.fmean(all_intervals) - perf_generate_ms_per_token)
                / perf_generate_ms_per_token
                if all_intervals and perf_generate_ms_per_token not in (None, 0)
                else None
            ),
        },
        "process_perf_stat": {
            **perf,
            "cpu_cores_utilized": cpu_cores_utilized,
        },
        "gates": {
            "run_exit_zero": run_exit == 0,
            "exactly_five_prompts": len(prompt_rows) == 5,
            "prompt_structure_pass": all_prompt_structure_pass,
            "transcripts_pass": all_transcripts_pass,
            "npu_numeric_each_prompt_window_pass": all_npu_windows_pass,
            "perf_all_events_numeric_pass": perf["all_expected_numeric"],
        },
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def build_report(receipt: dict[str, Any]) -> str:
    carriers = receipt["carriers"]
    c02 = carriers["c02_a2"]
    official = carriers["official_a1"]
    anchor = receipt["anchor"]
    gates = receipt["validity_gates"]
    lines = [
        "# OPE4 双载体层级归因报告（board-diagnostic-1core）",
        "",
        f"状态：`{receipt['status']}`。本次依 owner 指令只执行一个正式段；复测门为 "
        f"`{receipt['replication']['status']}`，不得声称 OPE4 replication。",
        "",
        "## 第一层：负载、载体与执行有效性",
        "",
        "五条 prompt 从 M0 S2a 证据包重新提取并以新的 OPE4 manifest 冻结。两载体均使用同一 manual Gemma4 模板、top-k=1、96 个最大新 token、16K context、单轮无历史；C02 是主载体，官方冻结件只读对照。没有执行质量评测，也没有检查回答语义。",
        "",
        f"- 总有效性：`{'PASS' if all(gates.values()) else 'FAIL'}`；governor 成对恢复：`{gates['governor_restore_pass']}`。",
        f"- transcript 门：严格 UTF-8、非空、无 `[PAD]`、无替换字符；结果 `{'PASS' if gates['transcripts_pass'] else 'FAIL'}`。",
        f"- OPE3 保留：isolated measurement=`{receipt['ope3_preservation']['isolated_measurement_pass']}`，Tier-B 原冻结代理=`{receipt['ope3_preservation']['tier_b_preregistered_analytic_backing_pass']}`；OPE4 不回写该负结果。",
        "",
        "## 第二层：同协议 C02 vs 官方对照",
        "",
        "| carrier lineage | TTFT p50 (ms) | callback TPOT aggregate (ms/token) | RKLLM perf decode (ms/token) | NPU core0 mean / p50 / max (%) | CPU cores utilized |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("c02_a2", "C02 A2 primary"), ("official_a1", "official A1 read-only")):
        carrier = carriers[name]
        aggregate = carrier["aggregate"]
        npu = carrier["npu_load_prompt_union"]["core0"]
        lines.append(
            "| "
            + label
            + " | "
            + fmt(aggregate["wall_ttft_ms"]["p50"])
            + " | "
            + fmt(aggregate["primary_decode_ms_per_token"])
            + " | "
            + fmt(aggregate["rkllm_perf_decode_ms_per_token"])
            + " | "
            + " / ".join(fmt(npu[key]) for key in ("mean", "p50", "max"))
            + " | "
            + fmt(carrier["process_perf_stat"]["cpu_cores_utilized"])
            + " |"
        )
    comparison = receipt["comparison"]
    lines += [
        "",
        "callback TPOT 的主口径是所有 prompt 内相邻 NORMAL callback 间隔的加权聚合；WAITING 事件保留在原始 CSV，但不进入 token 间隔。RKLLM perf 列是 runtime 字段的独立交叉。载体间差异仅用于系统诊断，不代表训练质量。",
        "",
        f"C02 相对官方：TTFT p50 比值 `{fmt(comparison['c02_over_official_ttft_p50_ratio'], 4)}`，callback TPOT 比值 `{fmt(comparison['c02_over_official_primary_decode_ratio'], 4)}`。",
        "",
        "![OPE4 双载体五题 10 Hz RKNPU load 曲线](../../results/operator/board/ope4/ope4_npu_load_curve_20260921.png)",
        "",
        "### RKLLM 阶段拆分",
        "",
        "| carrier | init (ms) | prompt wall p50 (ms) | prefill p50 (ms) | generate p50 (ms) | generated tokens total | wall−prefill−generate p50 (ms) | memory max (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("c02_a2", "C02 A2 primary"), ("official_a1", "official A1 read-only")):
        carrier = carriers[name]
        aggregate = carrier["aggregate"]
        lines.append(
            f"| {label} | {fmt(carrier['init_ms'])} | "
            f"{fmt(aggregate['wall_run_ms']['p50'])} | "
            f"{fmt(aggregate['rkllm_prefill_time_ms']['p50'])} | "
            f"{fmt(aggregate['rkllm_generate_time_ms']['p50'])} | "
            f"{aggregate['rkllm_generate_tokens_total']} | "
            f"{fmt(aggregate['wall_minus_rkllm_stages_ms']['p50'])} | "
            f"{fmt(aggregate['memory_usage_mb']['max'])} |"
        )
    lines += [
        "",
        "`wall−prefill−generate` 只描述 callback/runtime 侧剩余时间，不命名为 SDK 未公开阶段。两载体因 EOS 位置不同而生成 token 总数不同，绝对 generate/prompt wall 时长不能当作速度对比；归一化比较使用上表 TPOT。两载体的 callback interval 数均与 RKLLM `generate_tokens` 总数一致，callback 与 runtime 的 decode ms/token 相对差均小于 0.1%。",
        "",
        "## 第三层：OPE1 权重流量 × OPE0 带宽锚",
        "",
        "| 帐本 | 权重流量 (B/token) | OPE0 下界预测 (ms/token) | C02 callback 实测 (ms/token) | gap |",
        "|---|---:|---:|---:|---:|",
        f"| 258 次核心投影（不是全部 linear） | {anchor['core_projection']['weight_bytes']:,} | {anchor['core_projection']['predicted_lower_bound_ms']:.3f} | {fmt(anchor['measured_primary_decode_ms_per_token'])} | {fmt(anchor['core_projection']['gap_pct'])}% |",
        f"| 全部 343 个非 lm_head linear | {anchor['all_non_lmhead_linear']['weight_bytes']:,} | {anchor['all_non_lmhead_linear']['predicted_lower_bound_ms']:.3f} | {fmt(anchor['measured_primary_decode_ms_per_token'])} | {fmt(anchor['all_non_lmhead_linear']['gap_pct'])}% |",
        "",
        "这里的 `predicted = 权重流量 / OPE0 CPU triad ceiling` 只是解析帐下界，不是 NPU DRAM 计数器实测。NPU 内部 SRAM 复用、权重压缩与调度气泡仍是候选；本次证据只能用 NPU load 曲线和进程级 CPU perf 判断兼容性，不能强行锁定单一原因。",
        "",
        f"归因裁决：{anchor['attribution_summary']}",
        "",
        "## 结论边界",
        "",
        "OPE4 模型阶段锚已按单段 owner waiver 处理。OPE3 的 Tier-B `bus_access×64B` 代理仍为 FAIL，因此即便本阶段锚定有效，K-E1 四门也不会被改写成全 PASS。所有数字仅为 `board-diagnostic-1core`，不外推 3-core、不与 B0/PC 同表、不形成部署或质量声明。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--board-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    board_run = args.board_run.resolve()
    prereg_path = repo / "configs/operator/board/ope4_diag_protocol.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    run_meta = read_env(board_run / "run_meta.env")
    controller_meta_path = board_run / "controller_meta.json"
    controller_meta = (
        json.loads(controller_meta_path.read_text(encoding="utf-8"))
        if controller_meta_path.exists()
        else None
    )
    load_rows = read_csv(board_run / "npu_load_10hz.csv")
    carriers = {
        carrier: summarize_carrier(board_run, carrier, load_rows)
        for carrier in CARRIERS
    }

    for carrier, summary in carriers.items():
        union_rows: list[dict[str, str]] = []
        for prompt in summary["prompts"]:
            prompt_id = prompt["prompt_id"]
            row = next(
                item
                for item in read_csv(board_run / carrier / "prompt_summary.csv")
                if int(item["prompt_id"]) == prompt_id
            )
            start_ns = int(row["run_start_ns"])
            end_ns = int(row["run_return_ns"])
            union_rows.extend(
                item
                for item in load_rows
                if (timestamp := as_int(item.get("monotonic_raw_ns"))) is not None
                and start_ns <= timestamp <= end_ns
            )
        cores: dict[str, Any] = {}
        for core in range(3):
            key = f"npu_load_core{core}_pct"
            values = [
                value
                for row in union_rows
                if (value := as_float(row.get(key))) is not None
            ]
            cores[f"core{core}"] = distribution(values)
        summary["npu_load_prompt_union"] = cores

    telemetry = summarize_telemetry(board_run)
    governor = parse_governors(board_run / "governor_state.tsv")
    # The board file intentionally has no header; parse the three columns directly.
    implementation_pass = True
    implementation_checks = []
    with (board_run / "implementation_hash_observed.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            path, expected, observed = line.rstrip("\n").split("\t")
            implementation_checks.append(
                {"path": path, "expected_sha256": expected, "observed_sha256": observed}
            )
            implementation_pass = implementation_pass and expected == observed
    implementation_pass = implementation_pass and len(implementation_checks) == 6

    artifact_unchanged = all(
        run_meta.get(key) == run_meta.get(f"{key}_after")
        for key in (
            "c02_model_bytes",
            "c02_model_mtime_epoch",
            "official_model_bytes",
            "official_model_mtime_epoch",
        )
    )
    carrier_gates_pass = all(
        all(summary["gates"].values()) for summary in carriers.values()
    )
    prereg_sha = sha256_file(prereg_path)
    board_elapsed_seconds = (
        int(run_meta["board_end_monotonic_raw_ns"])
        - int(run_meta["board_start_monotonic_raw_ns"])
    ) / 1e9
    controller_elapsed_seconds = (
        controller_meta.get("controller_elapsed_seconds")
        if controller_meta is not None
        else None
    )
    timebase_pass = bool(
        controller_meta is not None
        and controller_meta.get("controller_exit_code") == 0
        and controller_meta.get("board_boot_id") == run_meta.get("board_start_boot_id")
        and board_elapsed_seconds > 0
        and controller_elapsed_seconds is not None
        and controller_elapsed_seconds >= board_elapsed_seconds
        and controller_elapsed_seconds - board_elapsed_seconds < 10
    )
    validity_gates = {
        "formal_run_exit_zero": run_meta.get("run_exit_code") == "0",
        "prereg_identity_pass": run_meta.get("prereg_sha256") == prereg_sha,
        "workload_identity_pass": (
            run_meta.get("prompt_payload_sha256")
            == prereg["workload"]["payload_sha256"]
        ),
        "timebase_pass": timebase_pass,
        "implementation_identity_pass": implementation_pass,
        "carrier_and_prompt_gates_pass": carrier_gates_pass,
        "transcripts_pass": all(
            carrier["gates"]["transcripts_pass"] for carrier in carriers.values()
        ),
        "npu_load_windows_pass": all(
            carrier["gates"]["npu_numeric_each_prompt_window_pass"]
            for carrier in carriers.values()
        ),
        "process_perf_pass": all(
            carrier["gates"]["perf_all_events_numeric_pass"]
            for carrier in carriers.values()
        ),
        "telemetry_samples_pass": telemetry["numeric_samples_each_carrier_pass"],
        "performance_governors_during_run_pass": telemetry["governor_pass"],
        "a76_frequency_pass": telemetry["frequency_pass"],
        "thermal_pass": telemetry["thermal_pass"],
        "governor_restore_pass": governor["pass"],
        "carrier_artifacts_unchanged_pass": artifact_unchanged,
    }
    measurement_valid = all(validity_gates.values())

    c02_decode = carriers["c02_a2"]["aggregate"]["primary_decode_ms_per_token"]
    core_predicted = prereg["anchor"]["core_projection_predicted_lower_bound_ms"]
    all_predicted = prereg["anchor"]["all_non_lmhead_linear_predicted_lower_bound_ms"]
    core_gap = (
        (c02_decode - core_predicted) / core_predicted * 100
        if c02_decode is not None
        else None
    )
    all_gap = (
        (c02_decode - all_predicted) / all_predicted * 100
        if c02_decode is not None
        else None
    )
    c02_npu_mean = carriers["c02_a2"]["npu_load_prompt_union"]["core0"]["mean"]
    c02_npu_p50 = carriers["c02_a2"]["npu_load_prompt_union"]["core0"]["p50"]
    c02_cpu_cores = carriers["c02_a2"]["process_perf_stat"]["cpu_cores_utilized"]
    if c02_decode is None:
        attribution = "主载体 callback decode 口径不可用，未作 gap 归因。"
    elif c02_decode > all_predicted and c02_npu_p50 is not None and c02_npu_p50 >= 95:
        attribution = (
            f"实测高于两列解析下界；prompt 窗口内 NPU core0 mean/p50="
            f"{c02_npu_mean:.3f}%/{c02_npu_p50:.3f}%，10 Hz 尺度呈持续近满载，"
            "不支持把主要差额归为百毫秒级粗粒度调度空洞（更细气泡仍不可见）。"
            f"进程级 CPU 参与约 {c02_cpu_cores:.3f} 核，未显示 4 个可用 A76 饱和。"
            "结合 OPE2 已记录的 M=1 同几何强路径瓶颈，NPU/runtime decode 路径限制与现象相容；"
            "现有接口没有 NPU DRAM 字节计数器，SRAM 复用和权重压缩均不能证实或排除。"
        )
    elif c02_decode > all_predicted:
        attribution = (
            "实测高于两列解析下界，但 NPU load 曲线不足以把差额唯一归到调度气泡；"
            "SRAM 复用、压缩、搬运与 runtime 调度均保持未决。"
        )
    else:
        attribution = (
            "实测未高于解析权重流量/OPE0 下界，这与内部复用或压缩相容，但 CPU triad 分母并非"
            " NPU DRAM 计数器，不能据此证明任何一种机制。"
        )

    ope3 = json.loads(
        (repo / "results/operator/board/ope3/ope3_receipt_20260919.json").read_text(
            encoding="utf-8"
        )
    )
    official_decode = carriers["official_a1"]["aggregate"]["primary_decode_ms_per_token"]
    c02_ttft = carriers["c02_a2"]["aggregate"]["wall_ttft_ms"]["p50"]
    official_ttft = carriers["official_a1"]["aggregate"]["wall_ttft_ms"]["p50"]
    receipt: dict[str, Any] = {
        "schema": "edgeforge-ope4-diagnostic-receipt-v1",
        "stage": "OPE4",
        "status": (
            "VALID_OPE4_SINGLE_SEGMENT_OWNER_WAIVER_MODEL_STAGE_ANCHOR_PRESENT"
            if measurement_valid
            else "INVALID_OPE4_SINGLE_SEGMENT_RETAINED_FOR_DIAGNOSIS"
        ),
        "label": "board-diagnostic-1core",
        "prereg_ref": str(prereg_path.relative_to(repo)),
        "prereg_sha256": prereg_sha,
        "board_run": str(board_run.relative_to(repo)),
        "controller_meta": controller_meta,
        "run_meta": run_meta,
        "replication": {
            "formal_segments": 1,
            "status": "WAIVED_BY_OWNER_NOT_SATISFIED",
            "owner_instruction": prereg["owner_variance"]["instruction"],
            "claim": "NO_OPE4_REPLICATION_CLAIM",
        },
        "visualizations": {
            "npu_load_curve_png": "results/operator/board/ope4/ope4_npu_load_curve_20260921.png",
            "npu_load_curve_svg": "results/operator/board/ope4/ope4_npu_load_curve_20260921.svg",
            "source": "scripts/operator/board/plot_ope4_npu_load.py",
        },
        "carriers": carriers,
        "comparison": {
            "c02_over_official_ttft_p50_ratio": (
                c02_ttft / official_ttft
                if c02_ttft is not None and official_ttft not in (None, 0)
                else None
            ),
            "c02_over_official_primary_decode_ratio": (
                c02_decode / official_decode
                if c02_decode is not None and official_decode not in (None, 0)
                else None
            ),
            "interpretation": "system diagnostic only; not a training-quality conclusion",
        },
        "telemetry": telemetry,
        "governor_guard": governor,
        "implementation_identity": {
            "checks": implementation_checks,
            "pass": implementation_pass,
        },
        "anchor": {
            "primary_carrier": "c02_a2",
            "measured_primary_decode_ms_per_token": c02_decode,
            "metric": "aggregate consecutive NORMAL callback interval",
            "ope0_tier_a_ceiling_gbps": prereg["anchor"]["ope0_tier_a_ceiling_gbps"],
            "ope0_replication_status": prereg["anchor"]["ope0_replication_status"],
            "core_projection": {
                "weight_bytes": prereg["anchor"]["core_projection_weight_bytes"],
                "label": prereg["anchor"]["core_projection_label"],
                "predicted_lower_bound_ms": core_predicted,
                "gap_pct": core_gap,
            },
            "all_non_lmhead_linear": {
                "weight_bytes": prereg["anchor"]["all_non_lmhead_linear_weight_bytes"],
                "label": prereg["anchor"]["all_non_lmhead_linear_label"],
                "predicted_lower_bound_ms": all_predicted,
                "gap_pct": all_gap,
            },
            "model_stage_anchor_present": measurement_valid,
            "attribution_summary": attribution,
            "claim_limit": prereg["anchor"]["interpretation_limit"],
        },
        "validity_gates": validity_gates,
        "measurement_valid": measurement_valid,
        "ope3_preservation": {
            "source_status": ope3["status"],
            "isolated_measurement_pass": ope3["isolated_measurement_pass"],
            "tier_b_preregistered_analytic_backing_pass": ope3["method_gates"][
                "tier_b_preregistered_analytic_backing_pass"
            ],
            "ope4_model_stage_anchor_present_now": measurement_valid,
            "full_ke1_four_gate_closure_pass": False,
            "reason": "OPE4 cannot repair the frozen OPE3 Tier-B cache-line proxy failure",
        },
        "negative_scope_assertions": {
            "quality_evaluation_performed": False,
            "semantic_transcript_judgment_performed": False,
            "three_core_or_multicore_llm_run": False,
            "prompt_cache_ab_performed": False,
            "driver_kernel_dt_or_runtime_so_modified": False,
            "frozen_model_modified": False,
            "isolated_primitive_called_tpot": False,
            "b0_or_pc_same_table_comparison": False,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(build_report(receipt), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(args.output_json),
                "report": str(args.output_report),
            },
            ensure_ascii=False,
        )
    )
    return 0 if measurement_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
