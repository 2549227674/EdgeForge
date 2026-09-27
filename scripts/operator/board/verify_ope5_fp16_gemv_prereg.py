#!/usr/bin/env python3
"""Verify the OPE5-02 FP16 GEMV four-question scope and preregistration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_GIT_SHA = "213460d88311cba5d84add4d3ee57aae26716c6d"
EXPECTED_OWNER_INSTRUCTION = "不需要复测，取消同日复测，只测一次"
OPE2_CELL_ID = "formal__decode_mlp_down_proj__fp16xfp16_to_fp32__core0__normal"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def close(actual: float, expected: float, *, abs_tol: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=abs_tol)


def candidate_score(candidate: dict[str, Any]) -> int:
    dims = candidate["dimensions"]
    return (
        dims["authenticity"]
        + dims["ope4_gap_relevance"]
        + dims["novel_evidence"]
        + dims["capability_readiness"]
        + dims["deliverability"]
        - dims["cost_risk_penalty"]
    )


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    dims = candidate["dimensions"]
    return (
        candidate["scope_blocker"] is not None,
        -candidate_score(candidate),
        -dims["ope4_gap_relevance"],
        -dims["novel_evidence"],
        dims["cost_risk_penalty"],
        candidate["candidate_id"],
    )


def resolve_sources(
    repo: Path, sources: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    rows: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for source_id, source in sources.items():
        if "path" not in source:
            continue
        source_path = repo / source["path"]
        observed = sha256_file(source_path) if source_path.is_file() else None
        rows[source_id] = {
            "path": source["path"],
            "expected_sha256": source["sha256"],
            "observed_sha256": observed,
        }
        checks[f"source_{source_id}_sha256"] = observed == source["sha256"]
    return rows, checks


def find_ope1_shape(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest["linear_shapes"]["formal_main_and_card_prefill_probes"]
    matches = [row for row in rows if row["shape_id"] == "decode_mlp_down_proj"]
    if len(matches) != 1:
        raise ValueError(f"expected one decode_mlp_down_proj row, found {len(matches)}")
    return matches[0]


def read_ope2_cell(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if row["cell_id"] == OPE2_CELL_ID]
    if len(matches) != 1:
        raise ValueError(f"expected one {OPE2_CELL_ID} row, found {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scope",
        type=Path,
        default=Path("configs/operator/board/ope5_fp16_gemv_scope.json"),
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path("configs/operator/board/ope5_fp16_gemv_prereg.json"),
    )
    parser.add_argument("--scope-output", type=Path, required=True)
    parser.add_argument("--prereg-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    scope_path = (repo / args.scope).resolve()
    prereg_path = (repo / args.prereg).resolve()
    scope = read_json(scope_path)
    prereg = read_json(prereg_path)
    observed_git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()

    scope_source_rows, scope_checks = resolve_sources(repo, scope["sources"])
    triage = read_json(repo / scope["sources"]["ope5_backlog_triage"]["path"])
    ope1 = read_json(repo / scope["sources"]["ope1_shape_manifest"]["path"])
    ope3_receipt = read_json(repo / scope["sources"]["ope3_receipt"]["path"])
    ope3_table = read_json(repo / scope["sources"]["ope3_three_horizontal"]["path"])
    ope4 = read_json(repo / scope["sources"]["ope4_receipt"]["path"])
    ope5_03 = read_json(repo / scope["sources"]["ope5_03_receipt"]["path"])
    ope2_cell = read_ope2_cell(repo / scope["sources"]["ope2_cells"]["path"])

    ranked = sorted(triage["candidates"], key=candidate_sort_key)
    ope5_02 = next(row for row in ranked if row["candidate_id"] == "OPE5-02")
    ope1_shape = find_ope1_shape(ope1)
    ope1_fp16 = ope1_shape["analytics_by_type"]["fp16xfp16_to_fp32"]
    ope1_int8 = ope1_shape["analytics_by_type"]["int8xint8_to_int32"]
    ope3_a76 = next(row for row in ope3_table["rows"] if row["carrier"] == "a76-neon")

    scope_checks["scope_status"] = (
        scope["status"] == "PASS_OPE5_02_FOUR_QUESTIONS_PREREG_AUTHORIZED"
    )
    scope_checks["candidate_is_rank_two_and_ready"] = (
        ranked.index(ope5_02) == 1
        and scope["selection"]["candidate_id"] == "OPE5-02"
        and scope["selection"]["backlog_rank_at_ope5_triage"] == 2
        and ope5_02["disposition"] == "READY_AFTER_RMSNORM"
        and ope5_02["scope_blocker"] is None
    )
    scope_checks["prior_candidate_closed"] = (
        ope5_03["status"]
        == "VALID_OPE5_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_A76_RMSNORM"
    )
    scope_checks["four_questions_pass"] = all(
        item["status"].startswith("PASS") for item in scope["four_questions"].values()
    )
    scope_checks["shape_and_frequency_match_ope1"] = (
        ope1_shape["logical_mkn"] == [1, 10240, 2560]
        and scope["frozen_shape"]["m"] == ope1_shape["logical_mkn"][0]
        and scope["frozen_shape"]["k"] == ope1_shape["logical_mkn"][1]
        and scope["frozen_shape"]["n"] == ope1_shape["logical_mkn"][2]
        and scope["frozen_shape"]["calls_per_model_forward"]
        == ope1_shape["frequency"]["calls_per_model_forward"]
        == 42
        and scope["frozen_shape"]["observed_model_forwards"]
        == ope1_shape["frequency"]["observed_model_forwards"]
        == 21694
        and scope["frozen_shape"]["observed_invocations"]
        == ope1_shape["frequency"]["observed_invocations"]
        == 911148
    )
    account = scope["analytic_account"]
    io = ope1_fp16["logical_io_bytes"]
    scope_checks["fp16_analytic_account_matches_ope1"] = (
        account["activation_read_bytes_per_call"] == io["A_read"] == 20480
        and account["weight_read_bytes_per_call"] == io["B_read"] == 52428800
        and account["output_write_bytes_per_call"] == io["C_write"] == 10240
        and account["total_bytes_per_call"] == io["total"] == 52459520
        and account["ops_per_call"] == ope1_fp16["logical_ops"] == 52428800
    )
    expected_ops_per_byte = account["ops_per_call"] / account["total_bytes_per_call"]
    expected_42_bytes = account["total_bytes_per_call"] * 42
    expected_42_ops = account["ops_per_call"] * 42
    expected_floor_us = (
        account["total_bytes_per_call"] / account["ope0_ceiling_gbps"] / 1e9 * 1e6
    )
    expected_floor_42_ms = expected_floor_us * 42 / 1000
    expected_floor_percent = expected_floor_42_ms / 446.1262611013072 * 100
    scope_checks["analytic_arithmetic"] = (
        close(account["ops_per_byte"], expected_ops_per_byte)
        and account["bytes_for_42_call_projection"] == expected_42_bytes == 2203299840
        and account["ops_for_42_call_projection"] == expected_42_ops == 2202009600
        and close(account["roofline_floor_us_per_call"], expected_floor_us)
        and close(account["roofline_floor_ms_for_42_calls"], expected_floor_42_ms)
        and close(
            account["roofline_floor_percent_of_ope4_tpot"], expected_floor_percent
        )
    )
    scope_checks["ope2_anchor_exact"] = (
        ope2_cell["terminal_status"] == "PASS"
        and [int(ope2_cell[key]) for key in ["M", "K", "N"]] == [1, 10240, 2560]
        and ope2_cell["type"] == "fp16xfp16_to_fp32"
        and ope2_cell["core_mask"] == "core0"
        and ope2_cell["b_layout"] == "normal"
        and close(float(ope2_cell["p50_us"]), scope["anchors"]["ope2_npu_fp16"]["p50_us"])
        and close(float(ope2_cell["iqr_us"]), 0.583999)
        and close(
            float(ope2_cell["measured_gops"]),
            scope["anchors"]["ope2_npu_fp16"].get("measured_gops", 11.088513872),
        )
        and close(
            float(ope2_cell["measured_gbps"]),
            scope["anchors"]["ope2_npu_fp16"]["measured_gbps"],
        )
        and ope2_cell["correctness_pass"] == "True"
    )
    scope_checks["ope3_anchor_exact"] = (
        ope3_receipt["isolated_measurement_pass"] is True
        and ope3_table["shape_mkn"] == [1, 10240, 2560]
        and close(ope3_a76["p50_us"], scope["anchors"]["ope3_a76_int8"]["p50_us"])
        and close(
            ope3_a76["measured_gbps"],
            scope["anchors"]["ope3_a76_int8"]["measured_gbps"],
        )
        and scope["anchors"]["ope3_a76_int8"]["analytic_bytes"]
        == ope1_int8["logical_io_bytes"]["total"]
        == 26234880
    )
    scope_checks["ope4_anchor_exact"] = (
        ope4["measurement_valid"] is True
        and close(
            ope4["anchor"]["measured_primary_decode_ms_per_token"],
            scope["anchors"]["ope4"]["primary_decode_ms_per_token"],
        )
        and close(
            ope4["carriers"]["c02_a2"]["process_perf_stat"]["cpu_cores_utilized"],
            scope["anchors"]["ope4"]["primary_process_cpu_cores_utilized"],
        )
    )
    scope_checks["no_execution_or_implementation"] = (
        scope["board_execution_started"] is False
        and scope["candidate_implemented"] is False
        and scope["decision"]["implementation_authorized_by_this_file"] is False
        and scope["decision"]["board_execution_authorized_by_this_file"] is False
    )
    scope_checks["git_sha_at_freeze"] = (
        scope["git_sha_at_freeze"] == observed_git_sha == EXPECTED_GIT_SHA
    )

    prereg_source_rows, prereg_checks = resolve_sources(repo, prereg["sources"])
    prereg_checks["prereg_status"] = prereg["status"] == "PREREGISTERED_FROZEN_NOT_RUN"
    prereg_checks["scope_identity"] = (
        prereg["selection"]["scope_path"] == str(scope_path.relative_to(repo))
        and prereg["selection"]["scope_sha256"] == sha256_file(scope_path)
        and prereg["sources"]["scope_freeze"]["sha256"] == sha256_file(scope_path)
    )
    prereg_checks["selected_identity_and_four_questions"] = (
        prereg["selection"]["candidate_id"] == "OPE5-02"
        and prereg["selection"]["kernel_id"] == "ope5_neon_fp16_gemv"
        and prereg["selection"]["four_question_statuses"]
        == {key: value["status"] for key, value in scope["four_questions"].items()}
    )
    semantic = prereg["semantic_contract"]
    formal_shape = prereg["formal_shape"]
    prereg_checks["semantic_and_shape_contract"] = (
        [semantic["m"], semantic["k"], semantic["n"]] == [1, 10240, 2560]
        and formal_shape["logical_mkn"] == [1, 10240, 2560]
        and semantic["activation_dtype"] == "IEEE float16"
        and semantic["weight_dtype"] == "IEEE float16"
        and semantic["accumulator_dtype"] == "float32"
        and semantic["output_dtype"] == "float32"
        and semantic["weight_layout"] == "row-major contiguous [N,K]"
        and semantic["bias"] is False
        and formal_shape["calls_per_model_forward"] == 42
        and formal_shape["observed_invocations"] == 911148
    )
    prereg_account = prereg["analytic_account"]
    prereg_checks["analytic_account_exact"] = (
        prereg_account["total_bytes_per_call"] == 52459520
        and prereg_account["ops_per_call"] == 52428800
        and prereg_account["bytes_for_42_call_projection"] == 52459520 * 42
        and prereg_account["ops_for_42_call_projection"] == 52428800 * 42
        and close(
            prereg_account["fp16_to_int8_analytic_byte_ratio"],
            52459520 / 26234880,
        )
        and close(
            prereg_account["roofline_floor_us_per_call"],
            52459520 / 23.1951846685 / 1e9 * 1e6,
        )
        and close(
            prereg_account["roofline_floor_ms_for_42_calls"],
            52459520 / 23.1951846685 / 1e9 * 1e3 * 42,
        )
    )
    distributions = prereg["correctness_protocol"]["input_distributions"]
    expected_element_comparisons = (
        len(prereg["correctness_protocol"]["implementations_checked"])
        * len(distributions)
        * len(prereg["correctness_protocol"]["seeds"])
        * prereg["correctness_protocol"]["outputs_checked_per_case"]
    )
    exact_distribution = next(row for row in distributions if row["id"] == "quarter_grid_exact")
    random_distributions = [row for row in distributions if row["id"] != "quarter_grid_exact"]
    prereg_checks["correctness_population_and_tolerances"] = (
        prereg["correctness_protocol"]["must_complete_before_timing"] is True
        and prereg["correctness_protocol"]["oracle"].startswith("Independent FP64")
        and expected_element_comparisons
        == prereg["correctness_protocol"]["expected_element_comparisons"]
        == 69120
        and exact_distribution["atol"] == 0.0
        and exact_distribution["rtol"] == 0.0
        and all(row["atol"] == 0.05 and row["rtol"] == 0.001 for row in random_distributions)
    )
    prereg_checks["three_distinct_baselines"] = set(prereg["baselines"]) == {
        "scalar_fp32",
        "autovec_o3",
        "neon_fp16_f32acc",
    }
    protocol = prereg["measurement_protocol"]
    prereg_checks["four_a76_fixed_partition"] = (
        formal_shape["cpu_cores"] == [4, 5, 6, 7]
        and formal_shape["threads"] == 4
        and protocol["primary_cpus"] == [4, 5, 6, 7]
        and protocol["taskset"] == "4-7"
        and protocol["threads"] == 4
        and "N/4=640" in formal_shape["output_partition"]
    )
    prereg_checks["timing_population_and_working_set"] = (
        protocol["warmup_repetitions_per_variant"] == 50
        and protocol["timed_repetitions_per_variant_minimum"] == 200
        and protocol["timed_window_minimum_seconds_per_variant"] == 3.0
        and "52,428,800-byte" in protocol["working_set"]
        and "all six permutations" in protocol["interleave"]
        and protocol["clock"] == "CLOCK_MONOTONIC_RAW"
    )
    prereg_checks["single_segment_owner_waiver_no_retry"] = (
        protocol["formal_segments"] == 1
        and protocol["owner_instruction"] == EXPECTED_OWNER_INSTRUCTION
        and protocol["replication_status"] == "WAIVED_BY_OWNER_NOT_SATISFIED"
        and "no automatic retry" in protocol["stability_gate"]
        and "Do not silently rerun" in protocol["failure_policy"]
    )
    prereg_checks["no_meaningless_smoke"] = (
        protocol["smoke_policy"]
        == "No separate performance smoke, no reduced-shape timing, and no second formal segment. Randomized correctness is the only pre-timing execution gate."
    )
    prereg_checks["tier_b_contract"] = (
        prereg["tier_b"]["stat_events"]
        == ["cycles:u", "instructions:u", "l2d_cache_refill:u", "bus_access:u"]
        and prereg["tier_b"]["variants"]
        == ["scalar_fp32", "autovec_o3", "neon_fp16_f32acc"]
        and "rejection of bus_access count*64" in prereg["tier_b"]["bus_access_policy"]
        and "count*8 only" in prereg["tier_b"]["bus_access_policy"]
        and "not an analytic-byte acceptance gate" in prereg["tier_b"]["bus_access_policy"]
        and prereg["tier_b"]["perf_record"].startswith("Not required")
    )
    prereg_checks["anchors_exact"] = (
        prereg["anchors"]["ope3_a76_int8"]["p50_us"] == 944.7080005
        and prereg["anchors"]["ope3_a76_int8"]["analytic_bytes_per_call"] == 26234880
        and prereg["anchors"]["ope2_npu_fp16"]["cell_id"] == OPE2_CELL_ID
        and prereg["anchors"]["ope2_npu_fp16"]["p50_us"] == 4728.208
        and prereg["anchors"]["ope4"]["primary_decode_ms_per_token"]
        == 446.1262611013072
    )
    implementation_paths = [
        repo / prereg["implementation_identity"][key]
        for key in ["candidate_source_path", "runner_path", "measurement_payload_path"]
    ]
    prereg_checks["implementation_absent_at_freeze"] = all(
        not path.exists() for path in implementation_paths
    )
    prereg_checks["no_formal_execution"] = (
        prereg["formal_execution_started"] is False
        and prereg["candidate_implemented"] is False
        and prereg["implementation_identity"]["formal_source_locked"] is False
    )
    prereg_checks["git_sha_at_freeze"] = (
        prereg["sources"]["git_sha_at_freeze"] == observed_git_sha == EXPECTED_GIT_SHA
    )

    scope_status = (
        "PASS_OPE5_02_FOUR_QUESTIONS_PREREG_AUTHORIZED"
        if all(scope_checks.values())
        else "FAIL"
    )
    prereg_status = (
        "PASS_FROZEN_BEFORE_IMPLEMENTATION_NO_BOARD_EXECUTION"
        if all(prereg_checks.values())
        else "FAIL"
    )
    scope_receipt = {
        "schema_version": 1,
        "stage": "OPE5-02-FP16-GEMV-SCOPE",
        "status": scope_status,
        "recorded_at": scope["frozen_at"],
        "scope": {
            "path": str(scope_path.relative_to(repo)),
            "sha256": sha256_file(scope_path),
        },
        "source_hashes_at_freeze": scope_source_rows,
        "checks": scope_checks,
        "check_summary": {
            "passed": sum(scope_checks.values()),
            "total": len(scope_checks),
        },
        "selection": {
            "candidate_id": "OPE5-02",
            "backlog_rank": 2,
            "backlog_score": candidate_score(ope5_02),
            "prior_candidate_status": ope5_03["status"],
        },
        "four_questions": scope["four_questions"],
        "frozen_shape": scope["frozen_shape"],
        "analytic_account": scope["analytic_account"],
        "board_execution_started": False,
        "candidate_implemented": False,
        "next_action": "Freeze and verify the independent OPE5-02 preregistration before implementation.",
        "git_sha": observed_git_sha,
    }
    prereg_receipt = {
        "schema_version": 1,
        "stage": "OPE5-02-FP16-GEMV-PREREG",
        "status": prereg_status,
        "recorded_at": prereg["frozen_at"],
        "preregistration": {
            "path": str(prereg_path.relative_to(repo)),
            "sha256": sha256_file(prereg_path),
        },
        "scope": {
            "path": str(scope_path.relative_to(repo)),
            "sha256": sha256_file(scope_path),
            "status": scope_status,
        },
        "source_hashes_at_freeze": prereg_source_rows,
        "checks": prereg_checks,
        "check_summary": {
            "passed": sum(prereg_checks.values()),
            "total": len(prereg_checks),
        },
        "formal_shape": prereg["formal_shape"],
        "analytic_account": prereg["analytic_account"],
        "correctness_population": {
            "distributions": len(distributions),
            "seeds": len(prereg["correctness_protocol"]["seeds"]),
            "implementations": len(
                prereg["correctness_protocol"]["implementations_checked"]
            ),
            "outputs_per_case": prereg["correctness_protocol"][
                "outputs_checked_per_case"
            ],
            "expected_element_comparisons": expected_element_comparisons,
        },
        "owner_variance": {
            "formal_segments": 1,
            "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
            "same_day_retest_scheduled": False,
            "instability_retry_authorized": False,
        },
        "implementation": {
            "implemented": False,
            "source_locked": False,
            "formal_execution_started": False,
        },
        "next_action": "Implement the three frozen variants, write a source-lock receipt, then execute exactly one formal board segment.",
        "git_sha": observed_git_sha,
    }
    write_json(args.scope_output, scope_receipt)
    write_json(args.prereg_output, prereg_receipt)
    print(
        json.dumps(
            {
                "scope_status": scope_status,
                "scope_checks": f"{sum(scope_checks.values())}/{len(scope_checks)}",
                "prereg_status": prereg_status,
                "prereg_checks": f"{sum(prereg_checks.values())}/{len(prereg_checks)}",
                "candidate": "OPE5-02",
                "board_execution_started": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if scope_status != "FAIL" and prereg_status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
