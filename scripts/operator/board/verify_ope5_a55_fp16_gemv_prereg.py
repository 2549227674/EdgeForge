#!/usr/bin/env python3
"""Verify the OPE5-05 A55 FP16 GEMV scope and preregistration."""

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
EXPECTED_A76_STATUS = (
    "VALID_OPE5_02_SINGLE_SEGMENT_OWNER_WAIVER_ISOLATED_4XA76_FP16_GEMV"
)
VARIANTS = ("scalar_fp32", "autovec_o3", "neon_fp16_f32acc")
EVENTS = ("cycles:u", "instructions:u", "l2d_cache_refill:u", "bus_access:u")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    repo: Path, sources: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    rows: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for source_id, source in sources.items():
        if not isinstance(source, dict) or "path" not in source:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scope",
        type=Path,
        default=Path("configs/operator/board/ope5_a55_fp16_gemv_scope.json"),
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path("configs/operator/board/ope5_a55_fp16_gemv_prereg.json"),
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
    board_identity = read_json(
        repo / scope["sources"]["board_environment_identity"]["path"]
    )
    ope1 = read_json(repo / scope["sources"]["ope1_shape_manifest"]["path"])
    ope3 = read_json(repo / scope["sources"]["ope3_receipt"]["path"])
    ope5_03 = read_json(repo / scope["sources"]["ope5_03_receipt"]["path"])
    a76 = read_json(repo / scope["sources"]["ope5_02_receipt"]["path"])
    ope4 = read_json(repo / scope["sources"]["ope4_receipt"]["path"])
    telemetry = read_csv(repo / scope["sources"]["ope5_02_telemetry"]["path"])
    ope1_shape = find_ope1_shape(ope1)
    ope1_fp16 = ope1_shape["analytics_by_type"]["fp16xfp16_to_fp32"]
    ranked = sorted(triage["candidates"], key=candidate_sort_key)
    candidate = next(row for row in ranked if row["candidate_id"] == "OPE5-05")

    scope_checks["scope_status"] = (
        scope["status"] == "PASS_OPE5_05_FOUR_QUESTIONS_PREREG_AUTHORIZED"
    )
    scope_checks["candidate_is_rank_three_and_ready"] = (
        ranked.index(candidate) == 2
        and scope["selection"]["candidate_id"] == "OPE5-05"
        and scope["selection"]["backlog_rank_at_ope5_triage"] == 3
        and candidate["disposition"] == "READY_COMPARATOR_AFTER_PRIMARY_A76_ITEM"
        and candidate["scope_blocker"] is None
        and candidate_score(candidate) == 10
    )
    scope_checks["prior_a76_primary_valid"] = (
        a76["measurement_valid"] is True
        and a76["status"] == EXPECTED_A76_STATUS
        and a76["evidence_strength"]["formal_segments"] == 1
        and a76["evidence_strength"]["same_day_retest_performed"] is False
    )
    scope_checks["alternatives_remain_separate"] = (
        ope3["isolated_measurement_pass"] is True
        and ope5_03["measurement_valid"] is True
        and scope["selection"]["selected_primary"]
        == "OPE5-02 four-A76 FP16 GEMV"
        and scope["primary_choice_rationale"]["a76_rerun"] is False
    )
    scope_checks["four_questions_pass"] = all(
        item["status"].startswith("PASS") for item in scope["four_questions"].values()
    )
    frozen = scope["frozen_shape"]
    scope_checks["shape_frequency_and_semantics_match_ope1"] = (
        ope1_shape["logical_mkn"] == [1, 10240, 2560]
        and [frozen["m"], frozen["k"], frozen["n"]] == [1, 10240, 2560]
        and frozen["calls_per_model_forward"]
        == ope1_shape["frequency"]["calls_per_model_forward"]
        == 42
        and frozen["observed_model_forwards"]
        == ope1_shape["frequency"]["observed_model_forwards"]
        == 21694
        and frozen["observed_invocations"]
        == ope1_shape["frequency"]["observed_invocations"]
        == 911148
        and frozen["activation_dtype"] == "IEEE float16"
        and frozen["weight_dtype"] == "IEEE float16"
        and frozen["accumulator_output_dtype"] == "float32"
    )
    account = scope["analytic_account"]
    io = ope1_fp16["logical_io_bytes"]
    expected_floor_us = account["total_bytes_per_call"] / 23.1951846685 / 1e9 * 1e6
    scope_checks["analytic_account_matches_ope1"] = (
        account["activation_read_bytes_per_call"] == io["A_read"] == 20480
        and account["weight_read_bytes_per_call"] == io["B_read"] == 52428800
        and account["output_write_bytes_per_call"] == io["C_write"] == 10240
        and account["total_bytes_per_call"] == io["total"] == 52459520
        and account["ops_per_call"] == ope1_fp16["logical_ops"] == 52428800
        and close(account["ops_per_byte"], 52428800 / 52459520)
        and close(account["ope0_floor_us_per_call"], expected_floor_us)
    )
    anchor = scope["a76_primary_anchor"]
    receipt_timing = a76["timing"]
    receipt_perf = a76["tier_b"]["variants"]["neon_fp16_f32acc"]
    scope_checks["a76_anchor_exact"] = (
        anchor["status"] == a76["status"] == EXPECTED_A76_STATUS
        and anchor["variant_p50_us"]
        == {name: receipt_timing["variants"][name]["p50_us"] for name in VARIANTS}
        and close(
            anchor["candidate_analytic_gbps"],
            receipt_timing["variants"]["neon_fp16_f32acc"]["analytic_gbps"],
        )
        and close(
            anchor["candidate_cycles_per_invocation"],
            receipt_perf["per_invocation"]["cycles:u"],
        )
        and close(
            anchor["candidate_instructions_per_invocation"],
            receipt_perf["per_invocation"]["instructions:u"],
        )
        and anchor["recorded_max_frequency_khz"] == 2352000
    )
    features = set(board_identity["identity"]["cpu_features_first_core"])
    max_policy0 = {int(row["policy0_max_khz"]) for row in telemetry}
    a55_design = scope["a55_design_identity"]
    scope_checks["existing_a55_capability_evidence"] = (
        set(a55_design["required_features"]) <= features
        and a55_design["expected_cpu_cores"] == [0, 1, 2, 3]
        and a55_design["expected_cpufreq_policy"] == "policy0"
        and a55_design["observed_policy_max_frequency_khz"] == 1800000
        and max_policy0 == {1800000}
        and a55_design["compiler_target"] == "-mcpu=cortex-a55"
    )
    scope_checks["prelock_topology_assertions_frozen"] = (
        len(a55_design["prelock_assertions"]) == 5
        and any("related_cpus" in item for item in a55_design["prelock_assertions"])
        and any("perf events" in item for item in a55_design["prelock_assertions"])
    )
    scope_checks["ope4_is_limit_not_attribution"] = (
        ope4["measurement_valid"] is True
        and "does not identify the OPE4" in scope["four_questions"]["B_bottleneck"]["limit"]
        and "not evidence of closed-runtime placement" in scope["claim_limits"][0]
    )
    scope_checks["no_board_connection_execution_or_implementation"] = (
        scope["board_connection_performed"] is False
        and scope["board_execution_started"] is False
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
        prereg["selection"]["candidate_id"] == "OPE5-05"
        and prereg["selection"]["kernel_id"] == "ope5_a55_neon_fp16_gemv"
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
        and close(prereg_account["ops_per_byte"], 52428800 / 52459520)
        and prereg_account["ope0_shared_platform_ceiling_gbps"] == 23.1951846685
        and close(
            prereg_account["ope0_floor_us_per_call"],
            52459520 / 23.1951846685 / 1e9 * 1e6,
        )
    )
    prereg_anchor = prereg["immutable_a76_anchor"]
    prereg_checks["immutable_a76_anchor_exact"] = (
        prereg_anchor["receipt_sha256"]
        == sha256_file(repo / prereg_anchor["receipt_path"])
        == prereg["sources"]["ope5_02_receipt"]["sha256"]
        and prereg_anchor["status"] == EXPECTED_A76_STATUS
        and prereg_anchor["variant_p50_us"] == anchor["variant_p50_us"]
        and close(
            prereg_anchor["candidate_cycles_per_invocation"],
            anchor["candidate_cycles_per_invocation"],
        )
        and "No A76 benchmark" in prereg_anchor["reuse_policy"]
    )
    a55 = prereg["a55_platform_contract"]
    prereg_checks["a55_topology_and_frequency_contract"] = (
        formal_shape["cpu_cores"] == [0, 1, 2, 3]
        and formal_shape["threads"] == 4
        and a55["cpu_cores"] == [0, 1, 2, 3]
        and a55["cpufreq_policy"] == "policy0"
        and a55["expected_max_frequency_khz"] == 1800000
        and a55["compiler_target"] == "-mcpu=cortex-a55"
        and set(a55["required_features"]) <= features
        and close(a55["a76_to_a55_max_frequency_ratio"], 2352000 / 1800000)
        and len(a55["prelock_read_only_gates"]) == 5
    )
    provenance = prereg["implementation_provenance"]
    prereg_checks["source_lineage_allowlist"] = (
        provenance["a76_source_sha256"]
        == sha256_file(repo / provenance["a76_source_path"])
        and len(provenance["allowed_source_changes"]) == 3
        and "Kernel bodies" in provenance["forbidden_source_changes"]
        and "-mcpu=cortex-a55" in provenance["runner_change"]
        and "taskset 0-3" in provenance["runner_change"]
    )
    distributions = prereg["correctness_protocol"]["input_distributions"]
    expected_comparisons = (
        len(prereg["correctness_protocol"]["implementations_checked"])
        * len(distributions)
        * len(prereg["correctness_protocol"]["seeds"])
        * prereg["correctness_protocol"]["outputs_checked_per_case"]
    )
    exact = next(row for row in distributions if row["id"] == "quarter_grid_exact")
    random_rows = [row for row in distributions if row["id"] != "quarter_grid_exact"]
    prereg_checks["correctness_population_and_tolerances"] = (
        prereg["correctness_protocol"]["must_complete_before_timing"] is True
        and prereg["correctness_protocol"]["oracle"].startswith("Independent FP64")
        and tuple(prereg["correctness_protocol"]["implementations_checked"])
        == VARIANTS
        and prereg["correctness_protocol"]["seeds"] == [20260924, 20260925, 20260926]
        and expected_comparisons
        == prereg["correctness_protocol"]["expected_element_comparisons"]
        == 69120
        and exact["atol"] == 0.0
        and exact["rtol"] == 0.0
        and all(row["atol"] == 0.05 and row["rtol"] == 0.001 for row in random_rows)
    )
    prereg_checks["three_distinct_a55_baselines"] = set(prereg["baselines"]) == set(VARIANTS)
    protocol = prereg["measurement_protocol"]
    prereg_checks["four_a55_fixed_partition"] = (
        protocol["primary_cpus"] == [0, 1, 2, 3]
        and protocol["taskset"] == "0-3"
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
    prereg_checks["governor_and_telemetry_contract"] = (
        protocol["governors_during_window"]
        == {
            "policy0": "performance",
            "policy4": "unchanged/read-only",
            "policy6": "unchanged/read-only",
            "npu": "unchanged/read-only",
        }
        and protocol["telemetry_hz"] == 1
        and "policy0" in protocol["frequency_gate"]
        and "policy4/policy6/NPU" in protocol["frequency_gate"]
    )
    prereg_checks["single_segment_owner_waiver_no_retry"] = (
        protocol["formal_segments"] == 1
        and protocol["owner_instruction"] == EXPECTED_OWNER_INSTRUCTION
        and protocol["replication_status"] == "WAIVED_BY_OWNER_NOT_SATISFIED"
        and "no automatic retry" in protocol["stability_gate"]
        and "Do not silently rerun" in protocol["failure_policy"]
    )
    prereg_checks["no_smoke_or_a76_rerun"] = (
        protocol["smoke_policy"]
        == "No separate performance smoke, no reduced-shape timing, no A76 rerun, and no second A55 formal segment. Randomized correctness is the only pre-timing benchmark execution gate."
    )
    tier_a = prereg["tier_a"]
    prereg_checks["cluster_comparison_formulas_frozen"] = (
        "A55_variant_p50_us" in tier_a["a55_to_a76_latency_ratio_formula"]
        and close(2352000 / 1800000, 1.3066666666666666)
        and "do not divide" in tier_a["max_frequency_ratio_context"]
        and set(tier_a["cluster_direction_classification"])
        == {"a55_faster", "parity", "a55_slower"}
    )
    tier_b = prereg["tier_b"]
    prereg_checks["tier_b_contract_no_bus_multiplier"] = (
        tuple(tier_b["stat_events"]) == EVENTS
        and tuple(tier_b["variants"]) == VARIANTS
        and "29466634.99509323" in tier_b["primary_cross_cluster_metric"]
        and "not directly ratioed" in tier_b["cross_microarchitecture_policy"]
        and "Do not multiply" in tier_b["bus_access_policy"]
        and "by 8 or 64" in tier_b["bus_access_policy"]
        and tier_b["perf_record"].startswith("Not required")
    )
    implementation_paths = [
        repo / prereg["implementation_identity"][key]
        for key in ("candidate_source_path", "runner_path", "measurement_payload_path")
    ]
    future_lock = repo / "results/operator/board/ope5/ope5_a55_fp16_gemv_source_lock_20260921.json"
    future_segments = list(
        (repo / "results/operator/board/ope5").glob("ope5_a55_fp16_gemv_segment*")
    )
    prereg_checks["implementation_absent_at_freeze"] = (
        all(not path.exists() for path in implementation_paths)
        and not future_lock.exists()
        and not future_segments
    )
    prereg_checks["no_board_connection_or_formal_execution"] = (
        prereg["board_connection_performed"] is False
        and prereg["formal_execution_started"] is False
        and prereg["candidate_implemented"] is False
        and prereg["implementation_identity"]["formal_source_locked"] is False
    )
    prereg_checks["git_sha_at_freeze"] = (
        prereg["sources"]["git_sha_at_freeze"] == observed_git_sha == EXPECTED_GIT_SHA
    )

    scope_status = (
        "PASS_OPE5_05_FOUR_QUESTIONS_PREREG_AUTHORIZED"
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
        "stage": "OPE5-05-A55-FP16-GEMV-SCOPE",
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
            "candidate_id": "OPE5-05",
            "backlog_rank": 3,
            "backlog_score": candidate_score(candidate),
            "selected_primary": scope["selection"]["selected_primary"],
            "prior_primary_status": a76["status"],
        },
        "four_questions": scope["four_questions"],
        "frozen_shape": scope["frozen_shape"],
        "a55_design_identity": scope["a55_design_identity"],
        "a76_primary_anchor": scope["a76_primary_anchor"],
        "board_connection_performed": False,
        "board_execution_started": False,
        "candidate_implemented": False,
        "next_action": "Freeze and verify the independent OPE5-05 A55 preregistration before implementation.",
        "git_sha": observed_git_sha,
    }
    prereg_receipt = {
        "schema_version": 1,
        "stage": "OPE5-05-A55-FP16-GEMV-PREREG",
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
        "immutable_a76_anchor": prereg["immutable_a76_anchor"],
        "correctness_population": {
            "distributions": len(distributions),
            "seeds": len(prereg["correctness_protocol"]["seeds"]),
            "implementations": len(
                prereg["correctness_protocol"]["implementations_checked"]
            ),
            "outputs_per_case": prereg["correctness_protocol"]["outputs_checked_per_case"],
            "expected_element_comparisons": expected_comparisons,
        },
        "owner_variance": {
            "formal_segments": 1,
            "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
            "a76_rerun_authorized": False,
            "same_day_a55_retest_scheduled": False,
            "instability_retry_authorized": False,
        },
        "implementation": {
            "implemented": False,
            "source_locked": False,
            "formal_execution_started": False,
        },
        "board_connection_performed": False,
        "next_action": "Implement the allowlisted A55 source derivative and one-shot runner, complete read-only topology/codegen audit and source-lock, then execute exactly one A55 formal segment.",
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
                "candidate": "OPE5-05",
                "selected_primary": "OPE5-02 FP16 GEMV",
                "board_connection_performed": False,
                "board_execution_started": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if scope_status != "FAIL" and prereg_status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
