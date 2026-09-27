#!/usr/bin/env python3
"""Independently verify the OPE5-01 attention scope and preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCOPE_STATUS = "PASS_OPE5_01_FOUR_QUESTIONS_DISTINCT_SEQUENCE_PREREG_AUTHORIZED"
PREREG_STATUS = "PASS_FROZEN_BEFORE_IMPLEMENTATION_NO_BOARD_EXECUTION"
EXPECTED_GIT_SHA = "213460d88311cba5d84add4d3ee57aae26716c6d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_sources(repo: Path, sources: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, bool]]:
    rows: list[dict[str, str]] = []
    checks: dict[str, bool] = {}
    for name, item in sources.items():
        if name == "git_sha_at_freeze":
            continue
        path = repo / item["path"]
        observed = sha256(path)
        rows.append(
            {
                "name": name,
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "observed_sha256": observed,
            }
        )
        checks[f"source_sha256_{name}"] = observed == item["sha256"]
    return rows, checks


def attention_candidate(triage: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in triage["candidates"] if row["candidate_id"] == "OPE5-01")


def expected_output_comparisons() -> int:
    per_distribution_seed = 8 * 256 + 5 * 8 * 512
    return per_distribution_seed * 2 * 2 * 3


def all_pass(checks: dict[str, bool]) -> bool:
    return all(checks.values())


def receipt(
    *,
    stage: str,
    status: str,
    recorded_at: str,
    checks: dict[str, bool],
    sources: list[dict[str, str]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    passed = sum(bool(value) for value in checks.values())
    return {
        "schema_version": 1,
        "stage": stage,
        "status": status if passed == len(checks) else "FAIL",
        "recorded_at": recorded_at,
        "check_summary": {"passed": passed, "total": len(checks)},
        "checks": checks,
        "sources": sources,
        **extra,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scope",
        type=Path,
        default=Path("configs/operator/board/ope5_attention_sequence_scope.json"),
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path("configs/operator/board/ope5_attention_sequence_prereg.json"),
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
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()

    scope_sources, scope_checks = resolve_sources(repo, scope["sources"])
    triage = read_json(repo / scope["sources"]["ope5_backlog_triage"]["path"])
    shape_census = read_json(repo / scope["sources"]["shape_census"]["path"])
    ope1 = read_json(repo / scope["sources"]["ope1_shape_manifest"]["path"])
    ope2 = read_json(repo / scope["sources"]["ope2_receipt"]["path"])
    terminals = read_jsonl(repo / scope["sources"]["ope2_matrix_terminals"]["path"])
    board = read_json(repo / scope["sources"]["board_environment_identity"]["path"])
    board_completion = read_json(
        repo / scope["sources"]["board_environment_completion"]["path"]
    )
    k3 = read_json(repo / scope["sources"]["pc_k3_preregistration"]["path"])
    k3_correctness = read_json(
        repo / scope["sources"]["pc_k3_correctness_receipt"]["path"]
    )
    k3_reconnect = read_json(
        repo / scope["sources"]["pc_k3_reconnect_adjudication"]["path"]
    )
    ope4 = read_json(repo / scope["sources"]["ope4_receipt"]["path"])
    candidate = attention_candidate(triage)
    standby = ope1["decode_attention_standby"]
    attention_rows = [row for row in terminals if row.get("scope") == "attention_standby"]

    scope_checks["scope_status"] = scope["status"] == SCOPE_STATUS
    scope_checks["candidate_reopens_deferred_item"] = (
        candidate["disposition"] == "DEFER_DISTINCT_SEQUENCE_CARRIER_REQUIRED"
        and scope["selection"]["candidate_id"] == "OPE5-01"
        and scope["selection"]["previous_disposition"]
        == "DEFER_DISTINCT_SEQUENCE_CARRIER_REQUIRED"
        and scope["selection"]["decision"]
        == "REOPEN_WITH_DISTINCT_HYBRID_SEQUENCE_CARRIER"
    )
    scope_checks["four_questions_pass"] = all(
        item["status"].startswith("PASS")
        for item in scope["four_questions"].values()
    )
    population = scope["frozen_population"]
    scope_checks["ope1_population_exact"] = (
        standby["population"]["decode_forwards"] == population["decode_forwards"] == 21694
        and standby["population"]["min"] == 807
        and standby["population"]["max"] == 7359
        and standby["shapes"][0]["logical_mkn"] == [1, 256, 512]
        and standby["shapes"][0]["frequency"]["layer_calls_per_model_forward"] == 35
        and standby["shapes"][0]["frequency"]["observed_layer_invocations"] == 759290
        and standby["shapes"][1]["frequency"]["layer_calls_per_model_forward"] == 7
        and standby["shapes"][1]["frequency"]["observed_layer_invocations_total_population"] == 151858
        and population["total_layer_calls_per_forward"] == 42
        and population["total_observed_invocations"] == 911148
    )
    census_decode = [
        row
        for row in shape_census["op_families"]["attention"]["shapes"]
        if row["phase"] == "decode"
    ]
    census_sliding = [row for row in census_decode if row["variant"] == "sliding_attention_eager"]
    census_full = [row for row in census_decode if row["variant"] == "full_attention_eager"]
    scope_checks["shape_census_attention_lineage"] = (
        len(census_sliding) == 1
        and sum(row["invocations"] for row in census_sliding) == 759290
        and sum(row["invocations"] for row in census_full) == 151858
        and sum(row["invocations"] for row in census_decode) == 911148
        and all(row["dims"][1] == 8 for row in census_decode)
        and census_sliding[0]["dims"] == [1, 8, 1, 512, 256]
    )
    scope_checks["ope2_attention_is_zero_attempt_reservation"] = (
        ope2["counts"]["by_scope"]["attention_standby"] == 108
        and ope2["counts"]["status_by_scope"]["attention_standby"] == {"N/A": 108}
        and len(attention_rows) == 108
        and all(row["terminal_reason"] == "N/A_OPE5_STANDBY" for row in attention_rows)
        and all(row["attempts"] == [] for row in attention_rows)
    )
    scope_checks["board_fp16_core0_capability_basis"] = (
        board["rknpu"]["fp16_to_fp32_present"] is True
        and board["rknpu"]["librknnrt_version"].startswith("2.3.2")
        and ope2["capability_adjudication"]["core0"] == "RUNNABLE"
        and board_completion["status"] == "PASS_CONTROLLER_CLOCK_PROTOCOL_PERF_AVAILABLE"
    )
    semantic = k3["semantic_contract"]
    scope_checks["k3_semantic_lineage_exact"] = (
        semantic["query_heads"] == 8
        and semantic["key_value_heads"] == 2
        and semantic["gqa_groups"] == 4
        and semantic["scaling"] == 1.0
        and "softmax in FP32" in semantic["operation"]
        and k3["target_population"]["total_projected_invocations"] == 911148
        and k3_correctness["status"] == "PASS"
        and k3_reconnect["status"] == "K3_C02_RECONNECTION_FAIL_CLOSE_DEPLOYMENT_PATH"
    )
    distinct = scope["distinct_sequence_carrier"]
    stages = " ".join(distinct["why_distinct_from_ope2"])
    scope_checks["carrier_is_sequence_not_matmul_replay"] = (
        "softmax" in stages
        and "KV" in stages
        and "QK^T" in stages
        and "PV" in stages
        and distinct["grouped_matmul_shapes"]["qk"] == "M=4,K=D,N=ceil(L/16)*16"
        and distinct["grouped_matmul_shapes"]["pv"] == "M=4,K=ceil(L/32)*32,N=D"
        and distinct["grouped_matmul_shapes"]["core_mask"] == "core0"
    )
    scope_checks["ope4_anchor_is_limit_only"] = (
        ope4["measurement_valid"] is True
        and ope4["anchor"]["model_stage_anchor_present"] is True
        and abs(ope4["anchor"]["measured_primary_decode_ms_per_token"] - 446.1262611013072) < 1e-12
        and "not an RKLLM component measurement" in scope["four_questions"]["B_bottleneck"]["limit"]
    )
    scope_checks["no_scope_execution_or_implementation"] = (
        scope["board_connection_performed"] is False
        and scope["board_execution_started"] is False
        and scope["candidate_implemented"] is False
        and scope["decision"]["implementation_authorized_by_this_file"] is False
        and scope["decision"]["board_execution_authorized_by_this_file"] is False
    )
    scope_checks["git_sha_at_freeze"] = (
        scope["git_sha_at_freeze"] == git_sha == EXPECTED_GIT_SHA
    )

    prereg_sources, prereg_checks = resolve_sources(repo, prereg["sources"])
    prereg_checks["prereg_status"] = prereg["status"] == "PREREGISTERED_FROZEN_NOT_RUN"
    prereg_checks["scope_identity"] = (
        prereg["selection"]["scope_path"] == str(scope_path.relative_to(repo))
        and prereg["selection"]["scope_sha256"] == sha256(scope_path)
        and prereg["sources"]["scope_freeze"]["sha256"] == sha256(scope_path)
    )
    prereg_checks["selected_identity_and_four_questions"] = (
        prereg["selection"]["candidate_id"] == "OPE5-01"
        and prereg["selection"]["kernel_id"] == "ope5_hybrid_gqa_decode_step_v1"
        and prereg["selection"]["four_question_statuses"]
        == {key: value["status"] for key, value in scope["four_questions"].items()}
    )
    contract = prereg["semantic_contract"]
    prereg_checks["semantic_contract_exact"] = (
        contract["query_heads"] == 8
        and contract["key_value_heads"] == 2
        and contract["gqa_groups"] == 4
        and contract["query_length"] == 1
        and contract["scaling"] == 1.0
        and contract["softmax_dtype"] == "float32"
        and contract["probability_carrier_dtype"] == "IEEE float16"
        and contract["output_dtype"] == "IEEE float16"
        and contract["no_physical_repeat_kv"] is True
    )
    kv = prereg["kv_state_contract"]
    prereg_checks["dual_form_kv_layout_frozen"] = (
        kv["sliding"]["head_dim"] == 256
        and kv["sliding"]["logical_tokens"] == 512
        and "[2,1024,256]" in kv["sliding"]["physical_layout"]
        and kv["full"]["head_dim"] == 512
        and kv["full"]["logical_tokens_inclusive"] == [807, 7359]
        and "[2,7360,512]" in kv["full"]["physical_layout"]
    )
    rknpu = prereg["rknpu_contract"]
    prereg_checks["grouped_rknpu_contract_frozen"] = (
        rknpu["api_type"] == "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32"
        and rknpu["core_mask"] == "RKNN_NPU_CORE_0"
        and rknpu["calls_per_attention_step"] == 4
        and rknpu["qk_per_kv_head"]["api_mkn"] == "[4,D,ceil(L/16)*16]"
        and rknpu["pv_per_kv_head"]["api_mkn"] == "[4,ceil(L/32)*32,D]"
        and "invalidates the segment" in rknpu["capability_gate"]
    )
    implementations = prereg["implementations"]
    prereg_checks["three_complete_sequence_implementations"] = (
        set(implementations) >= {
            "cpu_scalar_sequence",
            "cpu_neon_sequence",
            "hybrid_rknpu_sequence",
            "common_timed_boundary",
        }
        and all(
            "complete-sequence" in implementations[name]["role"]
            or name == "hybrid_rknpu_sequence"
            for name in ("cpu_scalar_sequence", "cpu_neon_sequence", "hybrid_rknpu_sequence")
        )
    )
    formal = prereg["formal_population"]
    cases = formal["performance_cases"]
    sliding = [row for row in cases if row["family"] == "sliding"]
    full = [row for row in cases if row["family"] == "full"]
    prereg_checks["performance_population_weights_exact"] = (
        len(cases) == 13
        and len(sliding) == 1
        and len(full) == 12
        and sum(row["projected_invocations"] for row in sliding) == 759290
        and sum(row["projected_invocations"] for row in full) == 151858
        and sum(row["projected_invocations"] for row in cases) == 911148
        and formal["weight_checks"]["total_calls_per_forward"] == 42
    )
    prereg_checks["all_performance_shapes_align_after_padding"] = all(
        row["D"] % 32 == 0
        and ((row["K"] + 15) // 16 * 16) % 16 == 0
        and ((row["K"] + 31) // 32 * 32) % 32 == 0
        for row in cases
    )
    correctness = prereg["formal_correctness_protocol"]
    prereg_checks["correctness_population_and_account_exact"] = (
        len(correctness["representative_cases"]) == 6
        and len(correctness["distributions"]) == 2
        and len(correctness["seeds"]) == 2
        and correctness["logical_trials"] == 24
        and len(correctness["implementations_checked"]) == 3
        and correctness["expected_output_element_comparisons"]
        == expected_output_comparisons()
        == 270336
        and correctness["tolerance"] == {"atol": 0.03125, "rtol": 0.02}
    )
    prereg_checks["state_and_padding_gates_frozen"] = (
        len(correctness["state_microcases"]) == 8
        and any("ring wrap" in item for item in correctness["state_microcases"])
        and any("padded positions" in item for item in correctness["state_microcases"])
        and any("byte-identical" in item for item in correctness["state_microcases"])
        and "Do not tune" in correctness["failure_rule"]
    )
    measurement = prereg["measurement_protocol"]
    prereg_checks["single_segment_no_retest_policy"] = (
        measurement["formal_segments"] == 1
        and measurement["performance_smoke"] is False
        and measurement["automatic_retry"] is False
        and measurement["same_day_retest"] is False
        and measurement["cross_day_retest"] is False
        and measurement["replication_status"] == "WAIVED_BY_OWNER_NOT_SATISFIED"
    )
    prereg_checks["timing_and_interleave_frozen"] = (
        measurement["cpu_affinity"] == [7]
        and measurement["warmup_repetitions_per_case_per_implementation"] == 10
        and measurement["minimum_measured_repetitions_per_case_per_implementation"] == 50
        and measurement["minimum_measured_elapsed_ms_per_case_per_implementation"] == 500
        and measurement["maximum_measured_repetitions_per_case_per_implementation"] == 5000
        and "six permutations" in measurement["interleave"]
        and "CLOCK_MONOTONIC_RAW" in measurement["timer"]
    )
    prereg_checks["telemetry_perf_and_restore_frozen"] = (
        measurement["telemetry"]["system_interval_seconds"] == 1.0
        and measurement["telemetry"]["npu_load_interval_seconds"] == 0.1
        and measurement["tier_b"]["cases"]
        == ["sliding_k512_d256", "full_k3123_d512"]
        and measurement["tier_b"]["events"]
        == ["cycles:u", "instructions:u", "task-clock", "context-switches", "cpu-migrations"]
        and "restore both" in measurement["governor_restore"]
    )
    lock = prereg["implementation_and_lock_policy"]
    prereg_checks["source_lock_before_only_formal_run"] = (
        len(lock["future_files"]) == 3
        and "without executing" in lock["allowed_before_source_lock"][1]
        and "board benchmark execution" in lock["forbidden_before_source_lock"]
        and "sole formal segment" in lock["post_lock_rule"]
        and "No source edit" in lock["post_lock_rule"]
    )
    prereg_checks["no_prereg_execution_or_implementation"] = (
        prereg["board_connection_performed"] is False
        and prereg["formal_execution_started"] is False
        and prereg["candidate_implemented"] is False
    )
    prereg_checks["git_sha_at_freeze"] = (
        prereg["sources"]["git_sha_at_freeze"] == git_sha == EXPECTED_GIT_SHA
    )

    recorded_at = prereg["frozen_at"]
    scope_receipt = receipt(
        stage="OPE5-01-DECODE-ATTENTION-SEQUENCE-SCOPE-VERIFY",
        status=SCOPE_STATUS,
        recorded_at=recorded_at,
        checks=scope_checks,
        sources=scope_sources,
        extra={
            "board_connection_performed": False,
            "candidate_implemented": False,
            "formal_execution_started": False,
            "distinct_carrier": "hybrid_gqa_decode_step_v1",
        },
    )
    prereg_receipt = receipt(
        stage="OPE5-01-DECODE-ATTENTION-SEQUENCE-PREREG-VERIFY",
        status=PREREG_STATUS,
        recorded_at=recorded_at,
        checks=prereg_checks,
        sources=prereg_sources,
        extra={
            "scope_receipt_status": scope_receipt["status"],
            "board_connection_performed": False,
            "implementation": {
                "implemented": False,
                "source_locked": False,
                "formal_execution_started": False,
            },
            "future_formal_segments": 1,
            "performance_smoke_authorized": False,
            "same_day_retest_authorized": False,
            "automatic_retry_authorized": False,
            "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
        },
    )

    for path, payload in (
        (args.scope_output, scope_receipt),
        (args.prereg_output, prereg_receipt),
    ):
        output = path if path.is_absolute() else repo / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "scope_status": scope_receipt["status"],
                "scope_checks": scope_receipt["check_summary"],
                "prereg_status": prereg_receipt["status"],
                "prereg_checks": prereg_receipt["check_summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if all_pass(scope_checks) and all_pass(prereg_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
