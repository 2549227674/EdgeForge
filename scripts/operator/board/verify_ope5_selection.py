#!/usr/bin/env python3
"""Verify OPE5 backlog adjudication and the selected RMSNorm preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--triage-config",
        type=Path,
        default=Path("configs/operator/board/ope5_backlog_triage.json"),
    )
    parser.add_argument(
        "--rmsnorm-prereg",
        type=Path,
        default=Path("configs/operator/board/ope5_rmsnorm_prereg.json"),
    )
    parser.add_argument("--triage-output", type=Path, required=True)
    parser.add_argument("--prereg-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    triage_path = (repo / args.triage_config).resolve()
    prereg_path = (repo / args.rmsnorm_prereg).resolve()
    triage = read_json(triage_path)
    prereg = read_json(prereg_path)

    triage_checks: dict[str, bool] = {}
    source_hashes: dict[str, dict[str, Any]] = {}
    mutable_status_indexes = {"execution_card", "ope4_handoff"}
    for source_id, source in triage["sources"].items():
        path = repo / source["path"]
        observed = sha256_file(path)
        source_hashes[source_id] = {
            "path": source["path"],
            "freeze_snapshot_sha256": source["sha256"],
            "current_sha256": observed,
            "policy": (
                "FREEZE_SNAPSHOT_MUTABLE_STATUS_INDEX"
                if source_id in mutable_status_indexes
                else "CURRENT_CONTENT_MUST_MATCH_FREEZE"
            ),
        }
        if source_id in mutable_status_indexes:
            triage_checks[f"source_{source_id}_freeze_snapshot_recorded"] = (
                path.is_file()
                and len(source["sha256"]) == 64
                and all(char in "0123456789abcdef" for char in source["sha256"])
            )
        else:
            triage_checks[f"source_{source_id}_sha256"] = observed == source["sha256"]

    candidates = triage["candidates"]
    ranked = sorted(candidates, key=candidate_sort_key)
    candidate_ids = [item["candidate_id"] for item in candidates]
    ranked_ids = [item["candidate_id"] for item in ranked]
    triage_checks["eight_unique_candidates"] = (
        len(candidates) == 8 and len(set(candidate_ids)) == 8
    )
    triage_checks["dimension_ranges"] = all(
        all(isinstance(value, int) and 0 <= value <= 3 for value in item["dimensions"].values())
        for item in candidates
    )
    triage_checks["selected_candidate"] = (
        ranked_ids[0]
        == triage["selection_rule"]["expected_selected_candidate"]
        == "OPE5-03"
    )
    triage_checks["expected_rank_order"] = ranked_ids == [
        "OPE5-03",
        "OPE5-02",
        "OPE5-05",
        "OPE5-01",
        "OPE5-07",
        "OPE5-06",
        "OPE5-04",
        "OPE5-08",
    ]
    triage_checks["selected_four_questions_pass"] = all(
        value.startswith("PASS")
        for value in ranked[0]["four_questions"].values()
    )
    triage_checks["scope_blockers_preserved"] = {
        item["candidate_id"]: item["scope_blocker"] for item in candidates
    } == {
        "OPE5-01": None,
        "OPE5-02": None,
        "OPE5-03": None,
        "OPE5-04": "BLOCKED_AUTHENTICITY_CURRENT_W8A8_CARRIER",
        "OPE5-05": None,
        "OPE5-06": None,
        "OPE5-07": None,
        "OPE5-08": "BLOCKED_UNTIL_QUALITY_EVALUATION_REOPENS",
    }

    terminal_path = repo / triage["sources"]["ope2_terminals"]["path"]
    terminals = [
        json.loads(line)
        for line in terminal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attention = [row for row in terminals if row.get("scope") == "attention_standby"]
    triage_checks["attention_108_scope_na"] = (
        len(attention) == 108
        and all(row.get("terminal_status") == "N/A" for row in attention)
        and all(row.get("terminal_reason") == "N/A_OPE5_STANDBY" for row in attention)
    )
    triage_checks["attention_zero_attempts"] = all(
        row.get("attempts") == [] for row in attention
    )

    census = read_json(repo / triage["sources"]["shape_census"]["path"])
    rmsnorm = census["op_families"]["rmsnorm"]
    decode_rows = [row for row in rmsnorm["shapes"] if row["phase"] == "decode"]
    triage_checks["rmsnorm_semantics_identity"] = (
        rmsnorm["op_semantics_sha"]
        == triage["sources"]["shape_census"]["rmsnorm_semantics_sha256"]
    )
    triage_checks["rmsnorm_population"] = (
        rmsnorm["calls_per_model_forward"] == 302
        and len(decode_rows) == 6
        and sum(row["calls_per_forward"] for row in decode_rows) == 302
        and {row["variant"] for row in decode_rows}
        == {
            "decoder_hidden_norms",
            "q_norm_sliding",
            "q_norm_full",
            "kv_norm_sliding_nonshared",
            "kv_norm_full_nonshared",
            "ple_projection_norm",
        }
    )

    ope2 = read_json(repo / triage["sources"]["ope2_receipt"]["path"])
    ope4 = read_json(repo / triage["sources"]["ope4_receipt"]["path"])
    pc_op2 = read_json(repo / triage["sources"]["pc_k1_op2_receipt"]["path"])
    pc_op3 = read_json(repo / triage["sources"]["pc_k1_op3_receipt"]["path"])
    pc_op4 = read_json(repo / triage["sources"]["pc_k1_op4_receipt"]["path"])
    triage_checks["upstream_terminal_states"] = (
        ope2["status"]
        == "PASS_RESUMED_PROTOCOL_AMENDED_MATRIX_WITH_CAPABILITY_NA_AND_DISCARDED_BOUNDARY"
        and ope4["measurement_valid"] is True
        and ope4["status"]
        == "VALID_OPE4_SINGLE_SEGMENT_OWNER_WAIVER_MODEL_STAGE_ANCHOR_PRESENT"
        and pc_op2["status"] == "PASS"
        and pc_op3["status"] == "PASS"
        and pc_op4["status"] == "PASS"
    )
    triage_checks["triage_did_not_authorize_board_execution"] = (
        triage["scope"]["board_execution_authorized_by_this_file"] is False
        and triage["scope"]["formal_measurement_started"] is False
    )

    prereg_checks: dict[str, bool] = {}
    prereg_checks["triage_identity"] = (
        prereg["selection"]["triage_path"]
        == str(triage_path.relative_to(repo))
        and prereg["selection"]["triage_sha256"] == sha256_file(triage_path)
    )
    prereg_checks["selected_identity"] = (
        prereg["selection"]["candidate_id"] == "OPE5-03"
        and prereg["selection"]["kernel_id"] == "ope5_neon_fp16_rmsnorm"
    )
    prereg_checks["four_questions_pass"] = all(
        value.startswith("PASS")
        for value in prereg["selection"]["four_questions"].values()
    )

    prereg_source_hashes: dict[str, dict[str, Any]] = {}
    for source_id, source in prereg["sources"].items():
        if "path" not in source:
            continue
        path = repo / source["path"]
        observed = sha256_file(path)
        prereg_source_hashes[source_id] = {
            "path": source["path"],
            "expected_sha256": source["sha256"],
            "observed_sha256": observed,
        }
        prereg_checks[f"source_{source_id}_sha256"] = observed == source["sha256"]

    formal_cases = prereg["formal_cases"]
    case_ids = [case["case_id"] for case in formal_cases]
    prereg_checks["eight_unique_formal_cases"] = (
        len(formal_cases) == 8 and len(set(case_ids)) == 8
    )
    prereg_checks["formal_call_population"] = (
        sum(case["calls_per_model_forward"] for case in formal_cases)
        == prereg["analytic_account"]["calls_per_model_forward"]
        == 302
    )
    prereg_checks["analytic_bytes_population"] = (
        sum(case["weighted_analytic_bytes"] for case in formal_cases)
        == prereg["analytic_account"]["bytes_per_model_forward"]
        == 3_840_000
    )
    prereg_checks["per_case_analytic_formulas"] = all(
        case["analytic_bytes_per_call"]
        == 4 * case["m"] * case["n"]
        + (2 * case["n"] if case["with_scale"] else 0)
        and case["weighted_analytic_bytes"]
        == case["analytic_bytes_per_call"] * case["calls_per_model_forward"]
        for case in formal_cases
    )
    expected_floor_ms = (
        prereg["analytic_account"]["bytes_per_model_forward"]
        / (prereg["sources"]["ope0_bandwidth_ceiling"]["ceiling_gbps_decimal"] * 1e9)
        * 1e3
    )
    expected_fraction = (
        expected_floor_ms
        / prereg["sources"]["ope4_receipt"]["primary_decode_ms_per_token"]
    )
    prereg_checks["anchor_arithmetic"] = (
        abs(expected_floor_ms - prereg["analytic_account"]["ope0_roofline_floor_ms"])
        < 1e-15
        and abs(
            expected_fraction
            - prereg["analytic_account"]["floor_fraction_of_ope4_tpot"]
        )
        < 1e-15
        and abs(
            expected_fraction * 100
            - prereg["analytic_account"]["floor_percent_of_ope4_tpot"]
        )
        < 1e-15
    )
    expected_comparisons = (
        len(formal_cases)
        * len(prereg["correctness_protocol"]["input_distributions"])
        * len(prereg["correctness_protocol"]["seeds"])
        * 2
    )
    prereg_checks["correctness_population"] = (
        expected_comparisons
        == prereg["correctness_protocol"]["expected_tensor_comparisons"]
        == 144
    )
    prereg_checks["single_segment_owner_waiver"] = (
        prereg["measurement_protocol"]["formal_segments"] == 1
        and prereg["measurement_protocol"]["replication_status"]
        == "WAIVED_BY_OWNER_NOT_SATISFIED"
        and prereg["measurement_protocol"]["owner_instruction"]
        == "不需要复测，取消同日复测，只测一次"
    )
    prereg_checks["no_meaningless_smoke"] = prereg["measurement_protocol"][
        "smoke_policy"
    ].startswith("No separate performance smoke")
    prereg_checks["old_tier_b_proxy_not_reintroduced"] = (
        "count*64" in prereg["tier_b"]["bus_access_policy"]
        and "not an analytic-backing acceptance gate"
        in prereg["tier_b"]["bus_access_policy"]
    )
    prereg_checks["implementation_absent_at_freeze"] = all(
        not (repo / prereg["implementation_identity"][key]).exists()
        for key in [
            "candidate_source_path",
            "runner_path",
            "measurement_payload_path",
        ]
    )
    prereg_checks["no_formal_execution"] = (
        prereg["formal_execution_started"] is False
        and prereg["candidate_implemented"] is False
        and prereg["implementation_identity"]["formal_source_locked"] is False
    )
    observed_git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    prereg_checks["git_sha_at_freeze"] = (
        observed_git_sha == prereg["sources"]["git_sha_at_freeze"]
    )

    triage_status = (
        "PASS_OPE5_BACKLOG_REORDERED_RMSNORM_SELECTED"
        if all(triage_checks.values())
        else "FAIL"
    )
    prereg_status = (
        "PASS_FROZEN_BEFORE_IMPLEMENTATION_NO_BOARD_EXECUTION"
        if all(prereg_checks.values())
        else "FAIL"
    )
    triage_receipt = {
        "schema_version": 1,
        "stage": "OPE5-BACKLOG-TRIAGE",
        "status": triage_status,
        "recorded_at": triage["frozen_at"],
        "config": {
            "path": str(triage_path.relative_to(repo)),
            "sha256": sha256_file(triage_path),
        },
        "source_hashes_at_freeze": source_hashes,
        "checks": triage_checks,
        "ranked_candidates": [
            {
                "rank": rank,
                "candidate_id": item["candidate_id"],
                "name": item["name"],
                "score": candidate_score(item),
                "scope_blocker": item["scope_blocker"],
                "disposition": item["disposition"],
                "four_questions": item["four_questions"],
            }
            for rank, item in enumerate(ranked, 1)
        ],
        "attention_raw_terminal_adjudication": {
            "rows": len(attention),
            "terminal_status": "N/A",
            "terminal_reason": "N/A_OPE5_STANDBY",
            "attempts": 0,
            "disposition": "SCOPE_NA_NOT_CAPABILITY_REJECTION",
        },
        "selected_candidate": "OPE5-03",
        "board_execution_started": False,
        "git_sha": observed_git_sha,
    }
    prereg_receipt = {
        "schema_version": 1,
        "stage": "OPE5-03-RMSNorm-PREREG",
        "status": prereg_status,
        "recorded_at": prereg["frozen_at"],
        "preregistration": {
            "path": str(prereg_path.relative_to(repo)),
            "sha256": sha256_file(prereg_path),
        },
        "triage": {
            "path": str(triage_path.relative_to(repo)),
            "sha256": sha256_file(triage_path),
            "status": triage_status,
        },
        "source_hashes_at_freeze": prereg_source_hashes,
        "checks": prereg_checks,
        "formal_population": {
            "cases": len(formal_cases),
            "calls_per_model_forward": 302,
            "analytic_bytes_per_model_forward": 3_840_000,
            "randomized_tensor_comparisons": 144,
            "ope0_roofline_floor_ms": expected_floor_ms,
            "floor_percent_of_ope4_tpot": expected_fraction * 100,
        },
        "owner_variance": {
            "formal_segments": 1,
            "replication_status": "WAIVED_BY_OWNER_NOT_SATISFIED",
            "second_segment_scheduled": False,
        },
        "implementation": {
            "implemented": False,
            "source_locked": False,
            "formal_execution_started": False,
        },
        "git_sha": observed_git_sha,
    }
    write_json(args.triage_output, triage_receipt)
    write_json(args.prereg_output, prereg_receipt)
    print(
        json.dumps(
            {
                "triage_status": triage_status,
                "prereg_status": prereg_status,
                "selected": ranked_ids[0],
                "rank_order": ranked_ids,
            },
            ensure_ascii=False,
        )
    )
    return 0 if triage_status != "FAIL" and prereg_status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
