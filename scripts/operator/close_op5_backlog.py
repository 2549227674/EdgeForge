#!/usr/bin/env python3
"""Close the current OP5 backlog from terminal machine receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from op0_common import atomic_write_json, now_iso


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/operator/pool/op5_current_backlog_close_receipt_20260916.json"
PATHS = {
    "backlog": ROOT / "results/operator/pool/op5_k2_selection.json",
    "k2": ROOT / "results/operator/pool/k2_r2_day1_receipt_20260916.json",
    "k3": ROOT
    / "results/operator/pool/k3_gqa_reconnect_r5_adjudication_receipt_20260916.json",
    "lm_head_selection": ROOT
    / "results/operator/pool/op5_lm_head_selection_r1_receipt_20260916.json",
    "lm_head_reconnection": ROOT
    / "results/operator/pool/op5_lm_head_reconnect_r1_adjudication_receipt_20260916.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    for name, path in PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing OP5 terminal evidence {name}: {path}")
    backlog = load(PATHS["backlog"])
    by_item = {row["item"]: row for row in backlog["backlog"]}
    inductor = by_item.get("Inductor generated-code comparison")
    kv = by_item.get("KV dequantization")
    if not inductor or inductor["disposition"] != "SUPPORTING_BASELINE_NOT_STANDALONE_KERNEL":
        raise AssertionError("Inductor supporting-baseline disposition drift")
    if not kv or kv["disposition"] != "BLOCKED_AUTHENTICITY":
        raise AssertionError("KV dequant authenticity block drift")

    k2 = load(PATHS["k2"])
    if k2.get("status") != "K2_R2_CLOSED_DAY1_NOT_POSITIVE":
        raise AssertionError("K2 is not terminally closed")
    k3 = load(PATHS["k3"])
    if k3.get("status") != "K3_C02_RECONNECTION_FAIL_CLOSE_DEPLOYMENT_PATH":
        raise AssertionError("K3 is not terminally closed")
    lm_selection = load(PATHS["lm_head_selection"])
    if not (
        lm_selection.get("status")
        == "EXISTING_CUDA_LM_HEAD_MIRROR_RECONNECTION_PROBE_AUTHORIZED_NO_CUSTOM_KERNEL"
        and lm_selection.get("bottleneck_gate") == "FAIL"
        and lm_selection.get("existing_path_eligible") is True
    ):
        raise AssertionError("lm_head selection boundary drift")
    lm_reconnection = load(PATHS["lm_head_reconnection"])
    if not (
        lm_reconnection.get("status")
        == "LM_HEAD_C02_LOCKSTEP_RECONNECTION_PASS_LOCAL_PATH_AUTHORIZED_AFTER_CLOCK_GATE_ADJUDICATION"
        and lm_reconnection.get("deployment_authorized") is True
    ):
        raise AssertionError("lm_head local path is not authorized")

    payload = {
        "schema_version": 1,
        "stage": "OP5_CURRENT_BACKLOG_CLOSE",
        "timestamp": now_iso(),
        "status": "OP5_CURRENT_BACKLOG_CLOSED_NO_REMAINING_EXECUTABLE_CANDIDATE",
        "sha_verification": "SKIPPED_BY_OWNER_DIRECTION",
        "all_current_items_disposed": True,
        "items": [
            {
                "rank": 1,
                "item": "NF4 double-quant decode GEMV",
                "terminal_status": k2["status"],
                "disposition": "CLOSED_NEGATIVE_NO_DAY2_NCU_OR_RECONNECTION",
                "evidence": str(PATHS["k2"]),
            },
            {
                "rank": 2,
                "item": "GQA decode attention (SWA 512 + full <=10k)",
                "terminal_status": k3["status"],
                "disposition": "CLOSED_C02_TOKEN_EQUIVALENCE_FAILURE",
                "evidence": str(PATHS["k3"]),
            },
            {
                "rank": 3,
                "item": "Inductor generated-code comparison",
                "terminal_status": inductor["disposition"],
                "disposition": "COVERED_AS_K2_K3_SUPPORTING_BASELINE_NO_STANDALONE_KERNEL",
                "evidence": str(PATHS["backlog"]),
            },
            {
                "rank": 4,
                "item": "lm_head 262k vocab + softcap fusion",
                "terminal_status": lm_reconnection["status"],
                "disposition": "LOCAL_C02_EAGER_CUDA_MIRROR_AUTHORIZED_NO_CUSTOM_KERNEL",
                "evidence": str(PATHS["lm_head_reconnection"]),
                "constraints": lm_reconnection["constraints"],
            },
            {
                "rank": 5,
                "item": "KV dequantization",
                "terminal_status": kv["disposition"],
                "disposition": "NO_QUANTIZED_KV_DEQUANT_OP_IN_CURRENT_CARRIER",
                "evidence": str(PATHS["backlog"]),
            },
        ],
        "closure_gates": {
            "k2_terminal": "PASS",
            "k3_terminal": "PASS",
            "inductor_not_a_standalone_candidate": "PASS",
            "lm_head_terminal_local_path": "PASS",
            "lm_head_custom_kernel_remains_closed": "PASS",
            "kv_dequant_authenticity_block_preserved": "PASS",
            "no_unresolved_current_backlog_item": "PASS",
        },
        "boundaries": [
            "The lm_head authorization is limited to the frozen local C02 PyTorch eager-mirror path and depends on the separate post-run clock-gate adjudication.",
            "The observed lm_head NVML margin to 7.5 GiB was narrow; deployment requires a dedicated GPU and runtime free-memory guard.",
            "No OP5 result is a cross-day or independent replication claim.",
            "KV dequant can reopen only after a carrier with an authentic quantized-KV dequant operation is observed and newly preregistered.",
            "OP-E remains blocked and is not opened by this receipt.",
        ],
        "next_action": (
            "No further execution is authorized inside the current OP5 backlog. A new kernel requires "
            "new census/authenticity evidence and a new preregistration; OP-E still requires owner "
            "authorization and its own execution card."
        ),
    }
    atomic_write_json(OUTPUT, payload)
    print(f"[{now_iso()}] OP5 backlog closed: {payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
