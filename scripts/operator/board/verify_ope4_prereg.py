#!/usr/bin/env python3
"""Independently verify the frozen OPE4 workload and measurement protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    prompt_manifest_path = repo / "configs/operator/board/ope4_five_prompt_manifest.json"
    prereg_path = repo / "configs/operator/board/ope4_diag_protocol.json"
    prompt_manifest = json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    archive = repo / prompt_manifest["source"]["archive"]
    checks["source_archive_sha256"] = (
        sha256_file(archive) == prompt_manifest["source"]["archive_sha256"]
    )
    with zipfile.ZipFile(archive) as bundle:
        member_bytes = bundle.read(prompt_manifest["source"]["member"])
    checks["source_member_sha256"] = (
        sha256_bytes(member_bytes) == prompt_manifest["source"]["member_sha256"]
    )
    source = member_bytes.decode("utf-8")
    match = re.search(
        r"printf '%s\\n' \\\n(?P<body>.*?)\n\s*\| timeout",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        extracted: list[str] = []
    else:
        extracted = re.findall(
            r"^\s*'([^']*)'\s*\\\s*$", match.group("body"), flags=re.MULTILINE
        )
    expected_prompts = [entry["text"] for entry in prompt_manifest["prompts"]]
    checks["source_literal_extraction"] = extracted == expected_prompts + ["exit"]

    payload_path = repo / prompt_manifest["payload_file"]
    payload = payload_path.read_bytes()
    checks["payload_sha256"] = sha256_bytes(payload) == prompt_manifest["payload_sha256"]
    checks["payload_bytes"] = len(payload) == prompt_manifest["payload_bytes"]
    checks["payload_trailing_lf"] = payload.endswith(b"\n")
    decoded_prompts = payload.decode("utf-8").splitlines()
    checks["payload_prompt_sequence"] = decoded_prompts == expected_prompts
    checks["prompt_count"] = len(decoded_prompts) == prompt_manifest["prompt_count"] == 5
    checks["per_prompt_identity"] = all(
        entry["prompt_id"] == index
        and len(entry["text"].encode("utf-8")) == entry["utf8_bytes"]
        and sha256_bytes(entry["text"].encode("utf-8")) == entry["sha256"]
        for index, entry in enumerate(prompt_manifest["prompts"], 1)
    )

    prompt_manifest_sha = sha256_file(prompt_manifest_path)
    checks["prereg_prompt_manifest_sha256"] = (
        prereg["workload"]["prompt_manifest_sha256"] == prompt_manifest_sha
    )
    checks["prereg_payload_sha256"] = (
        prereg["workload"]["payload_sha256"] == sha256_file(payload_path)
    )
    checks["single_segment_owner_waiver"] = (
        prereg["measurement_protocol"]["formal_segments"] == 1
        and prereg["measurement_protocol"]["carrier_order"]
        == "c02_a2-official_a1"
        and prereg["owner_variance"]["replication_status"]
        == "WAIVED_BY_OWNER_NOT_SATISFIED"
    )
    checks["manual_greedy_contract"] = (
        prereg["measurement_protocol"]["top_k"] == 1
        and prereg["measurement_protocol"]["keep_history"] == 0
        and prereg["measurement_protocol"]["prompt_cache"] is False
        and prereg["measurement_protocol"]["thinking"] is False
        and prereg["measurement_protocol"]["max_new_tokens"] == 96
        and prereg["measurement_protocol"]["max_context_len"] == 16384
        and prereg["measurement_protocol"]["template_prefix"]
        == "<bos><|turn>user\n"
        and prereg["measurement_protocol"]["template_postfix"]
        == "<turn|>\n<|turn>model\n"
    )
    checks["carrier_roles"] = (
        prereg["carriers"]["c02_a2"]["role"] == "primary"
        and prereg["carriers"]["official_a1"]["role"]
        == "read_only_comparison"
    )
    checks["anchor_accounts"] = (
        prereg["anchor"]["core_projection_weight_bytes"] == 3_890_216_960
        and prereg["anchor"]["all_non_lmhead_linear_weight_bytes"]
        == 3_972_792_320
    )
    ceiling_bytes_per_second = prereg["anchor"]["ope0_tier_a_ceiling_gbps"] * 1e9
    derived_core_ms = (
        prereg["anchor"]["core_projection_weight_bytes"]
        / ceiling_bytes_per_second
        * 1e3
    )
    derived_all_ms = (
        prereg["anchor"]["all_non_lmhead_linear_weight_bytes"]
        / ceiling_bytes_per_second
        * 1e3
    )
    checks["anchor_arithmetic"] = (
        abs(derived_core_ms - prereg["anchor"]["core_projection_predicted_lower_bound_ms"])
        < 1e-12
        and abs(
            derived_all_ms
            - prereg["anchor"]["all_non_lmhead_linear_predicted_lower_bound_ms"]
        )
        < 1e-12
    )

    implementation_hashes: dict[str, str] = {}
    implementation_pairs = [
        ("runner_source", "runner_source_sha256"),
        ("formal_runner", "formal_runner_sha256"),
        ("measurement_payload", "measurement_payload_sha256"),
        ("npu_sampler", "npu_sampler_sha256"),
    ]
    for path_key, hash_key in implementation_pairs:
        observed = sha256_file(repo / prereg["implementation"][path_key])
        implementation_hashes[path_key] = observed
        checks[f"implementation_{path_key}_sha256"] = (
            observed == prereg["implementation"][hash_key]
        )
    for path, hash_key in [
        ("scripts/operator/board/telemetry_sampler.sh", "telemetry_sampler_sha256"),
        ("scripts/operator/board/governor_guard.sh", "governor_guard_sha256"),
    ]:
        observed = sha256_file(repo / path)
        implementation_hashes[path] = observed
        checks[f"implementation_{Path(path).stem}_sha256"] = (
            observed == prereg["implementation"][hash_key]
        )

    status = "PASS" if all(checks.values()) else "FAIL"
    receipt = {
        "schema": "edgeforge-ope4-prereg-verification-v1",
        "stage": "OPE4",
        "status": status,
        "label": "board-diagnostic-1core",
        "prereg": {
            "path": str(prereg_path.relative_to(repo)),
            "sha256": sha256_file(prereg_path),
        },
        "prompt_manifest": {
            "path": str(prompt_manifest_path.relative_to(repo)),
            "sha256": prompt_manifest_sha,
        },
        "implementation_hashes": implementation_hashes,
        "checks": checks,
        "formal_execution_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "receipt": str(args.output)}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
