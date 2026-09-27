#!/usr/bin/env python3
"""Collect and compare the frozen T3 teacher logprob fingerprints.

The three collectors are intentionally separate processes so the 31B HF,
vLLM-BF16, and vLLM-FP8 implementations never share GPU memory.  Every
collector consumes the same frozen token-id JSONL; no tokenizer rendering is
performed during this gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any
import urllib.request


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_samples(path: Path, count: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                token_ids = [int(token_id) for token_id in row["input_ids"]]
                if len(token_ids) < 2:
                    raise ValueError(f"sample {row['uid']} has fewer than two tokens")
                samples.append({"uid": str(row["uid"]), "input_ids": token_ids})
                if len(samples) == count:
                    break
    if len(samples) != count:
        raise ValueError(f"expected {count} samples in {path}, found {len(samples)}")
    return samples


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("torch", "transformers", "trl", "vllm"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def common_receipt(mode: str, samples_path: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "T3_teacher_logprob_consistency",
        "mode": mode,
        "collected_at": datetime.now().astimezone().isoformat(),
        "packages": package_versions(),
        "samples_file": str(samples_path.resolve()),
        "samples_file_sha256": sha256_file(samples_path),
        "sample_count": len(samples),
        "scored_tokens": sum(len(sample["input_ids"]) - 1 for sample in samples),
    }


def record(
    sample: dict[str, Any], actual_logprobs: list[float], top1_ids: list[int], top1_logprobs: list[float]
) -> dict[str, Any]:
    expected = len(sample["input_ids"]) - 1
    if not (len(actual_logprobs) == len(top1_ids) == len(top1_logprobs) == expected):
        raise AssertionError(
            f"fingerprint length mismatch for {sample['uid']}: expected={expected}, "
            f"actual={len(actual_logprobs)}, top1_ids={len(top1_ids)}, top1_logprobs={len(top1_logprobs)}"
        )
    if not all(math.isfinite(value) for value in actual_logprobs + top1_logprobs):
        raise AssertionError(f"non-finite logprob for {sample['uid']}")
    token_bytes = json.dumps(sample["input_ids"], separators=(",", ":")).encode()
    return {
        "uid": sample["uid"],
        "sequence_length": len(sample["input_ids"]),
        "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "actual_logprobs": actual_logprobs,
        "top1_token_ids": top1_ids,
        "top1_logprobs": top1_logprobs,
    }


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10.0) as response:
        return json.load(response)


def collect_vllm(args: argparse.Namespace) -> None:
    from trl.generation.vllm_client import VLLMClient

    samples = load_samples(args.samples, args.count)
    sequences = [sample["input_ids"] for sample in samples]
    base_url = args.teacher_url.rstrip("/")
    health = fetch_json(f"{base_url}/health/")
    world_size = fetch_json(f"{base_url}/get_world_size/")
    client = VLLMClient(base_url=base_url, connection_timeout=60.0)
    result = client.get_sequence_logprobs(
        sequences=sequences,
        prompt_lengths=[1] * len(sequences),
        top_logprobs=1,
        temperature=1.0,
    )
    records = []
    for index, sample in enumerate(samples):
        records.append(
            record(
                sample,
                [float(row[0]) for row in result["actual_logprobs"][index]],
                [int(row[0]) for row in result["logprob_token_ids"][index]],
                [float(row[0]) for row in result["logprobs"][index]],
            )
        )
    receipt = common_receipt(args.label, args.samples, samples)
    receipt.update(
        {
            "teacher_url": base_url,
            "server_model_impl": args.server_model_impl,
            "server_quantization": args.server_quantization,
            "health": health,
            "world_size": world_size,
            "records": records,
        }
    )
    write_json(args.output, receipt)
    print(json.dumps({"status": "PASS_COLLECT", "mode": args.label, "output": str(args.output)}))


def collect_hf(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForImageTextToText

    samples = load_samples(args.samples, args.count)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval()
    records = []
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device="cuda:0")
            attention_mask = torch.ones_like(input_ids)
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=args.use_cache)
            logits = output.logits[0, :-1].float()
            targets = input_ids[0, 1:]
            log_normalizer = torch.logsumexp(logits, dim=-1)
            actual = logits.gather(1, targets[:, None]).squeeze(1) - log_normalizer
            top_values, top_ids = logits.max(dim=-1)
            records.append(
                record(
                    sample,
                    actual.cpu().tolist(),
                    top_ids.cpu().tolist(),
                    (top_values - log_normalizer).cpu().tolist(),
                )
            )
            del input_ids, attention_mask, output, logits, targets, log_normalizer, actual, top_values, top_ids
            if index % 4 == 0:
                print(json.dumps({"mode": args.label, "completed": index, "total": len(samples)}), flush=True)
    receipt = common_receipt(args.label, args.samples, samples)
    receipt.update(
        {
            "model": str(args.model.resolve()),
            "model_config_sha256": sha256_file(args.model / "config.json"),
            "attention_implementation": "sdpa",
            "dtype": "bfloat16",
            "use_cache": args.use_cache,
            "records": records,
        }
    )
    write_json(args.output, receipt)
    print(json.dumps({"status": "PASS_COLLECT", "mode": args.label, "output": str(args.output)}))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def compare_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    equal = 0
    total = 0
    actual_deltas: list[float] = []
    top1_logprob_deltas: list[float] = []
    per_sample = []
    for left_record, right_record in zip(left["records"], right["records"], strict=True):
        for key in ("uid", "sequence_length", "token_ids_sha256"):
            if left_record[key] != right_record[key]:
                raise AssertionError(f"record identity mismatch for {key}")
        sample_equal = sum(
            left_id == right_id
            for left_id, right_id in zip(
                left_record["top1_token_ids"], right_record["top1_token_ids"], strict=True
            )
        )
        sample_total = len(left_record["top1_token_ids"])
        equal += sample_equal
        total += sample_total
        sample_actual = [
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left_record["actual_logprobs"], right_record["actual_logprobs"], strict=True
            )
        ]
        actual_deltas.extend(sample_actual)
        top1_logprob_deltas.extend(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left_record["top1_logprobs"], right_record["top1_logprobs"], strict=True
            )
        )
        per_sample.append(
            {
                "uid": left_record["uid"],
                "tokens": sample_total,
                "top1_agreement": sample_equal / sample_total,
                "mean_abs_delta_actual_logprob": statistics.fmean(sample_actual),
            }
        )
    return {
        "left": left["mode"],
        "right": right["mode"],
        "tokens": total,
        "top1_equal_tokens": equal,
        "top1_agreement": equal / total,
        "actual_logprob_abs_delta": {
            "mean": statistics.fmean(actual_deltas),
            "p50": percentile(actual_deltas, 0.50),
            "p90": percentile(actual_deltas, 0.90),
            "p95": percentile(actual_deltas, 0.95),
            "p99": percentile(actual_deltas, 0.99),
            "max": max(actual_deltas),
        },
        "top1_logprob_abs_delta": {
            "mean": statistics.fmean(top1_logprob_deltas),
            "p99": percentile(top1_logprob_deltas, 0.99),
            "max": max(top1_logprob_deltas),
        },
        "per_sample": per_sample,
    }


def collect_runtime_identity() -> dict[str, Any]:
    def run(command: list[str]) -> str:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()

    return {
        "hostname": run(["hostname"]),
        "pid1_started_at_raw": run(["ps", "-p", "1", "-o", "lstart="]),
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "gpus": run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines(),
    }


def compare(args: argparse.Namespace) -> None:
    inputs = []
    for path in args.inputs:
        value = json.loads(path.read_text(encoding="utf-8"))
        inputs.append((path, value))
    by_mode = {value["mode"]: (path, value) for path, value in inputs}
    required = ("hf_bf16", "vllm_bf16", "vllm_fp8")
    if set(by_mode) != set(required):
        raise ValueError(f"expected modes {required}, got {tuple(by_mode)}")
    baseline_sha = by_mode[required[0]][1]["samples_file_sha256"]
    if any(by_mode[mode][1]["samples_file_sha256"] != baseline_sha for mode in required):
        raise AssertionError("collectors did not consume the same frozen samples file")
    pairs = {}
    for left_mode, right_mode in (
        ("hf_bf16", "vllm_bf16"),
        ("hf_bf16", "vllm_fp8"),
        ("vllm_bf16", "vllm_fp8"),
    ):
        pairs[f"{left_mode}_vs_{right_mode}"] = compare_pair(by_mode[left_mode][1], by_mode[right_mode][1])
    fp8_pair = pairs["vllm_bf16_vs_vllm_fp8"]
    gate_pass = fp8_pair["top1_agreement"] >= args.min_fp8_bf16_top1_agreement
    receipt = {
        "schema_version": 1,
        "stage": "T3_teacher_logprob_three_way_consistency",
        "verified_at": datetime.now().astimezone().isoformat(),
        "status": "PASS" if gate_pass else "FAIL",
        "samples_file_sha256": baseline_sha,
        "sample_count": by_mode[required[0]][1]["sample_count"],
        "scored_tokens": by_mode[required[0]][1]["scored_tokens"],
        "inputs": {
            mode: {"path": str(by_mode[mode][0]), "sha256": sha256_file(by_mode[mode][0])}
            for mode in required
        },
        "pairs": pairs,
        "gate": {
            "metric": "vllm_bf16_vs_vllm_fp8.top1_agreement",
            "minimum": args.min_fp8_bf16_top1_agreement,
            "actual": fp8_pair["top1_agreement"],
            "pass": gate_pass,
        },
        "runtime_identity": collect_runtime_identity(),
    }
    write_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output), "gate": receipt["gate"]}))
    if not gate_pass:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    vllm_parser = subparsers.add_parser("collect-vllm")
    vllm_parser.add_argument("--samples", type=Path, required=True)
    vllm_parser.add_argument("--count", type=int, default=32)
    vllm_parser.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    vllm_parser.add_argument(
        "--label",
        choices=(
            "vllm_bf16",
            "vllm_fp8",
            "vllm_language_only_bf16",
            "vllm_language_only_fp8",
            "vllm_transformers_bf16",
            "vllm_transformers_fp8",
        ),
        required=True,
    )
    vllm_parser.add_argument("--server-model-impl", choices=("vllm", "transformers"), required=True)
    vllm_parser.add_argument("--server-quantization", choices=("none", "fp8"), required=True)
    vllm_parser.add_argument("--output", type=Path, required=True)
    vllm_parser.set_defaults(func=collect_vllm)

    hf_parser = subparsers.add_parser("collect-hf")
    hf_parser.add_argument("--samples", type=Path, required=True)
    hf_parser.add_argument("--count", type=int, default=32)
    hf_parser.add_argument("--model", type=Path, required=True)
    hf_parser.add_argument("--label", choices=("hf_bf16", "hf_bf16_cache"), default="hf_bf16")
    hf_parser.add_argument("--use-cache", action="store_true")
    hf_parser.add_argument("--output", type=Path, required=True)
    hf_parser.set_defaults(func=collect_hf)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--inputs", type=Path, nargs=3, required=True)
    compare_parser.add_argument("--min-fp8-bf16-top1-agreement", type=float, default=0.99)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(func=compare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
