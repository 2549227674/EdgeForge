#!/usr/bin/env python3
"""Verify the latest T3 stack and optionally probe a live teacher server."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


CORE_VERSIONS = {
    "trl": "1.9.2",
    "vllm": "0.27.0",
    "peft": "0.20.0",
    "bitsandbytes": "0.50.0",
}
REQUIRED_SERVER_FIELDS = {
    "teacher_model_server_url",
    "loss_top_k",
    "loss_add_tail",
    "reverse_kl_top_1_mode",
    "beta",
    "num_generations",
    "generation_batch_size",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def read_text_if_present(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        return None


def training_freeze(python: Path | None) -> dict[str, Any] | None:
    if python is None:
        return None
    completed = subprocess.run(
        [str(python), "-m", "pip", "freeze"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "python": str(python),
        "sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "package_count": len(completed.stdout.splitlines()),
    }


def current_freeze(output: Path | None) -> dict[str, Any]:
    completed = subprocess.run(
        [os.sys.executable, "-m", "pip", "freeze"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if output is not None:
        atomic_bytes(output, completed.stdout)
    return {
        "sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "package_count": len(completed.stdout.splitlines()),
        "freeze_path": str(output.resolve()) if output is not None else None,
    }


def gpu_identity() -> list[dict[str, Any]]:
    query = "index,uuid,name,driver_version,memory.total,memory.used"
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    keys = ["index", "uuid", "name", "driver_version", "memory_total_mib", "memory_used_mib"]
    rows = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        row = dict(zip(keys, values, strict=True))
        row["index"] = int(row["index"])
        row["memory_total_mib"] = int(row["memory_total_mib"])
        row["memory_used_mib"] = int(row["memory_used_mib"])
        rows.append(row)
    return rows


def source_identity(obj: Any) -> dict[str, str]:
    path = Path(inspect.getsourcefile(obj) or "").resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def model_identity(model_dir: Path) -> dict[str, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    text_config = config.get_text_config()
    tokenizer_path = model_dir / "tokenizer.json"
    return {
        "path": str(model_dir.resolve()),
        "model_type": config.model_type,
        "architectures": list(config.architectures or []),
        "vocab_size": int(text_config.vocab_size),
        "tokenizer_json_bytes": tokenizer_path.stat().st_size,
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "config_json_sha256": sha256_file(model_dir / "config.json"),
    }


def fetch_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def probe_server(teacher_url: str, teacher_model: Path, vocab_size: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoTokenizer
    from trl.experimental.server_distillation import ServerDistillationTrainer
    from trl.generation.vllm_client import VLLMClient

    health = fetch_json(f"{teacher_url.rstrip('/')}/health/")
    world_size = fetch_json(f"{teacher_url.rstrip('/')}/get_world_size/")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model, local_files_only=True, trust_remote_code=False)
    conversations = [
        [{"role": "user", "content": "Reply with only the integer result of 2+2."}],
        [{"role": "user", "content": "Name the color produced by mixing blue and yellow."}],
    ]
    completions = ["4", "green"]
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    completion_ids: list[list[int]] = []
    for messages, completion in zip(conversations, completions, strict=True):
        # Transformers 5 may return a BatchEncoding from the tokenizing chat
        # template path.  Render first so the server always receives the
        # protocol's required list[list[int]], independent of that API detail.
        rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = tokenizer.encode(rendered_prompt, add_special_tokens=False)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
        sequences.append(list(prompt) + completion_tokens)
        prompt_lengths.append(len(prompt))
        completion_ids.append(completion_tokens)

    client = VLLMClient(base_url=teacher_url, connection_timeout=60.0)
    started = time.monotonic()
    result = client.get_sequence_logprobs(
        sequences=sequences,
        prompt_lengths=prompt_lengths,
        top_logprobs=1,
        temperature=1.0,
    )
    latency_seconds = time.monotonic() - started

    expected_lengths = [len(tokens) for tokens in completion_ids]
    actual_lengths = [len(values) for values in result["actual_logprobs"]]
    topk_lengths = [len(values) for values in result["logprobs"]]
    if actual_lengths != expected_lengths or topk_lengths != expected_lengths:
        raise AssertionError(
            f"teacher response length mismatch: expected={expected_lengths}, "
            f"actual={actual_lengths}, topk={topk_lengths}"
        )

    flat_actual = [float(row[0]) for sample in result["actual_logprobs"] for row in sample]
    flat_topk = [float(row[0]) for sample in result["logprobs"] for row in sample]
    flat_ids = [int(row[0]) for sample in result["logprob_token_ids"] for row in sample]
    if not flat_actual or not all(math.isfinite(value) for value in flat_actual + flat_topk):
        raise AssertionError("teacher server returned empty or non-finite required logprobs")
    if not all(0 <= token_id < vocab_size for token_id in flat_ids):
        raise AssertionError("teacher server returned a token id outside the shared vocabulary")

    batch = len(sequences)
    max_completion = max(expected_lengths)
    teacher_result = {
        "actual_logprobs": torch.full((batch, max_completion), float("-inf"), dtype=torch.float32),
        "topk_logprobs": torch.full((batch, max_completion, 1), float("-inf"), dtype=torch.float32),
        "topk_token_ids": torch.zeros((batch, max_completion, 1), dtype=torch.long),
    }
    labels = torch.full((batch, max_completion), -100, dtype=torch.long)
    tokens = torch.zeros((batch, max_completion), dtype=torch.long)
    for index, length in enumerate(expected_lengths):
        teacher_result["actual_logprobs"][index, :length] = torch.tensor(
            [row[0] for row in result["actual_logprobs"][index]], dtype=torch.float32
        )
        teacher_result["topk_logprobs"][index, :length, 0] = torch.tensor(
            [row[0] for row in result["logprobs"][index]], dtype=torch.float32
        )
        teacher_result["topk_token_ids"][index, :length, 0] = torch.tensor(
            [row[0] for row in result["logprob_token_ids"][index]], dtype=torch.long
        )
        tokens[index, :length] = torch.tensor(completion_ids[index], dtype=torch.long)
        labels[index, :length] = tokens[index, :length]

    student_logits = torch.zeros((batch, max_completion, vocab_size), dtype=torch.float32, requires_grad=True)
    student_log_probs = functional.log_softmax(student_logits, dim=-1)
    trainer = object.__new__(ServerDistillationTrainer)
    trainer.beta = 1.0
    trainer.loss_add_tail = True
    trainer.reverse_kl_top_1_mode = "sampled"
    loss = trainer._compute_server_sparse_top_1_divergence_loss(
        teacher_result=teacher_result,
        student_log_probs=student_log_probs,
        completion_tokens=tokens,
        labels=labels,
    )
    loss.backward()
    gradient = student_logits.grad
    if not math.isfinite(float(loss)) or gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError("TRL server-distillation reverse-KL backward produced non-finite values")

    return {
        "status": "PASS",
        "teacher_url": teacher_url,
        "health": health,
        "world_size": world_size,
        "binary_logprob_latency_seconds": latency_seconds,
        "sequence_count": batch,
        "sequence_lengths": [len(sequence) for sequence in sequences],
        "prompt_lengths": prompt_lengths,
        "completion_lengths": expected_lengths,
        "required_logprobs_finite": len(flat_actual),
        "top1_logprobs_finite": len(flat_topk),
        "loss_kernel": "ServerDistillationTrainer._compute_server_sparse_top_1_divergence_loss",
        "reverse_kl_loss": float(loss.detach()),
        "student_gradient_l2": float(torch.linalg.vector_norm(gradient)),
        "backward_finite": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8000")
    parser.add_argument("--probe-server", action="store_true")
    parser.add_argument("--training-python", type=Path)
    parser.add_argument("--expected-training-freeze-sha256")
    parser.add_argument("--server-wrapper", type=Path)
    parser.add_argument("--freeze-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from trl.experimental.server_distillation import ServerDistillationConfig, ServerDistillationTrainer

    installed = {name: importlib.metadata.version(name) for name in CORE_VERSIONS}
    mismatches = {
        name: {"expected": expected, "actual": installed[name]}
        for name, expected in CORE_VERSIONS.items()
        if installed[name] != expected
    }
    if mismatches:
        raise AssertionError(f"core version mismatch: {mismatches}")

    fields = {item.name for item in dataclasses.fields(ServerDistillationConfig)}
    missing_fields = sorted(REQUIRED_SERVER_FIELDS - fields)
    if missing_fields:
        raise AssertionError(f"ServerDistillationConfig is missing fields: {missing_fields}")

    with tempfile.TemporaryDirectory(prefix="t3-config-gate-") as output_dir:
        config = ServerDistillationConfig(
            output_dir=output_dir,
            report_to="none",
            use_cpu=True,
            bf16=False,
            teacher_model_server_url=args.teacher_url,
            beta=1.0,
            loss_top_k=1,
            loss_add_tail=True,
            reverse_kl_top_1_mode="sampled",
            num_generations=1,
            generation_batch_size=4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_length=4096,
            max_completion_length=1024,
        )
        try:
            ServerDistillationConfig(
                output_dir=output_dir,
                report_to="none",
                use_cpu=True,
                bf16=False,
                teacher_model_server_url=args.teacher_url,
                beta=1.0,
                loss_top_k=2,
            )
        except ValueError as error:
            invalid_topk_rejected = str(error)
        else:
            raise AssertionError("beta=1.0 with loss_top_k=2 was not rejected")

    student = model_identity(args.student_model)
    teacher = model_identity(args.teacher_model)
    if student["vocab_size"] != teacher["vocab_size"]:
        raise AssertionError("student and teacher vocab_size differ")
    if student["tokenizer_json_sha256"] != teacher["tokenizer_json_sha256"]:
        raise AssertionError("student and teacher tokenizer.json SHA-256 differ")

    original_training = training_freeze(args.training_python)
    if (
        original_training is not None
        and args.expected_training_freeze_sha256 is not None
        and original_training["sha256"] != args.expected_training_freeze_sha256
    ):
        raise AssertionError(
            "original training environment changed: "
            f"expected {args.expected_training_freeze_sha256}, got {original_training['sha256']}"
        )

    trl_requirements = importlib.metadata.requires("trl") or []
    declared_vllm_requirements = [requirement for requirement in trl_requirements if requirement.lower().startswith("vllm")]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "T3_latest_stack_api_compatibility",
        "status": "PASS_DYNAMIC" if args.probe_server else "PASS_STATIC",
        "verified_at": datetime.now().astimezone().isoformat(),
        "python": {"version": platform.python_version(), "executable": os.path.realpath(os.sys.executable)},
        "isolated_environment": current_freeze(args.freeze_output),
        "packages": {
            **installed,
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
            "datasets": importlib.metadata.version("datasets"),
        },
        "upstream_declared_vllm_requirements": declared_vllm_requirements,
        "declared_range_exception": {
            "present": True,
            "reason": "TRL 1.9.2 declares vllm<=0.25.1 in its optional vllm extra; owner requested latest vLLM 0.27.0 exploration.",
            "acceptance_condition": "dynamic server protocol and backward smoke must pass",
        },
        "api": {
            "config_class": "trl.experimental.server_distillation.ServerDistillationConfig",
            "trainer_class": "trl.experimental.server_distillation.ServerDistillationTrainer",
            "config_signature": str(inspect.signature(ServerDistillationConfig)),
            "trainer_signature": str(inspect.signature(ServerDistillationTrainer)),
            "required_fields": sorted(REQUIRED_SERVER_FIELDS),
            "missing_fields": missing_fields,
            "effective_config": {
                "teacher_model_server_url": config.teacher_model_server_url,
                "beta": config.beta,
                "loss_top_k": config.loss_top_k,
                "loss_add_tail": config.loss_add_tail,
                "reverse_kl_top_1_mode": config.reverse_kl_top_1_mode,
                "num_generations": config.num_generations,
                "generation_batch_size": config.generation_batch_size,
                "max_length": config.max_length,
                "max_completion_length": config.max_completion_length,
            },
            "invalid_topk_rejected": invalid_topk_rejected,
            "sources": {
                "config": source_identity(ServerDistillationConfig),
                "trainer": source_identity(ServerDistillationTrainer),
            },
        },
        "models": {"student": student, "teacher": teacher, "shared_tokenizer": True},
        "original_training_environment": original_training,
        "runtime_identity": {
            "hostname": platform.node(),
            "pid1_started_at_raw": subprocess.run(
                ["ps", "-p", "1", "-o", "lstart="], check=True, text=True, stdout=subprocess.PIPE
            ).stdout.strip(),
            "cgroup_memory_max": read_text_if_present(Path("/sys/fs/cgroup/memory.max")),
            "cgroup_cpu_max": read_text_if_present(Path("/sys/fs/cgroup/cpu.max")),
            "gpus": gpu_identity(),
        },
        "verifier": source_identity(main),
    }
    if args.server_wrapper is not None:
        receipt["server_wrapper"] = {
            "path": str(args.server_wrapper.resolve()),
            "sha256": sha256_file(args.server_wrapper),
        }
    if args.probe_server:
        receipt["dynamic_probe"] = probe_server(args.teacher_url, args.teacher_model, teacher["vocab_size"])
    atomic_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
