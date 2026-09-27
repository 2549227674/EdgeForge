"""Deterministic hybrid CPU/GPU replay primitives for operator-plane OP3.

Gemma 4's two very large embedding tables and tied language-model head remain
on CPU, matching the validated P0 carrier. The active language stack and KV
cache remain on CUDA. Every request receives a fresh cache and generates the
exact frozen reference completion budget with greedy argmax decoding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch

from k1_custom_op import K1InvocationTracker
from load_smoke_nf4 import get_submodule, memory_snapshot
from op0_common import now_iso, sha256, summarize, text_sha256


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_verified_tapes(
    root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed_manifest_sha = sha256(manifest_path)
    if observed_manifest_sha != expected_manifest_sha256:
        raise AssertionError(
            "tapes manifest hash mismatch: "
            f"{observed_manifest_sha} != {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tapes: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    for entry in manifest["tapes"]:
        tape_path = root / entry["path"]
        observed = sha256(tape_path)
        if observed != entry["sha256"]:
            raise AssertionError(
                f"tape hash mismatch for {entry['path']}: {observed} != {entry['sha256']}"
            )
        tape = json.loads(tape_path.read_text(encoding="utf-8"))
        if tape.get("tape_id") != entry["tape_id"]:
            raise AssertionError(f"tape ID mismatch: {tape_path}")
        if len(tape.get("requests", [])) != int(entry["requests"]):
            raise AssertionError(f"tape request count mismatch: {tape_path}")
        for request in tape["requests"]:
            request_id = request["request_id"]
            if request_id in request_ids:
                raise AssertionError(f"duplicate request ID: {request_id}")
            request_ids.add(request_id)
        tapes.append({"entry": entry, "tape": tape, "path": str(tape_path)})
    return manifest, tapes


def render_verified_requests(
    tokenizer: Any,
    tapes: list[dict[str, Any]],
    census: dict[str, Any],
    expected_template_file: Path,
    expected_template_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    if sha256(expected_template_file) != expected_template_sha256:
        raise AssertionError("frozen chat-template file hash mismatch")
    embedded_sha = text_sha256(tokenizer.chat_template)
    if embedded_sha != expected_template_sha256:
        expected_text = expected_template_file.read_text(encoding="utf-8").strip()
        if tokenizer.chat_template.strip() != expected_text:
            raise AssertionError("tokenizer chat template differs from frozen template")

    census_rows = {
        row["request_id"]: row for row in census["token_census"]["requests"]
    }
    rendered: dict[str, list[dict[str, Any]]] = {}
    observed_total_completion = 0
    for tape_record in tapes:
        entry = tape_record["entry"]
        tape = tape_record["tape"]
        rows: list[dict[str, Any]] = []
        for request_number, request in enumerate(tape["requests"], start=1):
            encoded = tokenizer.apply_chat_template(
                request["messages"], tokenize=True, add_generation_prompt=True
            )
            token_ids = [int(value) for value in encoded["input_ids"]]
            token_sha = token_ids_sha256(token_ids)
            census_row = census_rows.get(request["request_id"])
            if census_row is None:
                raise AssertionError(f"request absent from OP1 census: {request['request_id']}")
            if len(token_ids) != int(census_row["c02_rendered_prompt_tokens"]):
                raise AssertionError(f"rendered prompt length drift: {request['request_id']}")
            if token_sha != census_row["c02_rendered_token_ids_sha256"]:
                raise AssertionError(f"rendered token IDs drift: {request['request_id']}")
            completion_tokens = int(request["reference_completion_tokens"])
            if completion_tokens != int(census_row["b0_reference_completion_tokens"]):
                raise AssertionError(f"completion budget drift: {request['request_id']}")
            observed_total_completion += completion_tokens
            rows.append(
                {
                    "tape_id": tape["tape_id"],
                    "task_id": tape["task_id"],
                    "tape_sha256": entry["sha256"],
                    "request_number": request_number,
                    "request_id": request["request_id"],
                    "messages": request["messages"],
                    "rendered_token_ids": token_ids,
                    "rendered_prompt_tokens": len(token_ids),
                    "rendered_token_ids_sha256": token_sha,
                    "completion_tokens": completion_tokens,
                    "reference_cached_tokens": int(request["reference_cached_tokens"]),
                    "reference_prompt_tokens": int(request["reference_prompt_tokens"]),
                }
            )
        rendered[tape["tape_id"]] = rows
    if len(census_rows) != 11 or sum(len(rows) for rows in rendered.values()) != 11:
        raise AssertionError("OP3 requires exactly 11 frozen requests")
    if observed_total_completion != 21705:
        raise AssertionError(
            f"frozen completion total is {observed_total_completion}, expected 21705"
        )
    return rendered


def run_greedy_request(
    model: Any,
    request: dict[str, Any],
    *,
    device: int,
    seed: int,
    path: str,
    pass_number: int,
    tracker: K1InvocationTracker | None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_tokens: int = 64,
) -> dict[str, Any]:
    """Run one fixed-budget request and return auditable per-step metrics."""
    if path not in ("A", "B"):
        raise ValueError(f"unknown replay path: {path}")
    if (path == "B") != (tracker is not None and tracker.active):
        raise AssertionError("custom-op tracker activity does not match replay path")
    if progress_interval_tokens <= 0:
        raise ValueError("progress interval must be positive")

    text_model = get_submodule(model, "model.language_model")
    token_embedding = text_model.embed_tokens
    per_layer_embedding = text_model.embed_tokens_per_layer
    output_head = model.lm_head
    cuda_device = torch.device(f"cuda:{device}")
    num_layers = int(text_model.config.num_hidden_layers)
    per_layer_width = int(text_model.config.hidden_size_per_layer_input)
    softcap = text_model.config.final_logit_softcapping
    completion_tokens = int(request["completion_tokens"])
    prompt_ids = request["rendered_token_ids"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")
    attention_mask = torch.ones_like(input_ids, device="cpu")
    current_ids = input_ids
    past_key_values = None
    generated: list[int] = []
    step_seconds: list[float] = []
    embedding_cuda_deltas: list[int] = []
    eos_positions: list[int] = []
    raw_eos_ids = text_model.config.eos_token_id
    if isinstance(raw_eos_ids, int):
        eos_ids = {raw_eos_ids}
    else:
        eos_ids = set(int(value) for value in raw_eos_ids)
    nonfinite_observations: list[dict[str, Any]] = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    request_started = time.perf_counter()
    for token_index in range(completion_tokens):
        phase = "prefill" if token_index == 0 else "decode"
        if tracker is not None:
            tracker.set_phase(phase)

        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        allocation_before_embedding = torch.cuda.memory_allocated(device)
        inputs_embeds_cpu = token_embedding(current_ids)
        per_layer_inputs_cpu = per_layer_embedding(current_ids).reshape(
            *current_ids.shape, num_layers, per_layer_width
        )
        allocation_after_embedding = torch.cuda.memory_allocated(device)
        embedding_delta = allocation_after_embedding - allocation_before_embedding
        embedding_cuda_deltas.append(int(embedding_delta))
        if embedding_delta != 0:
            raise AssertionError(
                "CPU embedding lookup changed CUDA allocation: "
                f"request={request['request_id']} token={token_index} delta={embedding_delta}"
            )

        inputs_embeds = inputs_embeds_cpu.to(cuda_device)
        per_layer_inputs = per_layer_inputs_cpu.to(cuda_device)
        step_attention_mask = attention_mask.to(cuda_device)
        del inputs_embeds_cpu, per_layer_inputs_cpu

        with torch.inference_mode():
            output = text_model(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            last_hidden_cpu = output.last_hidden_state[:, -1:, :].to("cpu")
            logits = output_head(last_hidden_cpu)
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap

        hidden_finite = bool(torch.isfinite(last_hidden_cpu).all().item())
        logits_finite = bool(torch.isfinite(logits).all().item())
        if not hidden_finite or not logits_finite:
            nonfinite_observations.append(
                {
                    "generated_index": token_index,
                    "last_hidden_finite": hidden_finite,
                    "logits_finite": logits_finite,
                }
            )
            raise FloatingPointError(
                f"non-finite generation tensor at {request['request_id']} token {token_index}"
            )
        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)
        if next_token in eos_ids:
            eos_positions.append(token_index)

        current_ids = torch.tensor([[next_token]], dtype=torch.long, device="cpu")
        attention_mask = torch.cat(
            (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)), dim=1
        )
        del inputs_embeds, per_layer_inputs, step_attention_mask
        del output, last_hidden_cpu, logits
        torch.cuda.synchronize(device)
        if tracker is not None:
            tracker.flush_pending_events(require_synchronized=False)
        elapsed = time.perf_counter() - step_started
        step_seconds.append(elapsed)

        emitted = token_index + 1
        if (
            emitted == 1
            or emitted % progress_interval_tokens == 0
            or emitted == completion_tokens
        ):
            progress = {
                "path": path,
                "pass": pass_number,
                "tape_id": request["tape_id"],
                "request_id": request["request_id"],
                "generated_tokens": emitted,
                "completion_tokens": completion_tokens,
                "elapsed_seconds": time.perf_counter() - request_started,
                "latest_step_seconds": elapsed,
                "cache_sequence_length": int(past_key_values.get_seq_length()),
                "captured_at": now_iso(),
            }
            log(
                f"{path}{pass_number} {request['request_id']} "
                f"token {emitted}/{completion_tokens}; "
                f"step={elapsed * 1000.0:.3f} ms"
            )
            if progress_callback is not None:
                progress_callback(progress)

    total_seconds = time.perf_counter() - request_started
    final_cache_length = int(past_key_values.get_seq_length())
    expected_cache_length = len(prompt_ids) + completion_tokens - 1
    if final_cache_length != expected_cache_length:
        raise AssertionError(
            f"cache length mismatch for {request['request_id']}: "
            f"{final_cache_length} != {expected_cache_length}"
        )
    if len(generated) != completion_tokens:
        raise AssertionError("fixed completion budget was not fulfilled")

    incremental_seconds = sum(step_seconds[1:])
    incremental_tokens = max(0, completion_tokens - 1)
    peak_memory = memory_snapshot(torch, device)
    record = {
        "path": path,
        "pass": pass_number,
        "measurement_convention": "measured" if pass_number == 2 else "cache_prime",
        "tape_id": request["tape_id"],
        "task_id": request["task_id"],
        "tape_sha256": request["tape_sha256"],
        "request_number": request["request_number"],
        "request_id": request["request_id"],
        "seed": seed,
        "greedy": True,
        "eos_does_not_truncate": True,
        "rendered_prompt_tokens": len(prompt_ids),
        "rendered_token_ids_sha256": request["rendered_token_ids_sha256"],
        "completion_tokens": completion_tokens,
        "generated_token_ids": generated,
        "generated_token_ids_sha256": token_ids_sha256(generated),
        "eos_token_ids": sorted(eos_ids),
        "eos_generated_positions_zero_based": eos_positions,
        "ttft_ms": step_seconds[0] * 1000.0,
        "incremental_decode_seconds": incremental_seconds,
        "incremental_decode_tokens": incremental_tokens,
        "incremental_tpot_ms": (
            incremental_seconds / incremental_tokens * 1000.0
            if incremental_tokens
            else None
        ),
        "incremental_tokens_per_second": (
            incremental_tokens / incremental_seconds
            if incremental_tokens and incremental_seconds > 0
            else None
        ),
        "total_seconds": total_seconds,
        "total_tokens_per_second": completion_tokens / total_seconds,
        "step_seconds": step_seconds,
        "step_seconds_summary": summarize(step_seconds),
        "embedding_cuda_allocation_deltas_bytes": embedding_cuda_deltas,
        "embedding_cuda_allocation_delta_max_bytes": max(
            embedding_cuda_deltas, default=0
        ),
        "nonfinite_observations": nonfinite_observations,
        "final_cache_length": final_cache_length,
        "expected_cache_length": expected_cache_length,
        "memory": peak_memory,
    }
    del current_ids, input_ids, attention_mask, past_key_values
    torch.cuda.empty_cache()
    return record
