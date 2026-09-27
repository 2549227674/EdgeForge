#!/usr/bin/env python3
"""Build the two-channel C02-NF4 operator shape census for OP1.

Channel A verifies the frozen tapes, renders every request with the C02 chat
template, and derives per-family shapes and workload frequencies from the model
topology.  Channel B loads the formal NF4 carrier once and records two short
prefill/decode probes with module hooks plus torch.profiler.  The final census
keeps the legacy B0 token metrics parallel to, rather than mixed with, the
actual C02 Torch-carrier tokenization.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import torch

from load_smoke_nf4 import (
    CPU_MODULES,
    CUDA_MODULES,
    OFFLOADED_UNUSED_CPU_MODULES,
    PERSISTENT_CPU_MODULES,
    get_submodule,
    hook_snapshot,
    materialize_persistent_cpu_modules,
    memory_snapshot,
    module_devices,
    module_storage_snapshot,
    verify_formal_artifact,
)
from op0_common import (
    NvidiaSmiMonitor,
    atomic_write_json,
    now_iso,
    sha256,
    summarize,
    text_sha256,
)


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", file=sys.stderr, flush=True)


def git_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def median_int(values: list[int]) -> int:
    ordered = sorted(values)
    if len(ordered) % 2 != 1:
        raise ValueError("OP1 frozen request count should have an integer median")
    return ordered[len(ordered) // 2]


def verify_file_identity(repo_root: Path, relative: str, expected: str) -> Path:
    path = repo_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise ValueError(
            f"identity mismatch for {relative}: expected={expected}, observed={observed}"
        )
    return path


def load_and_verify_inputs(
    repo_root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    inputs = config["inputs"]
    manifest_path = verify_file_identity(
        repo_root, inputs["tapes_manifest"], inputs["tapes_manifest_sha256"]
    )
    model_path = repo_root / inputs["model"]
    verify_file_identity(
        repo_root,
        f"{inputs['model']}/config.json",
        inputs["model_config_sha256"],
    )
    verify_file_identity(
        repo_root,
        f"{inputs['model']}/chat_template.jinja",
        inputs["chat_template_sha256"],
    )
    verify_file_identity(
        repo_root,
        f"{inputs['model']}/tokenizer.json",
        inputs["tokenizer_sha256"],
    )
    verify_file_identity(
        repo_root, inputs["p0_nf4_receipt"], inputs["p0_nf4_receipt_sha256"]
    )
    verify_file_identity(
        repo_root, inputs["op0_receipt"], inputs["op0_receipt_sha256"]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tapes: dict[str, dict[str, Any]] = {}
    request_index: dict[str, Any] = {}
    for entry in manifest["tapes"]:
        tape_path = repo_root / entry["path"]
        observed = sha256(tape_path)
        if observed != entry["sha256"]:
            raise ValueError(
                f"tape SHA mismatch for {entry['path']}: {observed} != {entry['sha256']}"
            )
        tape = json.loads(tape_path.read_text(encoding="utf-8"))
        if tape["tape_id"] != entry["tape_id"]:
            raise ValueError(f"tape identity mismatch: {tape_path}")
        if len(tape["requests"]) != int(entry["requests"]):
            raise ValueError(f"tape request-count mismatch: {tape_path}")
        tapes[tape["tape_id"]] = tape
        for request in tape["requests"]:
            request_id = request["request_id"]
            if request_id in request_index:
                raise ValueError(f"duplicate request id: {request_id}")
            request_index[request_id] = {
                "tape": tape,
                "entry": entry,
                "request": request,
            }
    return manifest, request_index, json.loads(
        (model_path / "config.json").read_text(encoding="utf-8")
    )


def derive_topology(
    repo_root: Path, config: dict[str, Any], model_config: dict[str, Any]
) -> dict[str, Any]:
    text = model_config["text_config"]
    expected = config["topology_assertions"]
    index = json.loads(
        (repo_root / config["inputs"]["model"] / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )["weight_map"]
    language_prefix = "model.language_model.layers."
    layer_weight_names: dict[int, set[str]] = {}
    for layer_index in range(int(text["num_hidden_layers"])):
        prefix = f"{language_prefix}{layer_index}."
        layer_weight_names[layer_index] = {
            name[len(prefix) :] for name in index if name.startswith(prefix)
        }

    full_layers = [
        index
        for index, layer_type in enumerate(text["layer_types"])
        if layer_type == "full_attention"
    ]
    sliding_layers = [
        index
        for index, layer_type in enumerate(text["layer_types"])
        if layer_type == "sliding_attention"
    ]
    nonshared_layers = [
        index
        for index, names in layer_weight_names.items()
        if "self_attn.k_proj.weight" in names
    ]
    shared_layers = sorted(set(range(int(text["num_hidden_layers"]))) - set(nonshared_layers))
    nonshared_full = sorted(set(nonshared_layers) & set(full_layers))
    nonshared_sliding = sorted(set(nonshared_layers) & set(sliding_layers))
    shared_full = sorted(set(shared_layers) & set(full_layers))
    shared_sliding = sorted(set(shared_layers) & set(sliding_layers))
    full_head_layers = {
        int(layer): int(values["head_dim"])
        for layer, values in text.get("per_layer_config", {}).items()
    }

    observed = {
        "hidden_size": int(text["hidden_size"]),
        "intermediate_size": int(text["intermediate_size"]),
        "num_hidden_layers": int(text["num_hidden_layers"]),
        "num_attention_heads": int(text["num_attention_heads"]),
        "num_key_value_heads": int(text["num_key_value_heads"]),
        "num_kv_shared_layers": int(text["num_kv_shared_layers"]),
        "first_kv_shared_layer": min(shared_layers),
        "sliding_window": int(text["sliding_window"]),
        "sliding_layers_all": len(sliding_layers),
        "full_layers_all": len(full_layers),
        "nonshared_sliding_layers": len(nonshared_sliding),
        "nonshared_full_layers": len(nonshared_full),
        "sliding_head_dim": int(text["head_dim"]),
        "full_head_dim": max(full_head_layers.values()),
        "vocab_size": int(text["vocab_size"]),
        "hidden_size_per_layer_input": int(text["hidden_size_per_layer_input"]),
        "linear4bit_calls_per_forward": 343,
        "rmsnorm_calls_per_forward": 302,
        "attention_calls_per_forward": int(text["num_hidden_layers"]),
        "lm_head_calls_per_output_token": 1,
    }
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"model topology differs from frozen OP1 config: {mismatches}")

    for layer in nonshared_layers:
        required = {
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.k_norm.weight",
        }
        if not required.issubset(layer_weight_names[layer]):
            raise AssertionError(f"nonshared layer {layer} lacks K/V topology")
    for layer in shared_layers:
        forbidden = {
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.k_norm.weight",
        }
        if forbidden & layer_weight_names[layer]:
            raise AssertionError(f"shared layer {layer} unexpectedly owns K/V weights")

    return {
        **observed,
        "layer_types": list(text["layer_types"]),
        "full_layer_indices": full_layers,
        "sliding_layer_indices": sliding_layers,
        "nonshared_layer_indices": nonshared_layers,
        "shared_layer_indices": shared_layers,
        "nonshared_full_layer_indices": nonshared_full,
        "nonshared_sliding_layer_indices": nonshared_sliding,
        "shared_full_layer_indices": shared_full,
        "shared_sliding_layer_indices": shared_sliding,
        "shared_kv_sources": {
            "sliding_attention": max(nonshared_sliding),
            "full_attention": max(nonshared_full),
        },
        "global_head_overrides": full_head_layers,
        "weight_index_sha256": sha256(
            repo_root / config["inputs"]["model"] / "model.safetensors.index.json"
        ),
        "status": "PASS",
    }


def render_requests(
    repo_root: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    request_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    model_path = repo_root / config["inputs"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    embedded_template_sha = text_sha256(tokenizer.chat_template)
    file_template_sha = sha256(model_path / "chat_template.jinja")
    if embedded_template_sha != file_template_sha:
        # Ignore only surrounding whitespace in the comparison; the tokenized output
        # below is still hashed per request.
        if tokenizer.chat_template.strip() != (
            model_path / "chat_template.jinja"
        ).read_text(encoding="utf-8").strip():
            raise ValueError("tokenizer chat template differs from the frozen template file")

    rows: list[dict[str, Any]] = []
    for tape_entry in manifest["tapes"]:
        tape_id = tape_entry["tape_id"]
        tape = request_index[next(
            request_id
            for request_id, item in request_index.items()
            if item["tape"]["tape_id"] == tape_id
        )]["tape"]
        for request_number, request in enumerate(tape["requests"], start=1):
            encoded = tokenizer.apply_chat_template(
                request["messages"],
                tokenize=True,
                add_generation_prompt=True,
            )
            token_ids = [int(value) for value in encoded["input_ids"]]
            rendered_tokens = len(token_ids)
            reference_prompt = int(request["reference_prompt_tokens"])
            completion = int(request["reference_completion_tokens"])
            rows.append(
                {
                    "tape_id": tape_id,
                    "tape_sha256": tape_entry["sha256"],
                    "request_number": request_number,
                    "request_id": request["request_id"],
                    "message_count": len(request["messages"]),
                    "c02_rendered_prompt_tokens": rendered_tokens,
                    "c02_rendered_token_ids_sha256": token_ids_sha256(token_ids),
                    "b0_reference_prompt_tokens": reference_prompt,
                    "b0_reference_cached_tokens": int(request["reference_cached_tokens"]),
                    "b0_reference_completion_tokens": completion,
                    "prompt_delta_c02_minus_b0": rendered_tokens - reference_prompt,
                    "incremental_decode_forwards": completion - 1,
                    "c02_projected_context_envelope_tokens": rendered_tokens + completion,
                    "b0_reference_context_envelope_tokens": reference_prompt + completion,
                    "recorded_step_id": request.get("recorded_step_id"),
                }
            )
    return rows


def assert_card_facts(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = config["channel_a"]["expected_card_facts"]
    b0_prompt = [row["b0_reference_prompt_tokens"] for row in rows]
    completion = [row["b0_reference_completion_tokens"] for row in rows]
    cached = [row["b0_reference_cached_tokens"] for row in rows]
    observed = {
        "tapes": len({row["tape_id"] for row in rows}),
        "requests": len(rows),
        "b0_prompt_min": min(b0_prompt),
        "b0_prompt_median": median_int(b0_prompt),
        "b0_prompt_max": max(b0_prompt),
        "completion_min": min(completion),
        "completion_median": median_int(completion),
        "completion_max": max(completion),
        "cached_min": min(cached),
        "cached_median": median_int(cached),
        "cached_max": max(cached),
        "b0_deepest_prompt_plus_completion": max(
            row["b0_reference_context_envelope_tokens"] for row in rows
        ),
    }
    if observed != expected:
        raise AssertionError(
            f"execution-card tape facts changed: expected={expected}, observed={observed}"
        )
    return {"expected": expected, "observed": observed, "status": "PASS"}


def add_weights(shapes: list[dict[str, Any]], work_field: str) -> None:
    invocation_total = sum(int(shape["invocations"]) for shape in shapes)
    work_total = sum(int(shape[work_field]) for shape in shapes)
    for shape in shapes:
        shape["weight"] = round(shape["invocations"] / invocation_total, 12)
        shape["work_weight"] = round(shape[work_field] / work_total, 12)


def build_linear_shapes(
    rows: list[dict[str, Any]], topology: dict[str, Any]
) -> dict[str, Any]:
    hidden = topology["hidden_size"]
    intermediate = topology["intermediate_size"]
    ple = topology["hidden_size_per_layer_input"]
    layers = topology["num_hidden_layers"]
    variants = [
        ("q_proj_sliding", hidden, 8 * topology["sliding_head_dim"], 35),
        ("q_proj_full", hidden, 8 * topology["full_head_dim"], 7),
        ("k_proj_sliding_nonshared", hidden, 2 * topology["sliding_head_dim"], 20),
        ("k_proj_full_nonshared", hidden, 2 * topology["full_head_dim"], 4),
        ("v_proj_sliding_nonshared", hidden, 2 * topology["sliding_head_dim"], 20),
        ("v_proj_full_nonshared", hidden, 2 * topology["full_head_dim"], 4),
        ("o_proj_sliding", 8 * topology["sliding_head_dim"], hidden, 35),
        ("o_proj_full", 8 * topology["full_head_dim"], hidden, 7),
        ("mlp_gate_proj", hidden, intermediate, 42),
        ("mlp_up_proj", hidden, intermediate, 42),
        ("mlp_down_proj", intermediate, hidden, 42),
        ("ple_context_projection", hidden, layers * ple, 1),
        ("ple_input_gate", hidden, ple, 42),
        ("ple_residual_projection", ple, hidden, 42),
    ]
    if sum(item[3] for item in variants) != topology["linear4bit_calls_per_forward"]:
        raise AssertionError("derived Linear4bit call count is not 343")
    shapes: list[dict[str, Any]] = []
    for row in rows:
        tokens = row["c02_rendered_prompt_tokens"]
        for variant, reduction, output, calls in variants:
            invocations = calls
            shapes.append(
                {
                    "variant": variant,
                    "dims": [tokens, reduction, output],
                    "dimension_order": ["M_rows", "K_reduction", "N_output"],
                    "dtype": "bfloat16 activation / NF4 double-quantized weight",
                    "phase": "prefill",
                    "source_request_id": row["request_id"],
                    "calls_per_forward": calls,
                    "forward_count": 1,
                    "invocations": invocations,
                    "analytic_flops": 2 * tokens * reduction * output * invocations,
                }
            )
    decode_forwards = sum(row["incremental_decode_forwards"] for row in rows)
    for variant, reduction, output, calls in variants:
        invocations = calls * decode_forwards
        shapes.append(
            {
                "variant": variant,
                "dims": [1, reduction, output],
                "dimension_order": ["M_rows", "K_reduction", "N_output"],
                "dtype": "bfloat16 activation / NF4 double-quantized weight",
                "phase": "decode",
                "source_request_ids": [row["request_id"] for row in rows],
                "calls_per_forward": calls,
                "forward_count": decode_forwards,
                "invocations": invocations,
                "analytic_flops": 2 * reduction * output * invocations,
            }
        )
    add_weights(shapes, "analytic_flops")
    return {
        "op_semantics_sha": text_sha256(
            "y[m,n] = sum_k activation[m,k] * dequant_nf4(weight[n,k])"
        ),
        "carrier_implementation": "bitsandbytes.nn.Linear4bit",
        "calls_per_model_forward": topology["linear4bit_calls_per_forward"],
        "shapes": shapes,
    }


def build_rmsnorm_shapes(
    rows: list[dict[str, Any]], topology: dict[str, Any]
) -> dict[str, Any]:
    variants = [
        ("decoder_hidden_norms", 1, topology["hidden_size"], 211),
        ("q_norm_sliding", 8, topology["sliding_head_dim"], 35),
        ("q_norm_full", 8, topology["full_head_dim"], 7),
        ("kv_norm_sliding_nonshared", 2, topology["sliding_head_dim"], 40),
        ("kv_norm_full_nonshared", 2, topology["full_head_dim"], 8),
        ("ple_projection_norm", 42, topology["hidden_size_per_layer_input"], 1),
    ]
    if sum(item[3] for item in variants) != topology["rmsnorm_calls_per_forward"]:
        raise AssertionError("derived RMSNorm call count is not 302")
    shapes: list[dict[str, Any]] = []
    for row in rows:
        tokens = row["c02_rendered_prompt_tokens"]
        for variant, outer, width, calls in variants:
            normalized_rows = tokens * outer
            shapes.append(
                {
                    "variant": variant,
                    "dims": [normalized_rows, width],
                    "dimension_order": ["flattened_rows", "normalized_dim"],
                    "native_outer_shape_note": (
                        f"batch=1, tokens={tokens}, outer_multiplier={outer}"
                    ),
                    "dtype": "bfloat16 input/output; fp32 reduction",
                    "phase": "prefill",
                    "source_request_id": row["request_id"],
                    "calls_per_forward": calls,
                    "forward_count": 1,
                    "invocations": calls,
                    "analytic_elements": normalized_rows * width * calls,
                }
            )
    decode_forwards = sum(row["incremental_decode_forwards"] for row in rows)
    for variant, outer, width, calls in variants:
        invocations = calls * decode_forwards
        shapes.append(
            {
                "variant": variant,
                "dims": [outer, width],
                "dimension_order": ["flattened_rows", "normalized_dim"],
                "native_outer_shape_note": f"batch=1, tokens=1, outer_multiplier={outer}",
                "dtype": "bfloat16 input/output; fp32 reduction",
                "phase": "decode",
                "source_request_ids": [row["request_id"] for row in rows],
                "calls_per_forward": calls,
                "forward_count": decode_forwards,
                "invocations": invocations,
                "analytic_elements": outer * width * invocations,
            }
        )
    add_weights(shapes, "analytic_elements")
    return {
        "op_semantics_sha": text_sha256(
            "output = input * pow(mean(input^2)+eps,-0.5) * weight; fp32 reduction; v_norm has no scale"
        ),
        "epsilon": 1e-6,
        "implementation_semantics": {
            "with_scale": "normed_output * weight.float(); no implicit +1 offset in Transformers 5.15",
            "without_scale": "v_norm returns the normalized value without a learned scale",
            "op2_preregistration_correction": "replace the execution-card draft's (1+w) premise with the observed direct-scale semantics",
        },
        "calls_per_model_forward": topology["rmsnorm_calls_per_forward"],
        "variant_accounting": {
            "decoder_hidden_norms": "4 decoder residual norms + post-PLE norm per layer, plus final model norm = 211",
            "q_norm": "one per all 42 attention layers",
            "kv_norm": "k_norm + scale-free v_norm on 24 nonshared-KV layers = 48",
            "ple_projection_norm": "one module call over [B,T,42,256]",
        },
        "shapes": shapes,
    }


def build_attention_shapes(
    rows: list[dict[str, Any]], topology: dict[str, Any]
) -> dict[str, Any]:
    shapes: list[dict[str, Any]] = []
    variants = [
        ("sliding_attention_eager", 35, topology["sliding_head_dim"]),
        ("full_attention_eager", 7, topology["full_head_dim"]),
    ]
    for row in rows:
        tokens = row["c02_rendered_prompt_tokens"]
        for variant, calls, head_dim in variants:
            invocations = calls
            score_elements = 8 * tokens * tokens * invocations
            shapes.append(
                {
                    "variant": variant,
                    "dims": [1, 8, tokens, tokens, head_dim],
                    "dimension_order": ["batch", "query_heads", "Q", "K", "head_dim"],
                    "dtype": "bfloat16 Q/K/V and output; fp32 softmax accumulation",
                    "phase": "prefill",
                    "source_request_id": row["request_id"],
                    "calls_per_forward": calls,
                    "forward_count": 1,
                    "invocations": invocations,
                    "logical_sliding_window": topology["sliding_window"]
                    if variant.startswith("sliding")
                    else None,
                    "physical_eager_score_shape": [1, 8, tokens, tokens],
                    "materialized_score_elements": score_elements,
                    "analytic_attention_matmul_flops": 4
                    * 8
                    * tokens
                    * tokens
                    * head_dim
                    * invocations,
                }
            )

    sliding_decode_forwards = sum(row["incremental_decode_forwards"] for row in rows)
    sliding_kv = topology["sliding_window"]
    sliding_invocations = 35 * sliding_decode_forwards
    shapes.append(
        {
            "variant": "sliding_attention_eager",
            "dims": [1, 8, 1, sliding_kv, topology["sliding_head_dim"]],
            "dimension_order": ["batch", "query_heads", "Q", "K", "head_dim"],
            "dtype": "bfloat16 Q/K/V and output; fp32 softmax accumulation",
            "phase": "decode",
            "source_request_ids": [row["request_id"] for row in rows],
            "calls_per_forward": 35,
            "forward_count": sliding_decode_forwards,
            "invocations": sliding_invocations,
            "logical_sliding_window": sliding_kv,
            "physical_eager_score_shape": [1, 8, 1, sliding_kv],
            "materialized_score_elements": 8 * sliding_kv * sliding_invocations,
            "analytic_attention_matmul_flops": 4
            * 8
            * sliding_kv
            * topology["sliding_head_dim"]
            * sliding_invocations,
        }
    )

    for row in rows:
        decode_forwards = row["incremental_decode_forwards"]
        if decode_forwards <= 0:
            continue
        prompt = row["c02_rendered_prompt_tokens"]
        kv_min = prompt + 1
        kv_max = prompt + decode_forwards
        kv_sum = decode_forwards * prompt + decode_forwards * (decode_forwards + 1) // 2
        invocations = 7 * decode_forwards
        shapes.append(
            {
                "variant": "full_attention_eager",
                "dims": [1, 8, 1, kv_min, topology["full_head_dim"]],
                "dims_max": [1, 8, 1, kv_max, topology["full_head_dim"]],
                "dimension_order": ["batch", "query_heads", "Q", "K", "head_dim"],
                "dimension_range": {"kv_tokens_inclusive": [kv_min, kv_max]},
                "dtype": "bfloat16 Q/K/V and output; fp32 softmax accumulation",
                "phase": "decode",
                "source_request_id": row["request_id"],
                "calls_per_forward": 7,
                "forward_count": decode_forwards,
                "invocations": invocations,
                "materialized_score_elements": 8 * 7 * kv_sum,
                "analytic_attention_matmul_flops": 4
                * 8
                * 7
                * topology["full_head_dim"]
                * kv_sum,
            }
        )
    add_weights(shapes, "analytic_attention_matmul_flops")
    return {
        "op_semantics_sha": text_sha256(
            "repeat GQA KV 2->8 heads; QK matmul; causal/sliding mask; fp32 softmax; PV matmul"
        ),
        "implementation": "transformers Gemma4 eager_attention_forward",
        "calls_per_model_forward": topology["attention_calls_per_forward"],
        "implementation_detail": (
            "eager prefill materializes T-by-T scores even for sliding layers; the 512-token window is enforced by the mask"
        ),
        "shapes": shapes,
    }


def build_lm_head_shapes(
    rows: list[dict[str, Any]], topology: dict[str, Any]
) -> dict[str, Any]:
    prefill_count = len(rows)
    decode_count = sum(row["incremental_decode_forwards"] for row in rows)
    hidden = topology["hidden_size"]
    vocab = topology["vocab_size"]
    shapes = [
        {
            "variant": "tied_cpu_lm_head",
            "dims": [1, hidden, vocab],
            "dimension_order": ["M_rows", "K_reduction", "N_vocab"],
            "dtype": "bfloat16 CPU",
            "phase": "prefill",
            "source_request_ids": [row["request_id"] for row in rows],
            "trigger_prompt_tokens_range": [
                min(row["c02_rendered_prompt_tokens"] for row in rows),
                max(row["c02_rendered_prompt_tokens"] for row in rows),
            ],
            "calls_per_forward": 1,
            "forward_count": prefill_count,
            "invocations": prefill_count,
            "analytic_flops": 2 * hidden * vocab * prefill_count,
        },
        {
            "variant": "tied_cpu_lm_head",
            "dims": [1, hidden, vocab],
            "dimension_order": ["M_rows", "K_reduction", "N_vocab"],
            "dtype": "bfloat16 CPU",
            "phase": "decode",
            "source_request_ids": [row["request_id"] for row in rows],
            "calls_per_forward": 1,
            "forward_count": decode_count,
            "invocations": decode_count,
            "analytic_flops": 2 * hidden * vocab * decode_count,
        },
    ]
    add_weights(shapes, "analytic_flops")
    return {
        "op_semantics_sha": text_sha256(
            "logits = hidden_last @ tied_embedding_weight.T; tanh final logit softcap 30 follows"
        ),
        "implementation": "CPU torch.nn.Linear tied to token embedding",
        "final_logit_softcap": 30.0,
        "calls_per_output_token": 1,
        "shapes": shapes,
    }


def build_channel_a(
    repo_root: Path,
    config_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest, request_index, model_config = load_and_verify_inputs(repo_root, config)
    topology = derive_topology(repo_root, config, model_config)
    rows = render_requests(repo_root, config, manifest, request_index)
    card_facts = assert_card_facts(config, rows)
    c02_prompts = [row["c02_rendered_prompt_tokens"] for row in rows]
    b0_prompts = [row["b0_reference_prompt_tokens"] for row in rows]
    completions = [row["b0_reference_completion_tokens"] for row in rows]
    cached = [row["b0_reference_cached_tokens"] for row in rows]
    total_incremental = sum(row["incremental_decode_forwards"] for row in rows)
    op_families = {
        "nf4_linear": build_linear_shapes(rows, topology),
        "rmsnorm": build_rmsnorm_shapes(rows, topology),
        "attention": build_attention_shapes(rows, topology),
        "lm_head": build_lm_head_shapes(rows, topology),
    }
    prompt_mismatches = [
        row for row in rows if row["prompt_delta_c02_minus_b0"] != 0
    ]
    channel_a = {
        "schema_version": 1,
        "stage": "OP1-channel-A",
        "timestamp": now_iso(),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "carrier": config["carrier"],
        "source_identity": config["inputs"],
        "token_authority": config["channel_a"]["prompt_authorities"],
        "decode_accounting": config["channel_a"]["decode_accounting"],
        "card_fact_assertions": card_facts,
        "token_summary": {
            "tapes": len({row["tape_id"] for row in rows}),
            "requests": len(rows),
            "c02_rendered_prompt_tokens": summarize(c02_prompts, digits=3),
            "b0_reference_prompt_tokens": summarize(b0_prompts, digits=3),
            "b0_reference_cached_tokens": summarize(cached, digits=3),
            "b0_reference_completion_tokens": summarize(completions, digits=3),
            "incremental_decode_forwards": total_incremental,
            "total_model_forwards": len(rows) + total_incremental,
            "c02_projected_deepest_context_tokens": max(
                row["c02_projected_context_envelope_tokens"] for row in rows
            ),
            "b0_reference_deepest_context_tokens": max(
                row["b0_reference_context_envelope_tokens"] for row in rows
            ),
        },
        "prompt_metric_divergence": {
            "mismatched_requests": len(prompt_mismatches),
            "all_requests_mismatch": len(prompt_mismatches) == len(rows),
            "policy": (
                "C02 rendered lengths define formal Torch-carrier prefill shapes; B0 metrics remain a parallel systems-reference envelope"
            ),
            "silent_substitution_forbidden": True,
        },
        "requests": rows,
        "topology": topology,
        "op_families": op_families,
        "git_sha": git_sha(repo_root),
        "status": "PASS",
    }
    return channel_a, request_index


def tensor_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": list(value.stride()),
        "contiguous": value.is_contiguous(),
    }


def classify_linear(name: str, module: Any) -> str:
    if name.endswith("per_layer_model_projection"):
        return "ple_context_projection"
    if name.endswith("per_layer_input_gate"):
        return "ple_input_gate"
    if name.endswith("per_layer_projection"):
        return "ple_residual_projection"
    suffix_map = {
        ".q_proj": "q_proj",
        ".k_proj": "k_proj",
        ".v_proj": "v_proj",
        ".o_proj": "o_proj",
        ".gate_proj": "mlp_gate_proj",
        ".up_proj": "mlp_up_proj",
        ".down_proj": "mlp_down_proj",
    }
    base = next(
        (label for suffix, label in suffix_map.items() if name.endswith(suffix)),
        name.rsplit(".", 1)[-1],
    )
    if base in {"q_proj", "o_proj"}:
        output_features = int(module.out_features)
        return f"{base}_{'full' if output_features in (4096, 2560) and (base != 'o_proj' or int(module.in_features) == 4096) else 'sliding'}"
    if base in {"k_proj", "v_proj"}:
        return f"{base}_{'full' if int(module.out_features) == 1024 else 'sliding'}_nonshared"
    return base


def classify_rmsnorm(name: str, module: Any, model: Any) -> str:
    if name.endswith("per_layer_projection_norm"):
        return "ple_projection_norm"
    if name == "model.language_model.norm":
        return "model_final_norm"
    if ".self_attn.q_norm" in name:
        width = int(module.weight.numel()) if getattr(module, "weight", None) is not None else 0
        return f"q_norm_{'full' if width == 512 else 'sliding'}"
    if ".self_attn.k_norm" in name:
        width = int(module.weight.numel())
        return f"k_norm_{'full' if width == 512 else 'sliding'}_nonshared"
    if ".self_attn.v_norm" in name:
        # v_norm is scale-free and stores no dimension attribute. Infer its
        # width from the owning attention module.
        attention_name = name.rsplit(".", 1)[0]
        width = int(get_submodule(model, attention_name).head_dim)
        return f"v_norm_{'full' if width == 512 else 'sliding'}_nonshared"
    if name.endswith("input_layernorm"):
        return "decoder_input_layernorm"
    if name.endswith("post_attention_layernorm"):
        return "decoder_post_attention_layernorm"
    if name.endswith("pre_feedforward_layernorm"):
        return "decoder_pre_feedforward_layernorm"
    if name.endswith("post_feedforward_layernorm"):
        return "decoder_post_feedforward_layernorm"
    if name.endswith("post_per_layer_input_norm"):
        return "decoder_post_ple_norm"
    return name.rsplit(".", 1)[-1]


class ModuleShapeRecorder:
    """Record tensor metadata only; never retain model outputs or activations."""

    def __init__(self) -> None:
        self.context: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []

    def set_context(self, request_id: str, phase: str) -> None:
        self.context = {"request_id": request_id, "phase": phase}

    def clear_context(self) -> None:
        self.context = None

    def register(self, model: Any, bnb: Any) -> None:
        for name, module in model.named_modules():
            family = None
            variant = None
            if name.startswith("model.language_model.") and isinstance(
                module, bnb.nn.Linear4bit
            ):
                family = "linear4bit"
                variant = classify_linear(name, module)
            elif name.startswith("model.language_model.") and type(module).__name__ == "Gemma4RMSNorm":
                family = "rmsnorm"
                variant = classify_rmsnorm(name, module, model)
            elif name.startswith("model.language_model.layers.") and type(module).__name__ == "Gemma4TextAttention":
                family = "attention"
                variant = (
                    f"{module.layer_type}_{'shared' if module.is_kv_shared_layer else 'nonshared'}"
                )
            elif name == "lm_head":
                family = "lm_head"
                variant = "tied_cpu_lm_head"
            if family is None:
                continue

            def hook(
                hooked_module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                output: Any,
                *,
                module_name: str = name,
                module_family: str = family,
                module_variant: str = variant,
            ) -> None:
                if self.context is None:
                    return
                if args and isinstance(args[0], torch.Tensor):
                    input_tensor = args[0]
                else:
                    input_tensor = kwargs.get("hidden_states")
                outputs = output if isinstance(output, tuple) else (output,)
                output_metadata = [
                    item for value in outputs if (item := tensor_metadata(value)) is not None
                ]
                record = {
                    **self.context,
                    "family": module_family,
                    "variant": module_variant,
                    "module_name": module_name,
                    "module_class": type(hooked_module).__name__,
                    "input": tensor_metadata(input_tensor),
                    "outputs": output_metadata,
                }
                if module_family == "attention":
                    record["attention"] = {
                        "layer_idx": int(hooked_module.layer_idx),
                        "layer_type": hooked_module.layer_type,
                        "is_kv_shared_layer": bool(hooked_module.is_kv_shared_layer),
                        "store_full_length_kv": bool(hooked_module.store_full_length_kv),
                        "head_dim": int(hooked_module.head_dim),
                        "num_key_value_groups": int(hooked_module.num_key_value_groups),
                        "sliding_window": hooked_module.sliding_window,
                    }
                self.records.append(record)

            self.handles.append(
                module.register_forward_hook(hook, with_kwargs=True, always_call=False)
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def aggregate(self) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for record in self.records:
            key_payload = {
                key: record.get(key)
                for key in (
                    "request_id",
                    "phase",
                    "family",
                    "variant",
                    "module_class",
                    "input",
                    "outputs",
                )
            }
            if record.get("attention"):
                attention = dict(record["attention"])
                attention.pop("layer_idx", None)
                key_payload["attention"] = attention
            key = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
            if key not in groups:
                groups[key] = {
                    **key_payload,
                    "call_count": 0,
                    "module_names": [],
                    "layer_indices": [],
                }
            group = groups[key]
            group["call_count"] += 1
            group["module_names"].append(record["module_name"])
            if record.get("attention"):
                group["layer_indices"].append(record["attention"]["layer_idx"])
        output = []
        for group in groups.values():
            names = sorted(set(group.pop("module_names")))
            layers = sorted(set(group.pop("layer_indices")))
            group["module_count"] = len(names)
            group["module_names_sha256"] = text_sha256("\n".join(names))
            group["sample_module_names"] = names[:5]
            if layers:
                group["layer_indices"] = layers
            output.append(group)
        return sorted(
            output,
            key=lambda item: (
                item["request_id"],
                item["phase"],
                item["family"],
                item["variant"],
            ),
        )


def profiler_summary(profiler: Any) -> dict[str, Any]:
    keywords = (
        "bmm",
        "matmul",
        "linear",
        "mm",
        "softmax",
        "rms",
        "bitsandbytes",
        "elementwise",
    )
    interesting = []
    for event in profiler.key_averages(group_by_input_shape=True):
        lowered = event.key.lower()
        if not any(keyword in lowered for keyword in keywords):
            continue
        input_shapes = event.input_shapes
        if isinstance(input_shapes, tuple):
            input_shapes = list(input_shapes)
        interesting.append(
            {
                "operator": event.key,
                "count": int(event.count),
                "input_shapes": input_shapes,
                "cpu_time_total_us": round(float(event.cpu_time_total), 3),
                "self_cpu_time_total_us": round(float(event.self_cpu_time_total), 3),
                "device_time_total_us": round(
                    float(getattr(event, "device_time_total", 0.0)), 3
                ),
                "self_device_time_total_us": round(
                    float(getattr(event, "self_device_time_total", 0.0)), 3
                ),
            }
        )
    interesting.sort(
        key=lambda item: (item["device_time_total_us"], item["cpu_time_total_us"]),
        reverse=True,
    )
    return {
        "event_count": len(profiler.events()),
        "interesting_key_average_count": len(interesting),
        "interesting_key_averages": interesting,
        "timing_is_performance_evidence": False,
        "reason": "torch.profiler observer overhead is accepted here because OP1 uses shapes and call topology, not timing",
    }


def profile_callable(operation: Callable[[], Any]) -> tuple[Any, dict[str, Any], float]:
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        result = operation()
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    return result, profiler_summary(profiler), wall_seconds


def run_hybrid_forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask_cpu: torch.Tensor,
    past_key_values: Any,
    device: int,
) -> tuple[Any, int, dict[str, Any]]:
    text_model = get_submodule(model, "model.language_model")
    token_embedding = text_model.embed_tokens
    per_layer_embedding = text_model.embed_tokens_per_layer
    output_head = model.lm_head
    cuda_device = torch.device(f"cuda:{device}")
    before_embedding = torch.cuda.memory_allocated(device)
    inputs_embeds_cpu = token_embedding(input_ids)
    per_layer_inputs_cpu = per_layer_embedding(input_ids).reshape(
        *input_ids.shape,
        text_model.config.num_hidden_layers,
        text_model.config.hidden_size_per_layer_input,
    )
    after_embedding = torch.cuda.memory_allocated(device)
    embedding_delta = after_embedding - before_embedding
    if embedding_delta != 0:
        raise AssertionError(f"CPU embedding lookup changed CUDA allocation by {embedding_delta}")

    inputs_embeds = inputs_embeds_cpu.to(cuda_device)
    per_layer_inputs = per_layer_inputs_cpu.to(cuda_device)
    attention_mask = attention_mask_cpu.to(cuda_device)
    del inputs_embeds_cpu, per_layer_inputs_cpu
    with torch.inference_mode():
        output = text_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        last_hidden_cpu = output.last_hidden_state[:, -1:, :].to("cpu")
        logits = output_head(last_hidden_cpu)
        softcap = text_model.config.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        next_token = int(logits[0, -1].argmax().item())
    metadata = {
        "input_tokens": int(input_ids.shape[1]),
        "attention_mask_tokens": int(attention_mask_cpu.shape[1]),
        "embedding_cuda_allocation_delta_bytes": embedding_delta,
        "last_hidden_shape": list(last_hidden_cpu.shape),
        "logits_shape": list(logits.shape),
        "next_token": next_token,
        "cache_sequence_length": int(output.past_key_values.get_seq_length()),
    }
    del inputs_embeds, per_layer_inputs, attention_mask, last_hidden_cpu, logits
    return output.past_key_values, next_token, metadata


def expected_variant_counts() -> dict[str, dict[str, int]]:
    return {
        "linear4bit": {
            "q_proj_sliding": 35,
            "q_proj_full": 7,
            "k_proj_sliding_nonshared": 20,
            "k_proj_full_nonshared": 4,
            "v_proj_sliding_nonshared": 20,
            "v_proj_full_nonshared": 4,
            "o_proj_sliding": 35,
            "o_proj_full": 7,
            "mlp_gate_proj": 42,
            "mlp_up_proj": 42,
            "mlp_down_proj": 42,
            "ple_context_projection": 1,
            "ple_input_gate": 42,
            "ple_residual_projection": 42,
        },
        "rmsnorm": {
            "decoder_input_layernorm": 42,
            "decoder_post_attention_layernorm": 42,
            "decoder_pre_feedforward_layernorm": 42,
            "decoder_post_feedforward_layernorm": 42,
            "decoder_post_ple_norm": 42,
            "model_final_norm": 1,
            "q_norm_sliding": 35,
            "q_norm_full": 7,
            "k_norm_sliding_nonshared": 20,
            "k_norm_full_nonshared": 4,
            "v_norm_sliding_nonshared": 20,
            "v_norm_full_nonshared": 4,
            "ple_projection_norm": 1,
        },
        "attention": {
            "sliding_attention_nonshared": 20,
            "full_attention_nonshared": 4,
            "sliding_attention_shared": 15,
            "full_attention_shared": 3,
        },
        "lm_head": {"tied_cpu_lm_head": 1},
    }


def validate_recorded_context(
    records: list[dict[str, Any]], request_id: str, phase: str, prompt_tokens: int
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if record["request_id"] == request_id and record["phase"] == phase
    ]
    observed: dict[str, Counter[str]] = defaultdict(Counter)
    for record in selected:
        observed[record["family"]][record["variant"]] += 1
    expected = expected_variant_counts()
    mismatches = {}
    for family, family_expected in expected.items():
        family_observed = dict(sorted(observed[family].items()))
        if family_observed != family_expected:
            mismatches[family] = {
                "expected": family_expected,
                "observed": family_observed,
            }

    attention_shape_errors = []
    for record in selected:
        if record["family"] != "attention":
            continue
        if len(record["outputs"]) < 2:
            attention_shape_errors.append(
                {"module": record["module_name"], "reason": "attention weights missing"}
            )
            continue
        observed_shape = record["outputs"][1]["shape"]
        if phase == "prefill":
            expected_kv = prompt_tokens
        elif record["attention"]["layer_type"] == "sliding_attention":
            expected_kv = 512
        else:
            expected_kv = prompt_tokens + 1
        expected_shape = [1, 8, prompt_tokens if phase == "prefill" else 1, expected_kv]
        if observed_shape != expected_shape:
            attention_shape_errors.append(
                {
                    "module": record["module_name"],
                    "expected": expected_shape,
                    "observed": observed_shape,
                }
            )

    lm_heads = [record for record in selected if record["family"] == "lm_head"]
    lm_head_shape_pass = (
        len(lm_heads) == 1
        and lm_heads[0]["input"]["shape"] == [1, 1, 2560]
        and lm_heads[0]["outputs"][0]["shape"] == [1, 1, 262144]
    )
    return {
        "request_id": request_id,
        "phase": phase,
        "record_count": len(selected),
        "expected_variant_counts": expected,
        "observed_variant_counts": {
            family: dict(sorted(counter.items()))
            for family, counter in sorted(observed.items())
        },
        "variant_count_mismatches": mismatches,
        "attention_shape_errors": attention_shape_errors,
        "lm_head_shape_pass": lm_head_shape_pass,
        "pass": not mismatches and not attention_shape_errors and lm_head_shape_pass,
    }


def rmsnorm_semantics_evidence(model: Any) -> dict[str, Any]:
    import inspect

    text_model = get_submodule(model, "model.language_model")
    norm = text_model.layers[0].input_layernorm
    source = inspect.getsource(type(norm))
    sample_names = (
        "model.language_model.layers.0.input_layernorm",
        "model.language_model.layers.0.self_attn.q_norm",
        "model.language_model.norm",
    )
    weight_statistics = {}
    for name in sample_names:
        module = get_submodule(model, name)
        weight = module.weight.detach().float()
        weight_statistics[name] = {
            "shape": list(weight.shape),
            "min": float(weight.min().item()),
            "max": float(weight.max().item()),
            "mean": float(weight.mean().item()),
        }
    v_norm = text_model.layers[0].self_attn.v_norm
    evidence = {
        "class": type(norm).__name__,
        "source_file": inspect.getsourcefile(type(norm)),
        "class_source_sha256": text_sha256(source),
        "source_uses_direct_weight_scale": "normed_output * self.weight.float()" in source,
        "source_uses_one_plus_weight": "1 + self.weight" in source
        or "1.0 + self.weight" in source,
        "v_norm_with_scale": bool(v_norm.with_scale),
        "v_norm_has_weight": hasattr(v_norm, "weight"),
        "weight_statistics": weight_statistics,
        "execution_card_draft_claim": "(1+w) weight semantics",
        "observed_carrier_semantics": "direct weight scale for with_scale norms; no scale for v_norm",
    }
    evidence["draft_claim_contradicted"] = (
        evidence["source_uses_direct_weight_scale"]
        and not evidence["source_uses_one_plus_weight"]
        and not evidence["v_norm_with_scale"]
    )
    if not evidence["draft_claim_contradicted"]:
        raise AssertionError("RMSNorm source-semantics evidence is inconclusive")
    return evidence


def load_formal_nf4(
    repo_root: Path, config: dict[str, Any], device: int
) -> tuple[Any, Any, dict[str, Any]]:
    import bitsandbytes as bnb
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_path = repo_root / config["inputs"]["model"]
    log("OP1-b: re-verifying the complete C02 artifact inventory")
    integrity = verify_formal_artifact(
        model_path, config["inputs"]["model_manifest_sha256"]
    )
    torch.manual_seed(int(config["channel_b"]["seed"]))
    torch.cuda.manual_seed_all(int(config["channel_b"]["seed"]))
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    device_map: dict[str, int | str] = {
        **{name: "cpu" for name in CPU_MODULES},
        **{name: device for name in CUDA_MODULES},
    }
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    load_started = time.perf_counter()
    log("OP1-b: loading the formal carrier as NF4 with persistent CPU embeddings")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        attn_implementation=config["channel_b"]["attention_implementation"],
    )
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    cpu_materialization = materialize_persistent_cpu_modules(model, torch)

    active_quantized_modules = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("model.language_model.")
        and isinstance(module, bnb.nn.Linear4bit)
    ]
    quant_types = Counter(
        str(getattr(getattr(module.weight, "quant_state", None), "quant_type", None))
        for _, module in active_quantized_modules
    )
    devices = Counter(str(module.weight.device) for _, module in active_quantized_modules)
    expected_linear_count = int(config["topology_assertions"]["linear4bit_calls_per_forward"])
    if len(active_quantized_modules) != expected_linear_count:
        raise AssertionError(
            f"active Linear4bit count {len(active_quantized_modules)} != {expected_linear_count}"
        )
    if set(quant_types) != {"nf4"}:
        raise AssertionError(f"active language quant types are not all NF4: {quant_types}")
    if any(not name.startswith(f"cuda:{device}") for name in devices):
        raise AssertionError(f"active NF4 weights are not all on cuda:{device}: {devices}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    load_receipt = {
        "integrity": integrity,
        "load_seconds": load_seconds,
        "device_map_requested": device_map,
        "hf_device_map_observed": getattr(model, "hf_device_map", None),
        "cpu_materialization": cpu_materialization,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "torch.bfloat16",
            "active_language_linear4bit_count": len(active_quantized_modules),
            "quant_types": dict(sorted(quant_types.items())),
            "weight_devices": dict(sorted(devices.items())),
        },
        "memory_after_load": memory_snapshot(torch, device),
    }
    return model, tokenizer, {"bnb": bnb, "receipt": load_receipt}


def run_channel_b(
    repo_root: Path,
    config_path: Path,
    config: dict[str, Any],
    channel_a: dict[str, Any],
    request_index: dict[str, dict[str, Any]],
    device: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("OP1 channel B requires CUDA")
    model, tokenizer, loaded = load_formal_nf4(repo_root, config, device)
    bnb = loaded["bnb"]
    recorder = ModuleShapeRecorder()
    recorder.register(model, bnb)
    monitor = NvidiaSmiMonitor(interval_seconds=1.0)
    monitor.set_phase("op1_profile_setup")
    monitor.start()
    channel_a_rows = {row["request_id"]: row for row in channel_a["requests"]}
    selected_runs: list[dict[str, Any]] = []
    persistent_cpu_before = {
        name: module_storage_snapshot(get_submodule(model, name))
        for name in PERSISTENT_CPU_MODULES
    }
    try:
        for selection in config["channel_b"]["selected_requests"]:
            request_id = selection["request_id"]
            indexed = request_index.get(request_id)
            if indexed is None:
                raise KeyError(f"selected profiler request is absent: {request_id}")
            request = indexed["request"]
            encoded = tokenizer.apply_chat_template(
                request["messages"], tokenize=True, add_generation_prompt=True
            )
            token_ids = [int(value) for value in encoded["input_ids"]]
            static_row = channel_a_rows[request_id]
            if len(token_ids) != static_row["c02_rendered_prompt_tokens"]:
                raise AssertionError(f"channel B token count drift for {request_id}")
            if token_ids_sha256(token_ids) != static_row["c02_rendered_token_ids_sha256"]:
                raise AssertionError(f"channel B token-id identity drift for {request_id}")

            input_ids = torch.tensor([token_ids], dtype=torch.long, device="cpu")
            attention_mask = torch.ones_like(input_ids, device="cpu")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            monitor.set_phase(f"{request_id}:prefill")
            recorder.set_context(request_id, "prefill")
            log(
                f"OP1-b: profiling {request_id} prefill at {len(token_ids)} C02-rendered tokens"
            )
            prefill_result, prefill_profiler, prefill_wall_seconds = profile_callable(
                lambda: run_hybrid_forward(
                    model,
                    input_ids,
                    attention_mask,
                    past_key_values=None,
                    device=device,
                )
            )
            past_key_values, next_token, prefill_metadata = prefill_result

            decode_ids = torch.tensor([[next_token]], dtype=torch.long, device="cpu")
            decode_attention_mask = torch.ones(
                (1, len(token_ids) + 1), dtype=torch.long, device="cpu"
            )
            monitor.set_phase(f"{request_id}:decode")
            recorder.set_context(request_id, "decode")
            log(f"OP1-b: profiling {request_id} one-token incremental decode")
            decode_result, decode_profiler, decode_wall_seconds = profile_callable(
                lambda: run_hybrid_forward(
                    model,
                    decode_ids,
                    decode_attention_mask,
                    past_key_values=past_key_values,
                    device=device,
                )
            )
            next_past, second_token, decode_metadata = decode_result
            recorder.clear_context()
            torch.cuda.synchronize(device)
            selected_runs.append(
                {
                    "request_id": request_id,
                    "selection_reason": selection["selection_reason"],
                    "tape_id": indexed["tape"]["tape_id"],
                    "tape_sha256": indexed["entry"]["sha256"],
                    "b0_reference_prompt_tokens": static_row[
                        "b0_reference_prompt_tokens"
                    ],
                    "b0_reference_completion_tokens": static_row[
                        "b0_reference_completion_tokens"
                    ],
                    "c02_rendered_prompt_tokens": len(token_ids),
                    "c02_rendered_token_ids_sha256": token_ids_sha256(token_ids),
                    "prefill": {
                        "metadata": prefill_metadata,
                        "observer_wall_seconds": prefill_wall_seconds,
                        "profiler": prefill_profiler,
                    },
                    "decode": {
                        "metadata": decode_metadata,
                        "observer_wall_seconds": decode_wall_seconds,
                        "profiler": decode_profiler,
                        "second_token": second_token,
                    },
                    "memory_after_two_phase_probe": memory_snapshot(torch, device),
                }
            )
            del (
                input_ids,
                attention_mask,
                decode_ids,
                decode_attention_mask,
                past_key_values,
                next_past,
            )
            torch.cuda.empty_cache()
    finally:
        recorder.clear_context()
        recorder.remove()
        telemetry = monitor.stop()

    validations = []
    for run in selected_runs:
        request_id = run["request_id"]
        prompt_tokens = run["c02_rendered_prompt_tokens"]
        validations.append(
            validate_recorded_context(
                recorder.records, request_id, "prefill", prompt_tokens
            )
        )
        validations.append(
            validate_recorded_context(
                recorder.records, request_id, "decode", prompt_tokens
            )
        )
    if not validations or not all(validation["pass"] for validation in validations):
        raise AssertionError(f"channel B module-shape validation failed: {validations}")

    persistent_cpu_after = {
        name: module_storage_snapshot(get_submodule(model, name))
        for name in PERSISTENT_CPU_MODULES
    }
    persistent_unchanged = persistent_cpu_before == persistent_cpu_after
    if not persistent_unchanged:
        raise AssertionError("persistent CPU embedding/head storage changed during OP1")
    final_memory = memory_snapshot(torch, device)
    budget_bytes = int(float(config["channel_b"]["gpu_budget_gib"]) * 1024**3)
    budget_checks = {
        field: final_memory[field] < budget_bytes
        for field in ("peak_allocated_bytes", "peak_reserved_bytes", "device_used_bytes")
    }
    if not all(budget_checks.values()):
        raise AssertionError(f"OP1 GPU memory exceeded its budget: {budget_checks}")

    unused_offload_state = {
        name: {
            "hook": hook_snapshot(get_submodule(model, name)),
            "devices_recursive": module_devices(get_submodule(model, name)),
        }
        for name in OFFLOADED_UNUSED_CPU_MODULES
    }
    result = {
        "schema_version": 1,
        "stage": "OP1-channel-B",
        "timestamp": now_iso(),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "carrier": config["carrier"],
        "model_load": loaded["receipt"],
        "profiler_protocol": {
            **config["channel_b"]["torch_profiler"],
            "module_hooks": "metadata-only forward hooks on 343 Linear4bit, 302 RMSNorm, 42 attention, and CPU lm_head calls",
            "performance_numbers_eligible": False,
        },
        "selected_runs": selected_runs,
        "observed_shape_groups": recorder.aggregate(),
        "context_validations": validations,
        "rmsnorm_semantics_evidence": rmsnorm_semantics_evidence(model),
        "persistent_cpu_storage_unchanged": persistent_unchanged,
        "unused_multimodal_offload_state": unused_offload_state,
        "telemetry": telemetry,
        "memory_final": final_memory,
        "gpu_budget_gib": config["channel_b"]["gpu_budget_gib"],
        "gpu_budget_checks": budget_checks,
        "git_sha": git_sha(repo_root),
        "status": "PASS",
    }
    return result


def build_final_census(
    repo_root: Path,
    config_path: Path,
    config: dict[str, Any],
    channel_a: dict[str, Any],
    channel_b: dict[str, Any],
    channel_a_path: Path,
    channel_b_path: Path,
) -> dict[str, Any]:
    profiled_requests = {
        run["request_id"]: run["c02_rendered_prompt_tokens"]
        for run in channel_b["selected_runs"]
    }
    required_requests = {
        selection["request_id"] for selection in config["channel_b"]["selected_requests"]
    }
    selected_coverage_pass = set(profiled_requests) == required_requests
    context_pass = all(
        validation["pass"] for validation in channel_b["context_validations"]
    )
    token_counts_match = all(
        profiled_requests[request_id]
        == next(
            row["c02_rendered_prompt_tokens"]
            for row in channel_a["requests"]
            if row["request_id"] == request_id
        )
        for request_id in required_requests
    )
    rms_semantics = channel_b["rmsnorm_semantics_evidence"]
    if not (
        selected_coverage_pass
        and context_pass
        and token_counts_match
        and rms_semantics["draft_claim_contradicted"]
    ):
        raise AssertionError("OP1 final census gates did not close")
    return {
        "schema_version": 1,
        "stage": "OP1",
        "timestamp": now_iso(),
        "status": "PASS",
        "carrier": config["carrier"],
        "authority": {
            "formal_prefill_shapes": "C02 frozen-template tokenization from channel A, validated at selected lengths by C02-NF4 channel B",
            "decode_frequency_projection": "frozen B0 reference_completion_tokens, with first output charged to prefill",
            "parallel_reference_only": "B0 reference_prompt_tokens and cached_tokens",
            "census_finalization_rule": "observed C02 module topology and shapes override draft assumptions",
        },
        "source_identity": channel_a["source_identity"],
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "channel_receipts": {
            "channel_a": {
                "path": str(channel_a_path),
                "sha256": sha256(channel_a_path),
                "status": channel_a["status"],
            },
            "channel_b": {
                "path": str(channel_b_path),
                "sha256": sha256(channel_b_path),
                "status": channel_b["status"],
            },
        },
        "token_census": {
            **channel_a["token_summary"],
            "prompt_metric_divergence": channel_a["prompt_metric_divergence"],
            "requests": channel_a["requests"],
        },
        "topology": channel_a["topology"],
        "op_families": channel_a["op_families"],
        "channel_b_validation": {
            "selected_requests": profiled_requests,
            "selected_coverage_pass": selected_coverage_pass,
            "token_counts_match_channel_a": token_counts_match,
            "all_context_shape_checks_pass": context_pass,
            "context_validations": channel_b["context_validations"],
            "profiled_linear4bit_quantization": channel_b["model_load"]["quantization"],
            "persistent_cpu_storage_unchanged": channel_b[
                "persistent_cpu_storage_unchanged"
            ],
            "gpu_budget_checks": channel_b["gpu_budget_checks"],
            "profiler_timings_are_not_performance_evidence": True,
        },
        "op2_preregistration_inputs": {
            "k1_candidate": "Gemma4 RMSNorm",
            "authenticity": {
                "calls_per_model_forward": channel_a["topology"][
                    "rmsnorm_calls_per_forward"
                ],
                "decode_forward_count_projected": channel_a["token_summary"][
                    "incremental_decode_forwards"
                ],
                "shape_source": "op_families.rmsnorm.shapes",
            },
            "mandatory_semantics_correction": {
                "draft": rms_semantics["execution_card_draft_claim"],
                "observed": rms_semantics["observed_carrier_semantics"],
                "source_uses_direct_weight_scale": rms_semantics[
                    "source_uses_direct_weight_scale"
                ],
                "source_uses_one_plus_weight": rms_semantics[
                    "source_uses_one_plus_weight"
                ],
                "v_norm_has_weight": rms_semantics["v_norm_has_weight"],
            },
            "old_16k_32k_kv_guess_supported": False,
            "b0_reference_kv_envelope_tokens": [
                min(
                    row["b0_reference_prompt_tokens"]
                    for row in channel_a["requests"]
                ),
                channel_a["token_summary"]["b0_reference_deepest_context_tokens"],
            ],
            "c02_rendered_plus_reference_completion_envelope_tokens": [
                min(
                    row["c02_rendered_prompt_tokens"]
                    for row in channel_a["requests"]
                ),
                channel_a["token_summary"][
                    "c02_projected_deepest_context_tokens"
                ],
            ],
        },
        "limitations": [
            "Channel B profiles one prefill and one incremental decode per selected request; full B0 completion lengths are a frequency projection, not C02-generated observations.",
            "torch.profiler observer timings are excluded from all performance conclusions.",
            "B0 and C02 prompt-token metrics remain separate because every request diverged under exact C02 rendering.",
        ],
        "env_identity_ref": "results/operator/op0/env_identity_20260915.json",
        "git_sha": git_sha(repo_root),
    }


def write_failure(
    repo_root: Path,
    config_path: Path,
    phase: str,
    error: BaseException,
    receipt_path: Path,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    failure_path = (
        repo_root / "results/operator/failures" / f"op1_{phase}_{timestamp}.json"
    )
    failure = {
        "schema_version": 1,
        "stage": "OP1",
        "phase": phase,
        "timestamp": now_iso(),
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "command": sys.argv,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "status": "FAIL",
    }
    if torch.cuda.is_available():
        failure["memory_at_failure"] = memory_snapshot(torch, 0)
    atomic_write_json(failure_path, failure)
    atomic_write_json(
        receipt_path,
        {
            "schema_version": 1,
            "stage": "OP1",
            "phase": phase,
            "failure_ref": str(failure_path),
            "timestamp": now_iso(),
            "status": "FAIL",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/operator/op1_shape_census.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/operator/op1")
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--channel-a-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        (repo_root / args.config).resolve()
        if not args.config.is_absolute()
        else args.config.resolve()
    )
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "op1_receipt.json"
    channel_a_path = output_dir / "channel_a_static.json"
    channel_b_path = output_dir / "channel_b_profiler.json"
    final_path = repo_root / "results/operator/shape_census.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("stage") != "OP1":
        raise ValueError("unexpected OP1 config identity")
    phase = "channel_a"
    try:
        atomic_write_json(
            receipt_path,
            {
                "schema_version": 1,
                "stage": "OP1",
                "timestamp": now_iso(),
                "config": {"path": str(config_path), "sha256": sha256(config_path)},
                "status": "RUNNING",
            },
        )
        log("OP1-a: verifying tapes, rendering all requests, and deriving shapes")
        channel_a, request_index = build_channel_a(repo_root, config_path, config)
        atomic_write_json(channel_a_path, channel_a)
        if args.channel_a_only:
            atomic_write_json(
                receipt_path,
                {
                    "schema_version": 1,
                    "stage": "OP1",
                    "timestamp": now_iso(),
                    "channel_a": {
                        "path": str(channel_a_path),
                        "sha256": sha256(channel_a_path),
                        "status": "PASS",
                    },
                    "status": "CHANNEL_A_PASS",
                },
            )
            log(f"OP1 channel A complete: {channel_a_path}")
            return 0

        phase = "channel_b"
        channel_b = run_channel_b(
            repo_root,
            config_path,
            config,
            channel_a,
            request_index,
            args.device,
        )
        atomic_write_json(channel_b_path, channel_b)

        phase = "finalize"
        census = build_final_census(
            repo_root,
            config_path,
            config,
            channel_a,
            channel_b,
            channel_a_path,
            channel_b_path,
        )
        atomic_write_json(final_path, census)
        receipt = {
            "schema_version": 1,
            "stage": "OP1",
            "timestamp": now_iso(),
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "artifacts": {
                "channel_a": {
                    "path": str(channel_a_path),
                    "sha256": sha256(channel_a_path),
                },
                "channel_b": {
                    "path": str(channel_b_path),
                    "sha256": sha256(channel_b_path),
                },
                "shape_census": {
                    "path": str(final_path),
                    "sha256": sha256(final_path),
                },
            },
            "gates": {
                "channel_a": channel_a["status"],
                "channel_b": channel_b["status"],
                "selected_profile_coverage": "PASS"
                if census["channel_b_validation"]["selected_coverage_pass"]
                else "FAIL",
                "module_counts_and_shapes": "PASS"
                if census["channel_b_validation"]["all_context_shape_checks_pass"]
                else "FAIL",
                "nf4_carrier": "PASS",
                "cpu_embedding_persistence": "PASS"
                if channel_b["persistent_cpu_storage_unchanged"]
                else "FAIL",
                "gpu_budget": "PASS"
                if all(channel_b["gpu_budget_checks"].values())
                else "FAIL",
            },
            "summary": {
                "tapes": channel_a["token_summary"]["tapes"],
                "requests": channel_a["token_summary"]["requests"],
                "c02_prompt_tokens_min": channel_a["token_summary"][
                    "c02_rendered_prompt_tokens"
                ]["min"],
                "c02_prompt_tokens_p50": channel_a["token_summary"][
                    "c02_rendered_prompt_tokens"
                ]["p50"],
                "c02_prompt_tokens_max": channel_a["token_summary"][
                    "c02_rendered_prompt_tokens"
                ]["max"],
                "b0_reference_deepest_context_tokens": channel_a[
                    "token_summary"
                ]["b0_reference_deepest_context_tokens"],
                "c02_projected_deepest_context_tokens": channel_a[
                    "token_summary"
                ]["c02_projected_deepest_context_tokens"],
                "projected_incremental_decode_forwards": channel_a[
                    "token_summary"
                ]["incremental_decode_forwards"],
                "rmsnorm_calls_per_forward": channel_a["topology"][
                    "rmsnorm_calls_per_forward"
                ],
                "linear4bit_calls_per_forward": channel_a["topology"][
                    "linear4bit_calls_per_forward"
                ],
                "rmsnorm_draft_semantics_corrected": True,
            },
            "git_sha": git_sha(repo_root),
            "status": "PASS",
        }
        atomic_write_json(receipt_path, receipt)
        log(f"OP1 complete: {receipt_path}")
        return 0
    except BaseException as error:
        write_failure(repo_root, config_path, phase, error, receipt_path)
        log(f"OP1 failed in {phase}: {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
