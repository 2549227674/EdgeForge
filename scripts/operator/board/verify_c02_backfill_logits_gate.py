#!/usr/bin/env python3
"""C0-1b PC logits-equivalence gate for original C02 vs the transplant variant.

The preferred backend is BF16 on a GPU with at least 32 GiB memory. It avoids
bitsandbytes cross-process nondeterminism and is the authoritative path on the
48 GiB C0 cloud host. ``nf4`` remains a local 8 GiB fallback and imports the
already-proven CPU-embedding materialization and manual greedy loop from
``scripts/operator/load_smoke_nf4.py``.

Each carrier loads in a separate child process. Three prompts each produce
exactly 32 greedy tokens. If A/B differs, A/A is rerun to distinguish a
cross-process instability from an artifact-specific difference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any


PROMPTS = (
    "Write a bash one-liner that counts unique IP addresses in access.log.",
    "用一句话解释什么是 KV cache。",
    "def fib(n):",
)
MAX_NEW_TOKENS = 32
SEED = 20260917
TRANSPLANT_KEYS = tuple(
    f"model.language_model.layers.{layer}.self_attn.{suffix}.weight"
    for layer in range(24, 42)
    for suffix in ("k_proj", "v_proj", "k_norm")
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def environment_identity() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "safetensors",
                "tokenizers",
            )
        },
        "nvidia_smi": (completed.stdout + completed.stderr).strip(),
    }


def token_sha256(values: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def artifact_tensor_keys(model_dir: Path) -> set[str]:
    """Read the serialized tensor key set without instantiating Transformers."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return set(index["weight_map"])
    files = sorted(model_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError(
            f"expected one unindexed safetensors file in {model_dir}, found {len(files)}"
        )
    with files[0].open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {files[0]}")
        (header_length,) = struct.unpack("<Q", prefix)
        header = json.loads(handle.read(header_length))
    return {key for key in header if key != "__metadata__"}


def load_nf4_helpers() -> Any:
    helper_path = Path(__file__).resolve().parents[1] / "load_smoke_nf4.py"
    spec = importlib.util.spec_from_file_location("edgeforge_load_smoke_nf4", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import NF4 helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = [json_safe(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    return str(value)


def normalize_loading_info(info: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        value = info.get(key, [])
        if key == "mismatched_keys":
            normalized[key] = json_safe(value)
        else:
            normalized[key] = sorted(str(item) for item in value)
    error_messages = info.get("error_msgs", info.get("error_messages", []))
    normalized["error_messages"] = json_safe(error_messages)
    return normalized


def load_gemma4_config(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load config, working around the Transformers 5.14.1 validator bug."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        return config, {"mode": "native", "override_applied": False}
    except Exception as exc:
        if type(exc).__name__ != "AmbiguousGlobalPerLayerAttributeError":
            raise
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

        raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        raw["text_config"]["allow_global_per_layer_attribute_access"] = True
        config = Gemma4Config.from_dict(raw)
        return config, {
            "mode": "transformers_5_14_1_gemma4_heterogeneous_head_dim_workaround",
            "override_applied": True,
            "trigger_error_type": type(exc).__name__,
            "scope": "config construction only; both A and B use the same config",
        }


def load_model_bf16(
    model_dir: Path, device: int, torch: Any
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    from transformers import AutoModelForImageTextToText

    config, config_compatibility = load_gemma4_config(model_dir)

    loaded = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    model, loading_info = loaded
    return model, normalize_loading_info(loading_info), config_compatibility


def load_model_nf4(
    model_dir: Path, device: int, torch: Any
) -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    helper = load_nf4_helpers()
    device_map: dict[str, int | str] = {
        **{name: "cpu" for name in helper.CPU_MODULES},
        **{name: device for name in helper.CUDA_MODULES},
    }
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    config, config_compatibility = load_gemma4_config(model_dir)
    loaded = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    model, loading_info = loaded
    helper.materialize_persistent_cpu_modules(model, torch)
    return model, normalize_loading_info(loading_info), helper, config_compatibility


def worker(model_dir: Path, backend: str, device: int, seed: int) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device < 0 or device >= torch.cuda.device_count():
        raise ValueError(f"invalid CUDA device index: {device}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    helper = None
    if backend == "bf16":
        model, loading_info, config_compatibility = load_model_bf16(
            model_dir, device, torch
        )
    elif backend == "nf4":
        model, loading_info, helper, config_compatibility = load_model_nf4(
            model_dir, device, torch
        )
    else:
        raise ValueError(f"unsupported worker backend: {backend}")
    model.eval()
    state_dict_keys = set(model.state_dict())
    module_tree_audit = {
        "state_dict_key_count": len(state_dict_keys),
        "transplant_keys_present": sorted(state_dict_keys & set(TRANSPLANT_KEYS)),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, config=model.config
    )

    sequences: list[list[int]] = []
    prompt_rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(PROMPTS):
        encoded = tokenizer(prompt, return_tensors="pt")
        if backend == "bf16":
            input_ids = encoded["input_ids"].to(f"cuda:{device}")
            attention_mask = encoded["attention_mask"].to(f"cuda:{device}")
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    min_new_tokens=MAX_NEW_TOKENS,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            tokens = generated[0, input_ids.shape[1] :].tolist()
        else:
            assert helper is not None
            generation = helper.run_text_only_greedy(
                model,
                encoded["input_ids"],
                MAX_NEW_TOKENS,
                device,
                torch,
            )
            tokens = generation["generated_token_ids"]
        if len(tokens) != MAX_NEW_TOKENS:
            raise ValueError(
                f"prompt {prompt_index} generated {len(tokens)} tokens, "
                f"expected {MAX_NEW_TOKENS}"
            )
        sequences.append(tokens)
        prompt_rows.append(
            {
                "prompt_index": prompt_index,
                "input_tokens": int(encoded["input_ids"].shape[1]),
                "generated_tokens": len(tokens),
                "generated_token_ids_sha256": token_sha256(tokens),
            }
        )

    free, total = torch.cuda.mem_get_info(device)
    return {
        "model_dir": str(model_dir),
        "backend": backend,
        "device": device,
        "seed": seed,
        "sequences": sequences,
        "prompts": prompt_rows,
        "loading_info": loading_info,
        "module_tree_audit": module_tree_audit,
        "config_compatibility": config_compatibility,
        "cuda_memory": {
            "total_bytes": total,
            "free_bytes": free,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
    }


def run_worker(model_dir: Path, backend: str, device: int, seed: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        str(model_dir),
        "--backend",
        backend,
        "--device",
        str(device),
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    for line in completed.stdout.splitlines():
        if line.startswith("WORKER_JSON:"):
            result = json.loads(line[len("WORKER_JSON:") :])
            result["worker_stderr_tail"] = completed.stderr[-4000:]
            result["worker_returncode"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(
                    f"worker emitted data but exited {completed.returncode}: "
                    f"{completed.stderr[-4000:]}"
                )
            return result
    raise RuntimeError(
        f"worker failed for {model_dir} with exit {completed.returncode}\n"
        f"stdout:\n{completed.stdout[-2000:]}\nstderr:\n{completed.stderr[-6000:]}"
    )


def sequence_diff(left: list[list[int]], right: list[list[int]]) -> list[dict[str, Any]]:
    if len(left) != len(right):
        return [{"reason": "prompt_count", "left": len(left), "right": len(right)}]
    rows: list[dict[str, Any]] = []
    for prompt_index, (first, second) in enumerate(zip(left, right)):
        if first == second:
            continue
        divergence = next(
            (
                token_index
                for token_index, (a_token, b_token) in enumerate(zip(first, second))
                if a_token != b_token
            ),
            min(len(first), len(second)),
        )
        rows.append(
            {
                "prompt_index": prompt_index,
                "first_divergence_index": divergence,
                "left_length": len(first),
                "right_length": len(second),
            }
        )
    return rows


def select_backend(requested: str, device: int) -> tuple[str, int | None]:
    if requested != "auto":
        return requested, None
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    total = torch.cuda.get_device_properties(device).total_memory
    return ("bf16" if total >= 32 * 1024**3 else "nf4"), total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--original", type=Path, default=Path("models/C02-S.O-bf16-r1"))
    parser.add_argument(
        "--backfilled", type=Path, default=Path("models/C02-S.O-rkllm-input-r1")
    )
    parser.add_argument("--backend", choices=("auto", "bf16", "nf4"), default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("results/operator/board/c0/c0_1b_logits_gate_receipt.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.model is None:
            raise SystemExit("--worker requires --model")
        if args.backend == "auto":
            raise SystemExit("worker requires an explicit backend")
        result = worker(args.model, args.backend, args.device, args.seed)
        print("WORKER_JSON:" + json.dumps(result, ensure_ascii=False), flush=True)
        return 0

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "C0-1b-logits-gate",
        "card": "docs/operator/EdgeForge_算子面_板端_执行卡_R2.md",
        "timestamp_start": now_iso(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "environment": environment_identity(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "prompts": list(PROMPTS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": args.seed,
        "original": str(args.original),
        "backfilled": str(args.backfilled),
        "status": "FAIL",
    }
    try:
        if not args.original.is_dir():
            raise FileNotFoundError(f"missing original model: {args.original}")
        if not args.backfilled.is_dir():
            raise FileNotFoundError(f"missing backfilled model: {args.backfilled}")
        backend, auto_total = select_backend(args.backend, args.device)
        receipt["backend_requested"] = args.backend
        receipt["backend_selected"] = backend
        receipt["auto_detected_cuda_total_bytes"] = auto_total

        reference = run_worker(args.original, backend, args.device, args.seed)
        candidate = run_worker(args.backfilled, backend, args.device, args.seed)
        receipt["reference"] = reference
        receipt["candidate"] = candidate

        mismatch = sequence_diff(reference["sequences"], candidate["sequences"])
        receipt["ab_mismatch"] = mismatch
        reference_artifact_keys = artifact_tensor_keys(args.original)
        candidate_artifact_keys = artifact_tensor_keys(args.backfilled)
        candidate_unexpected = set(candidate["loading_info"]["unexpected_keys"])
        reference_unexpected = set(reference["loading_info"]["unexpected_keys"])
        expected_unexpected = set(TRANSPLANT_KEYS)
        loading_audit = {
            "expected_transplant_count": len(expected_unexpected),
            "reference_serialized_tensor_count": len(reference_artifact_keys),
            "candidate_serialized_tensor_count": len(candidate_artifact_keys),
            "reference_serialized_transplant_count": len(
                reference_artifact_keys & expected_unexpected
            ),
            "candidate_serialized_transplant_count": len(
                candidate_artifact_keys & expected_unexpected
            ),
            "reference_module_tree_transplant_keys": reference["module_tree_audit"][
                "transplant_keys_present"
            ],
            "candidate_module_tree_transplant_keys": candidate["module_tree_audit"][
                "transplant_keys_present"
            ],
            "candidate_loader_reported_unexpected_count": len(
                candidate_unexpected & expected_unexpected
            ),
            "candidate_not_reported_unexpected": sorted(
                expected_unexpected - candidate_unexpected
            ),
            "reference_loader_reported_transplants_unexpected": sorted(
                reference_unexpected & expected_unexpected
            ),
            "candidate_loader_other_unexpected": sorted(
                candidate_unexpected - expected_unexpected
            ),
            "reference_loader_other_unexpected": sorted(reference_unexpected),
            "interpretation": (
                "Transformers 5.14.1 prefilters single-file tensors against the module "
                "tree, so discarded extras need not appear in output_loading_info"
            ),
        }
        receipt["loading_audit"] = loading_audit
        loading_pass = (
            loading_audit["reference_serialized_transplant_count"] == 0
            and loading_audit["candidate_serialized_transplant_count"] == 54
            and not loading_audit["reference_module_tree_transplant_keys"]
            and not loading_audit["candidate_module_tree_transplant_keys"]
        )

        if not mismatch and loading_pass:
            receipt["status"] = "PASS"
            receipt["classification"] = (
                "three 32-token greedy sequences are bitwise identical; all 54 "
                "tensors exist in the derived serialization but not in either module tree"
            )
        elif not mismatch:
            receipt["status"] = "FAIL_LOADING_AUDIT"
            receipt["classification"] = (
                "sequences match, but serialization/module-tree exclusion was not proven"
            )
        else:
            reference_rerun = run_worker(args.original, backend, args.device, args.seed)
            aa_mismatch = sequence_diff(
                reference["sequences"], reference_rerun["sequences"]
            )
            receipt["reference_rerun"] = reference_rerun
            receipt["aa_rerun_mismatch"] = aa_mismatch
            if aa_mismatch:
                receipt["status"] = "INCONCLUSIVE_CROSS_PROCESS_NONDETERMINISM"
                receipt["classification"] = (
                    "A/A also differs; rerun with --backend bf16 on the 48 GiB host"
                )
            else:
                receipt["status"] = "FAIL"
                receipt["classification"] = (
                    "A/A is stable while A/B differs; C0-3 is blocked"
                )
        receipt["timestamp_end"] = now_iso()
        atomic_write_json(args.receipt, receipt)
        print(f"{receipt['status']}: receipt={args.receipt}", flush=True)
        return 0 if receipt["status"] == "PASS" else 1
    except BaseException as exc:
        receipt.update(
            {
                "status": "FAIL",
                "timestamp_end": now_iso(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(args.receipt, receipt)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
