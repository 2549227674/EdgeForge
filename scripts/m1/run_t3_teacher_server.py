#!/usr/bin/env python3
"""Run TRL's teacher server with the T3-only vLLM capacity knobs.

TRL 1.9.2 intentionally exposes the distillation-specific HTTP protocol, but
its ``vllm-serve`` CLI does not expose vLLM's online quantization or
``max_num_seqs`` arguments.  This wrapper keeps the upstream server and wire
protocol intact, injects those capacity arguments, and optionally bridges the
Transformers 5 heterogeneous Gemma 4 config to vLLM's legacy global fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from multiprocessing.connection import Connection
from pathlib import Path

from trl import TrlParser
from trl.scripts import vllm_serve as upstream


_UPSTREAM_LLM_WORKER = upstream.llm_worker


@dataclass
class T3TeacherServerArguments(upstream.ScriptArguments):
    """Upstream TRL server arguments plus the two missing T3 capacity knobs."""

    quantization: str | None = field(
        default=None,
        metadata={
            "help": "Online vLLM quantization method (for example 'fp8'). "
            "Leave unset for the model's native precision."
        },
    )
    max_num_seqs: int | None = field(
        default=None,
        metadata={"help": "Maximum number of sequences scheduled by each vLLM engine iteration."},
    )
    language_model_only: bool = field(
        default=False,
        metadata={
            "help": "Disable multimodal inputs and initialize only the language-model path. The T3 teacher "
            "receives token IDs only, so this is the intended mode for the Transformers backend."
        },
    )
    allow_global_per_layer_attribute_access: bool = field(
        default=False,
        metadata={
            "help": "Bridge Transformers 5's heterogeneous Gemma 4 config to the legacy global fields expected "
            "by vLLM 0.27. The values are read from the checkpoint config.json; the checkpoint is not modified."
        },
    )


def _gemma4_legacy_text_overrides(model: str) -> dict[str, object]:
    """Recover vLLM's legacy global fields from an unmodified local config."""

    config_path = Path(model) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "--allow-global-per-layer-attribute-access requires a local model "
            f"with a readable config.json; failed to read {config_path}: {exc}"
        ) from exc

    text_config = config.get("text_config", config)
    overrides: dict[str, object] = {
        "allow_global_per_layer_attribute_access": True,
    }
    for key in ("global_head_dim", "num_global_key_value_heads"):
        value = text_config.get(key)
        if value is not None:
            overrides[key] = value

    if "global_head_dim" not in overrides:
        raise ValueError(
            f"{config_path} has no text_config.global_head_dim; refusing to invent a compatibility value"
        )
    return overrides


def t3_llm_worker(
    script_args: T3TeacherServerArguments,
    data_parallel_rank: int,
    master_port: int,
    connection: Connection,
) -> None:
    """Inject T3-only kwargs, then delegate all behavior to TRL upstream."""

    import vllm

    real_llm = vllm.LLM

    def configured_llm(*args, **kwargs):
        if script_args.quantization is not None:
            kwargs["quantization"] = script_args.quantization
        if script_args.max_num_seqs is not None:
            kwargs["max_num_seqs"] = script_args.max_num_seqs
        if script_args.language_model_only:
            kwargs["language_model_only"] = True
        if script_args.allow_global_per_layer_attribute_access:
            kwargs["hf_overrides"] = {
                "text_config": _gemma4_legacy_text_overrides(script_args.model),
            }
        return real_llm(*args, **kwargs)

    vllm.LLM = configured_llm
    _UPSTREAM_LLM_WORKER(script_args, data_parallel_rank, master_port, connection)


def main() -> None:
    parser = TrlParser(T3TeacherServerArguments)
    (script_args,) = parser.parse_args_and_config()
    upstream.llm_worker = t3_llm_worker
    upstream.main(script_args)


if __name__ == "__main__":
    main()
