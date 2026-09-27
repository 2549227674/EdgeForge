#!/usr/bin/env python3
"""Train one frozen M1 QLoRA recipe with two-rank DDP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_rows(path: Path, batch_size: int = 16) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_step_groups(path: Path) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    current_step: int | None = None
    rows: list[dict[str, Any]] = []
    for row in iter_rows(path):
        step = int(row["optimizer_step"])
        if current_step is None:
            current_step = step
        if step != current_step:
            if step != current_step + 1:
                raise RuntimeError(f"non-contiguous optimizer step {current_step} -> {step}")
            yield current_step, rows
            current_step = step
            rows = []
        rows.append(row)
    if current_step is not None:
        yield current_step, rows


def tensorize(
    row: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
    return input_ids, labels, torch.ones_like(input_ids)


def reduce_sum(values: list[float | int], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


@torch.no_grad()
def evaluate(
    model: DistributedDataParallel,
    path: Path,
    use_cache: bool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    model.eval()
    local_count = 0
    local_loss_sum = 0.0
    local_loss_sq_sum = 0.0
    local_token_weighted_sum = 0.0
    local_supervised = 0
    local_input = 0
    started = time.perf_counter()
    for index, row in enumerate(iter_rows(path)):
        if index % world_size != rank:
            continue
        count = sum(label != -100 for label in row["labels"][1:])
        if not count:
            continue
        input_ids, labels, attention_mask = tensorize(row, device)
        output = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        value = float(output.loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite validation loss on rank {rank}, row {index}")
        local_count += 1
        local_loss_sum += value
        local_loss_sq_sum += value * value
        local_token_weighted_sum += value * count
        local_supervised += count
        local_input += input_ids.numel()
        del output, input_ids, labels, attention_mask
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    count, loss_sum, loss_sq_sum, token_sum, supervised, input_tokens = reduce_sum(
        [
            local_count,
            local_loss_sum,
            local_loss_sq_sum,
            local_token_weighted_sum,
            local_supervised,
            local_input,
        ],
        device,
    )
    elapsed_max = torch.tensor(elapsed, dtype=torch.float64, device=device)
    dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
    if count <= 0 or supervised <= 0:
        raise RuntimeError("holdout has no supervised rows")
    sample_mean = loss_sum / count
    variance = max(0.0, (loss_sq_sum - count * sample_mean * sample_mean) / max(1.0, count - 1))
    return {
        "records": int(count),
        "input_tokens": int(input_tokens),
        "supervised_tokens": int(supervised),
        "sample_mean_loss": sample_mean,
        "sample_loss_standard_error": math.sqrt(variance) / math.sqrt(count),
        "token_weighted_loss": token_sum / supervised,
        "elapsed_seconds": float(elapsed_max.cpu()),
        "world_size": world_size,
    }


def save_checkpoint(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    state: dict[str, Any],
    rank_rng_states: list[dict[str, Any]],
) -> None:
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.module.save_pretrained(adapter_dir, safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rank_rng_states": rank_rng_states,
            "state": state,
        },
        output_dir / "continuation_state.pt",
    )
    (output_dir / "READY").write_text("complete\n", encoding="utf-8")


def require_identity(path: Path, expected: str | None, label: str) -> str:
    actual = sha256(path)
    if expected and actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "race", "full"), default="race")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--race-steps", type=int, default=1024)
    parser.add_argument("--full-steps", type=int, default=8192)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-holdout-sha256")
    parser.add_argument("--expected-template-sha256")
    parser.add_argument("--expected-input-tokens", type=int)
    args = parser.parse_args()

    if not dist.is_torchelastic_launched():
        raise RuntimeError("launch with torchrun; single-process execution is forbidden")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != args.expected_world_size:
        raise RuntimeError(f"world size {world_size} != required {args.expected_world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume_from is None:
            raise FileExistsError(f"output directory is not empty: {args.output_dir}")
        if args.resume_from is not None and not (args.resume_from / "READY").is_file():
            raise FileNotFoundError(f"resume checkpoint is incomplete: {args.resume_from}")
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        plan_sha = require_identity(args.plan, args.expected_plan_sha256, "plan")
        holdout_sha = require_identity(args.holdout, args.expected_holdout_sha256, "holdout")
        template_sha = require_identity(
            args.model / "chat_template.jinja", args.expected_template_sha256, "template"
        )

        # Identical initialization followed by an explicit per-rank runtime seed.
        seed_everything(args.seed)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map={"": local_rank},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        if args.resume_from is not None:
            model = PeftModel.from_pretrained(
                model, args.resume_from / "adapter", is_trainable=True
            )
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules="all-linear",
                    task_type="CAUSAL_LM",
                ),
            )
        model.config.use_cache = args.use_cache
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = args.use_cache
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        rank_seed = args.seed + rank
        seed_everything(rank_seed)
        parameters = [parameter for parameter in ddp_model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
        warmup_steps = round(args.full_steps * args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.full_steps)
        trainable_parameters = sum(parameter.numel() for parameter in parameters)

        resume_payload: dict[str, Any] | None = None
        start_step = 0
        elapsed_offset = 0.0
        evaluations: list[dict[str, Any]] = []
        restored_rank_audit = None
        if args.resume_from is not None:
            resume_payload = torch.load(
                args.resume_from / "continuation_state.pt",
                map_location=device,
                weights_only=False,
            )
            previous_state = resume_payload["state"]
            if previous_state["plan_sha256"] != plan_sha:
                raise ValueError("resume plan SHA does not match current plan")
            if previous_state["holdout_sha256"] != holdout_sha:
                raise ValueError("resume holdout SHA does not match current holdout")
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
            start_step = int(previous_state["completed_steps"])
            elapsed_offset = float(previous_state["elapsed_seconds"])
            evaluations = list(previous_state.get("evaluations") or [])
            restored_rank_audit = next(
                item for item in previous_state["rank_audits"] if item["rank"] == rank
            )
            restored_rng = next(
                item for item in resume_payload["rank_rng_states"] if item["rank"] == rank
            )
            torch.set_rng_state(restored_rng["torch_rng_state"].cpu())
            torch.cuda.set_rng_state(restored_rng["cuda_rng_state"].cpu(), device)
        if not args.skip_evaluation and start_step == 0:
            initial_eval = evaluate(
                ddp_model, args.holdout, args.use_cache, device, rank, world_size
            )
            initial_eval["optimizer_step"] = 0
            evaluations.append(initial_eval)

        log_path = args.output_dir / "train_log.jsonl"
        log_mode = "a" if args.resume_from is not None else "w"
        log = log_path.open(log_mode, encoding="utf-8") if rank == 0 else None
        if log and evaluations and start_step == 0:
            log.write(json.dumps({"event": "validation", **evaluations[0]}, ensure_ascii=False) + "\n")
            log.flush()

        ddp_model.train()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        local_records = int(restored_rank_audit["records"]) if restored_rank_audit else 0
        local_input_tokens = int(restored_rank_audit["input_tokens"]) if restored_rank_audit else 0
        local_supervised_tokens = (
            int(restored_rank_audit["supervised_tokens"]) if restored_rank_audit else 0
        )
        completed_steps = start_step
        limit = args.stop_after_steps or args.race_steps

        for step, rows in iter_step_groups(args.plan):
            if step < start_step:
                continue
            if step >= limit:
                break
            if step != completed_steps:
                raise RuntimeError(f"expected optimizer step {completed_steps}, got {step}")
            local_rows = [row for index, row in enumerate(rows) if index % world_size == rank]
            if not local_rows:
                raise RuntimeError(f"rank {rank} received no rows for optimizer step {step}")
            group_counts = {int(row["optimizer_step_supervised_tokens"]) for row in rows}
            if len(group_counts) != 1:
                raise RuntimeError(f"inconsistent group supervision at step {step}")
            group_count = group_counts.pop()
            optimizer.zero_grad(set_to_none=True)
            local_step_loss = 0.0
            for local_index, row in enumerate(local_rows):
                count = int(row["shifted_supervised_tokens"])
                if count <= 0:
                    continue
                local_records += 1
                local_input_tokens += int(row["scheduled_tokens"])
                local_supervised_tokens += count
                sync_context = (
                    nullcontext()
                    if local_index == len(local_rows) - 1
                    else ddp_model.no_sync()
                )
                with sync_context:
                    input_ids, labels, attention_mask = tensorize(row, device)
                    output = ddp_model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        use_cache=args.use_cache,
                    )
                    contribution = output.loss * (count / group_count)
                    value = float(contribution.detach())
                    if not math.isfinite(value):
                        raise FloatingPointError(
                            f"non-finite loss at optimizer step {step}, rank {rank}"
                        )
                    # DDP averages gradients; multiply by world_size so the
                    # result equals the global token-weighted sum.
                    (contribution * world_size).backward()
                    local_step_loss += value
                    del output, contribution, input_ids, labels, attention_mask
            optimizer.step()
            scheduler.step()
            completed_steps += 1

            global_step_loss = reduce_sum([local_step_loss], device)[0]
            if log:
                payload = {
                    "event": "train_step",
                    "optimizer_step": completed_steps,
                    "weighted_loss": global_step_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
                }
                log.write(json.dumps(payload, ensure_ascii=False) + "\n")
                log.flush()

            if (
                not args.skip_evaluation
                and completed_steps < limit
                and completed_steps % args.eval_every == 0
            ):
                result = evaluate(
                    ddp_model, args.holdout, args.use_cache, device, rank, world_size
                )
                result["optimizer_step"] = completed_steps
                evaluations.append(result)
                if log:
                    log.write(json.dumps({"event": "validation", **result}, ensure_ascii=False) + "\n")
                    log.flush()
                ddp_model.train()

            if (
                not args.no_save
                and args.checkpoint_every > 0
                and completed_steps < limit
                and completed_steps % args.checkpoint_every == 0
            ):
                torch.cuda.synchronize(device)
                checkpoint_elapsed = torch.tensor(
                    time.perf_counter() - started, dtype=torch.float64, device=device
                )
                dist.all_reduce(checkpoint_elapsed, op=dist.ReduceOp.MAX)
                checkpoint_local_audit = {
                    "rank": rank,
                    "local_rank": local_rank,
                    "seed": rank_seed,
                    "records": local_records,
                    "input_tokens": local_input_tokens,
                    "supervised_tokens": local_supervised_tokens,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                }
                checkpoint_rank_audits: list[dict[str, Any] | None] = [None] * world_size
                dist.all_gather_object(checkpoint_rank_audits, checkpoint_local_audit)
                checkpoint_local_rng = {
                    "rank": rank,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(device),
                }
                checkpoint_rank_rng: list[dict[str, Any] | None] = [None] * world_size
                dist.all_gather_object(checkpoint_rank_rng, checkpoint_local_rng)
                checkpoint_state = {
                    "schema_version": 2,
                    "run_kind": args.run_kind,
                    "recipe": args.recipe,
                    "plan_sha256": plan_sha,
                    "holdout_sha256": holdout_sha,
                    "template_sha256": template_sha,
                    "completed_steps": completed_steps,
                    "full_schedule_steps": args.full_steps,
                    "elapsed_seconds": elapsed_offset + float(checkpoint_elapsed.cpu()),
                    "evaluations": evaluations,
                    "rank_audits": checkpoint_rank_audits,
                }
                checkpoint_dir = (
                    args.output_dir / "checkpoints" / f"step-{completed_steps:05d}"
                )
                if rank == 0:
                    if checkpoint_dir.exists():
                        raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
                    save_checkpoint(
                        ddp_model,
                        optimizer,
                        scheduler,
                        checkpoint_dir,
                        checkpoint_state,
                        [item for item in checkpoint_rank_rng if item is not None],
                    )
                    if log:
                        log.write(
                            json.dumps(
                                {
                                    "event": "checkpoint",
                                    "optimizer_step": completed_steps,
                                    "path": str(checkpoint_dir),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        log.flush()
                dist.barrier()

        expected_completed = limit
        if completed_steps != expected_completed:
            raise RuntimeError(
                f"completed {completed_steps} optimizer steps, expected {expected_completed}"
            )
        if not args.skip_evaluation:
            final_eval = evaluate(
                ddp_model, args.holdout, args.use_cache, device, rank, world_size
            )
            final_eval["optimizer_step"] = completed_steps
            evaluations.append(final_eval)
            if log:
                log.write(json.dumps({"event": "validation", **final_eval}, ensure_ascii=False) + "\n")
                log.flush()
        else:
            final_eval = None
        if log:
            log.close()

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        totals = reduce_sum(
            [local_records, local_input_tokens, local_supervised_tokens], device
        )
        elapsed_max = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
        peak = torch.tensor(
            [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
            dtype=torch.float64,
            device=device,
        )
        gathered_peak = [torch.zeros_like(peak) for _ in range(world_size)]
        dist.all_gather(gathered_peak, peak)
        local_audit = {
            "rank": rank,
            "local_rank": local_rank,
            "seed": rank_seed,
            "records": local_records,
            "input_tokens": local_input_tokens,
            "supervised_tokens": local_supervised_tokens,
            "peak_allocated_gib": float(peak[0].cpu()) / 2**30,
            "peak_reserved_gib": float(peak[1].cpu()) / 2**30,
        }
        rank_audits: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_audits, local_audit)
        local_rng = {
            "rank": rank,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
        }
        rank_rng_states: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_rng_states, local_rng)

        state = {
            "schema_version": 2,
            "run_kind": args.run_kind,
            "recipe": args.recipe,
            "base_seed": args.seed,
            "rank_seeds": [args.seed + index for index in range(world_size)],
            "world_size": world_size,
            "race_steps": args.race_steps,
            "completed_steps": completed_steps,
            "full_schedule_steps": args.full_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": args.learning_rate,
            "use_cache": args.use_cache,
            "activation_checkpointing": False,
            "attention_implementation": "sdpa",
            "plan_sha256": plan_sha,
            "holdout_sha256": holdout_sha,
            "template_sha256": template_sha,
            "records": int(totals[0]),
            "input_tokens": int(totals[1]),
            "supervised_tokens": int(totals[2]),
            "expected_input_tokens": args.expected_input_tokens,
            "elapsed_seconds": elapsed_offset + float(elapsed_max.cpu()),
            "input_tokens_per_second": int(totals[1])
            / (elapsed_offset + float(elapsed_max.cpu())),
            "trainable_parameters": trainable_parameters,
            "rank_audits": rank_audits,
            "peak_allocated_gib_by_rank": [float(item[0].cpu()) / 2**30 for item in gathered_peak],
            "peak_reserved_gib_by_rank": [float(item[1].cpu()) / 2**30 for item in gathered_peak],
            "evaluations": evaluations,
            "final_validation": final_eval,
        }
        if args.expected_input_tokens is not None and int(totals[1]) != args.expected_input_tokens:
            raise RuntimeError(
                f"global input tokens {int(totals[1])} != expected {args.expected_input_tokens}"
            )
        if rank == 0:
            if not args.no_save:
                save_checkpoint(
                    ddp_model,
                    optimizer,
                    scheduler,
                    args.output_dir,
                    state,
                    [item for item in rank_rng_states if item is not None],
                )
            report_path = args.output_dir / "report.json"
            report_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(state, ensure_ascii=False, indent=2))
        dist.barrier()
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
