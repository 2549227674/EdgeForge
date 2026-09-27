"""Batched dynamic-4K generation for TRL server distillation.

TRL 1.9.2 exposes one global ``max_new_tokens`` value and generates the
gradient-accumulation slices separately. T3 instead generates the complete
optimizer window together, caps the batch by its longest real prompt, then
stores every sequence without left padding or post-EOS padding so the server
loss remains correctly aligned and does not waste forward/backward compute.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, SequentialSampler
from trl.experimental.server_distillation import ServerDistillationTrainer
from trl.models.utils import unwrap_model_for_generation


class _RepeatCollatedBatch:
    """Yield one collated rollout window once for every cached training slice."""

    def __init__(self, dataloader: DataLoader, repeat_count: int) -> None:
        self.dataloader = dataloader
        self.repeat_count = repeat_count

    def __iter__(self):
        for batch in self.dataloader:
            for _ in range(self.repeat_count):
                yield batch

    def __len__(self) -> int:
        return len(self.dataloader) * self.repeat_count

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(epoch)

    def __getattr__(self, name: str):
        return getattr(self.dataloader, name)


class DynamicLengthServerDistillationTrainer(ServerDistillationTrainer):
    """Batch an optimizer window while enforcing a hard total-token envelope."""

    def __init__(
        self,
        *args: Any,
        dynamic_total_length_cap: int,
        dynamic_completion_length_cap: int,
        generation_eos_token_ids: list[int],
        generation_prefill_quadratic_budget: int = 275_000_000,
        rollout_buffer_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        trainer_args = kwargs.get("args")
        if getattr(trainer_args, "use_vllm", False):
            raise ValueError(
                "student vLLM is rejected for T3 before colocated engine construction: vLLM 0.27's native "
                "Gemma 4 path does not correctly initialize C01-S's 18 KV-sharing layers (missing K/V/K-norm "
                "weights and a Q-only 2048 vs QKV 3072 forward split), while model_impl=transformers also "
                "fails on heterogeneous per-layer head_dim. FlashInfer sampler/toolkit repair does not fix "
                "these model-semantics failures; use the qualified Transformers generation path. See "
                "results/m1/t3_student_vllm_compatibility_audit.md."
            )
        optimizer_window = int(getattr(trainer_args, "gradient_accumulation_steps", 1))
        self.rollout_buffer_size = rollout_buffer_size or optimizer_window
        if self.rollout_buffer_size < optimizer_window:
            raise ValueError("rollout_buffer_size must be at least one optimizer window")
        if self.rollout_buffer_size % optimizer_window:
            raise ValueError("rollout_buffer_size must be divisible by gradient_accumulation_steps")
        super().__init__(*args, **kwargs)
        if dynamic_total_length_cap <= 1:
            raise ValueError("dynamic_total_length_cap must be greater than one")
        if not 1 <= dynamic_completion_length_cap < dynamic_total_length_cap:
            raise ValueError("dynamic completion cap must be in [1, total cap)")
        self.dynamic_total_length_cap = dynamic_total_length_cap
        self.dynamic_completion_length_cap = dynamic_completion_length_cap
        if generation_prefill_quadratic_budget <= 0:
            raise ValueError("generation_prefill_quadratic_budget must be positive")
        self.generation_prefill_quadratic_budget = generation_prefill_quadratic_budget
        if not generation_eos_token_ids or any(token_id < 0 for token_id in generation_eos_token_ids):
            raise ValueError("generation_eos_token_ids must contain non-negative token IDs")
        self.generation_config.eos_token_id = list(generation_eos_token_ids)
        self.dynamic_length_events: list[dict[str, int]] = []
        self.dynamic_generation_batches: list[dict[str, float | int]] = []
        self.dynamic_trained_slice_indices: list[int] = []

    def _generation_subbatch_size(self, prompt_width: int, optimizer_window: int) -> int:
        """Cap SDPA prefill work while retaining the full optimizer-step batch."""

        raw_size = max(1, self.generation_prefill_quadratic_budget // (prompt_width**2))
        # Power-of-two subbatches make the exact call structure easy to freeze
        # and reproduce while keeping the proven 32-sequence long-prompt gate.
        power_of_two = 1 << (raw_size.bit_length() - 1)
        return min(optimizer_window, power_of_two)

    def _get_train_sampler(self, dataset=None):
        """Consume the frozen length-packed manifest order without reshuffling."""

        dataset = self.train_dataset if dataset is None else dataset
        if dataset is None:
            raise ValueError("training requires a frozen dataset")
        return SequentialSampler(dataset)

    def get_train_dataloader(self):
        """Load one large rollout window, then consume it in optimizer-sized groups."""

        if self.train_dataset is None:
            raise ValueError("training requires a frozen dataset")
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.rollout_buffer_size,
            sampler=self._get_train_sampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
        prepared = self.accelerator.prepare(dataloader)
        return _RepeatCollatedBatch(prepared, repeat_count=self.rollout_buffer_size)

    def _prepare_inputs(
        self,
        generation_batch: dict[str, torch.Tensor | Any],
    ) -> dict[str, torch.Tensor | Any]:
        """Generate once per rollout buffer and normalize each optimizer group independently."""

        if not self.model.training:
            return generation_batch
        if self._buffer_step % self.rollout_buffer_size == 0 or self._buffered_inputs is None:
            self._fill_buffer(generation_batch, self.rollout_buffer_size)

        slice_idx = self._buffer_step % self.rollout_buffer_size
        optimizer_window = self.args.gradient_accumulation_steps
        optimizer_start = (slice_idx // optimizer_window) * optimizer_window
        optimizer_stop = optimizer_start + optimizer_window
        local_completion_tokens = sum(
            int((entry["labels"] != -100).sum())
            for entry in self._buffered_inputs[optimizer_start:optimizer_stop]
            if entry is not None
        )
        self._buffered_num_items = self.accelerator.gather(
            torch.tensor(local_completion_tokens, device=self.accelerator.device)
        ).sum()
        inputs = self._buffered_inputs[slice_idx]
        self.dynamic_trained_slice_indices.append(slice_idx)
        self._buffer_step += 1
        return inputs

    def _completion_cap(self, prompt_width: int) -> int:
        completion_cap = min(
            self.dynamic_completion_length_cap,
            self.dynamic_total_length_cap - prompt_width,
        )
        if completion_cap <= 0:
            raise ValueError(
                f"prompt width {prompt_width} leaves no room under total cap "
                f"{self.dynamic_total_length_cap}"
            )
        return completion_cap

    def _real_prompt(self, inputs: dict[str, torch.Tensor | Any]) -> torch.Tensor:
        prompts = inputs["prompts"]
        if prompts.shape[0] != 1:
            raise ValueError(
                "T3 batched dynamic generation requires per_device_train_batch_size=1; "
                f"got slice batch {prompts.shape[0]}"
            )
        prompt = prompts[0]
        prompt_mask = inputs.get("prompt_attention_mask")
        if prompt_mask is not None:
            prompt = prompt[prompt_mask[0].bool()]
        elif self.processing_class.pad_token_id is not None:
            non_pad = (prompt != self.processing_class.pad_token_id).nonzero(as_tuple=True)[0]
            if len(non_pad):
                prompt = prompt[non_pad[0] :]
        if not prompt.numel():
            raise ValueError("empty prompts are not valid for on-policy generation")
        return prompt

    def _generate_student_completions(
        self,
        slices: list[dict[str, torch.Tensor | Any]],
        on_policy_indices: list[int],
    ) -> None:
        prompts = [self._real_prompt(slices[index]) for index in on_policy_indices]
        prompt_lengths = torch.tensor(
            [prompt.numel() for prompt in prompts],
            dtype=torch.long,
            device=prompts[0].device,
        )
        prompt_width = int(prompt_lengths.max().item())
        batch_completion_cap = self._completion_cap(prompt_width)
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            raise ValueError("a pad token is required for batched generation")

        padded_prompts = torch.full(
            (len(prompts), prompt_width),
            pad_token_id,
            dtype=prompts[0].dtype,
            device=prompts[0].device,
        )
        prompt_attention_mask = torch.zeros_like(padded_prompts)
        for row_index, prompt in enumerate(prompts):
            padded_prompts[row_index, -prompt.numel() :] = prompt
            prompt_attention_mask[row_index, -prompt.numel() :] = 1

        previous_config_cap = self.generation_config.max_new_tokens
        previous_kwargs_cap = self.generation_kwargs["max_new_tokens"]
        self.generation_config.max_new_tokens = batch_completion_cap
        self.generation_kwargs["max_new_tokens"] = batch_completion_cap
        generation_subbatch_size = self._generation_subbatch_size(prompt_width, len(prompts))
        generation_started = time.perf_counter()
        generated_tokens = 0
        subbatch_count = 0
        try:
            with unwrap_model_for_generation(
                self.model,
                self.accelerator,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model:
                for subbatch_start in range(0, len(prompts), generation_subbatch_size):
                    subbatch_stop = min(
                        subbatch_start + generation_subbatch_size,
                        len(prompts),
                    )
                    generated = unwrapped_model.generate(
                        input_ids=padded_prompts[subbatch_start:subbatch_stop],
                        attention_mask=prompt_attention_mask[subbatch_start:subbatch_stop],
                        generation_config=self.generation_config,
                        return_dict_in_generate=True,
                    ).sequences
                    subbatch_count += 1
                    completion_lengths = self._get_completion_lengths(generated, prompt_width)
                    completion_block = generated[:, prompt_width:]
                    generated_tokens += int(completion_lengths.sum().item())
                    for subbatch_row, row_index in enumerate(range(subbatch_start, subbatch_stop)):
                        slice_index = on_policy_indices[row_index]
                        prompt = prompts[row_index]
                        completion = completion_block[subbatch_row]
                        real_prompt_length = int(prompt.numel())
                        real_completion_length = int(completion_lengths[subbatch_row].item())
                        real_completion = completion[:real_completion_length]
                        generated_without_left_padding = torch.cat([prompt, real_completion]).unsqueeze(0)
                        new_attention_mask, new_labels = self._build_sequence_batch(
                            generated_without_left_padding,
                            real_prompt_length,
                            torch.tensor([real_prompt_length], device=generated.device),
                            torch.tensor([real_completion_length], device=generated.device),
                        )
                        updated = dict(slices[slice_index])
                        updated["input_ids"] = generated_without_left_padding
                        updated["attention_mask"] = new_attention_mask
                        updated["labels"] = new_labels
                        updated["prompts"] = prompt.unsqueeze(0)
                        updated["prompt_attention_mask"] = torch.ones_like(prompt).unsqueeze(0)
                        self._buffered_inputs[slice_index] = updated
                        if self.log_completions:
                            self._buffered_text_logs[slice_index] = (
                                [self.processing_class.decode(prompt.tolist(), skip_special_tokens=False)],
                                [
                                    self.processing_class.decode(
                                        real_completion.tolist(),
                                        skip_special_tokens=False,
                                    )
                                ],
                            )

                        padded_sequence_width = int(generated_without_left_padding.shape[1])
                        if padded_sequence_width > self.dynamic_total_length_cap:
                            raise RuntimeError(
                                f"dynamic length invariant failed: {padded_sequence_width} > "
                                f"{self.dynamic_total_length_cap}"
                            )
                        self.dynamic_length_events.append(
                            {
                                "slice_index": slice_index,
                                "prompt_width": real_prompt_length,
                                "individual_completion_cap": self._completion_cap(real_prompt_length),
                                "batch_completion_cap": batch_completion_cap,
                                "generated_completion_tokens": real_completion_length,
                                "terminal_token_id": int(real_completion[-1].item())
                                if real_completion_length
                                else None,
                                "stopped_by_configured_eos": bool(
                                    real_completion_length
                                    and int(real_completion[-1].item()) in self.generation_config.eos_token_id
                                ),
                                "hit_batch_completion_cap": real_completion_length >= batch_completion_cap,
                                "completion_ids_sha256": hashlib.sha256(
                                    real_completion.detach().cpu().numpy().astype("<i8").tobytes()
                                ).hexdigest(),
                                "padded_sequence_width": padded_sequence_width,
                            }
                        )
        finally:
            self.generation_config.max_new_tokens = previous_config_cap
            self.generation_kwargs["max_new_tokens"] = previous_kwargs_cap
        generation_seconds = time.perf_counter() - generation_started
        self.dynamic_generation_batches.append(
            {
                "prompts": len(prompts),
                "prompt_width": prompt_width,
                "batch_completion_cap": batch_completion_cap,
                "generation_subbatch_size": generation_subbatch_size,
                "generation_subbatches": subbatch_count,
                "prefill_quadratic_budget": self.generation_prefill_quadratic_budget,
                "generated_tokens": generated_tokens,
                "generation_seconds": generation_seconds,
                "generated_tokens_per_second": generated_tokens / generation_seconds,
            }
        )
