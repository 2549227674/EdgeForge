"""lm-eval adapter for prompt loglikelihood from the pinned llama.cpp build."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer


@register_model("edgeforge-llamacpp-logits")
class EdgeForgeLlamaCppLoglikelihood(TemplateLM):
    """Evaluate continuation token likelihoods without changing the backend."""

    def __init__(
        self,
        pretrained: str,
        model_path: str,
        scorer_path: str,
        max_length: int = 32768,
        batch_size: int | str = 1,
        device: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self._max_length = int(max_length)
        self._batch_size = batch_size
        self._device = device or "cuda"
        self._tokenizer_path = str(Path(pretrained).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_path, local_files_only=True
        )

        scorer = str(Path(scorer_path).resolve())
        model = str(Path(model_path).resolve())
        self._process = subprocess.Popen(
            [scorer, model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 1

        info = self._request({"op": "ping"})
        if info["n_ctx"] < self._max_length:
            raise RuntimeError(
                f"scorer context {info['n_ctx']} is shorter than max_length "
                f"{self._max_length}"
            )

        sentinel = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Tokenizer parity: A B C?"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        hf_tokens = self.tok_encode(sentinel)
        llama_tokens = self._request(
            {"op": "tokenize", "text": sentinel, "add_special": False}
        )["tokens"]
        if hf_tokens != llama_tokens:
            raise RuntimeError(
                "Hugging Face and GGUF tokenizers disagree on the parity probe"
            )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError(
                f"llama.cpp scorer exited with status {self._process.returncode}"
            )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("llama.cpp scorer pipes are unavailable")

        request_id = self._next_id
        self._next_id += 1
        payload = {"id": request_id, **payload}
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("llama.cpp scorer closed its output pipe")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "unknown scorer error"))
        if response.get("id") != request_id:
            raise RuntimeError("llama.cpp scorer response id mismatch")
        return response

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_path

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 0

    @property
    def batch_size(self) -> int | str:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    def tok_encode(self, string: str, **kwargs: Any) -> list[int]:
        kwargs.setdefault("add_special_tokens", False)
        return self.tokenizer.encode(string, **kwargs)

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    def _loglikelihood_tokens(
        self, requests: list[Any], disable_tqdm: bool = False, **_: Any
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool] | None] = [None] * len(requests)
        grouped: dict[tuple[int, ...], list[tuple[int, Any, list[int]]]] = defaultdict(list)
        for index, (cache_key, context_tokens, continuation_tokens) in enumerate(requests):
            grouped[tuple(context_tokens)].append(
                (index, cache_key, continuation_tokens)
            )

        progress = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="llama.cpp prompt loglikelihood",
        )
        for context_tokens, group in grouped.items():
            can_share_prefix = len(group) > 1 and all(
                len(continuation_tokens) == 1
                for _, _, continuation_tokens in group
            )
            if can_share_prefix:
                response = self._request(
                    {
                        "op": "score_options",
                        "context_tokens": list(context_tokens),
                        "continuations": [
                            continuation_tokens
                            for _, _, continuation_tokens in group
                        ],
                    }
                )
                scores = response["scores"]
                if len(scores) != len(group):
                    raise RuntimeError("llama.cpp scorer returned an incomplete option group")
                for (index, cache_key, _), score in zip(group, scores):
                    result = (
                        float(score["loglikelihood"]),
                        bool(score["is_greedy"]),
                    )
                    results[index] = result
                    self.cache_hook.add_partial("loglikelihood", cache_key, result)
                    progress.update(1)
                continue

            for index, cache_key, continuation_tokens in group:
                response = self._request(
                    {
                        "op": "score",
                        "context_tokens": list(context_tokens),
                        "continuation_tokens": continuation_tokens,
                    }
                )
                result = (
                    float(response["loglikelihood"]),
                    bool(response["is_greedy"]),
                )
                results[index] = result
                self.cache_hook.add_partial("loglikelihood", cache_key, result)
                progress.update(1)
        progress.close()
        if any(result is None for result in results):
            raise RuntimeError("llama.cpp scorer did not return every request")
        return [result for result in results if result is not None]

    def loglikelihood_rolling(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[float]:
        raise NotImplementedError("rolling loglikelihood is not used by MMLU")

    def generate_until(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[str]:
        raise NotImplementedError("generation is not used by MMLU")

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
