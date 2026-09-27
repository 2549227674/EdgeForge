#!/usr/bin/env python3
"""Register an EdgeForge local Gemma 4 endpoint with the pinned BFCL CLI."""

import os

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)


MODEL_ID = os.environ.get("EDGEFORGE_BFCL_MODEL_ID", "edgeforge-gemma4-e4b-FC")
ENDPOINT_MODEL = os.environ.get("EDGEFORGE_BFCL_ENDPOINT_MODEL", "gemma4-e4b")
DISPLAY_NAME = os.environ.get(
    "EDGEFORGE_BFCL_DISPLAY_NAME", "Gemma 4 E4B IT Q4_K_M (EdgeForge, FC)"
)

MODEL_CONFIG = ModelConfig(
    model_name=ENDPOINT_MODEL,
    display_name=DISPLAY_NAME,
    url="https://ai.google.dev/gemma",
    org="Google",
    license="gemma-terms-of-use",
    model_handler=OpenAICompletionsHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    # llama.cpp's tool schema normalizes dotted function names to underscores;
    # let BFCL restore them before AST scoring.
    underscore_to_dot=True,
)
MODEL_CONFIG_MAPPING[MODEL_ID] = MODEL_CONFIG
# BFCL's evaluator normalizes underscores in a model identifier into path
# separators before looking up the handler.  Register that internal spelling as
# well; it is the same endpoint and preserves B0's original identifier.
MODEL_CONFIG_MAPPING.setdefault(MODEL_ID.replace("_", "/"), MODEL_CONFIG)


if __name__ == "__main__":
    from bfcl_eval.__main__ import cli

    cli()
