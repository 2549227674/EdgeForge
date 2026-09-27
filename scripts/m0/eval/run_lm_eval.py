#!/usr/bin/env python3
"""Launch lm-eval's API backends with the installed Transformers 5 runtime.

lm-eval 0.4.9.1 eagerly imports its optional Hugging Face vision-language
registrar.  That registrar references a Transformers 4-only symbol, although
the API-only backends used by this baseline never use it.  Registering an empty
module here leaves the API evaluators untouched while allowing the Gemma 4
tokenizer to be loaded by Transformers 5.
"""

import sys
from types import ModuleType


sys.modules.setdefault("lm_eval.models.hf_vlms", ModuleType("lm_eval.models.hf_vlms"))

import edgeforge_llamacpp_loglikelihood  # noqa: F401, E402
from lm_eval.__main__ import cli_evaluate


if __name__ == "__main__":
    raise SystemExit(cli_evaluate())
