#!/usr/bin/env bash
# C0-5: run the first frozen prompt through the manual Gemma 4 template.
set -euo pipefail

if [ "$#" -ne 4 ]; then
  printf 'usage: %s RUNNER MODEL EXPECTED_MODEL_SHA256 OUTPUT_DIR\n' "$0" >&2
  exit 64
fi

runner=$1
model=$2
expected_model_sha=$3
output_dir=$4

test -x "$runner"
test -f "$model"
if [ "${#expected_model_sha}" -ne 64 ] || printf '%s' "$expected_model_sha" | grep -Eqv '^[0-9a-f]{64}$'; then
  printf 'invalid expected SHA-256\n' >&2
  exit 64
fi
if [ -e "$output_dir" ]; then
  printf 'refusing to reuse output directory: %s\n' "$output_dir" >&2
  exit 73
fi
mkdir -p "$output_dir"

observed_model_sha=$(sha256sum "$model")
observed_model_sha=${observed_model_sha%% *}
if [ "$observed_model_sha" != "$expected_model_sha" ]; then
  printf 'model SHA-256 mismatch: expected=%s observed=%s\n' \
    "$expected_model_sha" "$observed_model_sha" >&2
  exit 65
fi

prompt='请用一句话解释什么是模型量化。'
transcript="$output_dir/transcript.log"
started_at=$(date -Is)
started_uptime=$(cut -d' ' -f1 /proc/uptime)
set +e
printf '%s\nexit\n' "$prompt" \
  | timeout --signal=TERM --kill-after=30 900 "$runner" "$model" 64 16384 \
    >"$transcript" 2>&1
runner_exit=$?
set -e
ended_at=$(date -Is)
ended_uptime=$(cut -d' ' -f1 /proc/uptime)

utf8_valid=false
init_success=false
template_success=false
pad_token_seen=false
replacement_character_seen=false
nonempty_output=false
iconv -f UTF-8 -t UTF-8 "$transcript" >/dev/null 2>&1 && utf8_valid=true
grep -Fq 'rkllm init success' "$transcript" && init_success=true
grep -Fq 'manual_gemma4_template_ret=0' "$transcript" && template_success=true
grep -Fq '[PAD]' "$transcript" && pad_token_seen=true
LC_ALL=C grep -aq $'\357\277\275' "$transcript" && replacement_character_seen=true
grep -Eq '^robot: .+' "$transcript" && nonempty_output=true

status=FAIL
if [ "$runner_exit" -eq 0 ] && [ "$utf8_valid" = true ] \
  && [ "$init_success" = true ] && [ "$template_success" = true ] \
  && [ "$pad_token_seen" = false ] \
  && [ "$replacement_character_seen" = false ] \
  && [ "$nonempty_output" = true ]; then
  status=PASS
fi

RUNNER="$runner" MODEL="$model" EXPECTED_MODEL_SHA="$expected_model_sha" \
OUTPUT_DIR="$output_dir" STATUS="$status" RUNNER_EXIT="$runner_exit" \
UTF8_VALID="$utf8_valid" INIT_SUCCESS="$init_success" \
TEMPLATE_SUCCESS="$template_success" PAD_TOKEN_SEEN="$pad_token_seen" \
REPLACEMENT_CHARACTER_SEEN="$replacement_character_seen" \
NONEMPTY_OUTPUT="$nonempty_output" STARTED_AT="$started_at" ENDED_AT="$ended_at" \
STARTED_UPTIME="$started_uptime" ENDED_UPTIME="$ended_uptime" \
PROMPT="$prompt" python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

output = Path(os.environ["OUTPUT_DIR"])
transcript = output / "transcript.log"
payload = {
    "schema": "edgeforge-ope-c0-5-board-smoke/v1",
    "status": os.environ["STATUS"],
    "board_clock_trusted": False,
    "started_at_board_clock": os.environ["STARTED_AT"],
    "ended_at_board_clock": os.environ["ENDED_AT"],
    "started_uptime_seconds": float(os.environ["STARTED_UPTIME"]),
    "ended_uptime_seconds": float(os.environ["ENDED_UPTIME"]),
    "runner": {"path": os.environ["RUNNER"], "sha256": sha256(Path(os.environ["RUNNER"]))},
    "model": {
        "path": os.environ["MODEL"],
        "bytes": Path(os.environ["MODEL"]).stat().st_size,
        "sha256": os.environ["EXPECTED_MODEL_SHA"],
    },
    "contract": {
        "max_new_tokens": 64,
        "max_context_len": 16384,
        "top_k": 1,
        "template": "manual frozen Gemma 4 single-turn",
        "prompt": os.environ["PROMPT"],
    },
    "checks": {
        "runner_exit": int(os.environ["RUNNER_EXIT"]),
        "init_success": os.environ["INIT_SUCCESS"] == "true",
        "template_success": os.environ["TEMPLATE_SUCCESS"] == "true",
        "utf8_valid": os.environ["UTF8_VALID"] == "true",
        "pad_token_seen": os.environ["PAD_TOKEN_SEEN"] == "true",
        "replacement_character_seen": os.environ["REPLACEMENT_CHARACTER_SEEN"] == "true",
        "nonempty_output": os.environ["NONEMPTY_OUTPUT"] == "true",
    },
    "transcript": {"path": str(transcript), "bytes": transcript.stat().st_size, "sha256": sha256(transcript)},
}
(output / "receipt.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": payload["status"], "receipt": str(output / "receipt.json")}))
PY

test "$status" = PASS
