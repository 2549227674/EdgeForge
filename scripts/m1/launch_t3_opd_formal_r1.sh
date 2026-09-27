#!/usr/bin/env bash
set -euo pipefail

workspace=/root/autodl-tmp/edgeforge-m1/workspace
python_bin=/root/autodl-tmp/edgeforge-m1/t3-venv-py312/bin/python
prompt_pool=results/t3/opd_prompt_pool_formal_3840/prompts.parquet
prompt_pool_sha=e005b5f0f7ed4e01a1f2317f4821787cfed7a48e7a6ec46c84547666f2e5d5ca
state_dir=results/t3/c02_s_opd_formal_r1_state
receipt=results/m1/t3_opd_formal_r1_receipt.json
log=results/m1/t3_opd_formal_r1.log
pid_file=results/m1/t3_opd_formal_r1.pid

cd "$workspace"
actual_prompt_pool_sha=$(sha256sum "$prompt_pool" | awk '{print $1}')
if [[ "$actual_prompt_pool_sha" != "$prompt_pool_sha" ]]; then
  echo "prompt pool SHA mismatch: expected=$prompt_pool_sha actual=$actual_prompt_pool_sha" >&2
  exit 1
fi
if [[ -f "$pid_file" ]]; then
  prior_pid=$(<"$pid_file")
  if kill -0 "$prior_pid" 2>/dev/null; then
    echo "formal T3 is already running as PID $prior_pid" >&2
    exit 1
  fi
fi
if ! curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
  echo "teacher health endpoint is not ready" >&2
  exit 1
fi

mkdir -p "$(dirname "$log")" "$state_dir"
nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  "$python_bin" -u scripts/m1/run_t3_dynamic_length_smoke.py \
  --model models/C01-S-bf16-r1 \
  --plan results/t1/family_80_20_half_plan_r1/family_80_20.parquet \
  --prompt-pool "$prompt_pool" \
  --prompt-pool-offset 0 \
  --prompt-pool-count 3840 \
  --teacher-url http://127.0.0.1:8000 \
  --rank 256 \
  --alpha 16 \
  --learning-rate 5e-5 \
  --weight-decay 0 \
  --student-base-precision bf16 \
  --total-length-cap 4096 \
  --prompt-length-cap 3072 \
  --completion-length-cap 3072 \
  --generation-batch-size 64 \
  --optimizer-batch-size 4 \
  --max-steps 960 \
  --warmup-steps 29 \
  --lr-scheduler-type cosine \
  --state-output-dir "$state_dir" \
  --save-steps 320 \
  --save-total-limit 2 \
  --save-final-model \
  --seed 20260811 \
  --output "$receipt" \
  </dev/null >"$log" 2>&1 &
formal_pid=$!
printf '%s\n' "$formal_pid" >"$pid_file"
sleep 5
if ! kill -0 "$formal_pid" 2>/dev/null; then
  tail -n 80 "$log" >&2
  exit 1
fi
echo "PID=$formal_pid"
echo "LOG=$workspace/$log"
echo "RECEIPT=$workspace/$receipt"
echo "STATE_DIR=$workspace/$state_dir"
