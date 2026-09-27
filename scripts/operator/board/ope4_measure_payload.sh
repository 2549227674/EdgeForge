#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 8 )); then
  echo "usage: $0 OUTPUT_DIR PHASE_FILE RUNNER PERF PROMPTS ORDER C02_MODEL OFFICIAL_MODEL" >&2
  exit 2
fi

output_dir=$1
phase_file=$2
runner=$3
perf_bin=$4
prompts=$5
order=$6
c02_model=$7
official_model=$8
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)

case "$order" in
  c02_a2-official_a1)
    carriers=(c02_a2 official_a1)
    ;;
  official_a1-c02_a2)
    carriers=(official_a1 c02_a2)
    ;;
  *)
    echo "invalid carrier order: $order" >&2
    exit 2
    ;;
esac

telemetry_pid=
load_pid=
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  for pid in "$telemetry_pid" "$load_pid"; do
    if [[ -n $pid ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

printf '%s\n' performance_settle > "$phase_file"
"$script_dir/telemetry_sampler.sh" \
  "$output_dir/telemetry_1hz.csv" "$phase_file" 1 &
telemetry_pid=$!
python3 -u "$script_dir/npu_load_sampler.py" \
  "$output_dir/npu_load_10hz.csv" "$phase_file" 0.1 &
load_pid=$!
sleep 3

for carrier in "${carriers[@]}"; do
  case "$carrier" in
    c02_a2) model=$c02_model ;;
    official_a1) model=$official_model ;;
  esac
  carrier_dir=$output_dir/$carrier
  mkdir -p "$carrier_dir"
  printf 'carrier_%s\n' "$carrier" > "$phase_file"
  set +e
  timeout --signal=TERM --kill-after=60 1800 \
    "$perf_bin" stat --no-big-num -x, \
      -o "$carrier_dir/perf_stat.csv" \
      -e task-clock,context-switches,cpu-migrations,page-faults,cycles:u,instructions:u \
      -- taskset -c 4-7 "$runner" "$model" "$carrier" "$prompts" \
        "$carrier_dir" 96 16384 \
      > "$carrier_dir/runner_stdout.log" \
      2> "$carrier_dir/runner_stderr.log"
  carrier_rc=$?
  set -e
  printf '%s\n' "$carrier_rc" > "$carrier_dir/run_exit_code.txt"
  if (( carrier_rc != 0 )); then
    echo "carrier $carrier failed with exit $carrier_rc" >&2
    exit "$carrier_rc"
  fi
  sleep 5
done

printf '%s\n' measurement_complete > "$phase_file"
sleep 2
