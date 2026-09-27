#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# < 6 || $# > 7 )); then
  echo "usage: $0 OUTPUT_DIR PHASE_FILE PREREG BENCH RUNNER MODE [RESUME_TAG]" >&2
  exit 2
fi

output_dir=$1
phase_file=$2
prereg=$3
bench=$4
runner=$5
mode=$6
resume_tag=${7:-}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
suffix=
runner_resume=()
if [[ -n "$resume_tag" ]]; then
  suffix=.$resume_tag
  runner_resume=(--resume)
fi

telemetry_pid=
load_pid=
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  for pid in "$telemetry_pid" "$load_pid"; do
    if [[ -n "$pid" ]]; then
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
"$script_dir/telemetry_sampler.sh" "$output_dir/telemetry_1hz$suffix.csv" "$phase_file" 1 &
telemetry_pid=$!

printf '%s\n' 'board_wall_time,uptime_s,phase,npu_load_core0_pct,npu_load_core1_pct,npu_load_core2_pct' \
  > "$output_dir/npu_load_10hz$suffix.csv"
(
  while :; do
    phase=$(tr -d '\n' < "$phase_file" 2>/dev/null || printf 'NA')
    phase=${phase//,/;}
    load0=NA
    load1=NA
    load2=NA
    if [[ -r /sys/kernel/debug/rknpu/load ]]; then
      mapfile -t loads < <(grep -oE '[0-9]+%' /sys/kernel/debug/rknpu/load | tr -d '%')
      load0=${loads[0]:-NA}
      load1=${loads[1]:-NA}
      load2=${loads[2]:-NA}
    fi
    printf '%s,%s,%s,%s,%s,%s\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')" \
      "$(cut -d' ' -f1 /proc/uptime)" "$phase" "$load0" "$load1" "$load2" \
      >> "$output_dir/npu_load_10hz$suffix.csv"
    sleep 0.1
  done
) &
load_pid=$!

sleep 3
python3 -u "$runner" \
  --prereg "$prereg" \
  --bench "$bench" \
  --output-dir "$output_dir/matrix" \
  --phase-file "$phase_file" \
  --mode "$mode" \
  "${runner_resume[@]}" \
  > "$output_dir/runner_stdout$suffix.log" \
  2> "$output_dir/runner_stderr$suffix.log"

printf '%s\n' measurement_complete > "$phase_file"
sleep 2
