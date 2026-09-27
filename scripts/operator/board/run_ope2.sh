#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# < 3 || $# > 5 )); then
  echo "usage: sudo $0 OUTPUT_DIR PREREG MODE [RESUME_TAG] [RESUME_REASON]" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "run_ope2 must run as root for paired governors and debugfs telemetry" >&2
  exit 77
fi

output_dir=$1
prereg=$2
mode=$3
resume_tag=${4:-}
resume_reason=${5:-interrupted controller; durable terminals retained}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
workspace=/home/orangepi/edgeforge_ope_r1
bin_dir=$workspace/bin
rknn_root=/home/orangepi/rknn-llm/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api
suffix=
if [[ -n "$resume_tag" ]]; then
  if [[ ! "$resume_tag" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "invalid resume tag: $resume_tag" >&2
    exit 2
  fi
  suffix=.$resume_tag
fi
phase_file=$output_dir/phase$suffix.txt

if [[ -z "$resume_tag" ]]; then
  if [[ -e "$output_dir" ]]; then
    echo "refusing to reuse output directory: $output_dir" >&2
    exit 73
  fi
  mkdir -p "$output_dir" "$bin_dir"
else
  if [[ ! -d "$output_dir/matrix" ]]; then
    echo "resume matrix directory is missing: $output_dir/matrix" >&2
    exit 74
  fi
  if [[ -e "$output_dir/matrix/runner_summary.json" ]]; then
    echo "refusing to resume an already completed matrix" >&2
    exit 75
  fi
fi
mkdir -p "$bin_dir"
printf '%s\n' setup > "$phase_file"

if [[ -z "$resume_tag" ]]; then
  {
    printf 'mode=%s\n' "$mode"
    printf 'board_start_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
    printf 'board_start_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
    printf 'board_start_boot_id=%s\n' "$(cat /proc/sys/kernel/random/boot_id)"
    printf 'uname=%s\n' "$(uname -a)"
    printf 'gxx=%s\n' "$(g++ --version | head -1)"
    printf 'runtime=%s\n' "$(strings "$rknn_root/aarch64/librknnrt.so" | grep -m1 'librknnrt version:' || true)"
  } > "$output_dir/run_meta.env"
else
  {
    printf '%s_board_start_wall=%s\n' "$resume_tag" "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
    printf '%s_board_start_uptime_s=%s\n' "$resume_tag" "$(cut -d' ' -f1 /proc/uptime)"
    printf '%s_board_start_boot_id=%s\n' "$resume_tag" "$(cat /proc/sys/kernel/random/boot_id)"
    printf '%s_reason=%s\n' "$resume_tag" "$resume_reason"
  } >> "$output_dir/run_meta.env"
fi

g++ -O2 -std=gnu++17 -I"$rknn_root/include" \
  "$script_dir/ope2_matmul_bench.cpp" \
  -L"$rknn_root/aarch64" -lrknnrt -Wl,-rpath,"$rknn_root/aarch64" \
  -o "$bin_dir/ope2_matmul_bench" \
  > "$output_dir/build$suffix.log" 2>&1

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  printf '%s\n' restored_or_failure > "$phase_file"
  {
    printf 'board_end_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
    printf 'board_end_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
    if [[ -n "$resume_tag" ]]; then
      printf '%s_board_end_wall=%s\n' "$resume_tag" "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
      printf '%s_board_end_uptime_s=%s\n' "$resume_tag" "$(cut -d' ' -f1 /proc/uptime)"
    fi
    printf 'run_exit_code=%s\n' "$rc"
  } >> "$output_dir/run_meta.env"
  chown -R orangepi:orangepi "$output_dir" "$bin_dir/ope2_matmul_bench" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

"$script_dir/governor_guard.sh" "$output_dir/governor_state$suffix.tsv" -- \
  "$script_dir/ope2_measure_payload.sh" "$output_dir" "$phase_file" \
  "$prereg" "$bin_dir/ope2_matmul_bench" \
  "$script_dir/run_ope2_matrix.py" "$mode" "$resume_tag"
