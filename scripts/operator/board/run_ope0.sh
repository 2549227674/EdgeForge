#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 2 )); then
  echo "usage: sudo $0 OUTPUT_DIR SEGMENT_LABEL" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "run_ope0 must run as root so it can pair governor writes and sample debugfs" >&2
  exit 77
fi

output_dir=$1
segment_label=$2
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
workspace=/home/orangepi/edgeforge_ope_r1
bin_dir=$workspace/bin
rknn_root=/home/orangepi/rknn-llm/examples/multimodal_model_demo/deploy/3rdparty/librknnrt/Linux/librknn_api
rknn_include=$rknn_root/include
rknn_lib=$rknn_root/aarch64
phase_file=$output_dir/phase.txt

if [[ -e "$output_dir" ]]; then
  echo "refusing to reuse output directory: $output_dir" >&2
  exit 73
fi
mkdir -p "$output_dir" "$bin_dir"
printf '%s\n' setup > "$phase_file"

{
  printf 'segment_label=%s\n' "$segment_label"
  printf 'board_start_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
  printf 'board_start_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
  printf 'uname=%s\n' "$(uname -a)"
  printf 'gcc=%s\n' "$(gcc --version | head -1)"
} > "$output_dir/run_meta.env"

gcc -O3 -std=gnu11 -mcpu=cortex-a76 -fopenmp \
  "$script_dir/bandwidth_triad_neon.c" -o "$bin_dir/bandwidth_triad_neon" \
  > "$output_dir/build_triad.log" 2>&1
gcc -O2 -std=gnu11 -I"$rknn_include" \
  "$script_dir/matmul_bench.c" -L"$rknn_lib" -lrknnrt \
  -Wl,-rpath,"$rknn_lib" -o "$bin_dir/matmul_bench" \
  > "$output_dir/build_matmul.log" 2>&1

telemetry_pid=
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  printf '%s\n' restored_or_failure > "$phase_file"
  {
    printf 'board_end_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
    printf 'board_end_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
    printf 'run_exit_code=%s\n' "$rc"
  } >> "$output_dir/run_meta.env"
  chown -R orangepi:orangepi "$output_dir" "$bin_dir/bandwidth_triad_neon" "$bin_dir/matmul_bench" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

"$script_dir/telemetry_sampler.sh" "$output_dir/telemetry_1hz.csv" "$phase_file" 1 &
telemetry_pid=$!
sleep 2

"$script_dir/governor_guard.sh" "$output_dir/governor_state.tsv" -- \
  "$script_dir/ope0_measure_payload.sh" "$output_dir" "$phase_file" \
  "$bin_dir/bandwidth_triad_neon" "$bin_dir/matmul_bench"

printf '%s\n' post_restore > "$phase_file"
sleep 2
