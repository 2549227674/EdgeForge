#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 5 )); then
  echo "usage: sudo $0 OUTPUT_DIR PREREG SEGMENT_LABEL RUN_CORRECTNESS RUN_TIER_B" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "run_ope3 must run as root for paired governor writes and debugfs telemetry" >&2
  exit 77
fi

output_dir=$1
prereg=$2
segment_label=$3
run_correctness=$4
run_tier_b=$5
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
workspace=/home/orangepi/edgeforge_ope_r1
bin_dir=$workspace/bin
bench=$bin_dir/ke1_gemv_bench
perf_bin=$bin_dir/perf
phase_file=$output_dir/phase.txt

if [[ ! $segment_label =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "invalid segment label: $segment_label" >&2
  exit 2
fi
if [[ $run_correctness != yes && $run_correctness != no ]]; then
  echo "RUN_CORRECTNESS must be yes or no" >&2
  exit 2
fi
if [[ $run_tier_b != yes && $run_tier_b != no ]]; then
  echo "RUN_TIER_B must be yes or no" >&2
  exit 2
fi
if [[ -e $output_dir ]]; then
  echo "refusing to reuse output directory: $output_dir" >&2
  exit 73
fi
if [[ ! -r $prereg ]]; then
  echo "preregistration is missing: $prereg" >&2
  exit 74
fi
if [[ $(uname -m) != aarch64 ]]; then
  echo "formal OPE3 execution requires aarch64" >&2
  exit 75
fi
if ! grep -qE '(^| )asimddp( |$)' /proc/cpuinfo; then
  echo "asimddp is absent from /proc/cpuinfo" >&2
  exit 76
fi
if [[ ! -x $perf_bin && $run_tier_b == yes ]]; then
  echo "workspace perf is unavailable: $perf_bin" >&2
  exit 78
fi

mkdir -p "$output_dir" "$bin_dir"
printf '%s\n' setup > "$phase_file"

python3 - "$prereg" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    prereg = json.load(handle)
assert prereg["stage"] == "OPE3-K-E1"
assert prereg["status"] == "PREREGISTERED_FROZEN_NOT_RUN"
assert prereg["semantic_contract"]["shape_mkn"] == [1, 10240, 2560]
assert prereg["measurement_protocol"]["warmup_repetitions_per_variant"] == 50
assert prereg["measurement_protocol"]["timed_repetitions_per_variant"] == 200
assert prereg["measurement_protocol"]["cpu_set"] == [4, 5, 6, 7]
assert prereg["correctness_protocol"]["tolerance"] == {"atol": 0, "rtol": 0}
PY

gcc -O3 -std=gnu11 -mcpu=cortex-a76 -fopenmp -Wall -Wextra -Werror \
  -fopt-info-vec-all="$output_dir/vectorization.log" \
  "$script_dir/ke1_gemv_bench.c" -o "$bench" \
  > "$output_dir/build.stdout.log" 2> "$output_dir/build.stderr.log"

objdump -d --disassemble=dot_scalar "$bench" \
  > "$output_dir/disassembly_scalar.txt"
objdump -d --disassemble=dot_autovec "$bench" \
  > "$output_dir/disassembly_autovec.txt"
objdump -d --disassemble=dot_neon_sdot "$bench" \
  > "$output_dir/disassembly_neon_sdot.txt"
if grep -Eq '(^|[[:space:]])sdot([[:space:]]|$)' "$output_dir/disassembly_scalar.txt"; then
  echo "scalar kernel unexpectedly contains sdot" >&2
  exit 79
fi
if ! grep -Eq '(^|[[:space:]])sdot([[:space:]]|$)' "$output_dir/disassembly_neon_sdot.txt"; then
  echo "NEON candidate disassembly does not contain sdot" >&2
  exit 79
fi

{
  printf 'segment_label=%s\n' "$segment_label"
  printf 'run_correctness=%s\n' "$run_correctness"
  printf 'run_tier_b=%s\n' "$run_tier_b"
  printf 'board_start_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
  printf 'board_start_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
  printf 'board_start_boot_id=%s\n' "$(< /proc/sys/kernel/random/boot_id)"
  printf 'uname=%s\n' "$(uname -a)"
  printf 'gcc=%s\n' "$(gcc --version | head -1)"
  printf 'prereg_sha256=%s\n' "$(sha256sum "$prereg" | awk '{print $1}')"
  printf 'candidate_source_sha256=%s\n' "$(sha256sum "$script_dir/ke1_gemv_bench.c" | awk '{print $1}')"
  printf 'bench_sha256=%s\n' "$(sha256sum "$bench" | awk '{print $1}')"
  printf 'cache_line_bytes=%s\n' "$(< /sys/devices/system/cpu/cpu7/cache/index0/coherency_line_size)"
} > "$output_dir/run_meta.env"

printf '%s\n' $'zone\ttype\ttrip_index\ttrip_type\ttrip_temp_mC' \
  > "$output_dir/thermal_trip_points.tsv"
for zone_path in /sys/class/thermal/thermal_zone*; do
  [[ -d $zone_path ]] || continue
  zone=${zone_path##*thermal_zone}
  zone_type=$(<"$zone_path/type")
  for trip_temp_path in "$zone_path"/trip_point_*_temp; do
    [[ -r $trip_temp_path ]] || continue
    trip_name=${trip_temp_path##*/trip_point_}
    trip_index=${trip_name%_temp}
    trip_type_path="$zone_path/trip_point_${trip_index}_type"
    trip_type=NA
    [[ -r $trip_type_path ]] && trip_type=$(<"$trip_type_path")
    printf '%s\t%s\t%s\t%s\t%s\n' "$zone" "$zone_type" "$trip_index" \
      "$trip_type" "$(<"$trip_temp_path")" \
      >> "$output_dir/thermal_trip_points.tsv"
  done
done

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  printf '%s\n' restored_or_failure > "$phase_file"
  {
    printf 'board_end_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
    printf 'board_end_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
    printf 'run_exit_code=%s\n' "$rc"
  } >> "$output_dir/run_meta.env"
  chown -R orangepi:orangepi "$output_dir" "$bench" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

export OMP_DYNAMIC=false
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=true
export OMP_PLACES='{4},{5},{6},{7}'

"$script_dir/governor_guard.sh" "$output_dir/governor_state.tsv" -- \
  "$script_dir/ope3_measure_payload.sh" \
    "$output_dir" "$phase_file" "$bench" "$perf_bin" \
    "$run_correctness" "$run_tier_b" 50 200
