#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 4 )); then
  echo "usage: sudo $0 OUTPUT_DIR PREREG SEGMENT_LABEL CARRIER_ORDER" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "run_ope4 must run as root for paired governors and debugfs telemetry" >&2
  exit 77
fi

output_dir=$1
prereg=$2
segment_label=$3
carrier_order=$4
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
workspace=/home/orangepi/edgeforge_ope_r1
bin_dir=$workspace/bin
runner=$bin_dir/rkllm_ope4_diag
perf_bin=$bin_dir/perf
prompts=$workspace/data/operator/board/ope4_fixed_prompts.txt
c02_model=$workspace/models/C02-S.O-w8a8-rk3588-16k-1core-A2.rkllm
official_model=/home/orangepi/edgeforge_m0_frozen_r2/gemma-4-e4b-it-w8a8-rk3588-16k-1core.rkllm
include_dir=/home/orangepi/rknn-llm/rkllm-runtime/Linux/librkllm_api/include
lib_dir=/home/orangepi/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64
phase_file=$output_dir/phase.txt

if [[ ! $segment_label =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "invalid segment label: $segment_label" >&2
  exit 2
fi
[[ $carrier_order == c02_a2-official_a1 ]] || {
  echo "formal OPE4 owner-waived single segment requires c02_a2-official_a1" >&2
  exit 2
}
if [[ -e $output_dir ]]; then
  echo "refusing to reuse output directory: $output_dir" >&2
  exit 73
fi
for path in "$prereg" "$prompts" "$c02_model" "$official_model" \
            "$script_dir/rkllm_ope4_diag.cpp" "$perf_bin"; do
  [[ -r $path ]] || { echo "required path is unreadable: $path" >&2; exit 74; }
done
if [[ $(uname -m) != aarch64 ]]; then
  echo "formal OPE4 execution requires aarch64" >&2
  exit 75
fi

python3 - "$prereg" "$carrier_order" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    prereg = json.load(handle)
assert prereg["schema"] == "edgeforge-ope4-diagnostic-prereg-v1"
assert prereg["stage"] == "OPE4"
assert prereg["status"] == "PREREGISTERED_FROZEN_NOT_RUN"
assert prereg["label"] == "board-diagnostic-1core"
assert prereg["measurement_protocol"]["max_new_tokens"] == 96
assert prereg["measurement_protocol"]["max_context_len"] == 16384
assert prereg["measurement_protocol"]["cpu_set"] == [4, 5, 6, 7]
assert prereg["measurement_protocol"]["formal_segments"] == 1
assert prereg["measurement_protocol"]["carrier_order"] == "c02_a2-official_a1"
assert sys.argv[2] == prereg["measurement_protocol"]["carrier_order"]
assert prereg["anchor"]["core_projection_weight_bytes"] == 3890216960
assert prereg["anchor"]["all_non_lmhead_linear_weight_bytes"] == 3972792320
PY

expected_prompt_sha=$(python3 - "$prereg" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["workload"]["payload_sha256"])
PY
)
observed_prompt_sha=$(sha256sum "$prompts" | awk '{print $1}')
[[ $observed_prompt_sha == "$expected_prompt_sha" ]] || {
  echo "prompt payload SHA mismatch" >&2
  exit 76
}

python3 - "$prereg" "$c02_model" "$official_model" <<'PY'
import json
import os
import sys

p = json.load(open(sys.argv[1], encoding="utf-8"))
for carrier, path in (("c02_a2", sys.argv[2]), ("official_a1", sys.argv[3])):
    expected = p["carriers"][carrier]
    st = os.stat(path)
    assert path == expected["board_path"]
    assert st.st_size == expected["bytes"]
    assert int(st.st_mtime) == expected["mtime_epoch_seconds"]
PY

mkdir -p "$output_dir" "$bin_dir"
printf '%s\n' setup > "$phase_file"

g++ -O2 -std=c++17 -mcpu=cortex-a76 -Wall -Wextra -Werror \
  -I "$include_dir" "$script_dir/rkllm_ope4_diag.cpp" \
  -L "$lib_dir" -Wl,-rpath,"$lib_dir" -lrkllmrt -pthread \
  -o "$runner" \
  > "$output_dir/build.stdout.log" 2> "$output_dir/build.stderr.log"

implementation_manifest=$output_dir/implementation_hash_checks.tsv
python3 - "$prereg" > "$implementation_manifest" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))["implementation"]
for path_key, hash_key in [
    ("runner_source", "runner_source_sha256"),
    ("formal_runner", "formal_runner_sha256"),
    ("measurement_payload", "measurement_payload_sha256"),
    ("npu_sampler", "npu_sampler_sha256"),
]:
    print(p[path_key], p[hash_key], sep="\t")
print("scripts/operator/board/telemetry_sampler.sh", p["telemetry_sampler_sha256"], sep="\t")
print("scripts/operator/board/governor_guard.sh", p["governor_guard_sha256"], sep="\t")
PY
while IFS=$'\t' read -r relative_path expected_sha; do
  observed_sha=$(sha256sum "$workspace/$relative_path" | awk '{print $1}')
  printf '%s\t%s\t%s\n' "$relative_path" "$expected_sha" "$observed_sha" \
    >> "$output_dir/implementation_hash_observed.tsv"
  [[ $observed_sha == "$expected_sha" ]] || {
    echo "implementation SHA mismatch: $relative_path" >&2
    exit 76
  }
done < "$implementation_manifest"
observed_source_sha=$(sha256sum "$script_dir/rkllm_ope4_diag.cpp" | awk '{print $1}')

{
  printf 'segment_label=%s\n' "$segment_label"
  printf 'carrier_order=%s\n' "$carrier_order"
  printf 'board_start_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
  printf 'board_start_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
  printf 'board_start_monotonic_raw_ns=%s\n' "$(python3 -c 'import time; print(time.clock_gettime_ns(getattr(time,"CLOCK_MONOTONIC_RAW",time.CLOCK_MONOTONIC)))')"
  printf 'board_start_boot_id=%s\n' "$(< /proc/sys/kernel/random/boot_id)"
  printf 'uname=%s\n' "$(uname -a)"
  printf 'gxx=%s\n' "$(g++ --version | head -1)"
  printf 'prereg_sha256=%s\n' "$(sha256sum "$prereg" | awk '{print $1}')"
  printf 'prompt_payload_sha256=%s\n' "$observed_prompt_sha"
  printf 'runner_source_sha256=%s\n' "$observed_source_sha"
  printf 'runner_binary_sha256=%s\n' "$(sha256sum "$runner" | awk '{print $1}')"
  printf 'c02_model_path=%s\n' "$c02_model"
  printf 'c02_model_bytes=%s\n' "$(stat -c '%s' "$c02_model")"
  printf 'c02_model_mtime_epoch=%s\n' "$(stat -c '%Y' "$c02_model")"
  printf 'official_model_path=%s\n' "$official_model"
  printf 'official_model_bytes=%s\n' "$(stat -c '%s' "$official_model")"
  printf 'official_model_mtime_epoch=%s\n' "$(stat -c '%Y' "$official_model")"
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
    printf 'board_end_monotonic_raw_ns=%s\n' "$(python3 -c 'import time; print(time.clock_gettime_ns(getattr(time,"CLOCK_MONOTONIC_RAW",time.CLOCK_MONOTONIC)))')"
    printf 'c02_model_bytes_after=%s\n' "$(stat -c '%s' "$c02_model")"
    printf 'c02_model_mtime_epoch_after=%s\n' "$(stat -c '%Y' "$c02_model")"
    printf 'official_model_bytes_after=%s\n' "$(stat -c '%s' "$official_model")"
    printf 'official_model_mtime_epoch_after=%s\n' "$(stat -c '%Y' "$official_model")"
    printf 'run_exit_code=%s\n' "$rc"
  } >> "$output_dir/run_meta.env"
  chown -R orangepi:orangepi "$output_dir" "$runner" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

"$script_dir/governor_guard.sh" "$output_dir/governor_state.tsv" -- \
  "$script_dir/ope4_measure_payload.sh" \
    "$output_dir" "$phase_file" "$runner" "$perf_bin" "$prompts" \
    "$carrier_order" "$c02_model" "$official_model"
