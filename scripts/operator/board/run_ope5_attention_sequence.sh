#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 4 )); then
  echo "usage: sudo $0 OUTPUT_DIR PREREG SOURCE_LOCK SEGMENT_LABEL" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "run_ope5_attention_sequence must run as root for paired governors and debugfs telemetry" >&2
  exit 77
fi

output_dir=$1
prereg=$2
source_lock=$3
segment_label=$4
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
workspace=/home/orangepi/edgeforge_ope_r1
bench=$workspace/bin/ope5_attention_sequence_bench
perf_bin=$workspace/bin/perf
phase_file=$output_dir/phase.txt
policy6=/sys/devices/system/cpu/cpufreq/policy6/scaling_governor
policy4=/sys/devices/system/cpu/cpufreq/policy4/scaling_governor

[[ $segment_label == segment1 ]] || {
  echo "the OPE5-01 protocol authorizes segment1 only" >&2
  exit 2
}
if [[ -e $output_dir ]]; then
  echo "refusing to reuse output directory: $output_dir" >&2
  exit 73
fi
for path in "$prereg" "$source_lock" "$bench" "$perf_bin" \
            "$script_dir/ope5_attention_sequence_bench.cpp" \
            "$script_dir/ope5_attention_sequence_measure_payload.sh" \
            "$script_dir/telemetry_sampler.sh" \
            "$script_dir/npu_load_sampler.py"; do
  [[ -r $path ]] || { echo "required path is unreadable: $path" >&2; exit 74; }
done
if [[ $(uname -m) != aarch64 ]]; then
  echo "formal OPE5-01 execution requires aarch64" >&2
  exit 75
fi
for feature in asimd fphp asimdhp; do
  grep -qE "(^| )${feature}( |$)" /proc/cpuinfo || {
    echo "required CPU feature is absent: $feature" >&2
    exit 76
  }
done
[[ -r $policy6 && -w $policy6 && -r $policy4 ]] || {
  echo "CPU governor paths are unavailable" >&2
  exit 77
}

npu_node=
for path in /sys/class/devfreq/*; do
  [[ -e $path ]] || continue
  if [[ ${path##*/} == *npu* ]]; then
    npu_node=$path
    break
  fi
done
[[ -n $npu_node && -r $npu_node/governor && -w $npu_node/governor ]] || {
  echo "NPU governor path unavailable" >&2
  exit 77
}

python3 - "$workspace" "$prereg" "$source_lock" "$bench" <<'PY'
import hashlib
import json
import pathlib
import sys

workspace = pathlib.Path(sys.argv[1]).resolve()
prereg_path = pathlib.Path(sys.argv[2]).resolve()
lock_path = pathlib.Path(sys.argv[3]).resolve()
binary_path = pathlib.Path(sys.argv[4]).resolve()

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
lock = json.loads(lock_path.read_text(encoding="utf-8"))
assert prereg["stage"] == "OPE5-01-DECODE-ATTENTION-SEQUENCE"
assert prereg["status"] == "PREREGISTERED_FROZEN_NOT_RUN"
assert sha256(prereg_path) == "a0cc5d0c516f1b9e12028debec03743c386980041e4a5c6e2502e3d110cff464"
measurement = prereg["measurement_protocol"]
assert measurement["formal_segments"] == 1
assert measurement["performance_smoke"] is False
assert measurement["automatic_retry"] is False
assert measurement["same_day_retest"] is False
assert measurement["cross_day_retest"] is False
assert lock["stage"] == "OPE5-01-DECODE-ATTENTION-SEQUENCE-SOURCE-LOCK"
assert lock["status"] == "LOCKED_BEFORE_SOLE_FORMAL_SEGMENT"
assert lock["formal_execution_started"] is False
assert lock["board_benchmark_execution_count"] == 0
assert lock["preregistration"]["sha256"] == sha256(prereg_path)
for section in ("implementation_files", "runtime_dependencies"):
    for item in lock[section].values():
        path = workspace / item["path"]
        assert path.is_file(), path
        assert sha256(path) == item["sha256"], path
assert lock["board_binary"]["path"] == str(binary_path)
assert sha256(binary_path) == lock["board_binary"]["sha256"]
PY

mkdir -p "$output_dir"
printf '%s\n' setup > "$phase_file"

{
  printf 'segment_label=%s\n' "$segment_label"
  printf 'board_start_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
  printf 'board_start_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
  printf 'board_start_monotonic_raw_ns=%s\n' "$(python3 -c 'import time; print(time.clock_gettime_ns(getattr(time,"CLOCK_MONOTONIC_RAW",time.CLOCK_MONOTONIC)))')"
  printf 'board_start_boot_id=%s\n' "$(< /proc/sys/kernel/random/boot_id)"
  printf 'uname=%s\n' "$(uname -a)"
  printf 'gxx=%s\n' "$(g++ --version | head -1)"
  printf 'prereg_sha256=%s\n' "$(sha256sum "$prereg" | awk '{print $1}')"
  printf 'source_lock_sha256=%s\n' "$(sha256sum "$source_lock" | awk '{print $1}')"
  printf 'candidate_source_sha256=%s\n' "$(sha256sum "$script_dir/ope5_attention_sequence_bench.cpp" | awk '{print $1}')"
  printf 'formal_runner_sha256=%s\n' "$(sha256sum "$script_dir/run_ope5_attention_sequence.sh" | awk '{print $1}')"
  printf 'measurement_payload_sha256=%s\n' "$(sha256sum "$script_dir/ope5_attention_sequence_measure_payload.sh" | awk '{print $1}')"
  printf 'telemetry_sampler_sha256=%s\n' "$(sha256sum "$script_dir/telemetry_sampler.sh" | awk '{print $1}')"
  printf 'npu_load_sampler_sha256=%s\n' "$(sha256sum "$script_dir/npu_load_sampler.py" | awk '{print $1}')"
  printf 'perf_binary_sha256=%s\n' "$(sha256sum "$perf_bin" | awk '{print $1}')"
  printf 'bench_binary_sha256=%s\n' "$(sha256sum "$bench" | awk '{print $1}')"
  printf 'primary_cpu=7\n'
  printf 'npu_core=0\n'
  printf 'formal_segments=1\n'
  printf 'replication_status=WAIVED_BY_OWNER_NOT_SATISFIED\n'
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

before_policy6=$(<"$policy6")
before_policy4=$(<"$policy4")
before_npu=$(<"$npu_node/governor")

snapshot_governors() {
  local event=$1
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$event" \
    "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')" \
    "$(cut -d' ' -f1 /proc/uptime)" \
    "$(<"$policy4")" \
    "$(<"$policy6")" \
    "$(<"$npu_node/governor")" >> "$output_dir/governor_state.tsv"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  local restore_rc=0
  printf '%s\n' restoring_governors > "$phase_file"
  printf '%s\n' "$before_policy6" > "$policy6" || restore_rc=1
  printf '%s\n' "$before_npu" > "$npu_node/governor" || restore_rc=1
  snapshot_governors after_restore || restore_rc=1
  [[ $(<"$policy4") == "$before_policy4" ]] || restore_rc=1
  [[ $(<"$policy6") == "$before_policy6" ]] || restore_rc=1
  [[ $(<"$npu_node/governor") == "$before_npu" ]] || restore_rc=1
  printf '%s\n' restored_or_failure > "$phase_file"
  {
    printf 'board_end_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
    printf 'board_end_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
    printf 'board_end_monotonic_raw_ns=%s\n' "$(python3 -c 'import time; print(time.clock_gettime_ns(getattr(time,"CLOCK_MONOTONIC_RAW",time.CLOCK_MONOTONIC)))')"
    printf 'run_exit_code=%s\n' "$rc"
    printf 'governor_restore_exit_code=%s\n' "$restore_rc"
  } >> "$output_dir/run_meta.env"
  chown -R orangepi:orangepi "$output_dir" 2>/dev/null || true
  if (( restore_rc != 0 )); then
    exit 97
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

printf '%s\n' $'event\tboard_wall_time\tuptime_s\tpolicy4_governor\tpolicy6_governor\tnpu_governor' \
  > "$output_dir/governor_state.tsv"
snapshot_governors before
printf '%s\n' performance > "$policy6"
printf '%s\n' performance > "$npu_node/governor"
snapshot_governors after_set
[[ $(<"$policy4") == "$before_policy4" ]] || {
  echo "policy4 changed despite read-only contract" >&2
  exit 77
}
[[ $(<"$policy6") == performance && $(<"$npu_node/governor") == performance ]] || {
  echo "performance governors did not engage" >&2
  exit 77
}

"$script_dir/ope5_attention_sequence_measure_payload.sh" \
  "$output_dir" "$phase_file" "$bench" "$perf_bin" \
  "$script_dir/npu_load_sampler.py"
