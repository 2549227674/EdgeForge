#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 4 )); then
  echo "usage: sudo $0 OUTPUT_DIR PREREG SOURCE_LOCK SEGMENT_LABEL" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "run_ope5_a55_fp16_gemv must run as root for paired policy0 governor writes" >&2
  exit 77
fi

output_dir=$1
prereg=$2
source_lock=$3
segment_label=$4
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
workspace=/home/orangepi/edgeforge_ope_r1
bin_dir=$workspace/bin
bench=$bin_dir/ope5_a55_fp16_gemv_bench
perf_bin=$bin_dir/perf
phase_file=$output_dir/phase.txt
telemetry_script=$script_dir/telemetry_sampler.sh
policy0=/sys/devices/system/cpu/cpufreq/policy0/scaling_governor
policy4=/sys/devices/system/cpu/cpufreq/policy4/scaling_governor
policy6=/sys/devices/system/cpu/cpufreq/policy6/scaling_governor
policy0_related=/sys/devices/system/cpu/cpufreq/policy0/related_cpus
policy0_max=/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq

[[ $segment_label == segment1 ]] || {
  echo "the owner-waived OPE5-05 protocol authorizes segment1 only" >&2
  exit 2
}
if [[ -e $output_dir ]]; then
  echo "refusing to reuse output directory: $output_dir" >&2
  exit 73
fi
for path in "$prereg" "$source_lock" \
            "$script_dir/ope5_a55_fp16_gemv_bench.cpp" \
            "$script_dir/ope5_a55_fp16_gemv_measure_payload.sh" \
            "$telemetry_script" "$perf_bin"; do
  [[ -r $path ]] || { echo "required path is unreadable: $path" >&2; exit 74; }
done
if [[ $(uname -m) != aarch64 ]]; then
  echo "formal OPE5-05 execution requires aarch64" >&2
  exit 75
fi
[[ $(tr -s ' ' < "$policy0_related" | sed 's/^ //;s/ $//') == "0 1 2 3" ]] || {
  echo "policy0 related_cpus is not exactly 0 1 2 3" >&2
  exit 76
}
[[ $(<"$policy0_max") == 1800000 ]] || {
  echo "policy0 scaling_max_freq is not exactly 1800000 kHz" >&2
  exit 76
}
first_features=$(awk -F: '/^Features/{print $2; exit}' /proc/cpuinfo)
for feature in asimd fphp asimdhp; do
  grep -qE "(^| )${feature}( |$)" <<<"$first_features" || {
    echo "required first-core CPU feature is absent: $feature" >&2
    exit 76
  }
done
for cpu in 0 1 2 3; do
  taskset -c "$cpu" true || {
    echo "single-core affinity failed for CPU$cpu" >&2
    exit 76
  }
done
[[ -r $policy0 && -w $policy0 ]] || {
  echo "policy0 governor path is not readable/writable" >&2
  exit 77
}
for path in "$policy4" "$policy6"; do
  [[ -r $path ]] || {
    echo "read-only comparison governor path is unreadable: $path" >&2
    exit 77
  }
done

npu_node=
for path in /sys/class/devfreq/*; do
  [[ -e $path ]] || continue
  if [[ ${path##*/} == *npu* ]]; then
    npu_node=$path
    break
  fi
done
[[ -n $npu_node && -r $npu_node/governor ]] || {
  echo "NPU governor path is unavailable for read-only identity checks" >&2
  exit 77
}

python3 - "$workspace" "$prereg" "$source_lock" <<'PY'
import hashlib
import json
import pathlib
import sys

workspace = pathlib.Path(sys.argv[1]).resolve()
prereg_path = pathlib.Path(sys.argv[2]).resolve()
lock_path = pathlib.Path(sys.argv[3]).resolve()

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
lock = json.loads(lock_path.read_text(encoding="utf-8"))
assert prereg["stage"] == "OPE5-05-A55-FP16-GEMV"
assert prereg["status"] == "PREREGISTERED_FROZEN_NOT_RUN"
assert sha256(prereg_path) == "98a3fe20f7982f7fb58fca01f1fe23fa483f4ca11f387d24767805f1e7fb43ce"
assert prereg["selection"]["scope_sha256"] == "8f3a2c6949980d04096a1e995ff2aafae653aed031fdf0960b1261c86059d1f4"
assert prereg["formal_shape"]["logical_mkn"] == [1, 10240, 2560]
assert prereg["formal_shape"]["cpu_cores"] == [0, 1, 2, 3]
assert prereg["formal_shape"]["threads"] == 4
assert prereg["correctness_protocol"]["expected_element_comparisons"] == 69120
assert prereg["measurement_protocol"]["formal_segments"] == 1
assert prereg["measurement_protocol"]["owner_instruction"] == "不需要复测，取消同日复测，只测一次"
assert prereg["measurement_protocol"]["warmup_repetitions_per_variant"] == 50
assert prereg["measurement_protocol"]["timed_repetitions_per_variant_minimum"] == 200
assert prereg["measurement_protocol"]["timed_window_minimum_seconds_per_variant"] == 3.0
assert prereg["tier_b"]["stat_events"] == [
    "cycles:u", "instructions:u", "l2d_cache_refill:u", "bus_access:u"
]
assert "Do not multiply A55 bus_access" in prereg["tier_b"]["bus_access_policy"]
assert lock["stage"] == "OPE5-05-A55-FP16-GEMV-SOURCE-LOCK"
assert lock["status"] == "LOCKED_BEFORE_FORMAL_EXECUTION"
assert lock["formal_execution_started"] is False
assert lock["board_benchmark_process_started"] is False
assert lock["scope"]["sha256"] == prereg["selection"]["scope_sha256"]
assert lock["preregistration"]["sha256"] == sha256(prereg_path)
assert lock["prelock_audit"]["status"] == "PASS_A55_PRELOCK_NO_BENCHMARK_EXECUTION"
assert lock["prelock_audit"]["benchmark_execution_count"] == 0
for section in ("implementation_files", "runtime_dependencies", "reference_files"):
    for item in lock[section].values():
        path = workspace / item["path"]
        assert path.is_file(), path
        assert sha256(path) == item["sha256"], path
PY

mkdir -p "$output_dir" "$bin_dir"
printf '%s\n' setup > "$phase_file"

g++ -O3 -std=gnu++17 -mcpu=cortex-a55 -fno-fast-math -fno-math-errno \
  -pthread -Wall -Wextra -Werror \
  -fopt-info-vec-all="$output_dir/vectorization.log" \
  "$script_dir/ope5_a55_fp16_gemv_bench.cpp" -o "$bench" \
  > "$output_dir/build.stdout.log" 2> "$output_dir/build.stderr.log"

objdump -d --disassemble=scalar_fp32_rows "$bench" \
  > "$output_dir/disassembly_scalar_fp32.txt"
objdump -d --disassemble=autovec_o3_rows "$bench" \
  > "$output_dir/disassembly_autovec_o3.txt"
objdump -d --disassemble=neon_fp16_f32acc_rows "$bench" \
  > "$output_dir/disassembly_neon_fp16_f32acc.txt"
if grep -Eq 'fmla[[:space:]]+v' "$output_dir/disassembly_scalar_fp32.txt"; then
  echo "scalar_fp32 unexpectedly contains vector floating-point FMA" >&2
  exit 79
fi
if ! grep -Eq 'fcvtl' "$output_dir/disassembly_neon_fp16_f32acc.txt" || \
   ! grep -Eq 'fmla[[:space:]]+v' "$output_dir/disassembly_neon_fp16_f32acc.txt"; then
  echo "neon_fp16_f32acc disassembly is missing conversion or vector FMA" >&2
  exit 79
fi

python3 - "$source_lock" > "$output_dir/implementation_hash_checks.tsv" <<'PY'
import json
import sys

lock = json.load(open(sys.argv[1], encoding="utf-8"))
print("section", "path", "expected_sha256", sep="\t")
for section in ("implementation_files", "runtime_dependencies", "reference_files"):
    for item in lock[section].values():
        print(section, item["path"], item["sha256"], sep="\t")
PY
printf '%s\n' $'section\tpath\texpected_sha256\tobserved_sha256\tmatch' \
  > "$output_dir/implementation_hash_observed.tsv"
tail -n +2 "$output_dir/implementation_hash_checks.tsv" | \
while IFS=$'\t' read -r section relative_path expected_sha; do
  observed_sha=$(sha256sum "$workspace/$relative_path" | awk '{print $1}')
  match=no
  [[ $observed_sha == "$expected_sha" ]] && match=yes
  printf '%s\t%s\t%s\t%s\t%s\n' "$section" "$relative_path" \
    "$expected_sha" "$observed_sha" "$match" \
    >> "$output_dir/implementation_hash_observed.tsv"
  [[ $match == yes ]] || {
    echo "source-lock mismatch: $relative_path" >&2
    exit 76
  }
done

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
  printf 'candidate_source_sha256=%s\n' "$(sha256sum "$script_dir/ope5_a55_fp16_gemv_bench.cpp" | awk '{print $1}')"
  printf 'formal_runner_sha256=%s\n' "$(sha256sum "$script_dir/run_ope5_a55_fp16_gemv.sh" | awk '{print $1}')"
  printf 'measurement_payload_sha256=%s\n' "$(sha256sum "$script_dir/ope5_a55_fp16_gemv_measure_payload.sh" | awk '{print $1}')"
  printf 'telemetry_sampler_sha256=%s\n' "$(sha256sum "$telemetry_script" | awk '{print $1}')"
  printf 'perf_binary_sha256=%s\n' "$(sha256sum "$perf_bin" | awk '{print $1}')"
  printf 'bench_binary_sha256=%s\n' "$(sha256sum "$bench" | awk '{print $1}')"
  printf 'primary_cpus=0,1,2,3\n'
  printf 'threads=4\n'
  printf 'policy0_expected_max_khz=1800000\n'
  printf 'formal_segments=1\n'
  printf 'replication_status=WAIVED_BY_OWNER_NOT_SATISFIED\n'
  printf 'comparison_governor_policy=READ_ONLY_UNCHANGED\n'
  printf 'npu_governor_policy=READ_ONLY_UNCHANGED\n'
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

before_policy0=$(<"$policy0")
before_policy4=$(<"$policy4")
before_policy6=$(<"$policy6")
before_npu=$(<"$npu_node/governor")
governor_snapshot_ready=yes

snapshot_governors() {
  local event=$1
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$event" \
    "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')" \
    "$(cut -d' ' -f1 /proc/uptime)" \
    "$(<"$policy0")" \
    "$(<"$policy4")" \
    "$(<"$policy6")" \
    "$(<"$npu_node/governor")" >> "$output_dir/governor_state.tsv"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  local restore_rc=0
  printf '%s\n' restoring_policy0_governor > "$phase_file"
  if [[ ${governor_snapshot_ready:-no} == yes ]]; then
    printf '%s\n' "$before_policy0" > "$policy0" || restore_rc=1
    snapshot_governors after_restore || restore_rc=1
    if [[ $(<"$policy4") != "$before_policy4" || \
          $(<"$policy6") != "$before_policy6" || \
          $(<"$npu_node/governor") != "$before_npu" ]]; then
      echo "read-only comparison or NPU governor changed" >&2
      restore_rc=1
    fi
  fi
  printf '%s\n' restored_or_failure > "$phase_file"
  {
    printf 'board_end_wall=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
    printf 'board_end_uptime_s=%s\n' "$(cut -d' ' -f1 /proc/uptime)"
    printf 'board_end_monotonic_raw_ns=%s\n' "$(python3 -c 'import time; print(time.clock_gettime_ns(getattr(time,"CLOCK_MONOTONIC_RAW",time.CLOCK_MONOTONIC)))')"
    printf 'run_exit_code=%s\n' "$rc"
    printf 'governor_restore_exit_code=%s\n' "$restore_rc"
  } >> "$output_dir/run_meta.env"
  chown -R orangepi:orangepi "$output_dir" "$bench" 2>/dev/null || true
  if (( restore_rc != 0 )); then
    exit 97
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

printf '%s\n' $'event\tboard_wall_time\tuptime_s\tpolicy0_governor\tpolicy4_governor\tpolicy6_governor\tnpu_governor' \
  > "$output_dir/governor_state.tsv"
snapshot_governors before
printf '%s\n' performance > "$policy0"
snapshot_governors after_set
[[ $(<"$policy0") == performance ]] || {
  echo "policy0 performance governor did not engage" >&2
  exit 77
}
[[ $(<"$policy4") == "$before_policy4" && \
   $(<"$policy6") == "$before_policy6" && \
   $(<"$npu_node/governor") == "$before_npu" ]] || {
  echo "read-only comparison or NPU governor changed during policy0 setup" >&2
  exit 77
}

"$script_dir/ope5_a55_fp16_gemv_measure_payload.sh" \
  "$output_dir" "$phase_file" "$bench" "$perf_bin"
