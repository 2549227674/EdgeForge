#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 8 )); then
  echo "usage: $0 OUTPUT_DIR PHASE_FILE BENCH PERF RUN_CORRECTNESS RUN_TIER_B WARMUP REPS" >&2
  exit 2
fi

output_dir=$1
phase_file=$2
bench=$3
perf_bin=$4
run_correctness=$5
run_tier_b=$6
warmup=$7
reps=$8
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)

if [[ $run_correctness != yes && $run_correctness != no ]]; then
  echo "RUN_CORRECTNESS must be yes or no" >&2
  exit 2
fi
if [[ $run_tier_b != yes && $run_tier_b != no ]]; then
  echo "RUN_TIER_B must be yes or no" >&2
  exit 2
fi
if [[ $warmup != 50 || $reps != 200 ]]; then
  echo "formal OPE3 payload requires warmup=50 and reps=200" >&2
  exit 2
fi

telemetry_pid=
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if [[ -n $telemetry_pid ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

printf '%s\n' performance_settle > "$phase_file"
"$script_dir/telemetry_sampler.sh" "$output_dir/telemetry_1hz.csv" "$phase_file" 1 &
telemetry_pid=$!
sleep 2

if [[ $run_correctness == yes ]]; then
  printf '%s\n' randomized_correctness > "$phase_file"
  taskset -c 4-7 "$bench" \
    --mode correctness --threads 4 --first-cpu 4 \
    > "$output_dir/correctness.csv" \
    2> "$output_dir/correctness.stderr.log"
else
  printf '%s\n' NOT_RUN_REUSED_FROM_FIRST_SEGMENT \
    > "$output_dir/correctness.disposition.txt"
fi

printf '%s\n' formal_interleaved_three_baselines > "$phase_file"
taskset -c 4-7 "$bench" \
  --mode measure --threads 4 --first-cpu 4 \
  --warmup "$warmup" --reps "$reps" --seed 20260920 \
  > "$output_dir/timings.csv" \
  2> "$output_dir/timings.stderr.log"

run_perf_group() {
  local variant=$1
  local output_prefix=$2
  local events=$3
  printf 'tier_b_%s\n' "$variant" > "$phase_file"
  set +e
  "$perf_bin" stat --no-big-num -x, \
    -o "$output_prefix.stat.csv" -e "$events" -- \
    taskset -c 4-7 "$bench" \
      --mode perf --variant "$variant" --threads 4 --first-cpu 4 \
      --warmup 0 --reps 200 --seed 20260920 \
      > "$output_prefix.bench.csv" \
      2> "$output_prefix.stderr.log"
  local rc=$?
  set -e
  printf '%s\n' "$rc" > "$output_prefix.exit_code.txt"
  return "$rc"
}

if [[ $run_tier_b == yes ]]; then
  perf_events='cycles:u,instructions:u,l2d_cache_refill:u,bus_access:u'
  printf '%s\n' $'variant\tmode\tevent\texit_code' \
    > "$output_dir/perf_event_attempts.tsv"
  for variant in scalar autovec neon_sdot; do
    prefix="$output_dir/perf_${variant}_group"
    if run_perf_group "$variant" "$prefix" "$perf_events"; then
      printf '%s\t%s\t%s\t0\n' "$variant" group "$perf_events" \
        >> "$output_dir/perf_event_attempts.tsv"
      continue
    fi
    group_rc=$(<"$prefix.exit_code.txt")
    printf '%s\t%s\t%s\t%s\n' "$variant" group "$perf_events" "$group_rc" \
      >> "$output_dir/perf_event_attempts.tsv"
    for event in cycles:u instructions:u l2d_cache_refill:u bus_access:u; do
      event_slug=${event//:/_}
      isolated_prefix="$output_dir/perf_${variant}_${event_slug}"
      if run_perf_group "$variant" "$isolated_prefix" "$event"; then
        event_rc=0
      else
        event_rc=$(<"$isolated_prefix.exit_code.txt")
      fi
      printf '%s\t%s\t%s\t%s\n' "$variant" isolated "$event" "$event_rc" \
        >> "$output_dir/perf_event_attempts.tsv"
    done
  done

  printf '%s\n' tier_b_neon_sdot_record > "$phase_file"
  set +e
  "$perf_bin" record -o "$output_dir/perf.data" \
    -e cycles:u -F 99 --call-graph dwarf -- \
    taskset -c 4-7 "$bench" \
      --mode perf --variant neon_sdot --threads 4 --first-cpu 4 \
      --warmup 0 --reps 200 --seed 20260920 \
      > "$output_dir/perf_record_bench.csv" \
      2> "$output_dir/perf_record.stderr.log"
  perf_record_rc=$?
  set -e
  printf '%s\n' "$perf_record_rc" > "$output_dir/perf_record_exit_code.txt"
  if (( perf_record_rc == 0 )); then
    set +e
    "$perf_bin" report -i "$output_dir/perf.data" --stdio --no-children \
      > "$output_dir/perf_report.txt" \
      2> "$output_dir/perf_report.stderr.log"
    perf_report_rc=$?
    set -e
    printf '%s\n' "$perf_report_rc" > "$output_dir/perf_report_exit_code.txt"
  else
    printf '%s\n' NOT_RUN_PERF_RECORD_FAILED \
      > "$output_dir/perf_report.disposition.txt"
  fi
else
  printf '%s\n' NOT_RUN_FIRST_VALID_SEGMENT_ONLY \
    > "$output_dir/tier_b.disposition.txt"
fi

printf '%s\n' measurement_complete > "$phase_file"
sleep 2
