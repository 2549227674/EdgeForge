#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 4 )); then
  echo "usage: $0 OUTPUT_DIR PHASE_FILE BENCH PERF" >&2
  exit 2
fi

output_dir=$1
phase_file=$2
bench=$3
perf_bin=$4
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
warmup=50
repetitions=200
population_min_seconds=3.0
ring_frames=16
seed=20260921

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
"$script_dir/telemetry_sampler.sh" \
  "$output_dir/telemetry_1hz.csv" "$phase_file" 1 &
telemetry_pid=$!
sleep 2

printf '%s\n' randomized_correctness > "$phase_file"
timeout --signal=TERM --kill-after=10 300 \
  taskset -c 7 "$bench" --mode correctness \
  > "$output_dir/correctness.csv" \
  2> "$output_dir/correctness.stderr.log"

python3 - "$output_dir/correctness.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
assert len(rows) == 144, len(rows)
assert {row["variant"] for row in rows} == {"autovec_o3", "neon_fp16"}
assert len({row["case_id"] for row in rows}) == 8
assert {row["distribution"] for row in rows} == {
    "unit_normal", "wide_normal_std32", "near_zero_uniform_plus_minus_0.001"
}
assert {int(row["seed"]) for row in rows} == {20260921, 20260922, 20260923}
assert all(row["pass"] == "1" for row in rows)
assert all(row["all_finite"] == "1" for row in rows)
assert all(row["input_unchanged"] == "1" for row in rows)
assert all(row["scale_unchanged"] == "1" for row in rows)
assert all(row["scale_pointer_contract"] == "1" for row in rows)
PY
printf '%s\n' PASS_144_OF_144 > "$output_dir/correctness_gate.txt"

printf '%s\n' formal_interleaved_timing > "$phase_file"
timeout --signal=TERM --kill-after=10 900 \
  taskset -c 7 "$bench" \
    --mode measure \
    --warmup "$warmup" \
    --reps "$repetitions" \
    --population-min-seconds "$population_min_seconds" \
    --ring-frames "$ring_frames" \
    --seed "$seed" \
  > "$output_dir/timings.csv" \
  2> "$output_dir/timings.stderr.log"

python3 - "$output_dir/timings.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
per_case = [row for row in rows if row["scope"] == "per_case"]
population = [row for row in rows if row["scope"] == "population"]
assert len(per_case) == 24, len(per_case)
assert len(population) == 3, len(population)
assert {row["variant"] for row in population} == {
    "scalar_ref", "autovec_o3", "neon_fp16"
}
assert all(int(row["repetitions"]) == 200 for row in per_case)
assert all(int(row["repetitions"]) >= 200 for row in population)
assert all(int(row["calls_per_forward"]) == 302 for row in population)
assert all(int(row["analytic_bytes"]) == 3840000 for row in population)
assert all(int(row["permutations_used"]) == 6 for row in rows)
assert all(int(row["ring_frames"]) == 16 for row in rows)
assert all(float(row["iqr_over_p50"]) <= 0.30 for row in population)
assert all(float(row["p50_us"]) > 0 for row in rows)
PY
printf '%s\n' PASS_PRIMARY_POPULATION_STABLE > "$output_dir/timing_gate.txt"

perf_events='cycles:u,instructions:u,l2d_cache_refill:u,bus_access:u'
printf '%s\n' $'variant\tevents\texit_code' \
  > "$output_dir/perf_event_attempts.tsv"
for variant in scalar_ref autovec_o3 neon_fp16; do
  printf 'tier_b_%s\n' "$variant" > "$phase_file"
  prefix="$output_dir/perf_${variant}_population"
  set +e
  timeout --signal=TERM --kill-after=10 600 \
    "$perf_bin" stat --no-big-num -x, \
      -o "$prefix.stat.csv" -e "$perf_events" -- \
      taskset -c 7 "$bench" \
        --mode perf --variant "$variant" \
        --warmup "$warmup" --reps "$repetitions" \
        --population-min-seconds "$population_min_seconds" \
        --ring-frames "$ring_frames" --seed "$seed" \
      > "$prefix.bench.csv" 2> "$prefix.stderr.log"
  perf_rc=$?
  set -e
  printf '%s\n' "$perf_rc" > "$prefix.exit_code.txt"
  printf '%s\t%s\t%s\n' "$variant" "$perf_events" "$perf_rc" \
    >> "$output_dir/perf_event_attempts.tsv"
  if (( perf_rc != 0 )); then
    echo "Tier-B perf failed for $variant with exit $perf_rc" >&2
    exit "$perf_rc"
  fi
done

python3 - "$output_dir" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
events = {"cycles:u", "instructions:u", "l2d_cache_refill:u", "bus_access:u"}
for variant in ("scalar_ref", "autovec_o3", "neon_fp16"):
    stat_path = root / f"perf_{variant}_population.stat.csv"
    observed = {}
    for row in csv.reader(stat_path.open(encoding="utf-8")):
        if not row or row[0].startswith("#") or len(row) < 3:
            continue
        event = row[2].strip()
        if event not in events:
            continue
        value = row[0].strip()
        assert value not in {"<not counted>", "<not supported>", ""}, (variant, event, value)
        observed[event] = int(value)
    assert set(observed) == events, (variant, observed)
    bench_rows = list(csv.DictReader(
        (root / f"perf_{variant}_population.bench.csv").open(encoding="utf-8")
    ))
    assert len(bench_rows) == 1
    row = bench_rows[0]
    assert row["scope"] == "perf_population"
    assert row["variant"] == variant
    assert int(row["population_sequences"]) >= 200
    assert int(row["analytic_bytes_per_sequence"]) == 3840000
    assert float(row["elapsed_ms"]) >= 3000.0
    assert int(row["ring_frames"]) == 16
PY
printf '%s\n' PASS_ALL_FOUR_EVENTS_ALL_THREE_VARIANTS \
  > "$output_dir/tier_b_gate.txt"

printf '%s\n' measurement_complete > "$phase_file"
sleep 2
