#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 5 )); then
  echo "usage: $0 OUTPUT_DIR PHASE_FILE BENCH PERF NPU_LOAD_SAMPLER" >&2
  exit 2
fi

output_dir=$1
phase_file=$2
bench=$3
perf_bin=$4
npu_load_sampler=$5
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)

telemetry_pid=
npu_load_pid=
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  for pid in "$telemetry_pid" "$npu_load_pid"; do
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

printf '%s\n' telemetry_settle > "$phase_file"
"$script_dir/telemetry_sampler.sh" \
  "$output_dir/telemetry_1hz.csv" "$phase_file" 1 &
telemetry_pid=$!
python3 "$npu_load_sampler" \
  "$output_dir/npu_load_10hz.csv" "$phase_file" 0.1 &
npu_load_pid=$!
sleep 2

printf '%s\n' formal_capability_and_correctness > "$phase_file"
timeout --signal=TERM --kill-after=10 2400 \
  taskset -c 7 "$bench" --mode correctness --output-dir "$output_dir" \
  > "$output_dir/correctness.stdout.log" \
  2> "$output_dir/correctness.stderr.log"

python3 - "$output_dir" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
capability = list(csv.DictReader((root / "capability.csv").open(encoding="utf-8")))
assert len(capability) == 26, len(capability)
assert {row["stage"] for row in capability} == {"qk", "pv"}
assert all(row["core_mask"] == "core0" for row in capability)
assert all(row["api_type"] == "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32" for row in capability)
assert all(row["pass"] == "1" for row in capability)

correctness = list(csv.DictReader((root / "correctness.csv").open(encoding="utf-8")))
assert len(correctness) == 72, len(correctness)
assert {row["variant"] for row in correctness} == {
    "cpu_scalar_sequence", "cpu_neon_sequence", "hybrid_rknpu_sequence"
}
assert {row["distribution"] for row in correctness} == {"low_normal", "unit_normal"}
assert {int(row["seed"]) for row in correctness} == {2026092101, 2026092102}
assert sum(int(row["elements"]) for row in correctness) == 270336
assert all(int(row["mismatches"]) == 0 for row in correctness)
assert all(row["finite"] == "1" for row in correctness)
assert all(row["immutable_source"] == "1" for row in correctness)
assert all(float(row["pre_round_row_sum_error"]) <= 2e-6 for row in correctness)
assert all(float(row["fp16_row_sum_error"]) <= 0.01 for row in correctness)
assert all(row["pass"] == "1" for row in correctness)

gates = list(csv.DictReader((root / "state_gates.csv").open(encoding="utf-8")))
assert len(gates) == 8, len(gates)
assert all(row["pass"] == "1" for row in gates)
summary = json.loads((root / "correctness_summary.json").read_text(encoding="utf-8"))
assert summary == {
    "status": "PASS",
    "logical_trials": 24,
    "implementation_trials": 72,
    "output_element_comparisons": 270336,
    "mismatches": 0,
    "state_gates": 8,
}
PY
printf '%s\n' PASS_CAPABILITY_26_AND_CORRECTNESS_270336 \
  > "$output_dir/correctness_gate.txt"

printf '%s\n' formal_interleaved_sequence_timing > "$phase_file"
timeout --signal=TERM --kill-after=10 5400 \
  taskset -c 7 "$bench" --mode measure --output-dir "$output_dir" \
  > "$output_dir/timing.stdout.log" \
  2> "$output_dir/timing.stderr.log"

python3 - "$output_dir/timings.csv" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
assert len(rows) == 39, len(rows)
assert len({row["case_id"] for row in rows}) == 13
assert {row["variant"] for row in rows} == {
    "cpu_scalar_sequence", "cpu_neon_sequence", "hybrid_rknpu_sequence"
}
assert sum(int(row["projected_invocations"]) for row in rows if row["variant"] == "hybrid_rknpu_sequence") == 911148
assert all(int(row["warmup"]) == 10 for row in rows)
assert all(50 <= int(row["repetitions"]) <= 5000 for row in rows)
assert all(float(row["elapsed_ms"]) >= 500.0 for row in rows)
assert all(float(row["p50_us"]) > 0 for row in rows)
assert all(row["affinity_pass"] == "1" for row in rows)
PY
printf '%s\n' PASS_13_CASES_3_COMPLETE_SEQUENCES \
  > "$output_dir/timing_gate.txt"

perf_events='cycles:u,instructions:u,task-clock,context-switches,cpu-migrations'
printf '%s\n' $'case_id\tvariant\tevents\texit_code' \
  > "$output_dir/perf_event_attempts.tsv"
for case_id in sliding_k512_d256 full_k3123_d512; do
  for variant in cpu_scalar_sequence cpu_neon_sequence hybrid_rknpu_sequence; do
    printf 'tier_b_%s_%s\n' "$case_id" "$variant" > "$phase_file"
    prefix="$output_dir/perf_${case_id}_${variant}"
    set +e
    timeout --signal=TERM --kill-after=10 1200 \
      "$perf_bin" stat --no-big-num -x, \
        -o "$prefix.stat.csv" -e "$perf_events" -- \
        taskset -c 7 "$bench" --mode perf --case "$case_id" --variant "$variant" \
        > "$prefix.bench.csv" 2> "$prefix.stderr.log"
    perf_rc=$?
    set -e
    printf '%s\n' "$perf_rc" > "$prefix.exit_code.txt"
    printf '%s\t%s\t%s\t%s\n' "$case_id" "$variant" "$perf_events" "$perf_rc" \
      >> "$output_dir/perf_event_attempts.tsv"
    if (( perf_rc != 0 )); then
      echo "Tier-B perf failed for $case_id/$variant with exit $perf_rc" >&2
      exit "$perf_rc"
    fi
  done
done

python3 - "$output_dir" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
events = {"cycles:u", "instructions:u", "task-clock", "context-switches", "cpu-migrations"}
for case_id in ("sliding_k512_d256", "full_k3123_d512"):
    for variant in ("cpu_scalar_sequence", "cpu_neon_sequence", "hybrid_rknpu_sequence"):
        prefix = root / f"perf_{case_id}_{variant}"
        observed = {}
        for row in csv.reader(prefix.with_suffix(".stat.csv").open(encoding="utf-8")):
            if not row or row[0].startswith("#") or len(row) < 3:
                continue
            event = row[2].strip()
            if event not in events:
                continue
            value = row[0].strip()
            assert value not in {"<not counted>", "<not supported>", ""}, (case_id, variant, event, value)
            observed[event] = float(value)
        assert set(observed) == events, (case_id, variant, observed)
        rows = list(csv.DictReader(prefix.with_suffix(".bench.csv").open(encoding="utf-8")))
        assert len(rows) == 1, (case_id, variant, rows)
        row = rows[0]
        assert row["case_id"] == case_id
        assert row["variant"] == variant
        assert int(row["warmup"]) == 10
        assert 50 <= int(row["repetitions"]) <= 5000
        assert float(row["elapsed_ms"]) >= 500.0
        assert row["affinity_pass"] == "1"
PY
printf '%s\n' PASS_FIVE_EVENTS_SIX_SEQUENCE_BLOCKS \
  > "$output_dir/tier_b_gate.txt"

printf '%s\n' measurement_complete > "$phase_file"
sleep 2
