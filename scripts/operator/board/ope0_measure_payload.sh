#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 4 )); then
  echo "usage: $0 OUTPUT_DIR PHASE_FILE TRIAD_BIN MATMUL_BIN" >&2
  exit 2
fi

output_dir=$1
phase_file=$2
triad_bin=$3
matmul_bin=$4

printf '%s\n' performance_settle > "$phase_file"
sleep 3

printf '%s\n' bandwidth > "$phase_file"
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE \
  taskset -c 4-7 "$triad_bin" \
    --sizes-mib 64,128,256,512,1024 \
    --warmup 50 \
    --reps 200 \
    --threads 4 \
    --first-cpu 4 \
    > "$output_dir/bandwidth_samples.csv" \
    2> "$output_dir/bandwidth_stderr.log"

printf '%s\n' matmul > "$phase_file"
printf '%s\n' 'board_wall_time,uptime_s,npu_load_core0_pct,npu_load_core1_pct,npu_load_core2_pct' \
  > "$output_dir/npu_load_10hz.csv"

taskset -c 7 "$matmul_bin" \
  --m 4 --k 64 --n 32 \
  --warmup 50 \
  --reps 200 \
  --load-seconds 8 \
  > "$output_dir/matmul.log" 2>&1 &
matmul_pid=$!

sample_load() {
  while kill -0 "$matmul_pid" 2>/dev/null; do
    load0=NA
    load1=NA
    load2=NA
    if [[ -r /sys/kernel/debug/rknpu/load ]]; then
      mapfile -t loads < <(grep -oE '[0-9]+%' /sys/kernel/debug/rknpu/load | tr -d '%')
      load0=${loads[0]:-NA}
      load1=${loads[1]:-NA}
      load2=${loads[2]:-NA}
    fi
    printf '%s,%s,%s,%s,%s\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')" \
      "$(cut -d' ' -f1 /proc/uptime)" \
      "$load0" "$load1" "$load2" >> "$output_dir/npu_load_10hz.csv"
    sleep 0.1
  done
}

sample_load &
sampler_pid=$!
set +e
wait "$matmul_pid"
matmul_rc=$?
wait "$sampler_pid"
set -e
printf '%s\n' "$matmul_rc" > "$output_dir/matmul_exit_code.txt"
if (( matmul_rc != 0 )); then
  cat "$output_dir/matmul.log" >&2
  exit "$matmul_rc"
fi

printf '%s\n' performance_complete > "$phase_file"
sleep 2
