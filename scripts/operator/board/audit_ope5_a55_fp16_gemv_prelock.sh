#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 2 )); then
  echo "usage: $0 OUTPUT_DIR SOURCE" >&2
  exit 2
fi

output_dir=$1
source_file=$2
workspace=/home/orangepi/edgeforge_ope_r1
perf_bin=$workspace/bin/perf
audit_bin=$output_dir/ope5_a55_fp16_gemv_audit.bin
policy0_base=/sys/devices/system/cpu/cpufreq/policy0

[[ ! -e $output_dir ]] || { echo "refusing to reuse audit directory" >&2; exit 73; }
[[ -r $source_file && -x $perf_bin ]] || { echo "missing source or perf" >&2; exit 74; }
mkdir -p "$output_dir"
cleanup() {
  rm -f -- "$audit_bin"
}
trap cleanup EXIT

related_cpus=$(tr -s ' ' < "$policy0_base/related_cpus" | sed 's/^ //;s/ $//')
max_khz=$(<"$policy0_base/scaling_max_freq")
[[ $related_cpus == "0 1 2 3" ]] || { echo "unexpected policy0 related_cpus" >&2; exit 76; }
[[ $max_khz == 1800000 ]] || { echo "unexpected policy0 max frequency" >&2; exit 76; }

first_features=$(awk -F: '/^Features/{print $2; exit}' /proc/cpuinfo | sed 's/^ *//')
for feature in asimd fphp asimdhp; do
  grep -qE "(^| )${feature}( |$)" <<<"$first_features" || {
    echo "missing feature: $feature" >&2
    exit 76
  }
done

printf '%s\n' $'cpu\tobserved_allowed_list\tpass' > "$output_dir/affinity.tsv"
for cpu in 0 1 2 3; do
  observed=$(taskset -c "$cpu" sh -c "awk '/^Cpus_allowed_list:/{print \$2}' /proc/self/status")
  pass=no
  [[ $observed == "$cpu" ]] && pass=yes
  printf '%s\t%s\t%s\n' "$cpu" "$observed" "$pass" >> "$output_dir/affinity.tsv"
  [[ $pass == yes ]] || exit 76
done

perf_events='cycles:u,instructions:u,l2d_cache_refill:u,bus_access:u'
set +e
"$perf_bin" stat --no-big-num -x, -o "$output_dir/perf_true.stat.csv" \
  -e "$perf_events" -- taskset -c 0 true \
  > "$output_dir/perf_true.stdout.log" 2> "$output_dir/perf_true.stderr.log"
perf_rc=$?
set -e
printf '%s\n' "$perf_rc" > "$output_dir/perf_true.exit_code.txt"
(( perf_rc == 0 )) || exit "$perf_rc"

g++ -O3 -std=gnu++17 -mcpu=cortex-a55 -fno-fast-math -fno-math-errno \
  -pthread -Wall -Wextra -Werror \
  -fopt-info-vec-all="$output_dir/vectorization.log" \
  "$source_file" -o "$audit_bin" \
  > "$output_dir/build.stdout.log" 2> "$output_dir/build.stderr.log"
objdump -d --disassemble=scalar_fp32_rows "$audit_bin" \
  > "$output_dir/disassembly_scalar_fp32.txt"
objdump -d --disassemble=autovec_o3_rows "$audit_bin" \
  > "$output_dir/disassembly_autovec_o3.txt"
objdump -d --disassemble=neon_fp16_f32acc_rows "$audit_bin" \
  > "$output_dir/disassembly_neon_fp16_f32acc.txt"
if grep -Eq 'fmla[[:space:]]+v' "$output_dir/disassembly_scalar_fp32.txt"; then
  echo "scalar codegen gate failed" >&2
  exit 79
fi
if ! grep -Eq 'fcvtl' "$output_dir/disassembly_neon_fp16_f32acc.txt" || \
   ! grep -Eq 'fmla[[:space:]]+v' "$output_dir/disassembly_neon_fp16_f32acc.txt"; then
  echo "candidate codegen gate failed" >&2
  exit 79
fi

{
  printf 'recorded_at=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')"
  printf 'board=%s\n' "$(cat /proc/device-tree/model)"
  printf 'uname=%s\n' "$(uname -a)"
  printf 'compiler=%s\n' "$(g++ --version | head -1)"
  printf 'source_sha256=%s\n' "$(sha256sum "$source_file" | awk '{print $1}')"
  printf 'policy0_related_cpus=%s\n' "$related_cpus"
  printf 'policy0_max_khz=%s\n' "$max_khz"
  printf 'first_core_features=%s\n' "$first_features"
  printf 'perf_events=%s\n' "$perf_events"
  printf 'compile_invocations=1\n'
  printf 'benchmark_execution_count=0\n'
  printf 'compiler_target=-mcpu=cortex-a55\n'
} > "$output_dir/audit_meta.env"

rm -f -- "$audit_bin"
trap - EXIT
[[ ! -e $audit_bin ]] || exit 80
