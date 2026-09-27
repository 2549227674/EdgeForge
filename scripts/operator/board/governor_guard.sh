#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# < 3 )) || [[ $2 != -- ]]; then
  echo "usage: sudo $0 STATE.tsv -- COMMAND [ARG ...]" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "governor_guard must run as root" >&2
  exit 77
fi

state_tsv=$1
shift 2

policy4=/sys/devices/system/cpu/cpufreq/policy4/scaling_governor
policy6=/sys/devices/system/cpu/cpufreq/policy6/scaling_governor
npu_node=
for path in /sys/class/devfreq/*; do
  [[ -e "$path" ]] || continue
  if [[ "${path##*/}" == *npu* ]]; then
    npu_node=$path
    break
  fi
done

for path in "$policy4" "$policy6"; do
  [[ -r "$path" && -w "$path" ]] || {
    echo "governor path is not readable/writable: $path" >&2
    exit 77
  }
done
[[ -n "$npu_node" && -r "$npu_node/governor" && -w "$npu_node/governor" ]] || {
  echo "NPU governor path unavailable" >&2
  exit 77
}

before_policy4=$(<"$policy4")
before_policy6=$(<"$policy6")
before_npu=$(<"$npu_node/governor")

snapshot() {
  local event=$1
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$event" \
    "$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')" \
    "$(cut -d' ' -f1 /proc/uptime)" \
    "$(<"$policy4")" \
    "$(<"$policy6")" \
    "$(<"$npu_node/governor")" >> "$state_tsv"
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  local restore_rc=0
  printf '%s\n' "$before_policy4" > "$policy4" || restore_rc=1
  printf '%s\n' "$before_policy6" > "$policy6" || restore_rc=1
  printf '%s\n' "$before_npu" > "$npu_node/governor" || restore_rc=1
  snapshot after_restore || restore_rc=1
  if (( restore_rc != 0 )); then
    echo "failed to restore one or more governors" >&2
    exit 97
  fi
  exit "$rc"
}

mkdir -p "$(dirname "$state_tsv")"
printf '%s\n' $'event\tboard_wall_time\tuptime_s\tpolicy4_governor\tpolicy6_governor\tnpu_governor' > "$state_tsv"
snapshot before
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

printf '%s\n' performance > "$policy4"
printf '%s\n' performance > "$policy6"
printf '%s\n' performance > "$npu_node/governor"
snapshot after_set

"$@"
