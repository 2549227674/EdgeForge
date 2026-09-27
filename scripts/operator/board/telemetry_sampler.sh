#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 OUTPUT.csv PHASE_FILE [INTERVAL_SECONDS]" >&2
  exit 2
fi

output_csv=$1
phase_file=$2
interval_seconds=${3:-1}

read_one() {
  local path=$1
  if [[ -r "$path" ]]; then
    tr -d '\n' < "$path"
  else
    printf 'NA'
  fi
}

find_devfreq() {
  local needle=$1
  local path
  for path in /sys/class/devfreq/*; do
    [[ -e "$path" ]] || continue
    if [[ "${path##*/}" == *"$needle"* ]]; then
      printf '%s' "$path"
      return 0
    fi
  done
  return 1
}

npu_devfreq=$(find_devfreq npu || true)
dmc_devfreq=$(find_devfreq dmc || true)

thermal_paths=()
thermal_headers=()
for thermal_path in /sys/class/thermal/thermal_zone*; do
  [[ -d "$thermal_path" ]] || continue
  zone=${thermal_path##*thermal_zone}
  zone_type=$(read_one "$thermal_path/type")
  zone_type=${zone_type//[^a-zA-Z0-9_]/_}
  thermal_paths+=("$thermal_path")
  thermal_headers+=("thermal_zone${zone}_${zone_type}_mC")
done

mkdir -p "$(dirname "$output_csv")"
{
  printf '%s' 'board_wall_time,uptime_s,phase'
  printf '%s' ',policy0_khz,policy0_max_khz,policy0_governor'
  printf '%s' ',policy4_khz,policy4_max_khz,policy4_governor'
  printf '%s' ',policy6_khz,policy6_max_khz,policy6_governor'
  printf '%s' ',npu_hz,npu_max_hz,npu_governor,dmc_hz,dmc_governor'
  printf '%s' ',npu_load_core0_pct,npu_load_core1_pct,npu_load_core2_pct'
  for header in "${thermal_headers[@]}"; do
    printf ',%s' "$header"
  done
  printf '\n'
} > "$output_csv"

while :; do
  board_wall_time=$(date '+%Y-%m-%dT%H:%M:%S.%N%:z')
  uptime_s=$(cut -d' ' -f1 /proc/uptime)
  phase=$(read_one "$phase_file")
  phase=${phase//,/;}

  load0=NA
  load1=NA
  load2=NA
  if [[ -r /sys/kernel/debug/rknpu/load ]]; then
    mapfile -t loads < <(grep -oE '[0-9]+%' /sys/kernel/debug/rknpu/load | tr -d '%')
    load0=${loads[0]:-NA}
    load1=${loads[1]:-NA}
    load2=${loads[2]:-NA}
  fi

  {
    printf '%s,%s,%s' "$board_wall_time" "$uptime_s" "$phase"
    printf ',%s,%s,%s' \
      "$(read_one /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq)" \
      "$(read_one /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq)" \
      "$(read_one /sys/devices/system/cpu/cpufreq/policy0/scaling_governor)"
    printf ',%s,%s,%s' \
      "$(read_one /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq)" \
      "$(read_one /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq)" \
      "$(read_one /sys/devices/system/cpu/cpufreq/policy4/scaling_governor)"
    printf ',%s,%s,%s' \
      "$(read_one /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq)" \
      "$(read_one /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq)" \
      "$(read_one /sys/devices/system/cpu/cpufreq/policy6/scaling_governor)"
    if [[ -n "$npu_devfreq" ]]; then
      printf ',%s,%s,%s' \
        "$(read_one "$npu_devfreq/cur_freq")" \
        "$(read_one "$npu_devfreq/max_freq")" \
        "$(read_one "$npu_devfreq/governor")"
    else
      printf ',NA,NA,NA'
    fi
    if [[ -n "$dmc_devfreq" ]]; then
      printf ',%s,%s' \
        "$(read_one "$dmc_devfreq/cur_freq")" \
        "$(read_one "$dmc_devfreq/governor")"
    else
      printf ',NA,NA'
    fi
    printf ',%s,%s,%s' "$load0" "$load1" "$load2"
    for thermal_path in "${thermal_paths[@]}"; do
      printf ',%s' "$(read_one "$thermal_path/temp")"
    done
    printf '\n'
  } >> "$output_csv"

  sleep "$interval_seconds"
done
