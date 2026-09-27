#!/usr/bin/env bash
set -euo pipefail

workspace=${1:-/root/autodl-tmp/edgeforge-m1/workspace}
cd "$workspace"

run_dir=results/t2/c01_s_family_80_20_half_r1
candidate_dir=models/C01-S-bf16-r1
audit_file=results/t2/c01_s_completion_audit.json

while pgrep -f '[t]rain_recipe_race.py.*c01_s_family_80_20_half_r1' >/dev/null; do
  sleep 60
done

../venv/bin/python - <<'PY'
import json
import math
from pathlib import Path

run_dir = Path("results/t2/c01_s_family_80_20_half_r1")
report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
assert (run_dir / "READY").is_file()
assert report["run_kind"] == "full"
assert report["recipe"] == "family_80_20"
assert report["world_size"] == 2
assert report["completed_steps"] == 4096
assert report["full_schedule_steps"] == 4096
assert report["warmup_steps"] == 123
assert report["input_tokens"] == 148425311
assert report["expected_input_tokens"] == 148425311
assert report["plan_sha256"] == "61ece8bad05f4156b4b20fa734c2b7e1f804fc879aa50d7a28cceb790042820e"
assert report["holdout_sha256"] == "0a28a7f7b52a01f7a73ccdf927f98ac21902f8037ef236db4f86d5024a67d73b"
assert report["template_sha256"] == "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
assert [item["optimizer_step"] for item in report["evaluations"]] == [0, 1024, 2048, 3072, 4096]
assert report["final_validation"]["optimizer_step"] == 4096
for item in report["evaluations"]:
    assert item["records"] == 1948
    assert item["input_tokens"] == 3483799
    assert item["supervised_tokens"] == 1964016
    assert math.isfinite(item["sample_mean_loss"])
    assert math.isfinite(item["sample_loss_standard_error"])
    assert math.isfinite(item["token_weighted_loss"])
rank_audits = report["rank_audits"]
assert sorted(item["rank"] for item in rank_audits) == [0, 1]
assert sum(item["input_tokens"] for item in rank_audits) == 148425311
assert sum(item["records"] for item in rank_audits) == report["records"]
assert sum(item["supervised_tokens"] for item in rank_audits) == report["supervised_tokens"]
PY

test ! -e "$candidate_dir"
test ! -e "${candidate_dir}.incomplete"
test ! -e "$audit_file"
test ! -e "${audit_file}.incomplete"
../venv/bin/python scripts/m1/merge_lora_candidate.py \
  --base-model models/gemma-4-E4B-it \
  --adapter "$run_dir/adapter" \
  --training-report "$run_dir/report.json" \
  --output-dir "$candidate_dir" \
  --candidate C01-S

exec ../venv/bin/python scripts/m1/verify_c01_s_completion.py \
  --run-dir "$run_dir" \
  --candidate-dir "$candidate_dir" \
  --output "$audit_file"
