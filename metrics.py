#!/usr/bin/env python3
"""Compute EdgeForge's B0 agent metrics from ATIF trajectories (schema v2).

v2 keeps the two v1 parser columns byte-identical and adds trial-level,
recovery, clustered-statistics and token-split columns.  It is run over the
100 frozen B0 trajectories under ``results/baseline_e4b_q4km``.

Design decisions frozen after the 2026-08-05 input probe (branch B):

* ``parser_hard_error_rate`` / ``parser_soft_warning_rate`` reuse the exact v1
  definitions.  ``--verify-v1`` asserts they reproduce the committed
  ``parser_metrics.json`` counts (149/498/836/125) before anything else.
* Success is 0/100.  The zero rate is reported with a rule-of-three upper bound
  over the 20 task clusters (0.15), not a Wald SE (which is 0 and misleads).
* Parser rates carry a task-clustered bootstrap SE; the naive binomial SE is
  emitted but flagged ``do_not_cite`` (design effect ~20x).
* ``turns_per_trial`` is right-censored at max_turns=30 (7 trials) and once by
  an agent timeout; the headline statistic is the median.
* ``reasoning_content`` is absent on 125/836 responses.  The probe showed this
  absence is a MODEL BEHAVIOUR, not a logging gap: absent-accepted responses
  have ~6x lower completion tokens.  So ``no_reasoning_rate`` is a first-class
  column over all 836 responses, and thinking-token stats are conditional on
  the 711 responses that have reasoning.
* There is no persisted ``task_complete`` field, so no premature-complete rate
  is computed. F1 is the 92 zero-reward trials that neither exhausted the
  turn guard nor hit the agent timeout, and is reported as such.
* Command tokens come from ``tool_calls[].arguments.keystrokes`` (harbor-
  normalized), labelled as such; the original model JSON body is not persisted.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "edgeforge-agent-metrics/v2"
HARD_ERROR_PREFIX = "Previous response had parsing errors:"
SOFT_WARNING_PREFIX = "Previous response had warnings:"
BEFORE_JSON_WARNING = "Extra text detected before JSON object"
AFTER_JSON_WARNING = "Extra text detected after JSON object"
MAX_TURNS = 30
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 0

# Frozen v1 parser counts; --verify-v1 also re-reads the committed file.
V1_EXPECTED = {
    "agent_response_attempts": 836,
    "hard_error_responses": 149,
    "soft_warning_accepted_responses": 498,
    "hard_error_responses_with_extra_text_warning": 125,
}


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Root with <trial>/agent/trajectory.json files.")
    parser.add_argument("--job-result", required=True, type=Path,
                        help="Harbor job result.json (trial-level rewards/errors).")
    parser.add_argument("--trajectory-manifest", required=True, type=Path,
                        help="sha256 manifest; every listed file is verified.")
    parser.add_argument("--token-sidecar", type=Path,
                        help="count_response_tokens.py output; omit to skip token split.")
    parser.add_argument("--verify-v1", type=Path,
                        help="Committed parser_metrics.json to reproduce exactly.")
    parser.add_argument("--schema-branch", default="B", choices=["A", "B", "C"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def verify_manifest(root: Path, manifest: Path) -> dict[str, Any]:
    import hashlib

    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(maxsplit=1)
        expected[rel] = digest

    checked = 0
    mismatches: list[str] = []
    for rel in sorted(expected):
        path = Path(rel)
        if not path.exists():
            mismatches.append(f"missing:{rel}")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        checked += 1
        if got != expected[rel]:
            mismatches.append(f"mismatch:{rel}")
    if mismatches:
        raise ValueError(
            "trajectory manifest verification failed: " + "; ".join(mismatches[:5])
        )
    return {"manifest": relative_path(manifest),
            "files_checked": checked, "all_match": True}


def trajectory_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*/agent/trajectory.json"))
    if not files:
        raise FileNotFoundError(f"no trajectory.json under {root}")
    return files


def load_trajectory(path: Path) -> Mapping[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("steps"), list):
        raise ValueError(f"trajectory has no list-valued steps: {path}")
    return parsed


# --------------------------------------------------------------------------- #
# parser feedback (v1-identical)
# --------------------------------------------------------------------------- #
def observation_contents(step: Mapping[str, Any]) -> list[str]:
    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        return []
    results = observation.get("results")
    if not isinstance(results, list):
        return []
    return [r["content"] for r in results
            if isinstance(r, Mapping) and isinstance(r.get("content"), str)]


def parser_feedback(contents: Iterable[str]) -> list[str]:
    return [c for c in contents
            if c.startswith(HARD_ERROR_PREFIX) or c.startswith(SOFT_WARNING_PREFIX)]


def classify_parser(step: Mapping[str, Any]) -> dict[str, int]:
    feedback = parser_feedback(observation_contents(step))
    hard = any(c.startswith(HARD_ERROR_PREFIX) for c in feedback)
    before = any(BEFORE_JSON_WARNING in c for c in feedback)
    after = any(AFTER_JSON_WARNING in c for c in feedback)
    has_extra = before or after
    soft = has_extra and not hard
    return {
        "hard": int(hard),
        "soft": int(soft),
        "hard_with_extra": int(hard and has_extra),
    }


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cluster_bootstrap_se(clusters: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    """Bootstrap SE of a pooled rate by resampling task clusters."""
    names = list(clusters)

    def rate(sample: Sequence[str]) -> float:
        num = sum(clusters[t][0] for t in sample)
        den = sum(clusters[t][1] for t in sample)
        return num / den if den else 0.0

    point = rate(names)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [rate([rng.choice(names) for _ in names]) for _ in range(BOOTSTRAP_DRAWS)]
    mean = sum(draws) / len(draws)
    se = math.sqrt(sum((x - mean) ** 2 for x in draws) / (len(draws) - 1))
    den_total = sum(v[1] for v in clusters.values())
    naive = math.sqrt(point * (1 - point) / den_total) if den_total else 0.0
    return {
        "point": point,
        "cluster_se": se,
        "naive_binomial_se_do_not_cite": naive,
        "design_effect": (se / naive) ** 2 if naive else None,
        "n_clusters": len(names),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def describe(values: Sequence[float]) -> dict[str, Any]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "median": statistics.median(vals),
            "mean": round(sum(vals) / len(vals), 2),
            "p25": percentile(vals, 0.25), "p75": percentile(vals, 0.75),
            "min": min(vals), "max": max(vals)}


# --------------------------------------------------------------------------- #
# per-trial extraction
# --------------------------------------------------------------------------- #
def agent_steps(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [s for s in trajectory["steps"]
            if isinstance(s, Mapping) and s.get("source") == "agent"]


def recovery_turns(parser_flags: Sequence[int]) -> tuple[list[int], int]:
    """From a per-turn hard-error flag sequence, measure turns from each hard
    error to the next accepted turn.  Returns (recovery lengths, unrecovered)."""
    lengths: list[int] = []
    unrecovered = 0
    i = 0
    n = len(parser_flags)
    while i < n:
        if parser_flags[i] == 1:
            j = i + 1
            span = 1
            while j < n and parser_flags[j] == 1:
                span += 1
                j += 1
            if j < n:  # reached an accepted turn
                lengths.append(span)
            else:
                unrecovered += 1
            i = j
        else:
            i += 1
    return lengths, unrecovered


# --------------------------------------------------------------------------- #
# main summarize
# --------------------------------------------------------------------------- #
def summarize(args: argparse.Namespace) -> dict[str, Any]:
    files = trajectory_files(args.input)
    verification = verify_manifest(args.input, args.trajectory_manifest)

    job = json.loads(args.job_result.read_text(encoding="utf-8"))
    job_stats = job.get("stats", {})
    n_trials = job.get("n_total_trials")
    n_errored = job_stats.get("n_errored_trials", 0)
    n_retries = job_stats.get("n_retries", 0)
    evals = job_stats.get("evals", {})
    successes = sum(int(round(e.get("metrics", [{}])[0].get("mean", 0) * e["n_trials"]))
                    for e in evals.values())

    # per-response + per-trial aggregation
    hard = soft = hard_with_extra = 0
    responses = 0
    by_task: dict[str, list[int]] = {}          # hard count per trial, keyed by task
    by_task_resp: dict[str, list[int]] = {}       # response count per trial
    turns_list: list[int] = []
    censored_turns = 0
    all_recovery: list[int] = []
    unrecovered_lockin = 0
    reasoning_present = 0
    per_task_rows: dict[str, dict[str, int]] = {}
    per_trial_rows: list[dict[str, Any]] = []
    comp_with: list[int] = []
    comp_absent_accepted: list[int] = []
    comp_absent_hard: list[int] = []
    cached_missing: list[str] = []
    completion_per_trial: list[int] = []
    prompt_per_trial: list[int] = []
    cached_share_per_response: list[float] = []

    for path in files:
        trajectory = load_trajectory(path)
        trial = path.parts[-3]
        task = trial.split("__", 1)[0]
        steps = agent_steps(trajectory)
        n_steps = len(steps)
        turns_list.append(n_steps)
        if n_steps >= MAX_TURNS:
            censored_turns += 1

        hard_flags: list[int] = []
        trial_hard = 0
        for step in steps:
            responses += 1
            flags = classify_parser(step)
            hard += flags["hard"]
            soft += flags["soft"]
            hard_with_extra += flags["hard_with_extra"]
            hard_flags.append(flags["hard"])
            trial_hard += flags["hard"]

            metrics = step.get("metrics") or {}
            comp = metrics.get("completion_tokens")
            if "cached_tokens" not in metrics:
                cached_missing.append(f"{trial}:step{step.get('step_id')}")
            reasoning = step.get("reasoning_content")
            has_r = isinstance(reasoning, str) and bool(reasoning.strip())
            if has_r:
                reasoning_present += 1
                comp_with.append(comp)
            elif flags["hard"]:
                comp_absent_hard.append(comp)
            else:
                comp_absent_accepted.append(comp)

        rec, unrec = recovery_turns(hard_flags)
        all_recovery.extend(rec)
        unrecovered_lockin += 1 if unrec else 0

        by_task.setdefault(task, []).append(trial_hard)
        by_task_resp.setdefault(task, []).append(n_steps)
        row = per_task_rows.setdefault(
            task,
            {"trials": 0, "turns": 0, "hard": 0,
             "completion_tokens": 0, "prompt_tokens": 0, "cached_tokens": 0},
        )
        row["trials"] += 1
        row["turns"] += n_steps
        row["hard"] += trial_hard
        trial_completion = 0
        trial_prompt = 0
        trial_cached = 0
        for step in steps:
            step_metrics = step.get("metrics") or {}
            completion = step_metrics.get("completion_tokens")
            prompt = step_metrics.get("prompt_tokens")
            cached = step_metrics.get("cached_tokens")
            if isinstance(completion, int):
                trial_completion += completion
            if isinstance(prompt, int):
                trial_prompt += prompt
                if isinstance(cached, int):
                    trial_cached += cached
                    if prompt > 0:
                        cached_share_per_response.append(cached / prompt)
        completion_per_trial.append(trial_completion)
        prompt_per_trial.append(trial_prompt)
        row["completion_tokens"] += trial_completion
        row["prompt_tokens"] += trial_prompt
        row["cached_tokens"] += trial_cached
        per_trial_rows.append(
            {"trial": trial, "task": task, "turns": n_steps,
             "hard_errors": trial_hard, "completion_tokens": trial_completion,
             "prompt_tokens": trial_prompt, "cached_tokens": trial_cached}
        )

    if args.verify_v1:
        committed = json.loads(args.verify_v1.read_text(encoding="utf-8"))["counts"]
        got = {"agent_response_attempts": responses,
               "hard_error_responses": hard,
               "soft_warning_accepted_responses": soft,
               "hard_error_responses_with_extra_text_warning": hard_with_extra}
        for key, expected in V1_EXPECTED.items():
            if got[key] != expected or committed[key] != expected:
                raise ValueError(
                    f"v1 regression on {key}: recomputed={got[key]} "
                    f"committed={committed.get(key)} expected={expected}")

    # clustered SE for the hard-error rate
    clusters = {task: (sum(hs), sum(by_task_resp[task]))
                for task, hs in by_task.items()}
    hard_cluster = cluster_bootstrap_se(clusters)

    per_trial_hard = [r["hard_errors"] for r in per_trial_rows]

    # rule of three for zero success
    rule_of_three_tasks = 3 / len(by_task) if by_task else None
    rule_of_three_trials = 3 / n_trials if n_trials else None

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "schema_branch": args.schema_branch,
        "input_verification": verification,
        "denominators": {
            "trials": n_trials,
            "tasks": len(by_task),
            "agent_responses": responses,
            "responses_with_reasoning": reasoning_present,
        },
        "trial_level": {
            "success_rate": (successes / n_trials) if n_trials else None,
            "successes": successes,
            "failure_breakdown": {
                "f1_verifier_zero": n_trials - censored_turns - n_errored - successes,
                "f2_turn_exhausted": censored_turns,
                "f3_agent_timeout": n_errored,
                "f4_infrastructure_rerun": n_retries,
                "note": ("F1 = zero-reward trials that neither exhausted the "
                         "30-turn guard nor hit the agent timeout. No persisted "
                         "task_complete field exists, so no premature-complete "
                         "rate is derived from these."),
            },
            "success_rate_zero_ci": {
                "method": "rule_of_three",
                "upper_95_over_tasks": rule_of_three_tasks,
                "upper_95_over_trials_optimistic_do_not_cite": rule_of_three_trials,
                "note": ("Wald SE is 0 at p_hat=0 and must not be used. Report "
                         "the task-level rule-of-three bound (~0.15)."),
            },
            "turns": {
                **describe(turns_list),
                "restricted_mean_at_30": round(sum(turns_list) / len(turns_list), 2),
                "censored_at_max_turns": censored_turns,
                "censored_by_agent_timeout": n_errored,
                "headline_statistic": "median",
            },
        },
        "response_level": {
            "parser_hard_error_rate": hard / responses,
            "parser_soft_warning_rate": soft / responses,
            "parser_hard_error_rate_per_trial_median": statistics.median(
                h / t for h, t in zip(per_trial_hard,
                                      [r["turns"] for r in per_trial_rows])),
            "trials_with_zero_hard_errors": sum(1 for h in per_trial_hard if h == 0),
            "hard_soft_disjoint_note": ("Both use 836 responses as denominator; "
                                        "the 125 hard errors that also carried an "
                                        "extra-text warning are excluded from the "
                                        "soft column, so the columns do not add."),
        },
        "cluster_statistics": {
            "unit": "task",
            "parser_hard_error_rate": hard_cluster,
            "power_note": ("Clustered SE ~5.9pp gives a resolvable difference "
                           "near +/-12pp, the same order as the TB success SE, "
                           "not an order of magnitude better. The parser column's "
                           "real advantage over success is that it is off the "
                           "floor (17.8% has headroom; 0% does not)."),
            "m1_comparison_design": "task_paired_over_20_locked_tasks",
        },
        "recovery": {
            "hard_error_events_recovered": len(all_recovery),
            "recovery_turns": describe(all_recovery),
            "unrecovered_lockin_trials": unrecovered_lockin,
            "definition": ("A recovery event is a maximal run of consecutive "
                           "hard-error turns followed by an accepted turn; its "
                           "length is the run length. Runs that reach the end of "
                           "the trial are counted as lock-ins, not recoveries."),
        },
        "per_task": [
            {"task": task, **per_task_rows[task]}
            for task in sorted(per_task_rows)
        ],
        "per_trial": sorted(per_trial_rows, key=lambda r: r["trial"]),
        "reasoning_absence": {
            "no_reasoning_rate": (responses - reasoning_present) / responses,
            "no_reasoning_count": responses - reasoning_present,
            "denominator": responses,
            "interpretation": "model_behavior_not_logging_gap",
            "evidence": {
                "completion_tokens_with_reasoning": describe(comp_with),
                "completion_tokens_absent_accepted": describe(comp_absent_accepted),
                "completion_tokens_absent_hard": describe(comp_absent_hard),
                "note": ("Absent-accepted responses have ~6x lower completion "
                         "tokens than responses with reasoning, so the absence "
                         "reflects turns the model answered without thinking, "
                         "not unlogged thinking. thinking-token stats are "
                         "therefore conditional on responses_with_reasoning."),
            },
        },
        "cached_tokens_missing": {
            "count": len(cached_missing),
            "steps": cached_missing,
            "policy": "treated_as_zero_denominator_835",
        },
        "token_usage": {
            "completion_tokens_per_trial": describe(completion_per_trial),
            "prompt_tokens_per_trial": describe(prompt_per_trial),
            "cached_token_share_per_response": describe(cached_share_per_response),
            "cached_token_share_denominator": len(cached_share_per_response),
            "cached_token_share_note": (
                "cached_tokens / prompt_tokens for the 835 responses where "
                "cached_tokens was persisted; the one missing value is treated "
                "as zero in per-trial cached-token totals."
            ),
        },
    }

    if args.token_sidecar:
        result["tokens"] = token_split(args.token_sidecar)

    return result


def token_split(sidecar_path: Path) -> dict[str, Any]:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    records = sidecar["responses"]
    thinking = [r["thinking_tokens"] for r in records if r["has_reasoning"]]
    message = [r["message_tokens"] for r in records]
    command = [r["command_content_tokens"] for r in records]

    # Per-trial totals are retained for M1's task-paired comparison without
    # retaining any trajectory text.
    by_trial: dict[str, dict[str, int]] = {}
    residuals: list[int] = []
    for r in records:
        agg = by_trial.setdefault(r["trial"], {"completion": 0, "thinking": 0,
                                               "message": 0, "command": 0})
        comp = r.get("completion_tokens")
        if isinstance(comp, int):
            agg["completion"] += comp
        agg["thinking"] += r["thinking_tokens"]
        agg["message"] += r["message_tokens"]
        agg["command"] += r["command_content_tokens"]
        if isinstance(comp, int):
            residuals.append(comp - (r["thinking_tokens"] + r["message_tokens"]
                                     + r["command_content_tokens"]))

    residual_stats = describe(residuals)
    residual_ok = (residual_stats.get("n", 0) > 0
                   and isinstance(residual_stats.get("min"), (int, float))
                   and residual_stats["min"] >= 0)
    tool_calls_total = sum(r["tool_calls"] for r in records)
    turns_with_tool_calls = sum(1 for r in records if r["tool_calls"] > 0)
    turns_with_keystrokes = sum(
        1 for r in records
        if r.get("has_keystrokes", r["command_content_tokens"] > 0)
    )
    return {
        "tokenizer": sidecar.get("tokenizer"),
        "schema_branch": sidecar.get("schema_branch"),
        "split": {
            "thinking_tokens_conditional_on_reasoning": describe(thinking),
            "message_tokens_harbor_normalized": describe(message),
            "command_content_tokens_harbor_normalized": describe(command),
        },
        "residual_check": {
            "definition": ("completion_tokens - (thinking + message + command); "
                           "positive and non-constant under branch B because the "
                           "original JSON envelope is not persisted. Stability, "
                           "not near-zero, is the acceptance criterion."),
            **residual_stats,
            "stable": residual_ok,
        },
        "tool_call_usage": {
            "tool_calls_total": tool_calls_total,
            "turns_with_tool_calls": turns_with_tool_calls,
            "turns_with_keystrokes": turns_with_keystrokes,
            "tool_calls_per_turn": (
                tool_calls_total / turns_with_tool_calls
                if turns_with_tool_calls else None
            ),
        },
        "per_trial": [
            {"trial": trial, **totals,
             "residual": totals["completion"] - totals["thinking"]
             - totals["message"] - totals["command"]}
            for trial, totals in sorted(by_trial.items())
        ],
        "note": ("message and command_content are harbor-normalized views, not "
                 "the original model JSON body; message_tokens is a lower bound "
                 "on non-thinking prose."),
    }


def write_json(payload: Mapping[str, Any], output: Path | None) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        payload = summarize(args)
        write_json(payload, args.output)
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        print(f"metrics.py: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
