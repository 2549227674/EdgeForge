# M0 evaluation baseline

Frozen 2026-08-05. The machine-readable contract is `eval_config.yaml`; the
detailed audit trail is
`docs/m0/03_baseline.md`.

## Link probe

- 2026-08-02 — TB 2.1 probe `terminal-bench/kv-store-grpc`: endpoint
  request/reply, container command execution, and verifier result were all
  observed (`probe_link_closed=true`). Reward `0.0` is a link probe only and
  does not enter task selection or baseline scoring. Evidence:
  `results/m0_tb_probe/2026-08-02__16-31-43/`.

## B0 longitudinal baseline

Model: `gemma-4-e4b-it` PTQ `Q4_K_M`, alias `gemma4-e4b`, 131072 context,
single slot, q8_0 K/V cache, `n_predict=32768`.

| Metric | Frozen value |
|---|---:|
| TB 2.1 locked 20-task subset, 5 attempts each | 0/100 |
| Terminus hard parser errors | 149/836 (17.823%) |
| Terminus soft format warnings | 498/836 (59.569%) |
| BFCL v4 `simple_python` | 363/400 (90.75%) |
| MMLU frozen fast 500, 5-shot | 0.598 ± 0.0200 |
| GSM8K frozen fast 200, strict / flexible | 0.840 / 0.845 |
| HumanEval | 5 scored, 0 passed (pass@1=0); smoke only, no formal full baseline claimed |
| Cache-warm TTFT p50 / p95 | 349.459 / 1762.491 ms |
| Cache-warm throughput p50 | 52.029 tok/s |
| Cache-warm TPOT p50 | 19.220 ms/token |
| Parser hard-error cluster SE / design effect | 5.929pp / 20.064× |
| Parser hard-error per-trial median / zero-hard trials | 0 / 64 of 100 |
| Turns per trial, median / restricted mean / censored at 30 | 6.0 / 8.36 / 7 |
| Parser recovery events, median turns / lock-in trials | 42, 1.0 / 3 |
| No-reasoning responses | 125/836 (14.952%) |
| Completion / prompt tokens per trial, median | 5801 / 19831.5 |
| Cached-token share per response, median (n=835) | 81.613% |
| Thinking / message / command-content tokens, response median | 273 (n=711) / 159 / 4 |
| Harbor tool calls | 1340 across 559 turns (2.397 per tool-call turn) |

TB failure classes F1/F2/F3/F4 are 92/7/1/0; `finish_reason=length` is 0.
Per-trial peak-context p50/p95 is 6292.5/22340.55 tokens. The final 5-task
sanity was 0/5; the list was frozen after one swap round as predeclared, so
the score is retained rather than selecting tasks until it rises.

Agent-metric notes: (1) 0/100 success is reported with the task-clustered
rule-of-three upper 95% bound of 15%, rather than a Wald SE of zero. (2) Parser
hard/soft rates remain disjoint 149/836 and 498/836; the hard rate's clustered
SE is the reportable uncertainty, while the 1.324pp naive binomial SE is marked
do-not-cite. (3) `reasoning_content` is absent on 125 responses, including 111
parser-accepted short completions (median 110 tokens versus 677 with reasoning),
so it is a model-behaviour signal, not a logging gap. (4) F1 means zero reward
without turn exhaustion or agent timeout; it is not a `task_complete`-derived
"premature completion" rate. Token segment counts are Harbor-normalized views;
the positive branch-B residual is stable (min 1, median 54 tokens).

## Official QAT-Q4_0 deployment anchor

This is a separately labelled anchor, not the longitudinal B0 before row.
Endpoint/tools smoke passed; BFCL `simple_python` is 364/400 (91.00%);
GSM8K strict/flexible is 0.850/0.865. Its cache-warm replay has TTFT p50/p95
377.423/848.977 ms, throughput p50 52.188 tok/s, and TPOT p50
19.161 ms/token. The 20-task × 5 TB slow gate remains deferred to the M6
three-point comparison.

## Artifact boundary

Versioned artifacts include configs, runners, task/manifests, five B0-derived
tapes, the 100 raw `trajectory.json` files and their SHA-256 manifest, relevant
static logs, and selected small result snapshots. GGUF inputs remain
local/ignored and are pinned by hashes. lm-eval samples,
trial cast/pane files, old smoke directories, and data-pipeline intermediates
were removed during the 2026-08-11 asset consolidation; aggregate results,
frozen dataset exports, and the canonical `mix_records/` pool remain.
