# EdgeForge M1 agent-16 calibration audit

## Verdict

PASS. The calibration asset contains exactly 16 unique source UIDs, excludes all 1,500 frozen holdout UIDs, respects the 16,384-token per-record cap after decode/re-encode, and falls inside the frozen total-token gate. No W8A8 build or evaluation was run.

## Frozen identities

- Selection rule: `manifests/m1_calib_agent16_selection_rule.json` / `689d6eeb0c5e23c0ab239f427df88e1aaf488cfde6739e39c810c32d982f982b`
- Source pool manifest: `manifests/mix_records_sha256.json` / `cb3168198af82795315f5ae5fec355e74ca4affd171ba5e3b595ea3f2152bcf9`; composite pool identity `ab88cad43f97b65d8f5743d359364bef9da7c75716bf8b39a0e405f13cd965bd`
- Candidate index: `results/t1/segmented_4096_r6/segment_audit.jsonl.gz` / `01de75a2c26a80d55676fe76997a7435ca876601f5dcbdfa7a3787f391696d15`
- Holdout: `manifests/m1_holdout.json` / `c6ae96e93474b639ee4217a0d14506e383d1f406320b01a0b0aa1ec75c1b9374`
- Chat template: `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`
- Tokenizer identity: `72c512577c9d59deb174c82bdab1185760e6c2468a01a61f6908351197caaa45`
- Dataset: `data/m1/calibration/agent16/data_quant.json` / `c2b1c7a062a60ba6a6aaa2d198045d74407472b03712d1dff1dced1493c2863a`
- External verify-only receipt: `results/m1/agent16_calibration_verification.json` / `f55919c46dc2c3e82854803b8470ac890d60f1c30ea1b023a62505a502ce8d7a`

## Selection and undercoverage

After holdout exclusion, the candidate index contains 152,597 eligible records in 49 nonempty family×task_type strata. Since 49 > 16, this asset does not claim complete stratum coverage. The frozen rule first apportions 13 anthropic-label / 3 other records by Hamilton 80/20, then applies its declared over-capacity/under-capacity stratum rule and deterministic seed hash ordering.

| family_group | family | task_type | eligible | selected |
|---|---|---|---:|---:|
| `anthropic` | `claimed_anthropic_from_source_label` | `chat_or_reasoning` | 144867 | 12 |
| `anthropic` | `claimed_anthropic_from_source_label` | `tool_agent` | 4632 | 1 |
| `other` | `glm-5.2` | `tool_agent` | 229 | 1 |
| `other` | `kimi-k2.5` | `hermes:Repository Tasks/Bug Fix` | 234 | 1 |
| `other` | `kimi-k2.5` | `hermes:Terminal & Coding/Terminal Tasks` | 229 | 1 |

The `anthropic` bucket is only a source-label sampling group. It is not evidence of a verified teacher identity.

## Acceptance gates

- Records / unique UIDs: 16 / 16.
- Holdout intersection: 0; exclusion proof SHA-256 `2a3a8120e8deae92ec071e91113c73fbfa239a1fde9ef32d530e343f7c2b8cb6`.
- Total effective tokens: 116,577; allowed range 90,357–135,535.
- Per-record cap: all <=16,384 = `true`; truncated records = 0.
- Render contract: `add_generation_prompt=false`, `enable_thinking=true`, `preserve_thinking=true`.
- Dataset schema: `list[{input: rendered_calibration_text, target: ""}]`.
- Determinism: a second in-process re-render matched every rendered/calibration SHA and the complete dataset bytes; a separate external verify-only rerun is frozen in `results/m1/agent16_calibration_verification.json`.
- Privacy: source messages and rendered bodies appear only in the private `data_quant.json`; logs, receipt, manifest, and this audit contain identity/count/hash metadata only.

## Runtime isolation

- Host: `autodl-container-atdlpbjbca-22473649`; host logical CPUs: 208; cgroup CPU quota: 44.0 cores; cgroup memory limit: 236223201280 bytes.
- CUDA visibility: ``; workers: 1; `OMP_NUM_THREADS=1`; niceness: 19.
- Generation wall clock: 12.947 s; process max RSS: 1,289,268 KiB.
- C01-S training directories, model output, merge watcher, GPUs, W8A8 build, and B0-before evaluation were not accessed or invoked by this script.

## Absolute cloud paths

- Dataset: `/root/autodl-tmp/edgeforge-m1/workspace/data/m1/calibration/agent16/data_quant.json`
- Identity manifest: `/root/autodl-tmp/edgeforge-m1/workspace/manifests/m1_calib_agent16_sha256.json`
- Receipt: `/root/autodl-tmp/edgeforge-m1/workspace/results/m1/agent16_calibration_receipt.json`
- External verification receipt: `/root/autodl-tmp/edgeforge-m1/workspace/results/m1/agent16_calibration_verification.json`
- Audit: `/root/autodl-tmp/edgeforge-m1/workspace/results/m1/agent16_calibration_audit.md`
- Generator: `/root/autodl-tmp/edgeforge-m1/workspace/scripts/m1/prepare_agent16_calibration.py`
- Selection rule: `/root/autodl-tmp/edgeforge-m1/workspace/manifests/m1_calib_agent16_selection_rule.json`
