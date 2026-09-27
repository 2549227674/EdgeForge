# EdgeForge M0 §5 · Line C data card

Status: all automated M0 Line-C gates and the documented Codex reviews completed on 2026-08-06. The canonical post-gate pool contains 154,097 records at `data/pipeline/mix_records/`; no family was physically truncated. Two records containing unresolved, unverified final-mix detector candidates were conservatively excluded under Gate 3. The renderer was checked on a deterministic 20-trajectory sample only; full rendering is an M1 operation.

## Archive and provenance

The nine sources were frozen through a mixture of raw repository files and Hugging Face `refs/convert/parquet` exports, then SHA-256 recorded in `manifests/data_archive_sha256.json`. The manifest is authoritative for file-level checksums, revisions, and completeness; [`SOURCES.md`](SOURCES.md) explains why the local `archive` and `archive_parquet` layouts differ and identifies the canonical input for each source. `gate1_upstream_metadata.json` separately retains Dataset Viewer default-revision observations, so viewer and export revisions are not conflated.

| repo_id | frozen revision | upstream/local rows | completeness | license | pipeline disposition |
|---|---|---:|---:|---|---|
| `AletheiaResearch/GLM-5.2-Agent` | `d032fa9…` | 319 / 319 | 100% | unlabeled | train candidate |
| `Crownelius/Complete-FABLE.5-traces-2M` | `e9e7757…` | 228,968 / 228,968 | 100% | MIT | train candidate; aggregate provenance retained |
| `Glint-Research/Fable-5-traces` | `7c96478…` | 4,665 / 4,665 | 100% | AGPL-3.0 | one post-gate record retained |
| `Infatoshi/kernelbench-hard-traces` | `6cae9dc…` | 383 / 383 | 100% | MIT | never training |
| `Infatoshi/kernelbench-mega-traces` | `0f6b0f5…` | 75 / 75 | 100% | MIT | never training |
| `WithinUsAI/claude_mythos_distilled_25k` | `2c5e638…` | 25,000 / 25,000 | 100% | Apache-2.0 | excluded after degeneracy/dedup |
| `armand0e/claude-opus-4.8-pi-traces` | `7014bac…` | 4 / 4 | 100% | unlabeled | no post-dedup records |
| `armand0e/qwen3.7-max-pi-traces` | `bae934b…` | 47 / 47 | 100% | unlabeled | train candidate |
| `lambda/hermes-agent-reasoning-traces` | `b92885e…` | 14,701 / 14,701 | 100% | Apache-2.0 | train candidate |

The Crown repository name's `2M` is not row-count evidence: the frozen export has 228,968 rows. Glint's 4,665 rows are prefix-expanded traces; Gate 4 identifies 218 source sessions, so rows are not used as a session count. Hermes is a harness-labelled dataset; its config provenance contributes Kimi-K2.5 and GLM-5.1 labels, not an inferred single teacher identity.

## Funnel

Gate 1 is file/row accounting. Gate 4 parses each row into structured IR and annotates its source-session identity; Gate 2 then folds records that share that identity and performs cross-source deduplication. These units must not be added indiscriminately across gates. Gate 4b is diagnostic-only, so its output count equals Gate 4; its L2 near-duplicate result is incorporated in Gate 2.

| Dataset | Gate 1 rows | Gate 4 | Gate 4b | Gate 2 (L2) | Gate 3 | Gate 5 | Gate 6 | Gate 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| GLM-5.2 | 319 | 319 | 319 | 234 | 234 | 232 | 232 | sample covered |
| Crownelius aggregate | 228,968 | 201,786 | 201,786 | 151,364 | 151,364 | 150,921 | 150,919 | sample covered |
| Glint Fable | 4,665 | 4,665 | 4,665 | 1 | 1 | 1 | 1 | retained; no eligible Gate 7 trajectory |
| Mythos-25k | 25,000 | 25,000 | 25,000 | 12 | 12 | 12 | 0 | excluded |
| Opus pi traces | 4 | 4 | 4 | 0 | 0 | 0 | 0 | no records after Gate 2 |
| Qwen pi traces | 47 | 47 | 47 | 38 | 38 | 38 | 38 | sample covered |
| Hermes agent traces | 14,701 | 14,701 | 14,701 | 2,910 | 2,910 | 2,907 | 2,907 | sample covered |
| KernelBench hard | 383 | excluded | — | — | — | — | — | never training |
| KernelBench mega | 75 | excluded | — | — | — | — | — | never training |

Gate 4 coalesces only adjacent assistant event fragments, preserving their order, visible content, reasoning, and tool calls. This recovered 319/319 valid GLM records and 201,786 valid Crown records; it does not invent a model turn or tool call. Gate 2 used source-session L0/L1 plus 128-entry bottom-k 5-gram MinHash candidates followed by full Jaccard >=0.8 verification: 137,958 candidate pairs, 1,372 verified pairs, 1,324 removals, and four overlarge bands explicitly skipped. See `data/pipeline/gate2_dedup.md` and `gate2_l2_summary.json`.

The small-family counts have different causes. GLM and Qwen were complete but small upstream exports (319 and 47 records): exact first-user deduplication removed 85 and 9 respectively, and Gate 5 removed two additional GLM records, leaving 232 and 38. Glint's 4,665 rows were prefix-expanded snapshots of 218 source sessions, not 4,665 independent trajectories: same-session folding removed 4,447 rows, cross-source exact first-user deduplication removed 216, and verified L2 near-deduplication removed one of the remaining two. Its final count of one is therefore a deduplication result, not a family-sampling cap.

## Degeneracy and balancing

Mythos had a 100% top-prefix rate and 99.14% exact internal duplicates. After boilerplate stripping and deduplication it retained 12/25,000 records, below the predeclared 2% threshold, and was excluded. `gate4b_degeneracy.md` records all seven trainable-source diagnostic rates and review samples.

Gate 6 uses two descriptive axes: conservative source-family evidence and Hermes `category`/`subcategory` task type. Crown labels containing Claude/Fable/Opus/Sonnet/Mythos are reported only as `claimed_anthropic_from_source_label`; their labels are not treated as verified teacher identities. All 154,097 hard-gate-eligible records are retained: 150,919 Crown, 2,907 Hermes, 232 GLM-5.2, 38 Qwen, and one Glint record. The two remaining Gate 5 records were excluded by the Gate 3 final-mix security rule, not by family sampling. The raw pool is 97.94% Anthropic-style by source label. Rather than delete data, `data/mix.yaml` defines raw-uniform, family 80/20, and family 60/40 training-time sampling recipes; the M1 default must be selected by an equal optimizer-step and token-budget comparison.

## Security and decontamination

TruffleHog scanned the archive scope; the versioned security output contains detector counts and safe UID/rule associations only, never secret values. Redaction is applied after the raw-text decontamination scan and before rendering. The final scan is restricted to trainer-consumed message/tool payloads rather than UID or provenance metadata. Built-in rules produced 10,904 record-rule hits; two unverified payload candidates were conservatively excluded, and the remaining 215 candidates were replaced in 3,850 locations across 361 records. The rebuilt payload rescan returned zero candidates; see `gate3_final_mix_trufflehog.md`. `data/mix.yaml` provides rule-level record counts.

| Frozen test set | test denominator | canary hits | 13-gram-positive training records removed | frozen-subset hits |
|---|---:|---:|---:|---:|
| MMLU | 14,042 | 0 | 11 | 0 |
| GSM8K | 1,319 | 0 | 370 | 48 |
| HumanEval | 164 | 0 | 6 | 0 |
| Terminal-Bench 2.1 | 20 | 0 | 61 | 5 |

The Gate 5 scan started with 154,559 training records, removed 448, and retained 154,111. It proves no detected literal canary or normalized 13-gram overlap remains in this pool; it does not prove absence of semantic or upstream-pretraining contamination. No frozen test, manifest, or baseline configuration was modified. KernelBench exclusion is enforced by `data/pipeline/kernelbench_exclusion.json`.

## Rendering and release boundary

The native Gemma 4 template was verified byte-identical across the HF Jinja file, the B0 Q4_K_M GGUF, and the official QAT Q4_0 GGUF: 18,569 bytes and SHA-256 `0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5`. Gate 7 rendered 20 deterministic source-stratified multi-assistant tool trajectories with `enable_thinking=true` and `preserve_thinking=true`: 20/20 rendered, byte-round-tripped, and retained all source reasoning fields. The Codex review checked all 69 thought, 219 tool-call, 153 tool-response, and 141 turn boundaries; all passed. The token/mask worksheet remains locally ignored, while the versioned review records the sign-off.

The canonical pool includes one AGPL-3.0 Glint record. Per the project release policy, no weights are to be published; only reports and auditable metadata may be released.

## Deferred items

- No second-medium archive backup in this round; the risk is recorded in the execution card.
- SWE-bench Pro decontamination is deferred to M5.
- Full mix rendering is an M1 action; M0 performs only the documented 20-sample Gate 7 validation.
