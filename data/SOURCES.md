# Data source layout and completeness

This repository keeps audit metadata and pipeline reports in Git, but not the
multi-gigabyte training payload. The authoritative input identity is
[`manifests/data_archive_sha256.json`](../manifests/data_archive_sha256.json),
not either local download directory by itself.

## Why there are two local trees

- `data/archive/` is the original or legacy staging tree. Depending on the
  dataset and the download method, it can contain repository metadata, selected
  files, or only part of the default branch. It is useful for provenance and
  inspection, but it is not a uniform completeness boundary.
- `data/archive_parquet/` is the Hugging Face `refs/convert/parquet` export used
  by the M0 Line-C builder for datasets that expose that conversion. Its layout
  is normalized around parquet shards and therefore intentionally differs from
  the source repository layout.
- Some sources are canonically consumed from raw JSON/JSONL because no local
  converted-parquet tree is required. The builder and frozen manifest select the
  correct input per source.

The different layouts are therefore mostly a consequence of two acquisition
interfaces, not evidence that the completed M0 pool silently lost records.

## Frozen source map

| Source | Legacy/default-branch staging | Converted parquet | Canonical M0 input |
|---|---|---|---|
| AletheiaResearch/GLM-5.2-Agent | partial selection (16 of 321 relevant remote files observed) | complete; 319 rows | converted parquet |
| Crownelius/Complete-FABLE.5-traces-2M | partial legacy snapshot; an older single shard was superseded | complete four-shard export; 228,968 rows | converted parquet |
| Glint-Research/Fable-5-traces | partial selection (88 of 4,798 relevant remote files observed) | complete; 4,665 rows | converted parquet |
| Infatoshi/kernelbench-hard-traces | partial selection (25 of 613 relevant remote files observed) | complete; 383 rows | converted parquet; audit-only, excluded from training |
| Infatoshi/kernelbench-mega-traces | partial selection (47 of 78 relevant remote files observed) | complete; 75 rows | converted parquet; audit-only, excluded from training |
| WithinUsAI/claude_mythos_distilled_25k | complete three-file raw snapshot; 25,000 rows | not required locally | raw JSONL |
| armand0e/claude-opus-4.8-pi-traces | partial default-branch metadata/files (5 of 6 observed) | complete; 4 rows | converted parquet |
| armand0e/qwen3.7-max-pi-traces | complete 49-file raw snapshot; 47 records | not required locally | raw files |
| lambda/hermes-agent-reasoning-traces | data shards complete; local staging omitted one non-data README | not required locally; 14,701 rows | raw config parquet shards |

The M0 verification on 2026-08-11 established that:

- all nine current upstream default revisions matched the revisions frozen in
  the manifest;
- all six local converted-parquet trees matched their pinned upstream exports by
  file count, size, and SHA-256;
- all 129 files selected by the frozen manifest passed SHA-256 verification;
- the canonical post-gate pool contains 154,097 records: 150,919 Crown, 2,907
  Hermes, 232 Aletheia, 38 Qwen, and 1 Glint.

See [`docs/m0/05_data_pipeline.md`](../docs/m0/05_data_pipeline.md) and
[`data/data_card.md`](data_card.md) for gate-level accounting. The executable
source routing is in
[`scripts/m0/data/build_linec_ir.py`](../scripts/m0/data/build_linec_ir.py).

## Repository and backup policy

The raw archives, converted parquet, canonical `mix_records`, and generated IR
remain local and ignored by Git. They exceed ordinary GitHub file limits and
contain redistributable-source questions that should be reviewed independently
of code publication. A future second-medium backup should use a private,
versioned dataset store and reproduce the frozen manifest; it should not merge
the two directory layouts or rewrite the M0 identity in place.

No additional download or physical unification is required for M1: consume the
frozen canonical pool, or rebuild it from the manifest and the documented
builder. If an upstream refresh is desired, create a new manifest/version rather
than mutating this frozen M0 snapshot.
