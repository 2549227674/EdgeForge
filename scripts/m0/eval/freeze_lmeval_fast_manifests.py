#!/usr/bin/env python3
"""Create deterministic, auditable MMLU and GSM8K fast-tier manifests.

The manifests deliberately exclude examples used by the earlier five-question
protocol smokes.  They record both row indexes and content hashes, so a later
evaluation can prove it consumed the frozen rows from the pinned local shards.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
MMLU_ROOT = ROOT / "data/eval/mmlu/cais_mmlu"
GSM8K_TEST = ROOT / "data/eval/gsm8k/main/test-00000-of-00001.parquet"
OUT_DIR = ROOT / "manifests"
MMLU_SUBSET_ROOT = ROOT / "data/eval/mmlu/fast_500"
GSM8K_SUBSET = ROOT / "data/eval/gsm8k/fast_200/test-00000-of-00001.parquet"
MMLU_TASK_DIR = ROOT / "tasks/edgeforge_mmlu_fast"
GSM8K_TASK = ROOT / "tasks/edgeforge_gsm8k_fast_200.yaml"
SEED = 20260804
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"

# These are not eligible for a formal sample because their scores were already
# observed while proving the protocol path.  The MMLU smoke targeted only this
# subject; GSM8K used the leading five test rows.
MMLU_EXCLUDED_DOC_IDS = {"college_mathematics": [0, 1, 2, 3, 4]}
GSM8K_EXCLUDED_DOC_IDS = [0, 1, 2, 3, 4]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def freeze_mmlu() -> dict[str, Any]:
    subjects = sorted(
        path.name
        for path in MMLU_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if len(subjects) != 57:
        raise RuntimeError(f"expected 57 MMLU subjects, found {len(subjects)}")

    # 57 * 8 = 456.  The remaining 44 rows are distributed as one additional
    # sampled row across 44 subjects, preserving subject coverage at 8 or 9.
    extra_subjects = set(random.Random(SEED).sample(subjects, 44))
    source_files: dict[str, Any] = {}
    selections: list[dict[str, Any]] = []

    for subject in subjects:
        dev_path = MMLU_ROOT / subject / "dev-00000-of-00001.parquet"
        test_path = MMLU_ROOT / subject / "test-00000-of-00001.parquet"
        dev_rows = read_rows(dev_path)
        test_rows = read_rows(test_path)
        excluded = set(MMLU_EXCLUDED_DOC_IDS.get(subject, []))
        population = [index for index in range(len(test_rows)) if index not in excluded]
        count = 9 if subject in extra_subjects else 8
        if len(population) < count:
            raise RuntimeError(f"{subject} has only {len(population)} eligible rows")

        selected_ids = sorted(
            random.Random(f"{SEED}:mmlu:{subject}").sample(population, count)
        )
        source_files[subject] = {
            "dev": {
                "path": relative(dev_path),
                "rows": len(dev_rows),
                "sha256": sha256_file(dev_path),
            },
            "test": {
                "path": relative(test_path),
                "rows": len(test_rows),
                "sha256": sha256_file(test_path),
            },
        }
        for doc_id in selected_ids:
            selections.append(
                {
                    "subject": subject,
                    "doc_id": doc_id,
                    "row_sha256": canonical_hash(test_rows[doc_id]),
                }
            )

    if len(selections) != 500:
        raise RuntimeError(f"expected 500 selected MMLU rows, got {len(selections)}")

    return {
        "schema_version": "1.0",
        "dataset": "cais/mmlu",
        "revision": MMLU_REVISION,
        "split": "test",
        "num_fewshot": 5,
        "selection": {
            "method": "seeded_stratified_by_subject",
            "seed": SEED,
            "samples": 500,
            "subjects": len(subjects),
            "per_subject": "8 plus one extra row for 44 seed-selected subjects",
            "excluded_preexposed_smoke_doc_ids": MMLU_EXCLUDED_DOC_IDS,
        },
        "source_files": source_files,
        "selected": selections,
    }


def freeze_gsm8k() -> dict[str, Any]:
    rows = read_rows(GSM8K_TEST)
    excluded = set(GSM8K_EXCLUDED_DOC_IDS)
    population = [index for index in range(len(rows)) if index not in excluded]
    selected_ids = sorted(random.Random(f"{SEED}:gsm8k").sample(population, 200))
    selected = [
        {"doc_id": doc_id, "row_sha256": canonical_hash(rows[doc_id])}
        for doc_id in selected_ids
    ]

    return {
        "schema_version": "1.0",
        "dataset": "openai/gsm8k",
        "config": "main",
        "revision": GSM8K_REVISION,
        "split": "test",
        "num_fewshot": 5,
        "selection": {
            "method": "seeded_simple_random_sample_without_replacement",
            "seed": SEED,
            "samples": 200,
            "excluded_preexposed_smoke_doc_ids": GSM8K_EXCLUDED_DOC_IDS,
        },
        "source_file": {
            "path": relative(GSM8K_TEST),
            "rows": len(rows),
            "sha256": sha256_file(GSM8K_TEST),
        },
        "selected": selected,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_subset_parquet(source: Path, doc_ids: list[int], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(source).take(pa.array(doc_ids, type=pa.int64()))
    pq.write_table(table, destination, compression="snappy")


def write_mmlu_tasks(mmlu_manifest: dict[str, Any]) -> list[Path]:
    """Materialize each selected test slice as a normal lm-eval task."""
    MMLU_TASK_DIR.mkdir(parents=True, exist_ok=True)
    base_template = (
        ROOT
        / ".venv-lmeval/lib/python3.12/site-packages/lm_eval/tasks/mmlu/default"
        / "_default_template_yaml"
    )
    by_subject: dict[str, list[int]] = {}
    for row in mmlu_manifest["selected"]:
        by_subject.setdefault(row["subject"], []).append(row["doc_id"])

    task_names: list[str] = []
    inputs: list[Path] = []
    for subject in sorted(by_subject):
        task_name = f"edgeforge_mmlu_fast_{subject}"
        task_names.append(task_name)
        source = MMLU_ROOT / subject / "test-00000-of-00001.parquet"
        destination = MMLU_SUBSET_ROOT / subject / "test-00000-of-00001.parquet"
        write_subset_parquet(source, sorted(by_subject[subject]), destination)
        inputs.append(destination)

        task_config = {
            "include": str(base_template),
            "dataset_path": "parquet",
            "dataset_name": None,
            "dataset_kwargs": {
                "data_files": {
                    "dev": relative(MMLU_ROOT / subject / "dev-00000-of-00001.parquet"),
                    "test": relative(destination),
                }
            },
            "task": task_name,
            "task_alias": subject.replace("_", " "),
            "metadata": {"version": "1.0-edgeforge-fast-500"},
        }
        task_path = MMLU_TASK_DIR / f"{subject}.yaml"
        task_path.write_text(
            yaml.safe_dump(task_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        inputs.append(task_path)

    group_path = MMLU_TASK_DIR / "edgeforge_mmlu_fast_500.yaml"
    group_path.write_text(
        yaml.safe_dump(
            {
                "group": "edgeforge_mmlu_fast_500",
                "task": task_names,
                "aggregate_metric_list": [
                    {"metric": "acc", "weight_by_size": True}
                ],
                "metadata": {"version": "1.0-edgeforge-fast-500"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    inputs.append(group_path)
    return inputs


def write_gsm8k_task(gsm8k_manifest: dict[str, Any]) -> list[Path]:
    source = GSM8K_TEST
    selected_ids = [row["doc_id"] for row in gsm8k_manifest["selected"]]
    write_subset_parquet(source, selected_ids, GSM8K_SUBSET)
    task_config = {
        "include": str(ROOT / "tasks/edgeforge_gsm8k.yaml"),
        "task": "edgeforge_gsm8k_fast_200",
        "dataset_path": "parquet",
        "dataset_kwargs": {
            "data_files": {
                "train": relative(ROOT / "data/eval/gsm8k/main/train-00000-of-00001.parquet"),
                "test": relative(GSM8K_SUBSET),
            }
        },
        "metadata": {"version": "3.0-edgeforge-fast-200"},
    }
    GSM8K_TASK.write_text(
        yaml.safe_dump(task_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return [GSM8K_SUBSET, GSM8K_TASK]


def write_checksum(path: Path, paths: list[Path]) -> None:
    path.write_text(
        "\n".join(
            f"{sha256_file(item)}  {relative(item)}" for item in sorted(paths)
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not MMLU_ROOT.is_dir() or not GSM8K_TEST.is_file():
        raise SystemExit("pinned local MMLU/GSM8K data is missing")
    OUT_DIR.mkdir(exist_ok=True)

    mmlu_path = OUT_DIR / "mmlu_fast_500.json"
    gsm8k_path = OUT_DIR / "gsm8k_fast_200.json"
    mmlu_manifest = freeze_mmlu()
    gsm8k_manifest = freeze_gsm8k()
    write_json(mmlu_path, mmlu_manifest)
    write_json(gsm8k_path, gsm8k_manifest)

    execution_inputs = write_mmlu_tasks(mmlu_manifest)
    execution_inputs.extend(write_gsm8k_task(gsm8k_manifest))
    write_checksum(OUT_DIR / "lm_eval_fast_inputs.sha256", execution_inputs)

    checksum_path = OUT_DIR / "lm_eval_fast_manifests.sha256"
    checksum_path.write_text(
        "\n".join(
            [
                f"{sha256_file(mmlu_path)}  manifests/{mmlu_path.name}",
                f"{sha256_file(gsm8k_path)}  manifests/{gsm8k_path.name}",
                f"{sha256_file(Path(__file__))}  scripts/{Path(__file__).name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
