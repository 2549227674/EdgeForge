#!/usr/bin/env python3
"""Download the canonical parquet exports required by M0 line C gate 1.

The script intentionally records the commit resolved from ``refs/convert/parquet``
before downloading.  That makes a re-run reproducible even though HF datasets may
advance after the pipeline has started.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPOSITORIES = (
    "Crownelius/Complete-FABLE.5-traces-2M",
    "AletheiaResearch/GLM-5.2-Agent",
    "armand0e/claude-opus-4.8-pi-traces",
    "Glint-Research/Fable-5-traces",
    "Infatoshi/kernelbench-hard-traces",
    "Infatoshi/kernelbench-mega-traces",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/archive_parquet"))
    parser.add_argument("--record", type=Path, default=Path("data/pipeline/gate1_downloads.json"))
    parser.add_argument(
        "--repo",
        action="append",
        choices=REPOSITORIES,
        help="Download one named repository; repeat the option for multiple repositories.",
    )
    args = parser.parse_args()

    api = HfApi()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records_by_repo = {}
    if args.record.exists():
        previous = json.loads(args.record.read_text(encoding="utf-8"))
        records_by_repo = {item["repo_id"]: item for item in previous.get("downloads", [])}

    for repo_id in args.repo or REPOSITORIES:
        info = api.dataset_info(repo_id, revision="refs/convert/parquet")
        destination = args.output_root / repo_id.replace("/", "__")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=info.sha,
            local_dir=destination,
        )
        files = sorted(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file() and ".cache/huggingface" not in path.as_posix()
        )
        records_by_repo[repo_id] = (
            {
                "repo_id": repo_id,
                "requested_revision": "refs/convert/parquet",
                "resolved_revision": info.sha,
                "local_dir": destination.as_posix(),
                "files": files,
            }
        )
        _write_record(args.record, records_by_repo)
        print(f"{repo_id}: {info.sha} -> {destination}")


def _write_record(path: Path, records_by_repo: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now(UTC).isoformat(),
                "downloads": [records_by_repo[repo_id] for repo_id in sorted(records_by_repo)],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
