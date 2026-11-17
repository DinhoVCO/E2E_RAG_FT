from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tesis_unicamp.query_expansion.schemas import ExpandedQueryRecord


def save_expanded_queries(
    output_dir: Path,
    split: str,
    records: list[ExpandedQueryRecord],
) -> Path:
    path = output_dir / split / "expanded_queries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    return path


def save_run_settings(output_dir: Path, settings: dict[str, Any]) -> Path:
    path = output_dir / "run_settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)
    return path


def save_expanded_queries_bundle(
    output_dir: Path,
    splits: dict[str, list[ExpandedQueryRecord]],
    *,
    run_settings: dict[str, Any] | None = None,
) -> Path:
    for split_name, records in splits.items():
        save_expanded_queries(output_dir, split_name, records)
    if run_settings is not None:
        save_run_settings(output_dir, run_settings)
    return output_dir


def load_expanded_queries(output_dir: Path, split: str) -> list[ExpandedQueryRecord]:
    path = output_dir / split / "expanded_queries.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing expanded queries for split {split!r}: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
