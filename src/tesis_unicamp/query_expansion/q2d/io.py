from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tesis_unicamp.query_expansion.q2d.schemas import Q2dRecord


def save_q2d_records(
    output_dir: Path,
    split: str,
    records: list[Q2dRecord],
) -> Path:
    path = output_dir / split / "q2d_expansions.json"
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


def save_q2d_bundle(
    output_dir: Path,
    splits: dict[str, list[Q2dRecord]],
    *,
    run_settings: dict[str, Any] | None = None,
) -> Path:
    for split_name, records in splits.items():
        save_q2d_records(output_dir, split_name, records)
    if run_settings is not None:
        save_run_settings(output_dir, run_settings)
    return output_dir


def load_q2d_records(output_dir: Path, split: str) -> list[Q2dRecord]:
    path = output_dir / split / "q2d_expansions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Query2Doc expansions for split {split!r}: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
