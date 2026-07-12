from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tesis_unicamp.generation.rag.schemas import GeneratedAnswerRecord


def save_generated_answers(
    output_dir: Path,
    split: str,
    records: list[GeneratedAnswerRecord],
) -> Path:
    path = output_dir / split / "generated_answers.json"
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


def save_generated_answers_bundle(
    output_dir: Path,
    splits: dict[str, list[GeneratedAnswerRecord]],
    *,
    run_settings: dict[str, Any] | None = None,
) -> Path:
    for split_name, records in splits.items():
        save_generated_answers(output_dir, split_name, records)
    if run_settings is not None:
        save_run_settings(output_dir, run_settings)
    return output_dir


def load_generated_answers(output_dir: Path, split: str) -> list[GeneratedAnswerRecord]:
    path = output_dir / split / "generated_answers.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing generated answers for split {split!r}: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
