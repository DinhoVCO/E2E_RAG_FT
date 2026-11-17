from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragas.dataset_schema import EvaluationResult


def save_ragas_results(
    output_dir: Path,
    split: str,
    *,
    per_sample_scores: list[dict[str, Any]],
    summary: dict[str, Any],
    run_settings: dict[str, Any] | None = None,
    scores_filename: str = "ragas_scores.json",
    summary_filename: str = "ragas_summary.json",
) -> Path:
    """Persist per-sample scores, summary, and run settings."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    scores_path = split_dir / scores_filename
    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(per_sample_scores, handle, ensure_ascii=False, indent=2)

    summary_path = split_dir / summary_filename
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if run_settings is not None:
        settings_path = output_dir / "run_settings.json"
        with settings_path.open("w", encoding="utf-8") as handle:
            json.dump(run_settings, handle, ensure_ascii=False, indent=2)

    return split_dir


def evaluation_result_to_rows(
    result: EvaluationResult,
    *,
    include_inputs: bool = True,
) -> list[dict[str, Any]]:
    """Merge RAGAS scores with the evaluated samples."""
    rows: list[dict[str, Any]] = []
    for index, score_row in enumerate(result.scores):
        row = dict(score_row)
        if include_inputs and result.dataset is not None:
            sample = result.dataset[index]
            sample_row = sample.model_dump()
            for key, value in sample_row.items():
                if key not in row:
                    row[key] = value
        rows.append(row)
    return rows


def build_summary(result: EvaluationResult) -> dict[str, float]:
    return {metric_name: float(value) for metric_name, value in result._repr_dict.items()}
