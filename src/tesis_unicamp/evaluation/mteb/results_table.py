"""Build comparison tables from MTEB retrieval result JSON files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

METRICS: tuple[str, ...] = ("ndcg_at_10", "map_at_10", "recall_at_10")
METRIC_LABELS: tuple[str, ...] = ("ndcg@10", "map@10", "recall@10")

DEFAULT_DATASET_ORDER: tuple[str, ...] = (
    "BioASQ-RAG-Resplit",
    "NarrativeQA-RAG",
    "QASPER-RAG",
    "TelcoDPR-RAG",
)

DATASET_SLUG_TO_TASK: dict[str, str] = {
    "bioasq-resplit": "BioASQ-RAG-Resplit",
    "narrativeqa": "NarrativeQA-RAG",
    "qasper": "QASPER-RAG",
    "telco-dpr": "TelcoDPR-RAG",
}

TASK_TO_SHORT_NAME: dict[str, str] = {
    "BioASQ-RAG-Resplit": "BioASQ",
    "BioASQ-RAG-Resplit-Title": "BioASQ",
    "NarrativeQA-RAG": "NarrativeQA",
    "NarrativeQA-RAG-Title": "NarrativeQA",
    "QASPER-RAG": "QASPER",
    "QASPER-RAG-Title": "QASPER",
    "TelcoDPR-RAG": "TelcoDPR",
    "TelcoDPR-RAG-Title": "TelcoDPR",
}

EXPERIMENT_TYPE_ORDER: tuple[str, ...] = (
    "Qwen3-Embedding-4B",
    "Qwen3-Embedding-4B FT",
    "Qwen3-Embedding-4B Title",
    "Qwen3-Embedding-4B FT Title",
)

BASELINE_EXPERIMENT = "Qwen3-Embedding-4B"
EXPERIMENT_INDEX_NAME = "Experiment"
TITLE_TASK_SUFFIX = "-Title"

_OFFLINE_PATTERN = re.compile(r"^vllm-offline-(.+)-b128(?:-title)?$")
_LORA_PATTERN = re.compile(r"^vllm-lora-(.+)-b128-e\d+(?:-title)?$")


def canonical_task_name(task_name: str) -> str:
    """Map *-Title tasks to their base dataset task name for table columns."""
    if task_name.endswith(TITLE_TASK_SUFFIX):
        return task_name[: -len(TITLE_TASK_SUFFIX)]
    return task_name


def parse_experiment_folder(experiment_name: str) -> tuple[str, str, bool] | None:
    """Return ``(experiment_type, dataset_task, is_title_task)`` for known folders."""
    is_title = experiment_name.endswith("-title")

    offline_match = _OFFLINE_PATTERN.match(experiment_name)
    if offline_match:
        dataset_slug = offline_match.group(1)
        dataset_task = DATASET_SLUG_TO_TASK.get(dataset_slug)
        if dataset_task is None:
            return None
        experiment_type = (
            "Qwen3-Embedding-4B Title" if is_title else "Qwen3-Embedding-4B"
        )
        return experiment_type, dataset_task, is_title

    lora_match = _LORA_PATTERN.match(experiment_name)
    if lora_match:
        dataset_slug = lora_match.group(1)
        dataset_task = DATASET_SLUG_TO_TASK.get(dataset_slug)
        if dataset_task is None:
            return None
        experiment_type = (
            "Qwen3-Embedding-4B FT Title" if is_title else "Qwen3-Embedding-4B FT"
        )
        return experiment_type, dataset_task, is_title

    return None


@dataclass(frozen=True)
class MtebResultRecord:
    """One parsed MTEB result file."""

    model_name: str
    experiment_name: str
    task_name: str
    split: str
    metrics: dict[str, float]
    source_path: Path


def _extract_split_scores(
    payload: dict[str, Any],
    *,
    split: str | None,
) -> tuple[str, dict[str, float]]:
    scores_by_split = payload.get("scores") or {}
    if not scores_by_split:
        raise ValueError("Missing 'scores' in result JSON.")

    if split is not None:
        if split not in scores_by_split:
            available = ", ".join(sorted(scores_by_split))
            raise ValueError(
                f"Split {split!r} not found. Available splits: {available}"
            )
        split_name = split
    else:
        if "test" in scores_by_split:
            split_name = "test"
        else:
            split_name = next(iter(scores_by_split))

    split_scores = scores_by_split[split_name]
    if not split_scores:
        raise ValueError(f"No score entries for split {split_name!r}.")

    raw_metrics = split_scores[0]
    metrics: dict[str, float] = {}
    for metric in METRICS:
        if metric not in raw_metrics:
            raise ValueError(f"Metric {metric!r} not found in {split_name!r} scores.")
        metrics[metric] = float(raw_metrics[metric])
    return split_name, metrics


def load_result_file(
    path: Path,
    *,
    split: str | None = "test",
) -> MtebResultRecord:
    """Load one MTEB result JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_name = str(payload.get("task_name") or path.stem)
    split_name, metrics = _extract_split_scores(payload, split=split)

    parts = path.parts
    model_name: str | None = None
    experiment_name: str | None = None
    for idx, part in enumerate(parts):
        if (
            part == "results"
            and idx + 4 < len(parts)
            and parts[idx + 1] == "mteb"
            and parts[idx + 2] == "results"
        ):
            model_name = parts[idx + 3]
            experiment_name = parts[idx + 4]
            break

    if model_name is None or experiment_name is None:
        raise ValueError(
            f"Could not infer model/experiment from path: {path}"
        )

    return MtebResultRecord(
        model_name=model_name,
        experiment_name=experiment_name,
        task_name=task_name,
        split=split_name,
        metrics=metrics,
        source_path=path,
    )


def discover_result_files(results_root: Path) -> list[Path]:
    """Find task result JSON files under ``results/mteb/results``."""
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    files: list[Path] = []
    for path in sorted(results_root.rglob("*.json")):
        if path.name in {"model_meta.json"}:
            continue
        files.append(path)
    return files


def load_results(
    results_root: Path,
    *,
    split: str | None = "test",
) -> list[MtebResultRecord]:
    """Load all MTEB result files from a results directory."""
    records: list[MtebResultRecord] = []
    for path in discover_result_files(results_root):
        records.append(load_result_file(path, split=split))
    return records


def _resolve_datasets(
    records: list[MtebResultRecord],
    datasets: list[str] | None,
) -> list[str]:
    if datasets:
        return datasets

    seen: set[str] = set()
    ordered: list[str] = []
    for task_name in DEFAULT_DATASET_ORDER:
        if any(
            canonical_task_name(record.task_name) == task_name for record in records
        ):
            seen.add(task_name)
            ordered.append(task_name)

    for record in records:
        canonical = canonical_task_name(record.task_name)
        if canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered


def build_grouped_experiment_table(
    records: list[MtebResultRecord],
    *,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Build a 4-row table grouped by experiment type and in-domain dataset."""
    if not records:
        raise ValueError("No result records to build table from.")

    dataset_order = _resolve_datasets(records, datasets)
    rows: dict[str, dict[tuple[str, str], float | None]] = {
        experiment_type: {} for experiment_type in EXPERIMENT_TYPE_ORDER
    }

    for record in records:
        parsed = parse_experiment_folder(record.experiment_name)
        if parsed is None:
            continue

        experiment_type, dataset_task, is_title = parsed
        if dataset_task not in dataset_order:
            continue

        expected_title = f"{dataset_task}{TITLE_TASK_SUFFIX}" if is_title else dataset_task
        if record.task_name != expected_title:
            continue

        row = rows[experiment_type]
        for metric, label in zip(METRICS, METRIC_LABELS, strict=True):
            column = (dataset_task, label)
            row[column] = record.metrics[metric]

    multi_columns = pd.MultiIndex.from_product(
        [dataset_order, list(METRIC_LABELS)],
        names=["dataset", "metric"],
    )
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=multi_columns)
    frame = frame.reindex(EXPERIMENT_TYPE_ORDER)
    frame.index.name = EXPERIMENT_INDEX_NAME
    return frame


def flatten_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns for CSV export."""
    flat = table.copy()
    flat.columns = [
        f"{TASK_TO_SHORT_NAME.get(dataset, dataset)} | {metric}"
        for dataset, metric in flat.columns.to_list()
    ]
    return flat


def format_table_for_display(
    table: pd.DataFrame,
    *,
    precision: int = 4,
) -> pd.DataFrame:
    """Return a copy formatted for terminal or HTML display."""
    display = table.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{precision}f}"
        )
    return display


BASELINE_COLOR = "#f3f4f6"
NEUTRAL_COLOR = (255, 255, 255)
BETTER_COLOR = (33, 102, 172)  # #2166ac
WORSE_COLOR = (239, 138, 98)  # #ef8a62


def _lerp_rgb(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    intensity: float,
) -> str:
    t = max(0.0, min(1.0, intensity))
    channels = [
        int(start[idx] + (end[idx] - start[idx]) * t) for idx in range(3)
    ]
    return f"rgb({channels[0]}, {channels[1]}, {channels[2]})"


def _heatmap_color(
    value: float | None,
    baseline: float | None,
    *,
    is_baseline: bool,
    max_abs_delta: float,
) -> str:
    if is_baseline:
        return BASELINE_COLOR
    if value is None or baseline is None or pd.isna(value) or pd.isna(baseline):
        return "#ffffff"

    delta = float(value) - float(baseline)
    if abs(delta) < 1e-9:
        return "#ffffff"

    scale = max(max_abs_delta, abs(delta), 1e-9)
    intensity = min(abs(delta) / scale, 1.0)

    if delta > 0:
        return _lerp_rgb(NEUTRAL_COLOR, BETTER_COLOR, intensity)

    return _lerp_rgb(NEUTRAL_COLOR, WORSE_COLOR, intensity)


def table_to_heatmap_html(
    table: pd.DataFrame,
    *,
    title: str = "MTEB Retrieval Results",
    baseline: str = BASELINE_EXPERIMENT,
    precision: int = 4,
) -> str:
    """Render an HTML heatmap table relative to a baseline experiment row."""
    if baseline not in table.index:
        raise ValueError(f"Baseline experiment {baseline!r} not found in table.")

    numeric = table.apply(pd.to_numeric, errors="coerce")
    baseline_row = numeric.loc[baseline]
    display = format_table_for_display(table, precision=precision)

    dataset_labels = [
        TASK_TO_SHORT_NAME.get(dataset, dataset)
        for dataset in table.columns.get_level_values("dataset").unique()
    ]
    metric_labels = list(METRIC_LABELS)

    header_rows = [
        "<tr>"
        f'<th rowspan="2">{EXPERIMENT_INDEX_NAME}</th>'
        + "".join(
            f'<th colspan="{len(metric_labels)}">{label}</th>'
            for label in dataset_labels
        )
        + "</tr>",
        "<tr>"
        + "".join(
            f"<th>{metric}</th>" for _ in dataset_labels for metric in metric_labels
        )
        + "</tr>",
    ]

    tbody_rows: list[str] = []
    for experiment in table.index:
        is_baseline = experiment == baseline
        cells = [f"<td>{experiment}</td>"]
        for column in table.columns:
            raw_value = numeric.at[experiment, column]
            baseline_value = baseline_row[column]
            if not is_baseline and pd.notna(raw_value) and pd.notna(baseline_value):
                max_abs_delta = numeric.loc[
                    numeric.index != baseline, column
                ].sub(baseline_value).abs().max()
                if pd.isna(max_abs_delta):
                    max_abs_delta = 0.0
            else:
                max_abs_delta = 0.0

            color = _heatmap_color(
                None if pd.isna(raw_value) else float(raw_value),
                None if pd.isna(baseline_value) else float(baseline_value),
                is_baseline=is_baseline,
                max_abs_delta=float(max_abs_delta),
            )
            value_text = display.at[experiment, column]
            cells.append(
                f'<td style="background-color: {color};">{value_text}</td>'
            )
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    legend = """
<div class="legend">
  <span class="legend-item"><span class="swatch baseline"></span> Baseline (Qwen3-Embedding-4B)</span>
  <span class="legend-item"><span class="swatch better"></span> Better than baseline</span>
  <span class="legend-item"><span class="swatch worse"></span> Worse than baseline</span>
</div>
"""

    styles = """
<style>
  body { font-family: sans-serif; margin: 24px; color: #111827; }
  h1 { font-size: 1.25rem; margin-bottom: 8px; }
  p.subtitle { margin-top: 0; margin-bottom: 16px; color: #4b5563; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
  th, td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; }
  th:first-child, td:first-child { text-align: left; white-space: nowrap; font-weight: 600; }
  th { background: #f9fafb; position: sticky; top: 0; }
  .legend { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
  .legend-item { display: inline-flex; align-items: center; gap: 8px; }
  .swatch { width: 16px; height: 16px; border: 1px solid #d1d5db; display: inline-block; }
  .swatch.baseline { background: #f3f4f6; }
  .swatch.better { background: #2166ac; }
  .swatch.worse { background: #ef8a62; }
</style>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  {styles}
</head>
<body>
  <h1>{title}</h1>
  <p class="subtitle">Per-dataset comparison against the <strong>{baseline}</strong> baseline.</p>
  {legend}
  <table class="mteb-results-heatmap">
    <thead>
      {''.join(header_rows)}
    </thead>
    <tbody>
      {''.join(tbody_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    return html
