"""Build comparison tables for RFG retrieval results across k values."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tesis_unicamp.evaluation.mteb.results_table import (
    BASELINE_COLOR,
    METRIC_LABELS,
    METRICS,
    _heatmap_color,
    format_table_for_display,
    load_result_file,
)

K_INDEX_NAME = "K"
BASELINE_K = 0
K_VALUES: tuple[int, ...] = (0, 1, 3, 5, 7, 10)
DEFAULT_MODEL_NAME = "Qwen__Qwen3-Embedding-4B"
RFG_EXPERIMENT_SUFFIX = "emb-base-gen-base"


@dataclass(frozen=True)
class RfgDatasetConfig:
    slug: str
    short_name: str
    baseline_experiment: str
    baseline_task_file: str


DATASET_CONFIGS: tuple[RfgDatasetConfig, ...] = (
    RfgDatasetConfig(
        slug="bioasq-resplit",
        short_name="BioASQ",
        baseline_experiment="vllm-offline-bioasq-resplit-b128",
        baseline_task_file="BioASQ-RAG-Resplit.json",
    ),
    RfgDatasetConfig(
        slug="narrativeqa",
        short_name="NarrativeQA",
        baseline_experiment="vllm-offline-narrativeqa-b128",
        baseline_task_file="NarrativeQA-RAG.json",
    ),
    RfgDatasetConfig(
        slug="qasper",
        short_name="QASPER",
        baseline_experiment="vllm-offline-qasper-b128",
        baseline_task_file="QASPER-RAG.json",
    ),
    RfgDatasetConfig(
        slug="telco-dpr",
        short_name="TelcoDPR",
        baseline_experiment="vllm-offline-telco-dpr-b128",
        baseline_task_file="TelcoDPR-RAG.json",
    ),
)


def _extract_metrics(payload: dict[str, Any], *, split: str) -> dict[str, float]:
    scores_by_split = payload.get("scores") or {}
    if split not in scores_by_split:
        available = ", ".join(sorted(scores_by_split))
        raise ValueError(f"Split {split!r} not found. Available splits: {available}")

    split_scores = scores_by_split[split]
    if not split_scores:
        raise ValueError(f"No score entries for split {split!r}.")

    raw_metrics = split_scores[0]
    metrics: dict[str, float] = {}
    for metric in METRICS:
        if metric not in raw_metrics:
            raise ValueError(f"Metric {metric!r} not found in {split!r} scores.")
        metrics[metric] = float(raw_metrics[metric])
    return metrics


def load_metrics_from_json(path: Path, *, split: str = "test") -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _extract_metrics(payload, split=split)


def _baseline_result_path(
    baseline_root: Path,
    dataset: RfgDatasetConfig,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    return (
        baseline_root
        / model_name
        / dataset.baseline_experiment
        / dataset.baseline_task_file
    )


def _rfg_experiment_name(dataset_slug: str, k: int) -> str:
    return f"rfg-{dataset_slug}-{RFG_EXPERIMENT_SUFFIX}-k{k}"


def _rfg_result_path(
    rfg_root: Path,
    dataset: RfgDatasetConfig,
    k: int,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    split: str = "test",
) -> Path:
    experiment_dir = (
        rfg_root
        / dataset.slug
        / "results"
        / model_name
        / _rfg_experiment_name(dataset.slug, k)
    )
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Missing RFG experiment directory: {experiment_dir}")

    candidates = sorted(
        path
        for path in experiment_dir.glob("*.json")
        if path.name != "model_meta.json"
    )
    if not candidates:
        raise FileNotFoundError(f"No result JSON found under {experiment_dir}")
    if len(candidates) > 1:
        ctx_rfg = [path for path in candidates if path.name.endswith("-ctx-rfg.json")]
        if len(ctx_rfg) == 1:
            return ctx_rfg[0]
    return candidates[0]


def load_rfg_metrics(
    *,
    k: int,
    dataset: RfgDatasetConfig,
    baseline_root: Path,
    rfg_root: Path,
    split: str = "test",
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, float]:
    if k == BASELINE_K:
        path = _baseline_result_path(baseline_root, dataset, model_name=model_name)
    else:
        path = _rfg_result_path(
            rfg_root,
            dataset,
            k,
            model_name=model_name,
            split=split,
        )

    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")

    if k == BASELINE_K:
        record = load_result_file(path, split=split)
        return record.metrics

    return load_metrics_from_json(path, split=split)


def build_rfg_k_table(
    *,
    baseline_root: Path,
    rfg_root: Path,
    split: str = "test",
    model_name: str = DEFAULT_MODEL_NAME,
    k_values: tuple[int, ...] = K_VALUES,
    datasets: tuple[RfgDatasetConfig, ...] = DATASET_CONFIGS,
) -> pd.DataFrame:
    """Build a table with one row per k and dataset metric columns."""
    rows: dict[int, dict[tuple[str, str], float | None]] = {}
    missing: list[str] = []

    for k in k_values:
        row_values: dict[tuple[str, str], float | None] = {}
        for dataset in datasets:
            try:
                metrics = load_rfg_metrics(
                    k=k,
                    dataset=dataset,
                    baseline_root=baseline_root,
                    rfg_root=rfg_root,
                    split=split,
                    model_name=model_name,
                )
            except FileNotFoundError as exc:
                missing.append(str(exc))
                for metric_label in METRIC_LABELS:
                    row_values[(dataset.slug, metric_label)] = None
                continue

            for metric_key, metric_label in zip(METRICS, METRIC_LABELS, strict=True):
                row_values[(dataset.slug, metric_label)] = metrics[metric_key]
        rows[k] = row_values

    if missing:
        print("Warning: missing result files:")
        for message in missing:
            print(f"  - {message}")

    dataset_slugs = [dataset.slug for dataset in datasets]
    multi_columns = pd.MultiIndex.from_product(
        [dataset_slugs, list(METRIC_LABELS)],
        names=["dataset", "metric"],
    )
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=multi_columns)
    frame = frame.reindex(list(k_values))
    frame.index.name = K_INDEX_NAME
    return frame


def flatten_rfg_columns(
    table: pd.DataFrame,
    datasets: tuple[RfgDatasetConfig, ...] = DATASET_CONFIGS,
) -> pd.DataFrame:
    short_names = {dataset.slug: dataset.short_name for dataset in datasets}
    flat = table.copy().reset_index()
    flat.columns = [
        K_INDEX_NAME,
        *[
            f"{short_names.get(dataset, dataset)} | {metric}"
            for dataset, metric in table.columns.to_list()
        ],
    ]
    return flat


def table_to_rfg_heatmap_html(
    table: pd.DataFrame,
    *,
    title: str,
    datasets: tuple[RfgDatasetConfig, ...] = DATASET_CONFIGS,
    baseline_k: int = BASELINE_K,
    precision: int = 4,
) -> str:
    """Render an HTML heatmap table relative to k=0."""
    if baseline_k not in table.index:
        raise ValueError(f"Baseline k={baseline_k} not found in table.")

    numeric = table.apply(pd.to_numeric, errors="coerce")
    baseline_row = numeric.loc[baseline_k]
    display = format_table_for_display(table, precision=precision)

    short_names = {dataset.slug: dataset.short_name for dataset in datasets}
    dataset_labels = [
        short_names.get(dataset, dataset)
        for dataset in table.columns.get_level_values("dataset").unique()
    ]
    metric_labels = list(METRIC_LABELS)

    header_rows = [
        "<tr>"
        f'<th rowspan="2">{K_INDEX_NAME}</th>'
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
    for k_value in table.index:
        is_baseline = k_value == baseline_k
        cells = [f"<td>{k_value}</td>"]
        for column in table.columns:
            raw_value = numeric.loc[k_value, column]
            baseline_value = baseline_row[column]
            if not is_baseline and pd.notna(raw_value) and pd.notna(baseline_value):
                max_abs_delta = numeric.loc[
                    numeric.index != baseline_k, column
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
            value_text = display.loc[k_value, column]
            cells.append(f'<td style="background-color: {color};">{value_text}</td>')
        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    legend = """
<div class="legend">
  <span class="legend-item"><span class="swatch baseline"></span> Baseline (k=0)</span>
  <span class="legend-item"><span class="swatch better"></span> Better than baseline</span>
  <span class="legend-item"><span class="swatch worse"></span> Worse than baseline</span>
</div>
"""

    styles = f"""
<style>
  body {{ font-family: sans-serif; margin: 24px; color: #111827; }}
  h1 {{ font-size: 1.25rem; margin-bottom: 8px; }}
  p.subtitle {{ margin-top: 0; margin-bottom: 16px; color: #4b5563; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: center; white-space: nowrap; font-weight: 600; }}
  th {{ background: #f9fafb; position: sticky; top: 0; }}
  .legend {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 8px; }}
  .swatch {{ width: 16px; height: 16px; border: 1px solid #d1d5db; display: inline-block; }}
  .swatch.baseline {{ background: {BASELINE_COLOR}; }}
  .swatch.better {{ background: #2166ac; }}
  .swatch.worse {{ background: #ef8a62; }}
</style>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  {styles}
</head>
<body>
  <h1>{title}</h1>
  <p class="subtitle">
    Per-dataset comparison against <strong>k=0</strong>
    (vllm-offline baseline retrieval without RFG).
  </p>
  {legend}
  <table class="rfg-results-heatmap">
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
