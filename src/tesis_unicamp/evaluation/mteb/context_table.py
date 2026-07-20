"""Build comparison tables for context retrieval results across k values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tesis_unicamp.evaluation.mteb.rfg_table import (
    BASELINE_K,
    DATASET_CONFIGS,
    DEFAULT_MODEL_NAME,
    K_INDEX_NAME,
    load_metrics_from_json,
)
from tesis_unicamp.evaluation.mteb.results_table import (
    BASELINE_COLOR,
    METRIC_LABELS,
    METRICS,
    _heatmap_color,
    format_table_for_display,
    load_result_file,
)

ContextRowKey = int | str
CONTEXT_K_ROWS: tuple[ContextRowKey, ...] = (0, "stage1", 1, 3, 5, 7, 10)
CONTEXT_TITLE_K_ROWS: tuple[ContextRowKey, ...] = (0, "stage1", 1, 3, 5, 7, 10)
CONTEXT_ADAPTER_PREFIX = "Qwen3-Emb-4b-lora-ctx"


@dataclass(frozen=True)
class ContextDatasetConfig:
    slug: str
    short_name: str
    baseline_experiment: str
    baseline_task_file: str


TITLE_DATASET_CONFIGS: tuple[ContextDatasetConfig, ...] = (
    ContextDatasetConfig(
        slug="bioasq-resplit",
        short_name="BioASQ",
        baseline_experiment="vllm-offline-bioasq-resplit-b128-title",
        baseline_task_file="BioASQ-RAG-Resplit-Title.json",
    ),
    ContextDatasetConfig(
        slug="narrativeqa",
        short_name="NarrativeQA",
        baseline_experiment="vllm-offline-narrativeqa-b128-title",
        baseline_task_file="NarrativeQA-RAG-Title.json",
    ),
    ContextDatasetConfig(
        slug="qasper",
        short_name="QASPER",
        baseline_experiment="vllm-offline-qasper-b128-title",
        baseline_task_file="QASPER-RAG-Title.json",
    ),
    ContextDatasetConfig(
        slug="telco-dpr",
        short_name="TelcoDPR",
        baseline_experiment="vllm-offline-telco-dpr-b128-title",
        baseline_task_file="TelcoDPR-RAG-Title.json",
    ),
)


@dataclass(frozen=True)
class ContextTableConfig:
    """Configuration for context or context-title result tables."""

    experiment_title_suffix: str = ""
    datasets: tuple[ContextDatasetConfig, ...] = DATASET_CONFIGS
    row_keys: tuple[ContextRowKey, ...] = CONTEXT_K_ROWS
    heatmap_subtitle: str = (
        "Per-dataset comparison against <strong>k=0</strong> "
        "(vllm-offline baseline retrieval without context reranking)."
    )


CONTEXT_TABLE_CONFIG = ContextTableConfig()
CONTEXT_TITLE_TABLE_CONFIG = ContextTableConfig(
    experiment_title_suffix="-title",
    datasets=TITLE_DATASET_CONFIGS,
    row_keys=CONTEXT_TITLE_K_ROWS,
    heatmap_subtitle=(
        "Per-dataset comparison against <strong>k=0</strong> "
        "(vllm-offline title baseline retrieval without context reranking)."
    ),
)


def _context_adapter_dir(dataset_slug: str) -> str:
    return f"{CONTEXT_ADAPTER_PREFIX}-{dataset_slug}"


def _context_experiment_name(
    dataset_slug: str,
    row_key: ContextRowKey,
    *,
    title_suffix: str = "",
) -> str:
    if row_key == "stage1":
        return f"ctx-lora-{dataset_slug}-stage1{title_suffix}"
    return f"ctx-lora-{dataset_slug}-k{row_key}{title_suffix}"


def _baseline_result_path(
    baseline_root: Path,
    dataset: ContextDatasetConfig,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    return (
        baseline_root
        / model_name
        / dataset.baseline_experiment
        / dataset.baseline_task_file
    )


def _context_result_path(
    context_root: Path,
    dataset: ContextDatasetConfig,
    row_key: ContextRowKey,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    title_suffix: str = "",
) -> Path:
    experiment_dir = (
        context_root
        / dataset.slug
        / _context_adapter_dir(dataset.slug)
        / "results"
        / model_name
        / _context_experiment_name(
            dataset.slug,
            row_key,
            title_suffix=title_suffix,
        )
    )
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Missing context experiment directory: {experiment_dir}")

    candidates = sorted(
        path
        for path in experiment_dir.glob("*.json")
        if path.name != "model_meta.json"
    )
    if not candidates:
        raise FileNotFoundError(f"No result JSON found under {experiment_dir}")
    return candidates[0]


def load_context_metrics(
    *,
    row_key: ContextRowKey,
    dataset: ContextDatasetConfig,
    baseline_root: Path,
    context_root: Path,
    split: str = "test",
    model_name: str = DEFAULT_MODEL_NAME,
    title_suffix: str = "",
) -> dict[str, float]:
    if row_key == BASELINE_K:
        path = _baseline_result_path(baseline_root, dataset, model_name=model_name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing result file: {path}")
        record = load_result_file(path, split=split)
        return record.metrics

    path = _context_result_path(
        context_root,
        dataset,
        row_key,
        model_name=model_name,
        title_suffix=title_suffix,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    return load_metrics_from_json(path, split=split)


def build_context_k_table(
    *,
    baseline_root: Path,
    context_root: Path,
    split: str = "test",
    model_name: str = DEFAULT_MODEL_NAME,
    config: ContextTableConfig = CONTEXT_TABLE_CONFIG,
) -> pd.DataFrame:
    """Build a table with one row per k/stage1 and dataset metric columns."""
    rows: dict[ContextRowKey, dict[tuple[str, str], float | None]] = {}
    missing: list[str] = []

    for row_key in config.row_keys:
        row_values: dict[tuple[str, str], float | None] = {}
        for dataset in config.datasets:
            try:
                metrics = load_context_metrics(
                    row_key=row_key,
                    dataset=dataset,
                    baseline_root=baseline_root,
                    context_root=context_root,
                    split=split,
                    model_name=model_name,
                    title_suffix=config.experiment_title_suffix,
                )
            except FileNotFoundError as exc:
                missing.append(str(exc))
                for metric_label in METRIC_LABELS:
                    row_values[(dataset.slug, metric_label)] = None
                continue

            for metric_key, metric_label in zip(METRICS, METRIC_LABELS, strict=True):
                row_values[(dataset.slug, metric_label)] = metrics[metric_key]
        rows[row_key] = row_values

    if missing:
        print("Warning: missing result files:")
        for message in missing:
            print(f"  - {message}")

    dataset_slugs = [dataset.slug for dataset in config.datasets]
    multi_columns = pd.MultiIndex.from_product(
        [dataset_slugs, list(METRIC_LABELS)],
        names=["dataset", "metric"],
    )
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=multi_columns)
    frame = frame.reindex(list(config.row_keys))
    frame.index.name = K_INDEX_NAME
    return frame


def flatten_context_columns(
    table: pd.DataFrame,
    datasets: tuple[ContextDatasetConfig, ...] | None = None,
) -> pd.DataFrame:
    if datasets is None:
        dataset_slugs = list(table.columns.get_level_values("dataset").unique())
        short_names = {
            dataset.slug: dataset.short_name
            for dataset in (*DATASET_CONFIGS, *TITLE_DATASET_CONFIGS)
            if dataset.slug in dataset_slugs
        }
    else:
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


def table_to_context_heatmap_html(
    table: pd.DataFrame,
    *,
    title: str,
    datasets: tuple[ContextDatasetConfig, ...] | None = None,
    baseline_k: ContextRowKey = BASELINE_K,
    precision: int = 4,
    subtitle: str = CONTEXT_TABLE_CONFIG.heatmap_subtitle,
) -> str:
    """Render an HTML heatmap table relative to k=0."""
    if baseline_k not in table.index:
        raise ValueError(f"Baseline k={baseline_k!r} not found in table.")

    if datasets is None:
        dataset_slugs = list(table.columns.get_level_values("dataset").unique())
        datasets_by_slug = {
            dataset.slug: dataset
            for dataset in (*DATASET_CONFIGS, *TITLE_DATASET_CONFIGS)
        }
        datasets = tuple(
            datasets_by_slug[slug] for slug in dataset_slugs if slug in datasets_by_slug
        )

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
    for row_key in table.index:
        is_baseline = row_key == baseline_k
        cells = [f"<td>{row_key}</td>"]
        for column in table.columns:
            raw_value = numeric.loc[row_key, column]
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
            value_text = display.loc[row_key, column]
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
  <p class="subtitle">{subtitle}</p>
  {legend}
  <table class="context-results-heatmap">
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
