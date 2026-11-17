"""Build comparison tables from RAGAS traditional reference generation metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

METRICS: tuple[str, ...] = (
    "rouge_score(mode=fmeasure)",
    "bleu_score",
    "chrf_score",
    "non_llm_string_similarity",
)
METRIC_LABELS: tuple[str, ...] = ("ROUGE-F", "BLEU", "chrF", "String Sim")

DATASET_ORDER: tuple[str, ...] = (
    "bioasq-resplit",
    "narrativeqa",
    "qasper",
    "telco-dpr",
)

DATASET_SHORT_NAME: dict[str, str] = {
    "bioasq-resplit": "BioASQ",
    "narrativeqa": "NarrativeQA",
    "qasper": "QASPER",
    "telco-dpr": "TelcoDPR",
}

BASELINE_SUFFIX = "emb-base-gen-base-top5"
BASELINE_RETRIEVER = "Retriever qwen3-Emb-4b"
BASELINE_GENERATOR = "Generator qwen3-8b"

BASELINE_COLOR = "#f3f4f6"
NEUTRAL_COLOR = (255, 255, 255)
BETTER_COLOR = (33, 102, 172)
WORSE_COLOR = (239, 138, 98)


@dataclass(frozen=True)
class GenerationExperimentRow:
    """One configured generation experiment row."""

    row_id: str
    retriever: str
    generator: str
    suffix: str
    is_baseline: bool = False


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


ALL_GENERATION_EXPERIMENTS: tuple[GenerationExperimentRow, ...] = (
    GenerationExperimentRow(
        "gen-base-noretrieval",
        "Retriever -",
        "Generator qwen3-8b",
        "gen-base-noretrieval",
    ),
    GenerationExperimentRow(
        "gen-base-title-noretrieval-2k",
        "Retriever - (2k)",
        "Generator qwen3-8b (title)",
        "gen-base-title-noretrieval-2k",
    ),
    GenerationExperimentRow(
        "gen-qa-noretrieval",
        "Retriever -",
        "Generator qwen3-8b FT QA",
        "gen-qa-noretrieval",
    ),
    GenerationExperimentRow(
        "gen-qa-title-noretrieval-2k",
        "Retriever - (2K)",
        "Generator qwen3-8b FT QA (title)",
        "gen-qa-title-noretrieval-2k",
    ),
    GenerationExperimentRow(
        "gen-lora-noretrieval",
        "Retriever -",
        "Generator qwen3-8b FT",
        "gen-lora-noretrieval",
    ),
    GenerationExperimentRow(
        "gen-lora-title-noretrieval-2k",
        "Retriever - (2k)",
        "Generator qwen3-8b FT (title)",
        "gen-lora-title-noretrieval-2k",
    ),
    GenerationExperimentRow(
        BASELINE_SUFFIX,
        BASELINE_RETRIEVER,
        BASELINE_GENERATOR,
        BASELINE_SUFFIX,
        is_baseline=True,
    ),
    GenerationExperimentRow(
        "emb-base-gen-base-title-top5-2k",
        "Retriever qwen3-Emb-4b (2k)",
        "Generator qwen3-8b (title)",
        "emb-base-gen-base-title-top5-2k",
    ),
    GenerationExperimentRow(
        "emb-lora-gen-base-top5",
        "Retriever qwen3-Emb-4b FT",
        "Generator qwen3-8b",
        "emb-lora-gen-base-top5",
    ),
    GenerationExperimentRow(
        "emb-lora-gen-base-title-top5-2k",
        "Retriever qwen3-Emb-4b FT (2k)",
        "Generator qwen3-8b (title)",
        "emb-lora-gen-base-title-top5-2k",
    ),
    GenerationExperimentRow(
        "emb-base-gen-qa-top5",
        "Retriever qwen3-Emb-4b",
        "Generator qwen3-8b FT QA",
        "emb-base-gen-qa-top5",
    ),
    GenerationExperimentRow(
        "emb-base-gen-qa-title-top5-2k",
        "Retriever qwen3-Emb-4b (2k)",
        "Generator qwen3-8b FT QA (title)",
        "emb-base-gen-qa-title-top5-2k",
    ),
    GenerationExperimentRow(
        "emb-base-gen-lora-top5",
        "Retriever qwen3-Emb-4b",
        "Generator qwen3-8b FT",
        "emb-base-gen-lora-top5",
    ),
    GenerationExperimentRow(
        "emb-base-gen-lora-title-top5-2k",
        "Retriever qwen3-Emb-4b (2k)",
        "Generator qwen3-8b FT (title)",
        "emb-base-gen-lora-title-top5-2k",
    ),
    GenerationExperimentRow(
        "emb-lora-gen-lora-top5",
        "Retriever qwen3-Emb-4b FT",
        "Generator qwen3-8b FT",
        "emb-lora-gen-lora-top5",
    ),
    GenerationExperimentRow(
        "emb-lora-gen-lora-title-top5-2k",
        "Retriever qwen3-Emb-4b FT (2k)",
        "Generator qwen3-8b FT (title)",
        "emb-lora-gen-lora-title-top5-2k",
    ),
)

TABLE_2_ROW_IDS: tuple[str, ...] = (
    "gen-base-title-noretrieval-2k",
    "gen-qa-title-noretrieval-2k",
    "gen-lora-title-noretrieval-2k",
    "emb-base-gen-base-title-top5-2k",
    "emb-lora-gen-base-title-top5-2k",
    "emb-base-gen-qa-title-top5-2k",
    "emb-base-gen-lora-title-top5-2k",
    "emb-lora-gen-lora-title-top5-2k",
)

TABLE_3_ROW_IDS: tuple[str, ...] = (
    "gen-base-noretrieval",
    "gen-lora-noretrieval",
    BASELINE_SUFFIX,
    "emb-lora-gen-base-top5",
    "emb-base-gen-lora-top5",
    "emb-lora-gen-lora-top5",
)

TABLE_4_ROW_IDS: tuple[str, ...] = (
    "gen-base-noretrieval",
    "gen-qa-noretrieval",
    "gen-lora-noretrieval",
    BASELINE_SUFFIX,
    "emb-lora-gen-base-top5",
    "emb-base-gen-qa-top5",
    "emb-base-gen-lora-top5",
    "emb-lora-gen-lora-top5",
)

GENERATION_TABLE_PRESETS: dict[str, tuple[str, ...]] = {
    "full": tuple(row.row_id for row in ALL_GENERATION_EXPERIMENTS),
    "title_2k": TABLE_2_ROW_IDS,
    "no_qa": TABLE_3_ROW_IDS,
    "with_qa": TABLE_4_ROW_IDS,
}


def summary_path(
    results_root: Path,
    dataset: str,
    suffix: str,
    *,
    split: str = "test",
) -> Path:
    experiment_name = f"{dataset}-{suffix}"
    return (
        results_root
        / dataset
        / experiment_name
        / split
        / "reference_traditional_summary.json"
    )


def load_summary_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics: dict[str, float] = {}
    for metric in METRICS:
        if metric not in payload:
            raise ValueError(f"Metric {metric!r} not found in {path}")
        metrics[metric] = float(payload[metric])
    return metrics


def get_experiment_rows(row_ids: tuple[str, ...] | list[str]) -> list[GenerationExperimentRow]:
    by_id = {row.row_id: row for row in ALL_GENERATION_EXPERIMENTS}
    missing = [row_id for row_id in row_ids if row_id not in by_id]
    if missing:
        raise ValueError(f"Unknown experiment row ids: {', '.join(missing)}")
    return [by_id[row_id] for row_id in row_ids]


def build_generation_table(
    results_root: Path,
    experiment_rows: list[GenerationExperimentRow],
    *,
    datasets: list[str] | None = None,
    split: str = "test",
) -> pd.DataFrame:
    """Build a generation comparison table for the selected experiment rows."""
    dataset_order = list(datasets or DATASET_ORDER)
    rows: dict[tuple[str, str], dict[tuple[str, str], float | None]] = {}

    for experiment in experiment_rows:
        key = (experiment.retriever, experiment.generator)
        row_values = rows.setdefault(key, {})
        for dataset in dataset_order:
            path = summary_path(
                results_root,
                dataset,
                experiment.suffix,
                split=split,
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing summary file: {path}")
            metrics = load_summary_metrics(path)
            for metric_key, metric_label in zip(METRICS, METRIC_LABELS, strict=True):
                row_values[(dataset, metric_label)] = metrics[metric_key]

    multi_columns = pd.MultiIndex.from_product(
        [dataset_order, list(METRIC_LABELS)],
        names=["dataset", "metric"],
    )
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=multi_columns)
    frame.index = pd.MultiIndex.from_frame(
        pd.DataFrame(frame.index.tolist(), columns=["Retriever", "Generator"])
    )
    ordered_keys = [(row.retriever, row.generator) for row in experiment_rows]
    frame = frame.reindex(ordered_keys)
    return frame


def load_baseline_row(
    results_root: Path,
    *,
    datasets: list[str] | None = None,
    split: str = "test",
) -> pd.Series:
    """Load baseline metric values flattened across dataset/metric columns."""
    baseline_rows = get_experiment_rows([BASELINE_SUFFIX])
    baseline_table = build_generation_table(
        results_root,
        baseline_rows,
        datasets=datasets,
        split=split,
    )
    return baseline_table.iloc[0]


def flatten_generation_columns(table: pd.DataFrame) -> pd.DataFrame:
    flat = table.copy()
    flat = flat.reset_index()
    metric_columns = [
        f"{DATASET_SHORT_NAME.get(dataset, dataset)} | {metric}"
        for dataset, metric in table.columns.to_list()
    ]
    flat.columns = ["Retriever", "Generator", *metric_columns]
    return flat


def format_table_for_display(
    table: pd.DataFrame,
    *,
    precision: int = 4,
) -> pd.DataFrame:
    display = table.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{precision}f}"
        )
    return display


def table_to_heatmap_html(
    table: pd.DataFrame,
    baseline_row: pd.Series,
    *,
    title: str,
    baseline_label: str,
    precision: int = 4,
) -> str:
    """Render an HTML heatmap table relative to a baseline configuration."""
    numeric = table.apply(pd.to_numeric, errors="coerce")
    display = format_table_for_display(table, precision=precision)

    dataset_labels = [
        DATASET_SHORT_NAME.get(dataset, dataset)
        for dataset in table.columns.get_level_values("dataset").unique()
    ]
    metric_labels = list(METRIC_LABELS)

    header_rows = [
        "<tr>"
        '<th rowspan="2">Retriever</th>'
        '<th rowspan="2">Generator</th>'
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
        retriever, generator = row_key
        is_baseline = (
            retriever == BASELINE_RETRIEVER
            and generator == BASELINE_GENERATOR
        )
        cells = [f"<td>{retriever}</td>", f"<td>{generator}</td>"]
        for column in table.columns:
            raw_value = numeric.loc[row_key, column]
            baseline_value = baseline_row[column]
            if not is_baseline and pd.notna(raw_value) and pd.notna(baseline_value):
                max_abs_delta = numeric.loc[
                    ~(
                        (numeric.index.get_level_values("Retriever") == BASELINE_RETRIEVER)
                        & (numeric.index.get_level_values("Generator") == BASELINE_GENERATOR)
                    ),
                    column,
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
  <span class="legend-item"><span class="swatch baseline"></span> Baseline row</span>
  <span class="legend-item"><span class="swatch better"></span> Better than baseline</span>
  <span class="legend-item"><span class="swatch worse"></span> Worse than baseline</span>
</div>
"""

    styles = """
<style>
  body { font-family: sans-serif; margin: 24px; color: #111827; }
  h1 { font-size: 1.25rem; margin-bottom: 8px; }
  p.subtitle { margin-top: 0; margin-bottom: 16px; color: #4b5563; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin-bottom: 32px; }
  th, td { border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; }
  th:nth-child(1), th:nth-child(2), td:nth-child(1), td:nth-child(2) {
    text-align: left; white-space: nowrap; font-weight: 600;
  }
  th { background: #f9fafb; position: sticky; top: 0; }
  .legend { display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }
  .legend-item { display: inline-flex; align-items: center; gap: 8px; }
  .swatch { width: 16px; height: 16px; border: 1px solid #d1d5db; display: inline-block; }
  .swatch.baseline { background: #f3f4f6; }
  .swatch.better { background: #2166ac; }
  .swatch.worse { background: #ef8a62; }
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
  <p class="subtitle">Per-dataset comparison against the baseline <strong>{baseline_label}</strong>.</p>
  {legend}
  <table class="generation-results-heatmap">
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
