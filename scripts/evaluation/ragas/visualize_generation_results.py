#!/usr/bin/env python3
"""Build grouped generation comparison tables from RAGAS traditional metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from tesis_unicamp.evaluation.ragas.generation_table import (
    ALL_GENERATION_EXPERIMENTS,
    BASELINE_GENERATOR,
    BASELINE_RETRIEVER,
    BASELINE_SUFFIX,
    GENERATION_TABLE_PRESETS,
    build_generation_table,
    flatten_generation_columns,
    format_table_for_display,
    get_experiment_rows,
    load_baseline_row,
    table_to_heatmap_html,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_results_root() -> Path:
    return _project_root() / "results" / "ragas-reference" / "traditional"


def _default_reports_dir() -> Path:
    return _project_root() / "reports" / "generation_ragas_v1"


def _baseline_label() -> str:
    return (
        f"Retriever {BASELINE_RETRIEVER} | "
        f"Generator {BASELINE_GENERATOR} | "
        f"{{dataset}}-{BASELINE_SUFFIX}"
    )


TABLE_OUTPUTS: dict[str, tuple[str, str]] = {
    "full": ("generation_table_full", "Generation RAGAS — Full Comparison"),
    "title_2k": ("generation_table_title_2k", "Generation RAGAS — Title 2k Experiments"),
    "no_qa": ("generation_table_no_qa", "Generation RAGAS — Without QA Fine-Tuning"),
    "with_qa": ("generation_table_with_qa", "Generation RAGAS — With QA Fine-Tuning"),
}


def parse_args() -> argparse.Namespace:
    reports_dir = _default_reports_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Build generation comparison tables from RAGAS traditional "
            "reference metrics with heatmaps against the emb-base-gen-base-top5 baseline."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_default_results_root(),
        help="Root directory with RAGAS traditional results.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Evaluation split folder name (default: test).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for displayed values (default: 4).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=reports_dir,
        help="Directory for generated CSV/HTML reports.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(GENERATION_TABLE_PRESETS),
        default=None,
        help="Generate only one preset instead of all tables.",
    )
    return parser.parse_args()


def _write_table_outputs(
    *,
    preset: str,
    stem: str,
    title: str,
    results_root: Path,
    reports_dir: Path,
    split: str,
    precision: int,
    baseline_row,
) -> None:
    experiment_rows = get_experiment_rows(GENERATION_TABLE_PRESETS[preset])
    table = build_generation_table(
        results_root,
        experiment_rows,
        split=split,
    )
    display = format_table_for_display(table, precision=precision)
    print(f"\n=== {title} ===")
    print(display.to_string())

    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"{stem}.csv"
    html_path = reports_dir / f"{stem}.html"

    flatten_generation_columns(table).to_csv(csv_path, index=False)
    html = table_to_heatmap_html(
        table,
        baseline_row,
        title=title,
        baseline_label=_baseline_label(),
        precision=precision,
    )
    html_path.write_text(html, encoding="utf-8")
    print(f"CSV written to {csv_path}")
    print(f"HTML written to {html_path}")


def main() -> None:
    args = parse_args()
    baseline_row = load_baseline_row(args.results_root, split=args.split)

    presets = (
        (args.preset,) if args.preset is not None else tuple(GENERATION_TABLE_PRESETS)
    )
    for preset in presets:
        stem, title = TABLE_OUTPUTS[preset]
        _write_table_outputs(
            preset=preset,
            stem=stem,
            title=title,
            results_root=args.results_root,
            reports_dir=args.reports_dir,
            split=args.split,
            precision=args.precision,
            baseline_row=baseline_row,
        )


if __name__ == "__main__":
    main()
