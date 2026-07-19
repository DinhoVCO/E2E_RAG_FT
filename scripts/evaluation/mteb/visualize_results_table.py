#!/usr/bin/env python3
"""Build a grouped experiment comparison table from MTEB retrieval results."""

from __future__ import annotations

import argparse
from pathlib import Path

from tesis_unicamp.evaluation.mteb.results_table import (
    DEFAULT_DATASET_ORDER,
    build_grouped_experiment_table,
    flatten_columns,
    format_table_for_display,
    load_results,
    table_to_heatmap_html,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_results_root() -> Path:
    return _project_root() / "results" / "mteb" / "results"


def _default_reports_dir() -> Path:
    return _project_root() / "reports" / "retrieval_ft_v1"


def parse_args() -> argparse.Namespace:
    reports_dir = _default_reports_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Build a grouped MTEB experiment table with 4 experiment types "
            "(base, FT, Title, FT Title) and an HTML heatmap vs baseline."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_default_results_root(),
        help="Root directory with MTEB results (default: results/mteb/results).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Evaluation split to read from each JSON file (default: test).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Dataset task names to include as columns. "
            f"Default order: {', '.join(DEFAULT_DATASET_ORDER)}."
        ),
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for displayed values (default: 4).",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=reports_dir / "results_table.html",
        help=(
            "Path to write the heatmap HTML table "
            "(default: reports/retrieval_ft_v1/results_table.html)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=reports_dir / "results_table.csv",
        help=(
            "Path to write the flat CSV table "
            "(default: reports/retrieval_ft_v1/results_table.csv)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_results(args.results_root, split=args.split)
    if not records:
        raise SystemExit(f"No result files found under {args.results_root}")

    table = build_grouped_experiment_table(records, datasets=args.datasets)

    display = format_table_for_display(table, precision=args.precision)
    print(display.to_string())

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        flatten_columns(table).to_csv(args.output_csv)
        print(f"\nCSV written to {args.output_csv}")

    if args.output_html is not None:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        html = table_to_heatmap_html(
            table,
            title="MTEB Retrieval FT v1",
            precision=args.precision,
        )
        args.output_html.write_text(html, encoding="utf-8")
        print(f"HTML written to {args.output_html}")


if __name__ == "__main__":
    main()
