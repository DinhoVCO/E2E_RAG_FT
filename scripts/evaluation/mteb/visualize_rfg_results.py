#!/usr/bin/env python3
"""Build RFG retrieval comparison tables across k values."""

from __future__ import annotations

import argparse
from pathlib import Path

from tesis_unicamp.evaluation.mteb.rfg_table import (
    build_rfg_k_table,
    flatten_rfg_columns,
    format_table_for_display,
    table_to_rfg_heatmap_html,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_baseline_root() -> Path:
    return _project_root() / "results" / "mteb" / "results"


def _default_rfg_root() -> Path:
    return _project_root() / "results" / "mteb" / "rfg"


def _default_reports_dir() -> Path:
    return _project_root() / "reports" / "rfg_k_v1"


def parse_args() -> argparse.Namespace:
    reports_dir = _default_reports_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Build an RFG comparison table across k values using "
            "rfg-{dataset}-emb-base-gen-base-kN experiments, with k=0 "
            "as the vllm-offline baseline."
        )
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=_default_baseline_root(),
        help="Root directory with baseline MTEB results (default: results/mteb/results).",
    )
    parser.add_argument(
        "--rfg-root",
        type=Path,
        default=_default_rfg_root(),
        help="Root directory with RFG MTEB results (default: results/mteb/rfg).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Evaluation split to read from each JSON file (default: test).",
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
        help="Directory for generated CSV/HTML reports (default: reports/rfg_k_v1).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=reports_dir / "rfg_k_table.csv",
        help="Path to write the CSV table.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=reports_dir / "rfg_k_table.html",
        help="Path to write the heatmap HTML table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    table = build_rfg_k_table(
        baseline_root=args.baseline_root,
        rfg_root=args.rfg_root,
        split=args.split,
    )

    display = format_table_for_display(table, precision=args.precision)
    print(display.to_string())

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)

    flatten_rfg_columns(table).to_csv(args.output_csv, index=False)
    html = table_to_rfg_heatmap_html(
        table,
        title="RFG Retrieval — k Comparison",
        precision=args.precision,
    )
    args.output_html.write_text(html, encoding="utf-8")

    print(f"\nCSV written to {args.output_csv}")
    print(f"HTML written to {args.output_html}")


if __name__ == "__main__":
    main()
