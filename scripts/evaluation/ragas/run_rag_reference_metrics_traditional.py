"""Evaluate generated RAG answers with traditional NLP metrics (CPU only).

Metrics (response vs reference):
  - bleu_score
  - rouge_score
  - exact_match
  - string_present
  - chrf_score
  - non_llm_string_similarity

See: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/traditional/

Usage:
    python scripts/evaluation/ragas/run_rag_reference_metrics_traditional.py --all

    python scripts/evaluation/ragas/run_rag_reference_metrics_traditional.py \\
      --generation-dir datasets/generated/telco-dpr/generation-b128-vllm-offline-b128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.evaluation.ragas.reference_cli import (
    add_common_reference_args,
    add_generation_selection_args,
    run_reference_metric_batch,
)
from tesis_unicamp.evaluation.ragas.reference_runner import (
    evaluate_reference_traditional_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated RAG answers with traditional NLP metrics "
            "(no GPU, response vs reference only)."
        ),
    )
    add_generation_selection_args(parser)
    add_common_reference_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    exit_code = run_reference_metric_batch(
        args=args,
        metric_group="traditional",
        evaluate_one=evaluate_reference_traditional_metrics,
        evaluate_kwargs={},
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])
