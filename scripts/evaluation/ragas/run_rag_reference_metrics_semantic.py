"""Evaluate generated RAG answers with semantic similarity (GPU via vLLM).

Compares embeddings of response vs reference using a vLLM OpenAI-compatible
embedding server (recommended) or optional local Hugging Face model.

Usage:
    bash jobs/scripts/santos_dumont/run_rag_reference_metrics_semantic_h100.sh --all

    export RAGAS_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
    python scripts/evaluation/ragas/run_rag_reference_metrics_semantic.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.evaluation.ragas.reference_cli import (
    add_common_reference_args,
    add_generation_selection_args,
    run_reference_metric_batch,
)
from tesis_unicamp.evaluation.ragas.reference_runner import (
    evaluate_reference_semantic_similarity,
)
from tesis_unicamp.evaluation.ragas.runner import DEFAULT_EMBEDDING_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated RAG answers with semantic similarity "
            "(response vs reference only)."
        ),
    )
    add_generation_selection_args(parser)
    add_common_reference_args(parser)
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help=f"Hugging Face or vLLM embedding model id (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--embedding-device",
        default=os.getenv("RAGAS_EMBEDDING_DEVICE", "cuda"),
        help="Device for local Hugging Face embeddings (default: cuda).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("RAGAS_EMBEDDING_BATCH_SIZE", "32")),
        help="Batch size for local embedding encoding (default: 32).",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("RAGAS_EMBEDDING_BASE_URL"),
        help=(
            "Optional OpenAI-compatible embedding server URL. "
            "When set, skips local GPU loading and calls vLLM instead."
        ),
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("RAGAS_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="API key sent to vLLM OpenAI servers (default: EMPTY).",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=float(os.getenv("RAGAS_API_TIMEOUT", "300")),
        help="HTTP timeout in seconds for embedding API calls (default: 300).",
    )
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip the startup OpenAI /v1/models health check.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    exit_code = run_reference_metric_batch(
        args=args,
        metric_group="semantic-similarity",
        evaluate_one=evaluate_reference_semantic_similarity,
        evaluate_kwargs={
            "embedding_model": args.embedding_model,
            "embedding_device": args.embedding_device,
            "embedding_batch_size": args.embedding_batch_size,
            "embedding_base_url": args.embedding_base_url,
            "openai_api_key": args.openai_api_key,
            "api_timeout": args.api_timeout,
            "skip_server_check": args.skip_server_check,
        },
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])
