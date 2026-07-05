"""Reshuffle BioASQ RAG Hub splits and optionally upload to Hugging Face.

Usage:
    python scripts/create_dataset/resplit_bioasq_rag_dataset.py
    python scripts/create_dataset/resplit_bioasq_rag_dataset.py \\
        --push-to-hub DinoStackAI/bioasq-rag-13b-resplit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.datasets.preprocessing.rag.bioasq.constants import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_RESPLIT_DEV_RATIO,
    DEFAULT_RESPLIT_HUB_README_TEMPLATE,
    DEFAULT_RESPLIT_OUTPUT_DIR,
    DEFAULT_TEST_RATIO,
    PROJECT_ROOT,
)
from tesis_unicamp.datasets.preprocessing.rag.bioasq.resplit import (
    push_resplit_bioasq_rag_to_hub,
    resplit_bioasq_rag_from_hub,
)
from tesis_unicamp.datasets.utils.bioasq_rag import BIOASQ_RAG_DATASET_ID


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge train/dev/test from a BioASQ RAG Hub dataset, shuffle, "
            "resplit (20% test, 20% dev from remainder), and optionally upload."
        ),
    )
    parser.add_argument(
        "--source-repo-id",
        type=str,
        default=BIOASQ_RAG_DATASET_ID,
        help=f"Source Hugging Face dataset (default: {BIOASQ_RAG_DATASET_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESPLIT_OUTPUT_DIR,
        help="Directory to save processed JSON and HF dataset",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Fraction of all queries reserved for test (default: 0.2)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=DEFAULT_RESPLIT_DEV_RATIO,
        help="Fraction of post-test queries reserved for dev (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for shuffle and split",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Hugging Face repo id to upload the resplit dataset",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (defaults to HF_TOKEN in .env)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset on the Hub",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_RESPLIT_HUB_README_TEMPLATE,
        help="README template for Hugging Face",
    )
    return parser


def main() -> None:
    _load_env()
    args = _build_parser().parse_args()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    dataset_dict = resplit_bioasq_rag_from_hub(
        source_repo_id=args.source_repo_id,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    corpus_size = len(dataset_dict["corpus"])
    train_queries = len(dataset_dict["train"]["queries"])
    dev_queries = len(dataset_dict["dev"]["queries"])
    test_queries = len(dataset_dict["test"]["queries"])
    total_queries = train_queries + dev_queries + test_queries

    print(f"Source dataset: {args.source_repo_id}")
    print(f"Total queries merged: {total_queries}")
    print(f"Corpus documents: {corpus_size}")
    print(f"Train queries: {train_queries} ({train_queries / total_queries:.1%})")
    print(f"Dev queries: {dev_queries} ({dev_queries / total_queries:.1%})")
    print(f"Test queries: {test_queries} ({test_queries / total_queries:.1%})")
    print(f"Saved to: {args.output_dir}")

    if args.push_to_hub:
        if not token:
            raise SystemExit(
                "Missing Hugging Face token. Set HF_TOKEN in .env or pass --token."
            )
        push_resplit_bioasq_rag_to_hub(
            dataset_dict,
            repo_id=args.push_to_hub,
            source_repo_id=args.source_repo_id,
            output_dir=args.output_dir,
            token=token,
            private=args.private,
            readme_path=args.readme,
            test_ratio=args.test_ratio,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
            total_queries=total_queries,
        )
        print(f"Uploaded to Hugging Face Hub: {args.push_to_hub}")


if __name__ == "__main__":
    main()
