"""Build NarrativeQA RAG dataset and optionally upload to Hugging Face.

Usage:
    python scripts/create_narrativeqa_rag_dataset.py
    python scripts/create_narrativeqa_rag_dataset.py --push-to-hub username/narrativeqa-rag
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.builder import (
    build_narrativeqa_rag_dataset,
    push_narrativeqa_rag_to_hub,
)
from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.constants import (
    DEFAULT_HF_DATASET_ID,
    DEFAULT_HUB_README,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
)


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build NarrativeQA RAG dataset and optionally upload to Hugging Face.",
    )
    parser.add_argument(
        "--hf-dataset-id",
        type=str,
        default=DEFAULT_HF_DATASET_ID,
        help="Hugging Face dataset id to download from",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save processed JSON and HF dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Hugging Face repo id, e.g. username/narrativeqa-rag",
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
        default=DEFAULT_HUB_README,
        help="README for Hugging Face (default: datasets/processed/narrativeqa_rag/README.md)",
    )
    return parser


def main() -> None:
    _load_env()
    args = _build_parser().parse_args()

    token = args.token or os.getenv("HF_TOKEN")

    dataset_dict = build_narrativeqa_rag_dataset(
        hf_dataset_id=args.hf_dataset_id,
        output_dir=args.output_dir,
    )

    corpus_size = len(dataset_dict["corpus"])
    train_queries = len(dataset_dict["train"]["queries"])
    dev_queries = len(dataset_dict["dev"]["queries"])
    test_queries = len(dataset_dict["test"]["queries"])

    print(f"Corpus documents: {corpus_size}")
    print(f"Train queries: {train_queries}")
    print(f"Dev queries: {dev_queries}")
    print(f"Test queries: {test_queries}")
    print(f"Saved to: {args.output_dir}")

    if args.push_to_hub:
        push_narrativeqa_rag_to_hub(
            dataset_dict,
            repo_id=args.push_to_hub,
            token=token,
            private=args.private,
            readme_path=args.readme,
            hf_dataset_id=args.hf_dataset_id,
            output_dir=args.output_dir,
        )
        print(f"Uploaded to Hugging Face Hub: {args.push_to_hub}")


if __name__ == "__main__":
    main()
