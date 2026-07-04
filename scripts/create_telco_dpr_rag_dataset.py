"""Build Telco-DPR RAG dataset and optionally upload to Hugging Face.

Usage:
    python scripts/create_telco_dpr_rag_dataset.py
    python scripts/create_telco_dpr_rag_dataset.py --push-to-hub username/telco-dpr-rag
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.builder import (
    build_telco_dpr_rag_dataset,
    push_telco_dpr_rag_to_hub,
)
from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.constants import (
    DEFAULT_DEV_RATIO,
    DEFAULT_HF_DATASET_ID,
    DEFAULT_HUB_README,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_SEED,
    PROJECT_ROOT,
)


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Telco-DPR RAG dataset and optionally upload to Hugging Face.",
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
        "--dev-ratio",
        type=float,
        default=DEFAULT_DEV_RATIO,
        help="Fraction of training queries reserved for dev",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for train/dev split",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Hugging Face repo id, e.g. username/telco-dpr-rag",
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
        help="README for Hugging Face (default: datasets/processed/telco_dpr_rag/README.md)",
    )
    return parser


def main() -> None:
    _load_env()
    args = _build_parser().parse_args()

    token = args.token or os.getenv("HF_TOKEN")

    dataset_dict = build_telco_dpr_rag_dataset(
        hf_dataset_id=args.hf_dataset_id,
        output_dir=args.output_dir,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
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
        push_telco_dpr_rag_to_hub(
            dataset_dict,
            repo_id=args.push_to_hub,
            token=token,
            private=args.private,
            readme_path=args.readme,
            hf_dataset_id=args.hf_dataset_id,
            output_dir=args.output_dir,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
        )
        print(f"Uploaded to Hugging Face Hub: {args.push_to_hub}")


if __name__ == "__main__":
    main()
