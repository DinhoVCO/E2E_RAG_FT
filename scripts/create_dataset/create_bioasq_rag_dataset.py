"""Build BioASQ RAG dataset and optionally upload to Hugging Face.

Usage:
    python scripts/create_bioasq_rag_dataset.py
    python scripts/create_bioasq_rag_dataset.py --push-to-hub username/bioasq-rag-13b
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.datasets.preprocessing.rag.bioasq.builder import (
    build_bioasq_rag_dataset,
    push_bioasq_rag_to_hub,
)
from tesis_unicamp.datasets.preprocessing.rag.bioasq.constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DEV_RATIO,
    DEFAULT_GOLDEN_DIR,
    DEFAULT_HUB_README,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAINING_PATH,
    PROJECT_ROOT,
)


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build BioASQ RAG dataset and optionally upload to Hugging Face.",
    )
    parser.add_argument(
        "--training-path",
        type=Path,
        default=DEFAULT_TRAINING_PATH,
        help="Path to BioASQ training13b.json",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=DEFAULT_GOLDEN_DIR,
        help="Directory with Task13BGoldenEnriched JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save processed JSON and HF dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for PubMed abstract cache",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=DEFAULT_DEV_RATIO,
        help="Fraction of training questions reserved for dev",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for train/dev split",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Contact email for NCBI E-utilities (optional)",
    )
    parser.add_argument(
        "--force-refresh-pubmed",
        action="store_true",
        help="Ignore PubMed cache and re-fetch all abstracts",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help="Hugging Face repo id, e.g. username/bioasq-rag-13b",
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
        help="README for Hugging Face (default: datasets/processed/bioasq_rag/README.md)",
    )
    return parser


def main() -> None:
    _load_env()
    args = _build_parser().parse_args()

    email = args.email or os.getenv("NCBI_EMAIL")
    token = args.token or os.getenv("HF_TOKEN")

    dataset_dict = build_bioasq_rag_dataset(
        training_path=args.training_path,
        golden_dir=args.golden_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        email=email,
        force_refresh_pubmed=args.force_refresh_pubmed,
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
        push_bioasq_rag_to_hub(
            dataset_dict,
            repo_id=args.push_to_hub,
            token=token,
            private=args.private,
            readme_path=args.readme,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"Uploaded to Hugging Face Hub: {args.push_to_hub}")


if __name__ == "__main__":
    main()
