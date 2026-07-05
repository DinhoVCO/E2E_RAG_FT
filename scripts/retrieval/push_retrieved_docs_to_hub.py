"""Push retrieved_docs subset to Hugging Face Hub.

Usage:
    python scripts/retrieval/push_retrieved_docs_to_hub.py --dataset qasper
    python scripts/retrieval/push_retrieved_docs_to_hub.py --dataset bioasq --private
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.datasets.preprocessing.rag.bioasq.retrieval import (
    DEFAULT_RETRIEVED_DOCS_DIR as BIOASQ_OUTPUT,
    push_bioasq_retrieved_docs_to_hub,
)
from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.retrieval import (
    DEFAULT_RETRIEVED_DOCS_DIR as NARRATIVEQA_OUTPUT,
    push_narrativeqa_retrieved_docs_to_hub,
)
from tesis_unicamp.datasets.preprocessing.rag.qasper.retrieval import (
    DEFAULT_RETRIEVED_DOCS_DIR as QASPER_OUTPUT,
    push_qasper_retrieved_docs_to_hub,
)
from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.retrieval import (
    DEFAULT_RETRIEVED_DOCS_DIR as TELCO_OUTPUT,
    push_telco_dpr_retrieved_docs_to_hub,
)
from tesis_unicamp.datasets.utils.bioasq_rag import BIOASQ_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.narrativeqa_rag import NARRATIVEQA_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.qasper_rag import QASPER_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.telco_dpr_rag import TELCO_DPR_RAG_DATASET_ID

PROJECT_ROOT = Path(__file__).resolve().parents[2]


PUSH_HANDLERS = {
    "bioasq": (push_bioasq_retrieved_docs_to_hub, BIOASQ_OUTPUT, BIOASQ_RAG_DATASET_ID),
    "qasper": (push_qasper_retrieved_docs_to_hub, QASPER_OUTPUT, QASPER_RAG_DATASET_ID),
    "telco-dpr": (push_telco_dpr_retrieved_docs_to_hub, TELCO_OUTPUT, TELCO_DPR_RAG_DATASET_ID),
    "narrativeqa": (
        push_narrativeqa_retrieved_docs_to_hub,
        NARRATIVEQA_OUTPUT,
        NARRATIVEQA_RAG_DATASET_ID,
    ),
}


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload retrieved_docs config to Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(PUSH_HANDLERS),
        required=True,
        help="RAG dataset whose retrieved_docs to upload",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo (default: dinho1597/<dataset>-rag)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local retrieved_docs directory (default: datasets/retrieved/<dataset>)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token (default: HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload as a private dataset",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)

    push_fn, default_output, default_repo = PUSH_HANDLERS[args.dataset]
    output_dir = args.output_dir or default_output
    repo_id = args.repo_id or default_repo

    print(f"dataset: {args.dataset}")
    print(f"repo_id: {repo_id}")
    print(f"output_dir: {output_dir}")

    push_fn(
        repo_id=repo_id,
        output_dir=output_dir,
        token=args.token,
        private=args.private,
    )
    print(f"Pushed retrieved_docs to {repo_id!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
