"""Retrieve top-k documents from Qdrant for RAG query splits.

Usage:
    # Offline (recommended):
    CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k.py \\
        --dataset qasper --mode offline

    # One split only:
    python scripts/retrieval/retrieve_rag_top_k.py --dataset bioasq --splits test

    # Push results to Hugging Face (after retrieval):
    python scripts/retrieval/push_retrieved_docs_to_hub.py --dataset qasper
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.datasets.preprocessing.rag.bioasq.retrieval import (
    DEFAULT_QDRANT_COLLECTION as BIOASQ_COLLECTION,
)
from tesis_unicamp.datasets.preprocessing.rag.bioasq.retrieval import (
    DEFAULT_RETRIEVED_DOCS_DIR as BIOASQ_OUTPUT,
)
from tesis_unicamp.datasets.preprocessing.rag.bioasq.retrieval import (
    retrieve_bioasq_top_k,
    save_bioasq_retrieved_docs,
)
from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.retrieval import (
    DEFAULT_QDRANT_COLLECTION as NARRATIVEQA_COLLECTION,
)
from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.retrieval import (
    DEFAULT_RETRIEVED_DOCS_DIR as NARRATIVEQA_OUTPUT,
)
from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.retrieval import (
    retrieve_narrativeqa_top_k,
    save_narrativeqa_retrieved_docs,
)
from tesis_unicamp.datasets.preprocessing.rag.qasper.retrieval import (
    DEFAULT_QDRANT_COLLECTION as QASPER_COLLECTION,
)
from tesis_unicamp.datasets.preprocessing.rag.qasper.retrieval import (
    DEFAULT_RETRIEVED_DOCS_DIR as QASPER_OUTPUT,
)
from tesis_unicamp.datasets.preprocessing.rag.qasper.retrieval import (
    retrieve_qasper_top_k,
    save_qasper_retrieved_docs,
)
from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import RAG_SPLITS
from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.retrieval import (
    DEFAULT_QDRANT_COLLECTION as TELCO_COLLECTION,
)
from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.retrieval import (
    DEFAULT_RETRIEVED_DOCS_DIR as TELCO_OUTPUT,
)
from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.retrieval import (
    retrieve_telco_dpr_top_k,
    save_telco_dpr_retrieved_docs,
)
from tesis_unicamp.datasets.utils.bioasq_rag import BIOASQ_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.narrativeqa_rag import NARRATIVEQA_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.qasper_rag import QASPER_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.telco_dpr_rag import TELCO_DPR_RAG_DATASET_ID
from tesis_unicamp.embeddings import (
    DEFAULT_EMBED_BATCH_SIZE,
    EmbeddingConfig,
    OpenAIEmbedder,
    VLLMOfflineEmbedder,
)
from tesis_unicamp.vector_stores import QdrantVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class DatasetRetrievalSpec:
    retrieve_fn: Callable
    save_fn: Callable
    default_collection: str
    default_output_dir: Path
    hub_repo_id: str


DATASET_SPECS: dict[str, DatasetRetrievalSpec] = {
    "bioasq": DatasetRetrievalSpec(
        retrieve_fn=retrieve_bioasq_top_k,
        save_fn=save_bioasq_retrieved_docs,
        default_collection=BIOASQ_COLLECTION,
        default_output_dir=BIOASQ_OUTPUT,
        hub_repo_id=BIOASQ_RAG_DATASET_ID,
    ),
    "qasper": DatasetRetrievalSpec(
        retrieve_fn=retrieve_qasper_top_k,
        save_fn=save_qasper_retrieved_docs,
        default_collection=QASPER_COLLECTION,
        default_output_dir=QASPER_OUTPUT,
        hub_repo_id=QASPER_RAG_DATASET_ID,
    ),
    "telco-dpr": DatasetRetrievalSpec(
        retrieve_fn=retrieve_telco_dpr_top_k,
        save_fn=save_telco_dpr_retrieved_docs,
        default_collection=TELCO_COLLECTION,
        default_output_dir=TELCO_OUTPUT,
        hub_repo_id=TELCO_DPR_RAG_DATASET_ID,
    ),
    "narrativeqa": DatasetRetrievalSpec(
        retrieve_fn=retrieve_narrativeqa_top_k,
        save_fn=save_narrativeqa_retrieved_docs,
        default_collection=NARRATIVEQA_COLLECTION,
        default_output_dir=NARRATIVEQA_OUTPUT,
        hub_repo_id=NARRATIVEQA_RAG_DATASET_ID,
    ),
}


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _validate_cuda_for_offline(mode: str) -> None:
    if mode != "offline":
        return

    import torch

    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    slurm_visible = os.getenv("SLURM_JOB_GPUS") or os.getenv("SLURM_STEP_GPUS")
    count = torch.cuda.device_count()

    print(f"CUDA_VISIBLE_DEVICES: {visible or '(unset)'}")
    if slurm_visible:
        print(f"SLURM allocated GPU(s): {slurm_visible}")
    print(f"torch.cuda.device_count(): {count}")

    if count == 0:
        raise SystemExit(
            "No CUDA device is visible. Request a GPU in your SLURM session and "
            "avoid overriding CUDA_VISIBLE_DEVICES unless you know the mapped ids."
        )

    try:
        print(f"cuda:0 -> {torch.cuda.get_device_name(0)}")
    except RuntimeError as exc:
        raise SystemExit(
            "Cannot access cuda:0. If you set CUDA_VISIBLE_DEVICES manually, "
            "use an GPU index assigned to your job (usually 0 for the first "
            f"allocated GPU). Original error: {exc}"
        ) from exc


def _build_embedder(mode: str, model: str, batch_size: int):
    config = EmbeddingConfig(model=model, batch_size=batch_size)
    if mode == "online":
        return OpenAIEmbedder(
            config,
            base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        )
    return VLLMOfflineEmbedder(config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k documents from Qdrant for RAG query splits.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_SPECS),
        required=True,
        help="RAG dataset to retrieve for",
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline"),
        required=True,
        help="online: vLLM API server; offline: in-process LLM.embed",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL", DEFAULT_MODEL),
        help=f"Embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name (default: dataset-specific)",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant REST URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory under datasets/retrieved/ to write JSON + HF export",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("RETRIEVAL_TOP_K", DEFAULT_TOP_K)),
        help=f"Documents per query (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE)),
        help=f"Queries per embedding batch (default: {DEFAULT_EMBED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=RAG_SPLITS,
        default=list(RAG_SPLITS),
        help="Query splits to retrieve (default: train dev test)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    spec = DATASET_SPECS[args.dataset]

    collection = args.collection or spec.default_collection
    output_dir = args.output_dir or spec.default_output_dir
    splits = tuple(args.splits)

    _validate_cuda_for_offline(args.mode)
    embedder = _build_embedder(args.mode, args.model, args.batch_size)
    store = QdrantVectorStore(collection, url=args.qdrant_url)

    print(f"dataset: {args.dataset}")
    print(f"mode: {args.mode}")
    print(f"model: {args.model}")
    print(f"collection: {collection}")
    print(f"qdrant_url: {args.qdrant_url}")
    print(f"top_k: {args.top_k}")
    print(f"splits: {', '.join(splits)}")
    print(f"output_dir: {output_dir}")

    if store.count() == 0:
        raise SystemExit(
            f"Collection {collection!r} is empty. Index the corpus first "
            f"(see scripts/embeddings/index_*_corpus.py)."
        )

    retrieved = spec.retrieve_fn(
        embedder,
        store,
        top_k=args.top_k,
        splits=splits,
        batch_size=args.batch_size,
    )
    output_path = spec.save_fn(retrieved, output_dir=output_dir)

    for split in splits:
        count = len(retrieved[split])
        queries = count // args.top_k if args.top_k else 0
        print(f"  {split}: {queries} queries -> {count} retrieved_docs rows")

    print(f"Saved retrieved_docs to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
