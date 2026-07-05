"""Index the BioASQ RAG corpus into Qdrant using vLLM embeddings.

Usage:
    # Offline (recommended — single GPU job):
    python scripts/embeddings/index_bioasq_corpus.py --mode offline

    # Online (vLLM server must be running):
    bash jobs/scripts/vllm/serve_embedding_4b.sh
    python scripts/embeddings/index_bioasq_corpus.py --mode online
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.datasets.utils import index_bioasq_corpus
from tesis_unicamp.embeddings import EmbeddingConfig, OpenAIEmbedder, VLLMOfflineEmbedder
from tesis_unicamp.vector_stores import QdrantVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_COLLECTION = "bioasq-rag-13b-corpus"
DEFAULT_BATCH_SIZE = 32


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


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
        description="Index dinho1597/bioasq-rag-13b corpus into Qdrant.",
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
        default=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant REST URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMBED_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        help=f"Texts per embedding batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Corpus split to index (default: train)",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)

    embedder = _build_embedder(args.mode, args.model, args.batch_size)
    store = QdrantVectorStore(args.collection, url=args.qdrant_url)

    print(f"mode: {args.mode}")
    print(f"model: {args.model}")
    print(f"collection: {args.collection}")
    print(f"qdrant_url: {args.qdrant_url}")
    print(f"batch_size: {args.batch_size}")

    indexed = index_bioasq_corpus(
        embedder,
        store,
        split=args.split,
        batch_size=args.batch_size,
        recreate_collection=args.recreate_collection,
    )
    print(f"Indexed {indexed} documents into {args.collection!r} ({store.count()} points total)")


if __name__ == "__main__":
    main(sys.argv[1:])
