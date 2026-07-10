"""Retrieve top-k documents in memory (FAISS) for RAG test-split evaluation.

Embeds the corpus into an in-memory FAISS index, retrieves documents for the
selected query splits, and saves ``retrieved_docs`` locally. No Qdrant required.

Usage:
    # Offline (recommended):
    CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory.py \\
        --dataset qasper --mode offline

    # Custom run label (useful when comparing embedding models):
    CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory.py \\
        --dataset telco-dpr --mode offline --run-label vllm-lora-telco-dpr-b128
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.datasets.preprocessing.rag.bioasq.retrieval import (
    retrieve_bioasq_resplit_top_k,
    retrieve_bioasq_top_k,
    save_bioasq_resplit_retrieved_docs,
    save_bioasq_retrieved_docs,
)
from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.retrieval import (
    retrieve_narrativeqa_top_k,
    save_narrativeqa_retrieved_docs,
)
from tesis_unicamp.datasets.preprocessing.rag.qasper.retrieval import (
    retrieve_qasper_paper_scoped_top_k,
    retrieve_qasper_top_k,
    save_qasper_retrieved_docs,
)
from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import RAG_SPLITS
from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.retrieval import (
    retrieve_telco_dpr_top_k,
    save_telco_dpr_retrieved_docs,
)
from tesis_unicamp.datasets.utils import (
    index_bioasq_corpus,
    index_narrativeqa_corpus,
    index_qasper_corpus,
    index_telco_dpr_corpus,
)
from tesis_unicamp.datasets.utils.bioasq_rag import (
    BIOASQ_RAG_DATASET_ID,
    BIOASQ_RAG_RESPLIT_DATASET_ID,
    corpus_row_to_payload as bioasq_corpus_row_to_payload,
    corpus_row_to_point_id as bioasq_corpus_row_to_point_id,
    corpus_row_to_text as bioasq_corpus_row_to_text,
    load_bioasq_rag_resplit_corpus,
)
from tesis_unicamp.datasets.utils.indexing import index_dataset
from tesis_unicamp.datasets.utils.narrativeqa_rag import NARRATIVEQA_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.qasper_rag import QASPER_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.telco_dpr_rag import TELCO_DPR_RAG_DATASET_ID
from tesis_unicamp.embeddings import (
    DEFAULT_EMBED_BATCH_SIZE,
    EmbeddingConfig,
    OpenAIEmbedder,
    VLLMOfflineEmbedder,
)
from tesis_unicamp.vector_stores import InMemoryVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_TOP_K = 10
DEFAULT_RUN_LABEL = "inmemory-default"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "retrieved_inmemory"
DEFAULT_SPLITS = ("test",)


def index_bioasq_resplit_corpus(
    embedder,
    store,
    *,
    split: str = "train",
    batch_size: int | None = None,
    recreate_collection: bool = False,
    show_progress: bool = True,
) -> int:
    corpus = load_bioasq_rag_resplit_corpus(split=split)
    return index_dataset(
        corpus,
        embedder,
        store,
        text_fn=bioasq_corpus_row_to_text,
        id_fn=bioasq_corpus_row_to_point_id,
        payload_fn=bioasq_corpus_row_to_payload,
        batch_size=batch_size,
        recreate_collection=recreate_collection,
        show_progress=show_progress,
    )


@dataclass(frozen=True)
class DatasetRetrievalSpec:
    index_fn: Callable
    retrieve_fn: Callable
    save_fn: Callable
    hub_repo_id: str


DATASET_SPECS: dict[str, DatasetRetrievalSpec] = {
    "bioasq": DatasetRetrievalSpec(
        index_fn=index_bioasq_corpus,
        retrieve_fn=retrieve_bioasq_top_k,
        save_fn=save_bioasq_retrieved_docs,
        hub_repo_id=BIOASQ_RAG_DATASET_ID,
    ),
    "bioasq-resplit": DatasetRetrievalSpec(
        index_fn=index_bioasq_resplit_corpus,
        retrieve_fn=retrieve_bioasq_resplit_top_k,
        save_fn=save_bioasq_resplit_retrieved_docs,
        hub_repo_id=BIOASQ_RAG_RESPLIT_DATASET_ID,
    ),
    "qasper": DatasetRetrievalSpec(
        index_fn=index_qasper_corpus,
        retrieve_fn=retrieve_qasper_top_k,
        save_fn=save_qasper_retrieved_docs,
        hub_repo_id=QASPER_RAG_DATASET_ID,
    ),
    "telco-dpr": DatasetRetrievalSpec(
        index_fn=index_telco_dpr_corpus,
        retrieve_fn=retrieve_telco_dpr_top_k,
        save_fn=save_telco_dpr_retrieved_docs,
        hub_repo_id=TELCO_DPR_RAG_DATASET_ID,
    ),
    "narrativeqa": DatasetRetrievalSpec(
        index_fn=index_narrativeqa_corpus,
        retrieve_fn=retrieve_narrativeqa_top_k,
        save_fn=save_narrativeqa_retrieved_docs,
        hub_repo_id=NARRATIVEQA_RAG_DATASET_ID,
    ),
}


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _sanitize_run_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip())
    return sanitized or DEFAULT_RUN_LABEL


def _default_output_dir(dataset: str, run_label: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / dataset / _sanitize_run_label(run_label)


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


def _build_embedder(
    mode: str,
    model: str,
    batch_size: int,
    *,
    lora_path: str | None = None,
    max_lora_rank: int | None = None,
):
    config = EmbeddingConfig(model=model, batch_size=batch_size)
    if mode == "online":
        if lora_path is not None:
            raise ValueError("--lora-path is only supported with --mode offline.")
        return OpenAIEmbedder(
            config,
            base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        )
    lora_kwargs: dict[str, object] = {}
    if lora_path is not None:
        lora_kwargs["lora_path"] = lora_path
    if max_lora_rank is not None:
        lora_kwargs["max_lora_rank"] = max_lora_rank
    return VLLMOfflineEmbedder(config, **lora_kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Embed a RAG corpus in memory (FAISS), retrieve top-k docs, "
            "and save retrieved_docs for evaluation."
        ),
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
        help=(
            "Embedding model name or Hugging Face repo id "
            f"(default: {DEFAULT_MODEL} or VLLM_MODEL env var)."
        ),
    )
    parser.add_argument(
        "--lora-path",
        default=os.getenv("LORA_PATH"),
        help=(
            "LoRA adapter path or Hugging Face repo id. "
            "Only for --mode offline; --model must be the base model."
        ),
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=int(os.getenv("MAX_LORA_RANK", "16")),
        help="max_lora_rank passed to vLLM when --lora-path is set (default: 16).",
    )
    parser.add_argument(
        "--run-label",
        default=os.getenv("RETRIEVAL_RUN_LABEL", DEFAULT_RUN_LABEL),
        help=(
            "Subfolder name for this retrieval run (default: inmemory-default). "
            "Use it to separate results per embedding model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write JSON + HF export "
            "(default: datasets/retrieved_inmemory/<dataset>/<run-label>)"
        ),
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
        help=f"Embedding batch size (default: {DEFAULT_EMBED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--corpus-split",
        default="train",
        help="Corpus split to embed and index (default: train)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=RAG_SPLITS,
        default=list(DEFAULT_SPLITS),
        help="Query splits to retrieve (default: test)",
    )
    parser.add_argument(
        "--paper-scoped",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "QASPER only: restrict retrieval to each query's paper chunks via "
            "top_ranked (default: on for qasper, off for other datasets)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    spec = DATASET_SPECS[args.dataset]

    paper_scoped = args.paper_scoped
    if paper_scoped is None:
        paper_scoped = args.dataset == "qasper"
    if paper_scoped and args.dataset != "qasper":
        raise SystemExit("--paper-scoped is only supported for --dataset qasper")

    run_label = _sanitize_run_label(args.run_label)
    if paper_scoped and "paper-scoped" not in run_label.lower():
        run_label = f"{run_label}-paper-scoped"
    output_dir = args.output_dir or _default_output_dir(args.dataset, run_label)
    splits = tuple(args.splits)

    _validate_cuda_for_offline(args.mode)
    if args.lora_path and args.mode != "offline":
        raise SystemExit("--lora-path requires --mode offline.")

    embedder = _build_embedder(
        args.mode,
        args.model,
        args.batch_size,
        lora_path=args.lora_path,
        max_lora_rank=args.max_lora_rank if args.lora_path else None,
    )
    store = InMemoryVectorStore(collection_name=f"{args.dataset}-{run_label}")

    print(f"dataset: {args.dataset}")
    print(f"mode: {args.mode}")
    print(f"model: {args.model}")
    if args.lora_path:
        print(f"lora_path: {args.lora_path}")
        print(f"max_lora_rank: {args.max_lora_rank}")
    print(f"run_label: {run_label}")
    print(f"vector_store: InMemoryVectorStore (FAISS IndexFlatIP)")
    print(f"top_k: {args.top_k}")
    print(f"corpus_split: {args.corpus_split}")
    print(f"splits: {', '.join(splits)}")
    if args.dataset == "qasper":
        print(f"paper_scoped: {paper_scoped}")
    print(f"output_dir: {output_dir}")

    indexed = spec.index_fn(
        embedder,
        store,
        split=args.corpus_split,
        batch_size=args.batch_size,
        recreate_collection=True,
        show_progress=True,
    )
    print(f"Indexed {indexed} corpus documents in memory ({store.count()} vectors total)")

    if store.count() == 0:
        raise SystemExit("Corpus index is empty; nothing to retrieve.")

    if paper_scoped:
        retrieved = retrieve_qasper_paper_scoped_top_k(
            embedder,
            store,
            top_k=args.top_k,
            splits=splits,
            batch_size=args.batch_size,
        )
    else:
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
