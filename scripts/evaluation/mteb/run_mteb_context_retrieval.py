"""Two-stage MTEB evaluation for context-augmented embedding LoRA adapters.

Stage 1 embeds queries as ``Instruct + Query`` (no context) and retrieves
candidate documents in memory (FAISS), similar to
``retrieve_rag_top_k_inmemory.py``.

Stage 2 rebuilds each query with the top-k stage-1 documents using the same
format as context fine-tuning (``## Context:`` blocks), then runs MTEB
retrieval on the full corpus.

Usage:
    # Evaluate qasper ctx adapter (defaults: stage1-top-k=10, context-k=1 3 5 7 10)
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_context_retrieval.py \\
        --dataset qasper \\
        --lora-path DinoStackAI/Qwen3-Emb-4b-lora-ctx-qasper

    # Local adapter, custom context sizes
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_context_retrieval.py \\
        --dataset bioasq-resplit \\
        --lora-path models/qwen3-embedding-4b-lora-ctx/bioasq-resplit-ctx-b32-e10/final \\
        --context-k 5 10 \\
        --run-label bioasq-resplit-ctx-b32-e10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.embeddings import DEFAULT_EMBED_BATCH_SIZE
from tesis_unicamp.evaluation.mteb.context_retrieval import (
    CONTEXT_DATASET_IDS,
    DEFAULT_CONTEXT_K_VALUES,
    DEFAULT_STAGE1_TOP_K,
    ContextRetrievalEvalConfig,
    evaluate_context_retrieval,
    resolve_paper_scoped,
)
from tesis_unicamp.finetuning.embeddings.config import DEFAULT_BASE_MODEL
from tesis_unicamp.finetuning.embeddings.context.config import (
    MAX_DOC_TOKENS,
    MAX_QUERY_TOKENS,
    MAX_SEQ_LENGTH as CONTEXT_MAX_SEQ_LENGTH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "mteb" / "context"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _validate_cuda_for_offline(backend: str) -> None:
    if backend != "offline":
        return
    if os.getenv("CUDA_VISIBLE_DEVICES", "").strip() == "":
        print(
            "Warning: offline backend uses vLLM and typically requires "
            "CUDA_VISIBLE_DEVICES to be set.",
            file=sys.stderr,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage MTEB retrieval evaluation for context embedding LoRA adapters."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(CONTEXT_DATASET_IDS),
        required=True,
        help="RAG dataset to evaluate.",
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        help="Context LoRA adapter path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base embedding model (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--context-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_K_VALUES),
        metavar="K",
        help=(
            "One or more top-k values for stage 2. Each k uses the first k "
            "documents retrieved in stage 1 to build context-augmented queries "
            f"(default: {' '.join(map(str, DEFAULT_CONTEXT_K_VALUES))})."
        ),
    )
    parser.add_argument(
        "--stage1-top-k",
        type=int,
        default=DEFAULT_STAGE1_TOP_K,
        help=(
            "Documents retrieved per query in stage 1 "
            f"(default: {DEFAULT_STAGE1_TOP_K}; must be >= max(context-k))."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("offline", "online"),
        default="offline",
        help="Embedding backend (default: offline / vLLM).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help=f"Embedding batch size (default: {DEFAULT_EMBED_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=("train", "dev", "test"),
        help="Dataset splits to evaluate (default: test).",
    )
    parser.add_argument(
        "--corpus-split",
        default="train",
        help="Corpus split indexed for stage-1 retrieval (default: train).",
    )
    parser.add_argument(
        "--model-revision-template",
        default="ctx-lora-{dataset}-k{k}",
        help=(
            "MTEB results subfolder template for stage-2 runs "
            "(default: ctx-lora-{dataset}-k{k})."
        ),
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help=(
            "Label for output/cache folders under results/mteb/context/<dataset>/. "
            "Default: last segment of --lora-path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root output directory for stage-1 cache and stage-2 MTEB results "
            f"(default: {DEFAULT_OUTPUT_ROOT}/<dataset>/<run-label>)."
        ),
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=16,
        help="max_lora_rank for vLLM when loading the adapter (default: 16).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=CONTEXT_MAX_SEQ_LENGTH,
        help=f"Max tokens for stage-2 context queries (default: {CONTEXT_MAX_SEQ_LENGTH}).",
    )
    parser.add_argument(
        "--max-query-tokens",
        type=int,
        default=MAX_QUERY_TOKENS,
        help=f"Max query tokens when building stage-2 anchors (default: {MAX_QUERY_TOKENS}).",
    )
    parser.add_argument(
        "--max-doc-tokens",
        type=int,
        default=MAX_DOC_TOKENS,
        help=f"Max tokens per context document (default: {MAX_DOC_TOKENS}).",
    )
    search_scope = parser.add_mutually_exclusive_group()
    search_scope.add_argument(
        "--paper-scoped",
        action="store_true",
        help="QASPER only: restrict stage-1 retrieval to each query's paper chunks.",
    )
    search_scope.add_argument(
        "--full-corpus",
        action="store_true",
        help="Search the full corpus in stage 1 (default for non-qasper datasets).",
    )
    parser.add_argument(
        "--overwrite",
        choices=("always", "never", "only-missing", "only-cache"),
        default="always",
        help="MTEB overwrite strategy for stage 2 (default: always).",
    )
    return parser


def _resolve_paper_scoped(args: argparse.Namespace) -> bool | None:
    if args.paper_scoped:
        return True
    if args.full_corpus:
        return False
    return None


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)

    if args.backend == "offline":
        from tesis_unicamp.evaluation.mteb.runner import configure_vllm_multiprocessing

        configure_vllm_multiprocessing()

    _validate_cuda_for_offline(args.backend)
    if args.paper_scoped and args.dataset != "qasper":
        raise SystemExit("--paper-scoped is only supported for --dataset qasper.")

    run_label = args.run_label or args.lora_path.rstrip("/").rsplit("/", maxsplit=1)[-1]
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.dataset / run_label)

    config = ContextRetrievalEvalConfig(
        dataset=args.dataset,
        lora_path=args.lora_path,
        context_k_values=tuple(args.context_k),
        model=args.model,
        stage1_top_k=args.stage1_top_k,
        splits=tuple(args.splits),
        backend=args.backend,
        batch_size=args.batch_size,
        model_revision_template=args.model_revision_template,
        corpus_split=args.corpus_split,
        paper_scoped=_resolve_paper_scoped(args),
        output_dir=output_dir,
        overwrite=args.overwrite,
        max_lora_rank=args.max_lora_rank,
        max_seq_length=args.max_seq_length,
        max_query_tokens=args.max_query_tokens,
        max_doc_tokens=args.max_doc_tokens,
        run_label=run_label,
    )

    print(f"dataset: {config.dataset}")
    print(f"model: {config.model}")
    print(f"lora_path: {config.lora_path}")
    print(f"stage1_top_k: {config.stage1_top_k}")
    print(f"context_k: {list(config.context_k_values)}")
    print(f"splits: {', '.join(config.splits)}")
    print(f"backend: {config.backend}")
    if config.dataset == "qasper":
        print(f"paper_scoped: {resolve_paper_scoped(config.dataset, config.paper_scoped)}")
    print(f"output_dir: {output_dir}")
    print()

    evaluate_context_retrieval(config)
    print(f"Completed context MTEB evaluation for {len(config.context_k_values)} k value(s).")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
