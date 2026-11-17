"""Two-stage MTEB evaluation for context LoRA adapters with title-aware queries.

Stage 1 embeds queries as ``Instruct + ## Title: + Query`` (gold document title
from qrels), matching ``retrieve_rag_top_k_inmemory_title.py``.

Stage 2 rebuilds each query with the top-k stage-1 documents using the same
generative title RAG format as ``run_rag_experiment.py`` with
``experiments_title.yaml`` (``## Title:``, separate doc title/body,
2048-token chunk truncation).

Usage:
    # Stage 1 only
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_context_retrieval_title.py \\
        --dataset qasper \\
        --lora-path DinoStackAI/Qwen3-Emb-4b-lora-ctx-qasper \\
        --stage1-only

    # Full two-stage with default context sizes (k=1, 3, 5)
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_context_retrieval_title.py \\
        --dataset bioasq-resplit \\
        --lora-path models/qwen3-embedding-4b-lora-ctx/bioasq-resplit-ctx-b32-e10/final \\
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
    DEFAULT_STAGE1_TOP_K,
    DEFAULT_TITLE_CONTEXT_K_VALUES,
    DEFAULT_TITLE_MAX_SEQ_LENGTH,
    DEFAULT_TITLE_MAX_TOKENS_PER_CHUNK,
    DEFAULT_TITLE_MODEL_REVISION_TEMPLATE,
    DEFAULT_TITLE_NOTRUNC_MODEL_REVISION_TEMPLATE,
    DEFAULT_TITLE_NOTRUNC_OUTPUT_ROOT,
    DEFAULT_TITLE_OUTPUT_ROOT,
    DEFAULT_TITLE_STAGE1_ONLY_MODEL_REVISION_TEMPLATE,
    ContextRetrievalEvalConfig,
    evaluate_context_retrieval,
    resolve_paper_scoped,
)
from tesis_unicamp.finetuning.embeddings.config import DEFAULT_BASE_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / DEFAULT_TITLE_OUTPUT_ROOT
DEFAULT_NOTRUNC_OUTPUT_ROOT = PROJECT_ROOT / DEFAULT_TITLE_NOTRUNC_OUTPUT_ROOT


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
            "Two-stage MTEB retrieval evaluation for context LoRA adapters "
            "with title-aware stage-1 queries and generative title stage-2 prompts."
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
        "--stage1-only",
        action="store_true",
        help=(
            "Run MTEB once with Instruct + ## Title: + Query only (no context). "
            "Skips FAISS pre-retrieval and stage 2."
        ),
    )
    parser.add_argument(
        "--context-k",
        type=int,
        nargs="+",
        default=list(DEFAULT_TITLE_CONTEXT_K_VALUES),
        metavar="K",
        help=(
            "Top-k values for stage 2. Each k uses the first k stage-1 documents "
            "to build title-aware context queries "
            f"(default: {' '.join(map(str, DEFAULT_TITLE_CONTEXT_K_VALUES))})."
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
        "--stage1-only-model-revision",
        default=DEFAULT_TITLE_STAGE1_ONLY_MODEL_REVISION_TEMPLATE,
        help=(
            "MTEB results subfolder when --stage1-only is set "
            f"(default: {DEFAULT_TITLE_STAGE1_ONLY_MODEL_REVISION_TEMPLATE})."
        ),
    )
    parser.add_argument(
        "--model-revision-template",
        default=DEFAULT_TITLE_MODEL_REVISION_TEMPLATE,
        help=(
            "MTEB results subfolder template for stage-2 runs "
            f"(default: {DEFAULT_TITLE_MODEL_REVISION_TEMPLATE})."
        ),
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help=(
            "Label for output/cache folders under results/mteb/context_title/<dataset>/. "
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
        default=DEFAULT_TITLE_MAX_SEQ_LENGTH,
        help=(
            "Max tokens for stage-2 title context queries "
            f"(default: {DEFAULT_TITLE_MAX_SEQ_LENGTH}, same as generation title experiments)."
        ),
    )
    parser.add_argument(
        "--max-tokens-per-chunk",
        type=int,
        default=DEFAULT_TITLE_MAX_TOKENS_PER_CHUNK,
        help=(
            "Max tokens per retrieved document body in stage 2 "
            f"(default: {DEFAULT_TITLE_MAX_TOKENS_PER_CHUNK}; "
            "ignored with --no-truncate-stage2-docs)."
        ),
    )
    parser.add_argument(
        "--no-truncate-stage2-docs",
        action="store_true",
        help=(
            "Use full document bodies in stage 2 (no per-chunk truncation and no "
            "max-seq-length trimming). Results go to results/mteb/context_title_notrunc/."
        ),
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
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.no_truncate_stage2_docs:
        output_dir = DEFAULT_NOTRUNC_OUTPUT_ROOT / args.dataset / run_label
    else:
        output_dir = DEFAULT_OUTPUT_ROOT / args.dataset / run_label

    model_revision_template = args.model_revision_template
    if (
        args.no_truncate_stage2_docs
        and model_revision_template == DEFAULT_TITLE_MODEL_REVISION_TEMPLATE
    ):
        model_revision_template = DEFAULT_TITLE_NOTRUNC_MODEL_REVISION_TEMPLATE

    config = ContextRetrievalEvalConfig(
        dataset=args.dataset,
        lora_path=args.lora_path,
        context_k_values=tuple(args.context_k),
        model=args.model,
        stage1_top_k=args.stage1_top_k,
        splits=tuple(args.splits),
        backend=args.backend,
        batch_size=args.batch_size,
        model_revision_template=model_revision_template,
        stage1_only_model_revision_template=args.stage1_only_model_revision,
        stage1_only=args.stage1_only,
        corpus_split=args.corpus_split,
        paper_scoped=_resolve_paper_scoped(args),
        output_dir=output_dir,
        overwrite=args.overwrite,
        max_lora_rank=args.max_lora_rank,
        max_seq_length=args.max_seq_length,
        include_query_title=True,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        truncate_stage2_docs=not args.no_truncate_stage2_docs,
        run_label=run_label,
    )

    print(f"dataset: {config.dataset}")
    print(f"model: {config.model}")
    print(f"lora_path: {config.lora_path}")
    print(f"include_query_title: {config.include_query_title}")
    print(f"stage1_only: {config.stage1_only}")
    if not config.stage1_only:
        print(f"stage1_top_k: {config.stage1_top_k}")
        print(f"context_k: {list(config.context_k_values)}")
        print(f"truncate_stage2_docs: {config.truncate_stage2_docs}")
        if config.truncate_stage2_docs:
            print(f"max_tokens_per_chunk: {config.max_tokens_per_chunk}")
            print(f"max_seq_length: {config.max_seq_length}")
    else:
        print(
            "model_revision: "
            f"{args.stage1_only_model_revision.format(dataset=config.dataset)}"
        )
    print(f"splits: {', '.join(config.splits)}")
    print(f"backend: {config.backend}")
    if config.dataset == "qasper":
        print(f"paper_scoped: {resolve_paper_scoped(config.dataset, config.paper_scoped)}")
    print(f"output_dir: {output_dir}")
    print()

    evaluate_context_retrieval(config)
    if config.stage1_only:
        print("Completed stage-1-only title context MTEB evaluation.")
    else:
        print(
            f"Completed title context MTEB evaluation for "
            f"{len(config.context_k_values)} k value(s)."
        )


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
