"""Two-stage MTEB context evaluation with k=5 and 2048 tokens per context document.

Same prompt format as ``run_mteb_context_retrieval.py`` (Instruct + Query + ## Context:),
but fixed to stage-2 ``context-k=5``, ``max-doc-tokens=2048``, and a larger
``max-seq-length`` so all five documents can use the full 2048-token budget.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_context_retrieval_k5_2k.py \\
        --dataset qasper \\
        --lora-path DinoStackAI/Qwen3-Emb-4b-lora-ctx-qasper
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
    DEFAULT_TITLE_MAX_SEQ_LENGTH,
    ContextRetrievalEvalConfig,
    evaluate_context_retrieval,
    resolve_paper_scoped,
)
from tesis_unicamp.finetuning.embeddings.config import DEFAULT_BASE_MODEL
from tesis_unicamp.finetuning.embeddings.context.config import MAX_QUERY_TOKENS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "mteb" / "context_k5_2k"
DEFAULT_CONTEXT_K = (5,)
DEFAULT_MAX_DOC_TOKENS = 2048
DEFAULT_MODEL_REVISION_TEMPLATE = "ctx-lora-{dataset}-k5-2k"


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
            "Two-stage MTEB context evaluation with k=5 and 2048 tokens per "
            "context document."
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
        help="Run MTEB once with Instruct + Query only (no stage 2).",
    )
    parser.add_argument(
        "--stage1-top-k",
        type=int,
        default=DEFAULT_STAGE1_TOP_K,
        help=f"Stage-1 retrieval top-k (default: {DEFAULT_STAGE1_TOP_K}).",
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
        default="ctx-lora-{dataset}-stage1",
        help="MTEB results subfolder when --stage1-only is set.",
    )
    parser.add_argument(
        "--model-revision-template",
        default=DEFAULT_MODEL_REVISION_TEMPLATE,
        help=(
            "MTEB results subfolder template for stage-2 runs "
            f"(default: {DEFAULT_MODEL_REVISION_TEMPLATE})."
        ),
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help=(
            "Label for output/cache folders under "
            "results/mteb/context_k5_2k/<dataset>/. "
            "Default: last segment of --lora-path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root output directory "
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
            "Max tokens for stage-2 context queries "
            f"(default: {DEFAULT_TITLE_MAX_SEQ_LENGTH})."
        ),
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
        default=DEFAULT_MAX_DOC_TOKENS,
        help=f"Max tokens per context document (default: {DEFAULT_MAX_DOC_TOKENS}).",
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
    if args.stage1_top_k < max(DEFAULT_CONTEXT_K):
        raise SystemExit(
            f"--stage1-top-k ({args.stage1_top_k}) must be >= "
            f"{max(DEFAULT_CONTEXT_K)} for k=5 stage-2 evaluation."
        )

    run_label = args.run_label or args.lora_path.rstrip("/").rsplit("/", maxsplit=1)[-1]
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.dataset / run_label)

    config = ContextRetrievalEvalConfig(
        dataset=args.dataset,
        lora_path=args.lora_path,
        context_k_values=DEFAULT_CONTEXT_K,
        model=args.model,
        stage1_top_k=args.stage1_top_k,
        splits=tuple(args.splits),
        backend=args.backend,
        batch_size=args.batch_size,
        model_revision_template=args.model_revision_template,
        stage1_only_model_revision_template=args.stage1_only_model_revision,
        stage1_only=args.stage1_only,
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
    print(f"stage1_only: {config.stage1_only}")
    if not config.stage1_only:
        print(f"stage1_top_k: {config.stage1_top_k}")
        print(f"context_k: {list(config.context_k_values)}")
        print(f"max_doc_tokens: {config.max_doc_tokens}")
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
        print("Completed stage-1-only context MTEB evaluation (k5-2k runner).")
    else:
        print("Completed context MTEB evaluation for k=5 with 2048 tokens/doc.")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
