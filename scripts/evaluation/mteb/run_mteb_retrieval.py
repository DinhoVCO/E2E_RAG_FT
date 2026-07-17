"""Run MTEB information retrieval evaluation on project or custom RAG datasets.

Usage:
    # Predefined dataset (uses instruct query format by default):
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \\
        --dataset qasper --backend offline --splits test \\
        --model-revision vllm-offline-b128

    # SentenceTransformers model on a built-in MTEB task:
    python scripts/evaluation/mteb/run_mteb_retrieval.py \\
        --mteb-task NFCorpus --backend sentence-transformers \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --model-revision main

    # Custom Hugging Face dataset with corpus/queries/qrels configs:
    python scripts/evaluation/mteb/run_mteb_retrieval.py \\
        --hf-repo-id user/my-rag-dataset --task-name MyRAG \\
        --backend offline --splits test \\
        --model-revision vllm-offline-b128

    # Fine-tuned LoRA adapter with vLLM offline (base model + adapter):
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \\
        --dataset telco-dpr --backend offline --splits test \\
        --model Qwen/Qwen3-Embedding-4B \\
        --lora-path DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr \\
        --model-revision vllm-lora-telco-dpr-b128-e20

    # Custom results revision folder:
    python scripts/evaluation/mteb/run_mteb_retrieval.py \\
        --dataset qasper --backend offline --splits test \\
        --model-revision vllm-offline-b128
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import RAG_SPLITS
from tesis_unicamp.embeddings import DEFAULT_EMBED_BATCH_SIZE
from tesis_unicamp.evaluation.mteb.runner import evaluate_retrieval, resolve_model
from tesis_unicamp.evaluation.mteb.tasks import (
    RAG_RETRIEVAL_TASK_CONFIGS,
    create_custom_rag_retrieval_task,
    get_rag_retrieval_task,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "mteb"


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
        description="Evaluate information retrieval with MTEB.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        choices=sorted(RAG_RETRIEVAL_TASK_CONFIGS),
        help="Predefined project RAG dataset.",
    )
    source.add_argument(
        "--mteb-task",
        help="Built-in MTEB retrieval task name, e.g. NFCorpus or ArguAna.",
    )
    source.add_argument(
        "--hf-repo-id",
        help="Custom Hugging Face dataset repo with corpus/queries/qrels configs.",
    )

    parser.add_argument(
        "--task-name",
        help="Task name for --hf-repo-id. Defaults to the last segment of the repo id.",
    )
    parser.add_argument(
        "--task-description",
        default="Custom RAG retrieval dataset.",
        help="Description used when --hf-repo-id is provided.",
    )
    parser.add_argument(
        "--backend",
        choices=("offline", "online", "sentence-transformers"),
        default="offline",
        help="Embedding backend.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Embedding model name or Hugging Face repo id.",
    )
    parser.add_argument(
        "--model-revision",
        required=True,
        help=(
            "Label for MTEB results (subfolder name under results/mteb/results/). "
            "Example: lora-bioasq-resplit-b128-e10 or vllm-offline-b128."
        ),
    )
    parser.add_argument(
        "--hf-revision",
        default="main",
        help=(
            "Hugging Face git revision used when loading --model from the Hub "
            "(sentence-transformers backend only; default: main)."
        ),
    )
    parser.add_argument(
        "--lora-path",
        help=(
            "LoRA adapter path or Hugging Face repo id. "
            "Only for --backend offline; --model must be the base model."
        ),
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=16,
        help="max_lora_rank passed to vLLM when --lora-path is set (default: 16).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help="Embedding batch size for project backends.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=RAG_SPLITS,
        default=["test"],
        help="Dataset splits to evaluate for custom/project RAG tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where MTEB stores evaluation results.",
    )
    parser.add_argument(
        "--raw-queries",
        action="store_true",
        help="Do not apply the instruct query format used by project retrieval scripts.",
    )
    search_scope = parser.add_mutually_exclusive_group()
    search_scope.add_argument(
        "--paper-scoped",
        action="store_true",
        help="Restrict retrieval to each query's paper chunks via top_ranked.",
    )
    search_scope.add_argument(
        "--full-corpus",
        action="store_true",
        help="Search the full shared corpus instead of paper-scoped top_ranked.",
    )
    parser.add_argument(
        "--include-query-title",
        action="store_true",
        help=(
            "Prepend gold document title (## Title:) to each query before embedding, "
            "matching retrieve_rag_top_k_inmemory_title.py."
        ),
    )
    parser.add_argument(
        "--overwrite",
        choices=("always", "never", "only-missing", "only-cache"),
        default="always",
        help="MTEB overwrite strategy.",
    )
    return parser


def _resolve_use_top_ranked(args: argparse.Namespace) -> bool | None:
    if args.paper_scoped:
        return True
    if args.full_corpus:
        return False
    return None


def _resolve_tasks(args: argparse.Namespace):
    use_top_ranked = _resolve_use_top_ranked(args)

    if args.dataset:
        return [
            get_rag_retrieval_task(
                args.dataset,
                eval_splits=tuple(args.splits),
                use_top_ranked=use_top_ranked,
                include_query_title=args.include_query_title,
            )
        ]

    if args.mteb_task:
        import mteb

        return [mteb.get_task(args.mteb_task)]

    task_name = args.task_name
    if not task_name:
        task_name = args.hf_repo_id.rsplit("/", maxsplit=1)[-1]

    from tesis_unicamp.datasets.utils.bioasq_rag import query_to_instruct_text

    return [
        create_custom_rag_retrieval_task(
            name=task_name,
            hf_repo_id=args.hf_repo_id,
            description=args.task_description,
            eval_splits=tuple(args.splits),
            query_text_fn=None if args.raw_queries else query_to_instruct_text,
            use_top_ranked=bool(use_top_ranked),
        )
    ]


def main(argv: list[str] | None = None) -> None:
    _load_env()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.backend == "offline":
        from tesis_unicamp.evaluation.mteb.runner import configure_vllm_multiprocessing

        configure_vllm_multiprocessing()

    if args.hf_repo_id and not args.task_name:
        print(f"Using task name: {args.hf_repo_id.rsplit('/', maxsplit=1)[-1]}")

    if args.lora_path and args.backend != "offline":
        parser.error("--lora-path requires --backend offline.")

    _validate_cuda_for_offline(args.backend)
    tasks = _resolve_tasks(args)
    model = resolve_model(
        backend=args.backend,
        model=args.model,
        batch_size=args.batch_size,
        model_revision=args.model_revision,
        hf_revision=args.hf_revision,
        lora_path=args.lora_path,
        max_lora_rank=args.max_lora_rank if args.lora_path else None,
    )

    task_names = [task.metadata.name for task in tasks]
    print(f"backend: {args.backend}")
    print(f"model: {args.model}")
    if args.lora_path:
        print(f"lora_path: {args.lora_path}")
    print(f"model_revision: {args.model_revision}")
    if args.backend == "sentence-transformers" and not Path(args.model).exists():
        print(f"hf_revision: {args.hf_revision}")
    print(f"tasks: {', '.join(task_names)}")
    print(f"output_dir: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encode_kwargs: dict[str, object] = {"batch_size": args.batch_size}
    if args.backend == "sentence-transformers":
        # MTEB's task progress bar stays at 0/1 while the full corpus is encoded;
        # ST's internal batch bar makes long LoRA runs visible (~3 min for telco-dpr).
        encode_kwargs["show_progress_bar"] = True
        print(
            "Note: sentence-transformers encodes the full retrieval corpus before "
            "MTEB advances the task bar. Watch the 'Batches:' progress below.",
            flush=True,
        )

    results = evaluate_retrieval(
        model,
        tasks,
        output_folder=args.output_dir,
        overwrite_strategy=args.overwrite,
        encode_kwargs=encode_kwargs,
    )

    for task_result in results:
        print(task_result)


if __name__ == "__main__":
    main(sys.argv[1:])
