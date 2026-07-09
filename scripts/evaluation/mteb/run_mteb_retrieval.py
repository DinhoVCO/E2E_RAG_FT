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
    parser.add_argument(
        "--overwrite",
        choices=("always", "never", "only-missing", "only-cache"),
        default="always",
        help="MTEB overwrite strategy.",
    )
    return parser


def _resolve_tasks(args: argparse.Namespace):
    if args.dataset:
        return [
            get_rag_retrieval_task(
                args.dataset,
                eval_splits=tuple(args.splits),
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
        )
    ]


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)

    if args.backend == "offline":
        from tesis_unicamp.evaluation.mteb.runner import configure_vllm_multiprocessing

        configure_vllm_multiprocessing()

    if args.hf_repo_id and not args.task_name:
        print(f"Using task name: {args.hf_repo_id.rsplit('/', maxsplit=1)[-1]}")

    _validate_cuda_for_offline(args.backend)
    tasks = _resolve_tasks(args)
    model = resolve_model(
        backend=args.backend,
        model=args.model,
        batch_size=args.batch_size,
        model_revision=args.model_revision,
        hf_revision=args.hf_revision,
    )

    task_names = [task.metadata.name for task in tasks]
    print(f"backend: {args.backend}")
    print(f"model: {args.model}")
    print(f"model_revision: {args.model_revision}")
    if args.backend == "sentence-transformers" and not Path(args.model).exists():
        print(f"hf_revision: {args.hf_revision}")
    print(f"tasks: {', '.join(task_names)}")
    print(f"output_dir: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate_retrieval(
        model,
        tasks,
        output_folder=args.output_dir,
        overwrite_strategy=args.overwrite,
        encode_kwargs={"batch_size": args.batch_size},
    )

    for task_result in results:
        print(task_result)


if __name__ == "__main__":
    main(sys.argv[1:])
