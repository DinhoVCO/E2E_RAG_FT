"""Download files from Hugging Face Hub into the local cache.

Run on the **login node** before interactive GPU sessions on Santos Dumont,
where compute-node downloads (especially via XET) often hang.

Usage:
    # Corpus parquet only (for indexing scripts)
    python scripts/download_hf.py --preset rag-corpus --datasets qasper

    # Full dataset repo (corpus, queries, qrels, answers, retrieved_docs)
    python scripts/download_hf.py --preset rag-full --datasets qasper

    # One repo via snapshot
    python scripts/download_hf.py --snapshot --repo DinoStackAI/qasper-rag --repo-type dataset

    # Full model snapshot (e.g. embedding model for offline indexing)
    python scripts/download_hf.py --snapshot --repo Qwen/Qwen3-Embedding-4B --repo-type model

    # RAGAS judge + embedding models (before run_vllm_servers_h100.sh)
    python scripts/download_hf.py --preset ragas-models
    python scripts/download_hf.py --preset ragas-models --models judge
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable

from huggingface_hub import hf_hub_download, snapshot_download

RAG_CORPUS_FILE = "corpus/train-00000-of-00001.parquet"

RAG_DATASET_IDS: dict[str, str] = {
    "bioasq": "DinoStackAI/bioasq-rag-13b",
    "bioasq-resplit": "DinoStackAI/bioasq-rag-13b-resplit",
    "qasper": "DinoStackAI/qasper-rag",
    "telco-dpr": "DinoStackAI/telco-dpr-rag",
    "narrativeqa": "DinoStackAI/narrativeqa-rag",
}

RAGAS_MODEL_IDS: dict[str, str] = {
    "judge": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    "embedding": "Qwen/Qwen3-Embedding-8B",
}


def _configure_xet(disable_xet: bool) -> None:
    if disable_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def download_files(
    repo_id: str,
    files: Iterable[str],
    *,
    repo_type: str,
) -> list[str]:
    paths: list[str] = []
    for filename in files:
        path = hf_hub_download(repo_id, filename, repo_type=repo_type)
        paths.append(path)
        print(f"OK {repo_id}:{filename}\n    -> {path}")
    return paths


def download_snapshot(repo_id: str, *, repo_type: str) -> str:
    path = snapshot_download(repo_id, repo_type=repo_type)
    print(f"OK {repo_id} (snapshot)\n    -> {path}")
    return path


def download_rag_corpus(datasets: Iterable[str]) -> None:
    _validate_datasets(datasets)
    for name in datasets:
        repo_id = RAG_DATASET_IDS[name]
        download_files(repo_id, [RAG_CORPUS_FILE], repo_type="dataset")


def download_rag_full(datasets: Iterable[str]) -> None:
    """Download the full Hub dataset repo (all configs and splits)."""
    _validate_datasets(datasets)
    for name in datasets:
        repo_id = RAG_DATASET_IDS[name]
        download_snapshot(repo_id, repo_type="dataset")


def download_ragas_models(models: Iterable[str]) -> None:
    """Download RAGAS judge and/or embedding model snapshots."""
    _validate_ragas_models(models)
    for name in models:
        repo_id = RAGAS_MODEL_IDS[name]
        download_snapshot(repo_id, repo_type="model")


def _validate_datasets(datasets: Iterable[str]) -> None:
    unknown = sorted(set(datasets) - set(RAG_DATASET_IDS))
    if unknown:
        valid = ", ".join(sorted(RAG_DATASET_IDS))
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Valid: {valid}")


def _validate_ragas_models(models: Iterable[str]) -> None:
    unknown = sorted(set(models) - set(RAGAS_MODEL_IDS))
    if unknown:
        valid = ", ".join(sorted(RAGAS_MODEL_IDS))
        raise SystemExit(f"Unknown RAGAS model(s): {', '.join(unknown)}. Valid: {valid}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face Hub files into the local cache (run on login node).",
    )
    parser.add_argument(
        "--preset",
        choices=("rag-corpus", "rag-full", "ragas-models"),
        help=(
            "rag-corpus: corpus parquet only; rag-full: entire dataset repo; "
            "ragas-models: judge + embedding models for RAGAS/vLLM"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(RAG_DATASET_IDS),
        metavar="DATASET",
        help=(
            "With --preset rag-corpus|rag-full: which datasets to fetch "
            f"(default: all — {', '.join(RAG_DATASET_IDS)})"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(RAGAS_MODEL_IDS),
        metavar="MODEL",
        help=(
            "With --preset ragas-models: which models to fetch "
            f"(default: all — {', '.join(RAGAS_MODEL_IDS)})"
        ),
    )
    parser.add_argument(
        "--repo",
        help="Hub repo id, e.g. DinoStackAI/narrativeqa-rag or Qwen/Qwen3-Embedding-4B",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        metavar="PATH",
        help="File path inside the repo (repeatable). Ignored with --snapshot.",
    )
    parser.add_argument(
        "--repo-type",
        choices=("dataset", "model"),
        default="dataset",
        help="Hub repo type (default: dataset)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Download the full repo snapshot instead of individual --file entries",
    )
    parser.add_argument(
        "--use-xet",
        action="store_true",
        help="Allow XET transfers (disabled by default; XET often hangs on compute nodes)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _configure_xet(disable_xet=not args.use_xet)

    if args.preset == "rag-corpus":
        datasets = args.datasets or list(RAG_DATASET_IDS)
        download_rag_corpus(datasets)
        return

    if args.preset == "rag-full":
        datasets = args.datasets or list(RAG_DATASET_IDS)
        download_rag_full(datasets)
        return

    if args.preset == "ragas-models":
        models = args.models or list(RAGAS_MODEL_IDS)
        download_ragas_models(models)
        return

    if not args.repo:
        raise SystemExit(
            "Pass --preset rag-corpus|rag-full, or --repo with --file/--snapshot."
        )

    if args.snapshot:
        download_snapshot(args.repo, repo_type=args.repo_type)
        return

    if not args.files:
        raise SystemExit("Pass at least one --file, or use --snapshot for a full repo download.")

    download_files(args.repo, args.files, repo_type=args.repo_type)


if __name__ == "__main__":
    main(sys.argv[1:])
