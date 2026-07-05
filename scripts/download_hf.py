"""Download files from Hugging Face Hub into the local cache.

Run on the **login node** before interactive GPU sessions on Santos Dumont,
where compute-node downloads (especially via XET) often hang.

Usage:
    # All RAG corpus parquet files (for indexing scripts)
    python scripts/download_hf.py --preset rag-corpus

    # One or more RAG datasets
    python scripts/download_hf.py --preset rag-corpus --datasets narrativeqa qasper

    # Custom file(s)
    python scripts/download_hf.py \\
        --repo dinho1597/narrativeqa-rag \\
        --file corpus/train-00000-of-00001.parquet \\
        --repo-type dataset

    # Full model snapshot (e.g. embedding model for offline indexing)
    python scripts/download_hf.py --snapshot --repo Qwen/Qwen3-Embedding-4B
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable

from huggingface_hub import hf_hub_download, snapshot_download

from tesis_unicamp.datasets.utils.bioasq_rag import BIOASQ_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.narrativeqa_rag import NARRATIVEQA_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.qasper_rag import QASPER_RAG_DATASET_ID
from tesis_unicamp.datasets.utils.telco_dpr_rag import TELCO_DPR_RAG_DATASET_ID

RAG_CORPUS_FILE = "corpus/train-00000-of-00001.parquet"

RAG_CORPUS_DATASETS: dict[str, str] = {
    "bioasq": BIOASQ_RAG_DATASET_ID,
    "qasper": QASPER_RAG_DATASET_ID,
    "telco-dpr": TELCO_DPR_RAG_DATASET_ID,
    "narrativeqa": NARRATIVEQA_RAG_DATASET_ID,
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
    unknown = sorted(set(datasets) - set(RAG_CORPUS_DATASETS))
    if unknown:
        valid = ", ".join(sorted(RAG_CORPUS_DATASETS))
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Valid: {valid}")

    for name in datasets:
        repo_id = RAG_CORPUS_DATASETS[name]
        download_files(repo_id, [RAG_CORPUS_FILE], repo_type="dataset")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face Hub files into the local cache (run on login node).",
    )
    parser.add_argument(
        "--preset",
        choices=("rag-corpus",),
        help="Download a known bundle of files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(RAG_CORPUS_DATASETS),
        metavar="DATASET",
        help=(
            "With --preset rag-corpus: which datasets to fetch "
            f"(default: all — {', '.join(RAG_CORPUS_DATASETS)})"
        ),
    )
    parser.add_argument(
        "--repo",
        help="Hub repo id, e.g. dinho1597/narrativeqa-rag or Qwen/Qwen3-Embedding-4B",
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
        datasets = args.datasets or list(RAG_CORPUS_DATASETS)
        download_rag_corpus(datasets)
        return

    if not args.repo:
        raise SystemExit("Pass --preset rag-corpus or --repo with --file/--snapshot.")

    if args.snapshot:
        download_snapshot(args.repo, repo_type=args.repo_type)
        return

    if not args.files:
        raise SystemExit("Pass at least one --file, or use --snapshot for a full repo download.")

    download_files(args.repo, args.files, repo_type=args.repo_type)


if __name__ == "__main__":
    main(sys.argv[1:])
