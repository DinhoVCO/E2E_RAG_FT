"""Stage 1 retrieval for RFG (top-k documents for query expansion).

Thin wrapper around ``retrieve_rag_top_k_inmemory.py``. By default writes to
``datasets/retrieved_inmemory/`` (same location as the RAG pipeline), so
existing base/LoRA retrievals are reused when present.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_stage1_retrieval.py \\
        --dataset qasper --mode offline --run-label vllm-offline-b128

    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_stage1_retrieval.py \\
        --dataset telco-dpr --mode offline \\
        --lora-path DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr \\
        --run-label vllm-lora-telco-dpr-b128
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_SCRIPT = PROJECT_ROOT / "scripts" / "retrieval" / "retrieve_rag_top_k_inmemory.py"
DEFAULT_RUN_LABEL = "vllm-offline-b128"
DEFAULT_STAGE1_TOP_K = 5


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RFG stage 1: retrieve top-k docs for query expansion.",
    )
    parser.add_argument(
        "--dataset",
        choices=("bioasq", "bioasq-resplit", "qasper", "telco-dpr", "narrativeqa"),
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline"),
        required=True,
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL", "Qwen/Qwen3-Embedding-4B"),
    )
    parser.add_argument(
        "--lora-path",
        default=os.getenv("LORA_PATH"),
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=int(os.getenv("MAX_LORA_RANK", "16")),
    )
    parser.add_argument(
        "--run-label",
        default=os.getenv("RFG_RETRIEVAL_RUN_LABEL", DEFAULT_RUN_LABEL),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("RFG_STAGE1_TOP_K", DEFAULT_STAGE1_TOP_K)),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMBED_BATCH_SIZE", "128")),
    )
    parser.add_argument(
        "--corpus-split",
        default="train",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=("train", "dev", "test"),
    )
    parser.add_argument(
        "--paper-scoped",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser


def _retrieval_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(RETRIEVAL_SCRIPT),
        "--dataset",
        args.dataset,
        "--mode",
        args.mode,
        "--model",
        args.model,
        "--top-k",
        str(args.top_k),
        "--batch-size",
        str(args.batch_size),
        "--run-label",
        args.run_label,
        "--corpus-split",
        args.corpus_split,
        "--splits",
        *args.splits,
    ]
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    if args.lora_path:
        command.extend(["--lora-path", args.lora_path, "--max-lora-rank", str(args.max_lora_rank)])
    if args.dataset == "qasper":
        paper_scoped = args.paper_scoped if args.paper_scoped is not None else True
        command.append("--paper-scoped" if paper_scoped else "--no-paper-scoped")
    return command


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    command = _retrieval_command(args)
    print(f"$ {' '.join(command)}")
    if args.dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
