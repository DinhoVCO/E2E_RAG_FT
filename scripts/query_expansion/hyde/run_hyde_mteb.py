"""Run MTEB retrieval with HyDE averaged query embeddings.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/hyde/run_hyde_mteb.py \\
        --dataset bioasq-resplit \\
        --hyde-dir datasets/query_expansion/hyde/bioasq-resplit/bioasq-resplit-hyde \\
        --model-revision hyde-bioasq-resplit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.query_expansion.hyde.evaluation import HydeMtebEvalConfig, evaluate_hyde_mteb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "mteb" / "hyde"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyDE MTEB retrieval evaluation.")
    parser.add_argument(
        "--dataset",
        choices=("bioasq-resplit", "qasper", "telco-dpr", "narrativeqa"),
        required=True,
    )
    parser.add_argument("--hyde-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--lora-path", default=os.getenv("LORA_PATH"))
    parser.add_argument("--max-lora-rank", type=int, default=int(os.getenv("MAX_LORA_RANK", "16")))
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--backend", choices=("offline", "online"), default="offline")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBED_BATCH_SIZE", "128")))
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", choices=("always", "never", "only-missing", "only-cache"), default="always")
    parser.add_argument("--num-passages", type=int, default=int(os.getenv("HYDE_NUM_PASSAGES", "8")))
    search_scope = parser.add_mutually_exclusive_group()
    search_scope.add_argument("--paper-scoped", action="store_true")
    search_scope.add_argument("--full-corpus", action="store_true")
    parser.add_argument(
        "--no-include-query",
        action="store_true",
        help="Average passages only (do not include the query embedding)",
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

    if args.paper_scoped and args.dataset != "qasper":
        raise SystemExit("--paper-scoped is only supported for --dataset qasper")

    hyde_dir = args.hyde_dir
    if not hyde_dir.is_absolute():
        hyde_dir = PROJECT_ROOT / hyde_dir

    for split in args.splits:
        path = hyde_dir / split / "hyde_passages.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing HyDE passages: {path}")

    config = HydeMtebEvalConfig(
        dataset=args.dataset,
        hyde_dir=hyde_dir,
        model=args.model,
        lora_path=args.lora_path,
        splits=tuple(args.splits),
        backend=args.backend,
        batch_size=args.batch_size,
        model_revision=args.model_revision,
        output_dir=args.output_dir or (DEFAULT_OUTPUT_ROOT / args.dataset),
        overwrite=args.overwrite,
        max_lora_rank=args.max_lora_rank,
        paper_scoped=_resolve_paper_scoped(args),
        include_query=not args.no_include_query,
        num_passages=args.num_passages,
    )

    print("HyDE MTEB evaluation")
    evaluate_hyde_mteb(config)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
