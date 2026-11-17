"""Stage 2 MTEB retrieval evaluation for RFG expanded queries.

Embeds the long-form expanded queries (from stage 1 + generation) and
evaluates retrieval against the full corpus using MTEB metrics.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_stage2_mteb.py \\
        --dataset qasper \\
        --expanded-queries-dir datasets/query_expansion/qasper/qasper-rfg-emb-base-gen-base \\
        --model-revision rfg-qasper-emb-base-gen-base

    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_stage2_mteb.py \\
        --dataset telco-dpr \\
        --lora-path DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr \\
        --expanded-queries-dir datasets/query_expansion/telco-dpr/telco-dpr-rfg-emb-lora-gen-base \\
        --model-revision rfg-telco-dpr-emb-lora-gen-base
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.query_expansion.stage2_mteb import (
    RfgStage2MtebConfig,
    evaluate_rfg_stage2_mteb,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "mteb" / "rfg"
DEFAULT_SPLIT = "test"


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
        description="RFG stage 2: MTEB retrieval with expanded queries.",
    )
    parser.add_argument(
        "--dataset",
        choices=("bioasq-resplit", "qasper", "telco-dpr", "narrativeqa"),
        required=True,
        help="RAG dataset to evaluate",
    )
    parser.add_argument(
        "--expanded-queries-dir",
        type=Path,
        required=True,
        help="Directory with <split>/expanded_queries.json from stage 1+generation",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL", DEFAULT_MODEL),
        help=f"Embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--lora-path",
        default=os.getenv("LORA_PATH"),
        help="Embedding LoRA adapter (same as stage 1)",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=int(os.getenv("MAX_LORA_RANK", "16")),
        help="max_lora_rank for vLLM (default: 16)",
    )
    parser.add_argument(
        "--model-revision",
        required=True,
        help="MTEB results subfolder label",
    )
    parser.add_argument(
        "--backend",
        choices=("offline", "online"),
        default="offline",
        help="Embedding backend (default: offline)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMBED_BATCH_SIZE", "128")),
        help="Embedding batch size (default: 128)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[DEFAULT_SPLIT],
        choices=("train", "dev", "test"),
        help="Dataset splits to evaluate (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"MTEB output root (default: {DEFAULT_OUTPUT_ROOT}/<dataset>)",
    )
    parser.add_argument(
        "--overwrite",
        choices=("always", "never", "only-missing", "only-cache"),
        default="always",
        help="MTEB overwrite strategy (default: always)",
    )
    search_scope = parser.add_mutually_exclusive_group()
    search_scope.add_argument(
        "--paper-scoped",
        action="store_true",
        help="QASPER only: restrict retrieval to paper chunks",
    )
    search_scope.add_argument(
        "--full-corpus",
        action="store_true",
        help="Search the full corpus (default for non-qasper)",
    )
    parser.add_argument(
        "--raw-queries",
        action="store_true",
        help="Do not apply Instruct query format to expanded queries",
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

    expanded_dir = args.expanded_queries_dir
    if not expanded_dir.is_absolute():
        expanded_dir = PROJECT_ROOT / expanded_dir

    for split in args.splits:
        path = expanded_dir / split / "expanded_queries.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing expanded queries: {path}")

    _validate_cuda_for_offline(args.backend)

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.dataset)

    config = RfgStage2MtebConfig(
        dataset=args.dataset,
        expanded_queries_dir=expanded_dir,
        model=args.model,
        lora_path=args.lora_path,
        splits=tuple(args.splits),
        backend=args.backend,
        batch_size=args.batch_size,
        model_revision=args.model_revision,
        output_dir=output_dir,
        overwrite=args.overwrite,
        max_lora_rank=args.max_lora_rank,
        paper_scoped=_resolve_paper_scoped(args),
        apply_instruct_format=not args.raw_queries,
    )

    print("RFG stage 2 MTEB evaluation")
    print(f"dataset: {config.dataset}")
    print(f"model: {config.model}")
    if config.lora_path:
        print(f"lora_path: {config.lora_path}")
    print(f"expanded_queries_dir: {expanded_dir}")
    print(f"model_revision: {config.model_revision}")
    print(f"output_dir: {output_dir}")
    print()

    evaluate_rfg_stage2_mteb(config)
    print("Completed RFG stage 2 MTEB evaluation.")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
