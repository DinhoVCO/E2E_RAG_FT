"""Fine-tune Qwen3-Embedding-4B with LoRA on a RAG retrieval dataset.

Each dataset is fine-tuned individually using CachedMultipleNegativesRankingLoss,
InformationRetrievalEvaluator, and Weights & Biases logging.

Usage:
    # Single dataset (from repo root):
    CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \\
        --dataset qasper

    # All four datasets sequentially:
    bash jobs/scripts/finetuning/finetune_qwen3_embedding_all.sh

    # Custom output and training schedule:
    CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \\
        --dataset bioasq-resplit \\
        --output-dir models/qwen3-embedding-4b-lora/bioasq-resplit \\
        --epochs 3 \\
        --batch-size 128 \\
        --eval-steps 250 \\
        --save-steps 250
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.finetuning.embeddings.config import (
    DEFAULT_BASE_MODEL,
    EMBEDDING_FINETUNING_DATASET_IDS,
    MINI_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
)
from tesis_unicamp.finetuning.embeddings.trainer import FinetuningRunConfig, finetune_qwen3_embedding

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "qwen3-embedding-4b-lora"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _warn_if_multi_gpu() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    gpu_count = torch.cuda.device_count()
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "(unset)")
    print(f"visible_gpus: {gpu_count} (CUDA_VISIBLE_DEVICES={visible})")
    if gpu_count > 1:
        print(
            "Warning: multiple GPUs are visible; SentenceTransformerTrainer will "
            "use DataParallel and increase memory pressure. For one dataset per GPU, "
            "set CUDA_VISIBLE_DEVICES to a single id (e.g. CUDA_VISIBLE_DEVICES=0).",
            file=sys.stderr,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-Embedding-4B with LoRA on a RAG dataset.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(EMBEDDING_FINETUNING_DATASET_IDS),
        required=True,
        help="RAG dataset to fine-tune on (one run per dataset).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base embedding model (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints and final adapter (default: models/qwen3-embedding-4b-lora/<dataset>).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN_BATCH_SIZE,
        help=f"Per-device train/eval batch size (default: {TRAIN_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save a checkpoint every N steps (default: 500).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log metrics every N steps (default: 50).",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep (default: 3).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch size for InformationRetrievalEvaluator (default: 32).",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=MINI_BATCH_SIZE,
        help=(
            "Encode mini-batch size inside CachedMultipleNegativesRankingLoss "
            f"(default: {MINI_BATCH_SIZE}). Lower this if you hit OOM."
        ),
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Split used to build (query, document) training pairs (default: train).",
    )
    parser.add_argument(
        "--eval-split",
        default="dev",
        help="Split used by InformationRetrievalEvaluator (default: dev).",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name (default: WANDB_PROJECT from .env or qwen3-embedding-finetuning).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B run name (default: qwen3-embedding-4b-lora-<dataset>).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 training (default: disabled).",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable BF16 training (enabled by default on supported GPUs).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    _warn_if_multi_gpu()
    args = _build_parser().parse_args(argv)

    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.dataset)
    wandb_project = args.wandb_project or os.getenv(
        "WANDB_PROJECT",
        "qwen3-embedding-finetuning",
    )

    run_config = FinetuningRunConfig(
        dataset=args.dataset,
        output_dir=output_dir,
        model_name=args.model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        eval_batch_size=args.eval_batch_size,
        mini_batch_size=args.mini_batch_size,
        train_split=args.train_split,
        eval_split=args.eval_split,
        wandb_project=wandb_project,
        run_name=args.run_name,
        fp16=args.fp16,
        bf16=not args.no_bf16,
    )

    print(f"dataset: {args.dataset}")
    print(f"hub_repo: {EMBEDDING_FINETUNING_DATASET_IDS[args.dataset]}")
    print(f"model: {args.model}")
    print(f"output_dir: {output_dir}")
    print(f"batch_size: {args.batch_size}")
    print(f"mini_batch_size: {args.mini_batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"train_split: {args.train_split}")
    print(f"eval_split: {args.eval_split}")
    print(f"wandb_project: {wandb_project}")

    final_dir = finetune_qwen3_embedding(run_config)
    print(f"Training complete. Final model saved to: {final_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
