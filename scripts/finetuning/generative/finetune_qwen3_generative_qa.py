"""Fine-tune Qwen3-8B with LoRA on question-answer pairs (no RAG context).

Uses only the queries and answers train/dev splits from each RAG dataset.
Trainer, LoRA, early stopping, and W&B logging are shared with the RAG script.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/generative/finetune_qwen3_generative_qa.py \\
        --dataset qasper

    bash jobs/scripts/finetuning/finetune_qwen3_generative_qa.sh narrativeqa
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.finetuning.generative.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_METRIC_FOR_BEST_MODEL,
    EARLY_STOPPING_PATIENCE,
    GENERATIVE_FINETUNING_DATASET_IDS,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_QA_SEQ_LENGTH,
    TELCO_DPR_NUM_TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
)
from tesis_unicamp.finetuning.generative.trainer import (
    GenerativeFinetuningRunConfig,
    default_log_path,
    finetune_qwen3_generative,
    resolve_output_dir,
    resolve_run_name,
)
from tesis_unicamp.finetuning.generative.yaml_config import (
    load_finetuning_yaml,
    resolve_config_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "qwen3-8b-lora-qa"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs" / "qa"


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data: str) -> None:
        for handle in self.files:
            handle.write(data)
            handle.flush()

    def flush(self) -> None:
        for handle in self.files:
            handle.flush()

    def isatty(self) -> bool:
        return self.files[0].isatty() if self.files else False

    def fileno(self) -> int:
        return self.files[0].fileno()


def _setup_log_file(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_handle)
    sys.stderr = _Tee(sys.__stderr__, log_handle)


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _default_epochs(dataset: str) -> int:
    if dataset == "telco-dpr":
        return TELCO_DPR_NUM_TRAIN_EPOCHS
    return 3


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
            "Warning: multiple GPUs are visible; training will use device_map=auto. "
            "For one dataset per GPU, set CUDA_VISIBLE_DEVICES to a single id.",
            file=sys.stderr,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-8B with LoRA on QA pairs (no context).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML file with training hyperparameters. "
            f"Default: configs/qa/<dataset>.yaml when present ({CONFIGS_DIR})."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(GENERATIVE_FINETUNING_DATASET_IDS),
        required=True,
        help="Dataset to fine-tune on (one run per dataset).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base generative model (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for checkpoints and final adapter "
            "(default: models/qwen3-8b-lora-qa/<run_name>)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 3, telco-dpr: 6).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN_BATCH_SIZE,
        help=f"Per-device train/eval batch size (default: {TRAIN_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=(
            "Gradient accumulation steps "
            f"(default: {GRADIENT_ACCUMULATION_STEPS})."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio (default: 0.03).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate every N steps (default: 100).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save a checkpoint every N steps (default: 100).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log metrics every N steps (default: 10).",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep (default: 3).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_QA_SEQ_LENGTH,
        help=f"Maximum sequence length (default: {MAX_QA_SEQ_LENGTH}).",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Split used to build training examples (default: train).",
    )
    parser.add_argument(
        "--eval-split",
        default="dev",
        help="Split used for evaluation (default: dev).",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name (default: WANDB_PROJECT_STEP2 from .env).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Run name for W&B, model output folder, and log file "
            "(default: qwen3-8b-lora-qa-<dataset>-b<batch>-e<epochs>)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path (default: logs/<run_name>.log).",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable automatic logging to logs/<run_name>.log.",
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
    parser.add_argument(
        "--no-load-best-model",
        action="store_true",
        help=(
            "Disable automatic best-checkpoint selection "
            f"(default: pick best by {DEFAULT_METRIC_FOR_BEST_MODEL})."
        ),
    )
    parser.add_argument(
        "--metric-for-best-model",
        default=DEFAULT_METRIC_FOR_BEST_MODEL,
        help=f"Metric for best checkpoint selection (default: {DEFAULT_METRIC_FOR_BEST_MODEL}).",
    )
    parser.add_argument(
        "--greater-is-better",
        action="store_true",
        help="Treat metric_for_best_model as higher-is-better (default: False for eval_loss).",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help=(
            "Disable early stopping "
            f"(default: patience={EARLY_STOPPING_PATIENCE} eval steps)."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f"Early stopping patience in eval steps (default: {EARLY_STOPPING_PATIENCE}).",
    )
    return parser


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--dataset",
        choices=sorted(GENERATIVE_FINETUNING_DATASET_IDS),
    )
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = resolve_config_path(
        dataset=pre_args.dataset,
        config=pre_args.config,
        configs_dir=CONFIGS_DIR,
    )

    parser = _build_parser()
    yaml_defaults: dict = {}
    if config_path is not None:
        yaml_defaults = load_finetuning_yaml(config_path)
        parser.set_defaults(**yaml_defaults)

    args = parser.parse_args(argv)

    if config_path is not None:
        args.config = config_path
        yaml_dataset = yaml_defaults.get("dataset")
        if yaml_dataset is not None and yaml_dataset != args.dataset:
            parser.error(
                f"Dataset mismatch: --dataset {args.dataset!r} "
                f"does not match {yaml_dataset!r} in {config_path}"
            )

    if args.epochs is None:
        args.epochs = _default_epochs(args.dataset)

    return args


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _parse_args(argv)

    resolved_run_name = resolve_run_name(
        dataset=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        run_name=args.run_name,
        qa_only=True,
    )
    output_dir = resolve_output_dir(
        run_name=resolved_run_name,
        output_dir=args.output_dir,
        output_root=DEFAULT_OUTPUT_ROOT,
    )
    log_path = None if args.no_log_file else (
        args.log_file or default_log_path(
            run_name=resolved_run_name,
            logs_dir=DEFAULT_LOGS_DIR,
        )
    )
    if log_path is not None:
        _setup_log_file(log_path)

    _warn_if_multi_gpu()
    wandb_project = args.wandb_project or os.getenv(
        "WANDB_PROJECT_STEP2",
        "qwen3-generative-finetuning",
    )

    run_config = GenerativeFinetuningRunConfig(
        dataset=args.dataset,
        output_dir=output_dir,
        model_name=args.model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        max_seq_length=args.max_seq_length,
        train_split=args.train_split,
        eval_split=args.eval_split,
        wandb_project=wandb_project,
        run_name=resolved_run_name,
        fp16=args.fp16,
        bf16=not args.no_bf16,
        load_best_model=not args.no_load_best_model,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        qa_only=True,
    )

    print(f"dataset: {args.dataset}")
    print("training_mode: qa-only")
    if args.config is not None:
        print(f"config: {args.config}")
    if log_path is not None:
        print(f"log_file: {log_path}")
    print(f"hub_repo: {GENERATIVE_FINETUNING_DATASET_IDS[args.dataset]}")
    print(f"model: {args.model}")
    print(f"output_dir: {output_dir}")
    print(f"batch_size: {args.batch_size}")
    print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"epochs: {args.epochs}")
    print(f"max_seq_length: {args.max_seq_length}")
    print(f"train_split: {args.train_split}")
    print(f"eval_split: {args.eval_split}")
    print(f"wandb_project: {wandb_project}")
    print(f"wandb_run_name: {resolved_run_name}")
    print("enable_thinking: False")
    print(f"load_best_model: {run_config.load_best_model}")
    if run_config.load_best_model:
        print(f"metric_for_best_model: {run_config.metric_for_best_model}")
        print(f"greater_is_better: {run_config.greater_is_better}")
    print(f"early_stopping: {run_config.early_stopping}")
    if run_config.early_stopping:
        print(f"early_stopping_patience: {run_config.early_stopping_patience}")

    final_dir = finetune_qwen3_generative(run_config)
    print(f"Training complete. Final model saved to: {final_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
