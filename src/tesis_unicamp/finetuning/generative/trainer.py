from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from transformers import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer

from tesis_unicamp.finetuning.generative.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_METRIC_FOR_BEST_MODEL,
    DEFAULT_WANDB_PROJECT,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_SEQ_LENGTH,
    TRAIN_BATCH_SIZE,
)
from tesis_unicamp.finetuning.generative.datasets import (
    GenerativeFinetuningDatasetConfig,
    get_generative_finetuning_config,
    prepare_training_dataset,
    summarize_training_dataset,
)
from tesis_unicamp.finetuning.generative.model import load_qwen3_generative_with_lora

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerativeFinetuningRunConfig:
    dataset: str
    output_dir: Path
    model_name: str = DEFAULT_BASE_MODEL
    num_train_epochs: int = 3
    per_device_train_batch_size: int = TRAIN_BATCH_SIZE
    per_device_eval_batch_size: int = TRAIN_BATCH_SIZE
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    max_seq_length: int = MAX_SEQ_LENGTH
    train_split: str | None = None
    eval_split: str | None = None
    dataset_seed: int = 42
    wandb_project: str = DEFAULT_WANDB_PROJECT
    run_name: str | None = None
    fp16: bool = False
    bf16: bool = True
    load_best_model: bool = True
    metric_for_best_model: str = DEFAULT_METRIC_FOR_BEST_MODEL
    greater_is_better: bool = False
    early_stopping: bool = True
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE


def configure_wandb(*, project: str, run_name: str) -> None:
    # Force override: .env sets WANDB_PROJECT for embedding step 1; generative uses
    # WANDB_PROJECT_STEP2 passed here as `project`.
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_RUN_NAME"] = run_name


def default_wandb_run_name(config: GenerativeFinetuningRunConfig) -> str:
    return (
        f"qwen3-8b-lora-{config.dataset}"
        f"-b{config.per_device_train_batch_size}"
        f"-e{config.num_train_epochs}"
    )


def resolve_run_name(
    *,
    dataset: str,
    batch_size: int,
    epochs: int,
    run_name: str | None,
) -> str:
    if run_name:
        return run_name
    placeholder = GenerativeFinetuningRunConfig(
        dataset=dataset,
        output_dir=Path("."),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
    )
    return default_wandb_run_name(placeholder)


def resolve_output_dir(
    *,
    run_name: str,
    output_dir: Path | None,
    output_root: Path,
) -> Path:
    return output_dir or (output_root / run_name)


def default_log_path(*, run_name: str, logs_dir: Path) -> Path:
    return logs_dir / f"{run_name}.log"


def build_training_arguments(config: GenerativeFinetuningRunConfig) -> SFTConfig:
    run_name = config.run_name or default_wandb_run_name(config)
    configure_wandb(project=config.wandb_project, run_name=run_name)

    best_model_kwargs: dict[str, object] = {}
    if config.load_best_model:
        if config.save_steps % config.eval_steps != 0:
            raise ValueError(
                "When load_best_model is enabled, save_steps must be a multiple of "
                f"eval_steps (got save_steps={config.save_steps}, eval_steps={config.eval_steps}). "
                "Align them so every saved checkpoint has eval metrics."
            )
        best_model_kwargs = {
            "load_best_model_at_end": True,
            "metric_for_best_model": config.metric_for_best_model,
            "greater_is_better": config.greater_is_better,
        }
        logger.info(
            "Best checkpoint selection enabled (metric=%s, greater_is_better=%s).",
            config.metric_for_best_model,
            config.greater_is_better,
        )

    return SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_length=config.max_seq_length,
        assistant_only_loss=True,
        **best_model_kwargs,
    )


def build_trainer_callbacks(
    config: GenerativeFinetuningRunConfig,
) -> list[TrainerCallback]:
    if not config.early_stopping:
        return []
    if not config.load_best_model:
        raise ValueError(
            "early_stopping requires load_best_model to be enabled "
            "(eval metrics are needed to detect plateaus)."
        )
    logger.info(
        "Early stopping enabled (patience=%d eval steps).",
        config.early_stopping_patience,
    )
    return [
        EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
    ]


def write_best_model_metadata(
    *,
    output_dir: Path,
    metric_for_best_model: str,
    best_model_checkpoint: str | None,
    best_metric: float | None,
    greater_is_better: bool,
    early_stopping_patience: int | None,
) -> Path:
    metadata = {
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "best_model_checkpoint": best_model_checkpoint,
        "best_metric": best_metric,
        "early_stopping_patience": early_stopping_patience,
    }
    metadata_path = output_dir / "best_model.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def finetune_qwen3_generative(
    config: GenerativeFinetuningRunConfig,
    *,
    dataset_config: GenerativeFinetuningDatasetConfig | None = None,
) -> Path:
    dataset_config = dataset_config or get_generative_finetuning_config(config.dataset)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_qwen3_generative_with_lora(model_name=config.model_name)

    train_dataset = prepare_training_dataset(
        dataset_config,
        tokenizer,
        split=config.train_split,
        seed=config.dataset_seed,
    )
    eval_dataset = prepare_training_dataset(
        dataset_config,
        tokenizer,
        split=config.eval_split,
        seed=config.dataset_seed + 1,
    )

    train_summary = summarize_training_dataset(
        dataset_config,
        tokenizer,
        split=config.train_split,
        seed=config.dataset_seed,
    )
    eval_summary = summarize_training_dataset(
        dataset_config,
        tokenizer,
        split=config.eval_split,
        seed=config.dataset_seed + 1,
    )
    logger.info("Train dataset summary: %s", train_summary)
    logger.info("Eval dataset summary: %s", eval_summary)
    print(f"train_dataset: {train_summary}")
    print(f"eval_dataset: {eval_summary}")

    training_args = build_training_arguments(config)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=build_trainer_callbacks(config),
    )

    trainer.train()

    if config.load_best_model:
        write_best_model_metadata(
            output_dir=config.output_dir,
            metric_for_best_model=config.metric_for_best_model,
            best_model_checkpoint=trainer.state.best_model_checkpoint,
            best_metric=trainer.state.best_metric,
            greater_is_better=config.greater_is_better,
            early_stopping_patience=(
                config.early_stopping_patience if config.early_stopping else None
            ),
        )
        if trainer.state.best_model_checkpoint:
            print(
                "best_checkpoint: "
                f"{trainer.state.best_model_checkpoint} "
                f"({config.metric_for_best_model}={trainer.state.best_metric})"
            )
        else:
            print(
                "Warning: load_best_model was enabled but no best checkpoint was recorded.",
            )

    final_dir = config.output_dir / "final"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return final_dir
