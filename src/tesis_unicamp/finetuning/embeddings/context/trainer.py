from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
)

try:
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
except ImportError:
    from sentence_transformers.base.sampler import BatchSamplers

from transformers import AutoTokenizer

from tesis_unicamp.finetuning.embeddings.context.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_WANDB_PROJECT,
    MAX_SEQ_LENGTH,
    MINI_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
)
from tesis_unicamp.finetuning.embeddings.context.datasets import (
    ContextEmbeddingFinetuningDatasetConfig,
    build_ir_evaluator,
    default_ir_metric_for_best_model,
    get_context_embedding_finetuning_config,
    prepare_training_dataset,
    summarize_training_dataset,
)
from tesis_unicamp.finetuning.embeddings.model import load_qwen3_embedding_with_lora

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextFinetuningRunConfig:
    dataset: str
    output_dir: Path
    model_name: str = DEFAULT_BASE_MODEL
    num_train_epochs: int = 1
    per_device_train_batch_size: int = TRAIN_BATCH_SIZE
    per_device_eval_batch_size: int = TRAIN_BATCH_SIZE
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    eval_batch_size: int = 32
    mini_batch_size: int = MINI_BATCH_SIZE
    max_seq_length: int = MAX_SEQ_LENGTH
    train_split: str | None = None
    eval_split: str | None = None
    dataset_seed: int = 42
    wandb_project: str = DEFAULT_WANDB_PROJECT
    run_name: str | None = None
    fp16: bool = False
    bf16: bool = True
    load_best_model: bool = True
    metric_for_best_model: str | None = None


def configure_wandb(*, project: str, run_name: str) -> None:
    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_RUN_NAME", run_name)


def default_wandb_run_name(config: ContextFinetuningRunConfig) -> str:
    return (
        f"qwen3-embedding-4b-lora-ctx-{config.dataset}"
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
    placeholder = ContextFinetuningRunConfig(
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


def build_training_arguments(
    config: ContextFinetuningRunConfig,
    *,
    metric_for_best_model: str | None = None,
) -> SentenceTransformerTrainingArguments:
    run_name = config.run_name or default_wandb_run_name(config)
    configure_wandb(project=config.wandb_project, run_name=run_name)

    best_model_kwargs: dict[str, object] = {}
    if config.load_best_model:
        if config.save_steps % config.eval_steps != 0:
            raise ValueError(
                "When load_best_model is enabled, save_steps must be a multiple of "
                f"eval_steps (got save_steps={config.save_steps}, eval_steps={config.eval_steps}). "
                "Align them so every saved checkpoint has IR evaluation metrics."
            )
        resolved_metric = (
            config.metric_for_best_model
            or metric_for_best_model
        )
        if not resolved_metric:
            raise ValueError(
                "metric_for_best_model is required when load_best_model is enabled."
            )
        best_model_kwargs = {
            "load_best_model_at_end": True,
            "metric_for_best_model": resolved_metric,
            "greater_is_better": True,
        }
        logger.info(
            "Best checkpoint selection enabled (metric=%s).",
            resolved_metric,
        )

    return SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        bf16=config.bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to="wandb",
        run_name=run_name,
        **best_model_kwargs,
    )


def write_best_model_metadata(
    *,
    output_dir: Path,
    metric_for_best_model: str,
    best_model_checkpoint: str | None,
    best_metric: float | None,
) -> Path:
    metadata = {
        "metric_for_best_model": metric_for_best_model,
        "best_model_checkpoint": best_model_checkpoint,
        "best_metric": best_metric,
    }
    metadata_path = output_dir / "best_model.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def finetune_qwen3_embedding_context(
    config: ContextFinetuningRunConfig,
    *,
    dataset_config: ContextEmbeddingFinetuningDatasetConfig | None = None,
) -> Path:
    dataset_config = dataset_config or get_context_embedding_finetuning_config(config.dataset)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
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
    evaluator: InformationRetrievalEvaluator = build_ir_evaluator(
        dataset_config,
        tokenizer,
        split=config.eval_split,
        seed=config.dataset_seed + 1,
        batch_size=config.eval_batch_size,
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

    metric_for_best_model = (
        config.metric_for_best_model
        or default_ir_metric_for_best_model(
            dataset_config,
            split=config.eval_split,
        )
    )

    model_card_name = (
        f"Qwen3-Embedding-4B LoRA (context) finetuned on {dataset_config.name}"
    )
    model = load_qwen3_embedding_with_lora(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        model_card_name=model_card_name,
    )
    loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=config.mini_batch_size,
    )
    args = build_training_arguments(
        config,
        metric_for_best_model=metric_for_best_model,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

    if config.load_best_model:
        write_best_model_metadata(
            output_dir=config.output_dir,
            metric_for_best_model=metric_for_best_model,
            best_model_checkpoint=trainer.state.best_model_checkpoint,
            best_metric=trainer.state.best_metric,
        )
        if trainer.state.best_model_checkpoint:
            print(
                "best_checkpoint: "
                f"{trainer.state.best_model_checkpoint} "
                f"({metric_for_best_model}={trainer.state.best_metric})"
            )
        else:
            print(
                "Warning: load_best_model was enabled but no best checkpoint was recorded.",
            )

    final_dir = config.output_dir / "final"
    model.save_pretrained(str(final_dir))
    return final_dir
