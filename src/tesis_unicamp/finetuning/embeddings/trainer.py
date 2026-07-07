from __future__ import annotations

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

from tesis_unicamp.finetuning.embeddings.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_WANDB_PROJECT,
    MINI_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
)
from tesis_unicamp.finetuning.embeddings.datasets import (
    EmbeddingFinetuningDatasetConfig,
    build_ir_evaluator,
    get_embedding_finetuning_config,
    prepare_training_dataset,
)
from tesis_unicamp.finetuning.embeddings.model import load_qwen3_embedding_with_lora


@dataclass(frozen=True)
class FinetuningRunConfig:
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
    train_split: str | None = None
    eval_split: str | None = None
    wandb_project: str = DEFAULT_WANDB_PROJECT
    run_name: str | None = None
    fp16: bool = False
    bf16: bool = True


def configure_wandb(*, project: str, run_name: str) -> None:
    os.environ.setdefault("WANDB_PROJECT", project)
    os.environ.setdefault("WANDB_RUN_NAME", run_name)


def build_training_arguments(config: FinetuningRunConfig) -> SentenceTransformerTrainingArguments:
    run_name = config.run_name or f"qwen3-embedding-4b-lora-{config.dataset}"
    configure_wandb(project=config.wandb_project, run_name=run_name)

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
    )


def finetune_qwen3_embedding(
    config: FinetuningRunConfig,
    *,
    dataset_config: EmbeddingFinetuningDatasetConfig | None = None,
) -> Path:
    dataset_config = dataset_config or get_embedding_finetuning_config(config.dataset)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = prepare_training_dataset(
        dataset_config,
        split=config.train_split,
    )
    eval_dataset = prepare_training_dataset(
        dataset_config,
        split=config.eval_split,
    )
    evaluator: InformationRetrievalEvaluator = build_ir_evaluator(
        dataset_config,
        split=config.eval_split,
        batch_size=config.eval_batch_size,
    )

    model_card_name = (
        f"Qwen3-Embedding-4B LoRA finetuned on {dataset_config.name}"
    )
    model = load_qwen3_embedding_with_lora(
        model_name=config.model_name,
        model_card_name=model_card_name,
    )
    loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=config.mini_batch_size,
    )
    args = build_training_arguments(config)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()

    final_dir = config.output_dir / "final"
    model.save_pretrained(str(final_dir))
    return final_dir
