from __future__ import annotations

__all__ = [
    "ContextEmbeddingFinetuningDatasetConfig",
    "ContextFinetuningRunConfig",
    "build_ir_evaluator",
    "finetune_qwen3_embedding_context",
    "get_context_embedding_finetuning_config",
    "prepare_training_dataset",
]


def __getattr__(name: str):
    if name == "ContextEmbeddingFinetuningDatasetConfig":
        from tesis_unicamp.finetuning.embeddings.context.datasets import (
            ContextEmbeddingFinetuningDatasetConfig,
        )

        return ContextEmbeddingFinetuningDatasetConfig
    if name in {
        "build_ir_evaluator",
        "get_context_embedding_finetuning_config",
        "prepare_training_dataset",
    }:
        from tesis_unicamp.finetuning.embeddings.context import datasets

        return getattr(datasets, name)
    if name in {"ContextFinetuningRunConfig", "finetune_qwen3_embedding_context"}:
        from tesis_unicamp.finetuning.embeddings.context import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
