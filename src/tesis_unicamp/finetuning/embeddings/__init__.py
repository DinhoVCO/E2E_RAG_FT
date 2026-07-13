from tesis_unicamp.finetuning.embeddings.config import (
    DEFAULT_BASE_MODEL,
    EMBEDDING_FINETUNING_DATASET_IDS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    MINI_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
)

__all__ = [
    "DEFAULT_BASE_MODEL",
    "EMBEDDING_FINETUNING_DATASET_IDS",
    "EmbeddingFinetuningDatasetConfig",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_R",
    "MAX_SEQ_LENGTH",
    "MINI_BATCH_SIZE",
    "TRAIN_BATCH_SIZE",
    "build_ir_evaluator",
    "context",
    "finetune_qwen3_embedding",
    "get_embedding_finetuning_config",
    "prepare_training_dataset",
]


def __getattr__(name: str):
    if name == "EmbeddingFinetuningDatasetConfig":
        from tesis_unicamp.finetuning.embeddings.datasets import (
            EmbeddingFinetuningDatasetConfig,
        )

        return EmbeddingFinetuningDatasetConfig
    if name in {
        "build_ir_evaluator",
        "get_embedding_finetuning_config",
        "prepare_training_dataset",
    }:
        from tesis_unicamp.finetuning.embeddings import datasets

        return getattr(datasets, name)
    if name == "finetune_qwen3_embedding":
        from tesis_unicamp.finetuning.embeddings.trainer import finetune_qwen3_embedding

        return finetune_qwen3_embedding
    if name == "context":
        from tesis_unicamp.finetuning.embeddings import context

        return context
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
