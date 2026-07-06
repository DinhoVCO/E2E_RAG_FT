from tesis_unicamp.finetuning.embeddings.config import (
    DEFAULT_BASE_MODEL,
    EMBEDDING_FINETUNING_DATASET_IDS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    TRAIN_BATCH_SIZE,
)
from tesis_unicamp.finetuning.embeddings.datasets import (
    EmbeddingFinetuningDatasetConfig,
    build_ir_evaluator,
    get_embedding_finetuning_config,
    prepare_training_dataset,
)
from tesis_unicamp.finetuning.embeddings.trainer import finetune_qwen3_embedding

__all__ = [
    "DEFAULT_BASE_MODEL",
    "EMBEDDING_FINETUNING_DATASET_IDS",
    "EmbeddingFinetuningDatasetConfig",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_R",
    "MAX_SEQ_LENGTH",
    "TRAIN_BATCH_SIZE",
    "build_ir_evaluator",
    "finetune_qwen3_embedding",
    "get_embedding_finetuning_config",
    "prepare_training_dataset",
]
