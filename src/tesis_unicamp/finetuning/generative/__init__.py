from tesis_unicamp.finetuning.generative.config import (
    DEFAULT_BASE_MODEL,
    GENERATIVE_FINETUNING_DATASET_IDS,
    GRADIENT_ACCUMULATION_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_ANSWER_TOKENS,
    MAX_CONTEXT_DOCS,
    MAX_DOC_TOKENS,
    MAX_QUERY_TOKENS,
    MAX_SEQ_LENGTH,
    TRAIN_BATCH_SIZE,
)

__all__ = [
    "DEFAULT_BASE_MODEL",
    "GENERATIVE_FINETUNING_DATASET_IDS",
    "GRADIENT_ACCUMULATION_STEPS",
    "GenerativeFinetuningDatasetConfig",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_R",
    "MAX_ANSWER_TOKENS",
    "MAX_CONTEXT_DOCS",
    "MAX_DOC_TOKENS",
    "MAX_QUERY_TOKENS",
    "MAX_SEQ_LENGTH",
    "TRAIN_BATCH_SIZE",
    "finetune_qwen3_generative",
    "get_generative_finetuning_config",
    "prepare_training_dataset",
]


def __getattr__(name: str):
    if name == "GenerativeFinetuningDatasetConfig":
        from tesis_unicamp.finetuning.generative.datasets import (
            GenerativeFinetuningDatasetConfig,
        )

        return GenerativeFinetuningDatasetConfig
    if name in {"get_generative_finetuning_config", "prepare_training_dataset"}:
        from tesis_unicamp.finetuning.generative import datasets

        return getattr(datasets, name)
    if name == "finetune_qwen3_generative":
        from tesis_unicamp.finetuning.generative.trainer import finetune_qwen3_generative

        return finetune_qwen3_generative
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
