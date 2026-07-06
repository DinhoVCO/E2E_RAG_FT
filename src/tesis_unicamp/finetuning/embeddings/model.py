from __future__ import annotations

from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData

from tesis_unicamp.finetuning.embeddings.config import (
    DEFAULT_BASE_MODEL,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
)


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )


def load_qwen3_embedding_with_lora(
    *,
    model_name: str = DEFAULT_BASE_MODEL,
    max_seq_length: int = MAX_SEQ_LENGTH,
    model_card_name: str | None = None,
) -> SentenceTransformer:
    model = SentenceTransformer(
        model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=model_card_name or f"{model_name} LoRA finetuned",
        ),
        model_kwargs={"torch_dtype": "float32"},
    )
    model.max_seq_length = max_seq_length
    model.add_adapter(build_lora_config())
    return model
