from __future__ import annotations

DEFAULT_BASE_MODEL = "Qwen/Qwen3-Embedding-4B"

MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 128
# Encode mini-batches inside MNRL to avoid OOM at large per-device batch sizes.
MINI_BATCH_SIZE = 32

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

EMBEDDING_FINETUNING_DATASET_IDS: dict[str, str] = {
    "bioasq-resplit": "DinoStackAI/bioasq-rag-13b-resplit",
    "qasper": "DinoStackAI/qasper-rag",
    "telco-dpr": "DinoStackAI/telco-dpr-rag",
    "narrativeqa": "DinoStackAI/narrativeqa-rag",
}

DEFAULT_WANDB_PROJECT = "qwen3-embedding-finetuning"
DEFAULT_HUB_ORG = "DinoStackAI"
