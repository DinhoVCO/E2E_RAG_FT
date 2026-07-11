from __future__ import annotations

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"

MAX_SEQ_LENGTH = 3712
MAX_QUERY_TOKENS = 512
MAX_DOC_TOKENS = 512
MAX_ANSWER_TOKENS = 512
MAX_CONTEXT_DOCS = 5

TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

DEFAULT_NUM_TRAIN_EPOCHS = 3
TELCO_DPR_NUM_TRAIN_EPOCHS = 6

DEFAULT_INSTRUCTION = (
    "Responde a la query, utilizando los documentos como fundamento, "
    "caso los documentos no proporcienen informacion relevante intenta "
    "responder desde tu conocimiento."
)

RELEVANT_DOC_RATIO = 0.7
DEFAULT_DATASET_SEED = 42

GENERATIVE_FINETUNING_DATASET_IDS: dict[str, str] = {
    "bioasq-resplit": "DinoStackAI/bioasq-rag-13b-resplit",
    "qasper": "DinoStackAI/qasper-rag",
    "telco-dpr": "DinoStackAI/telco-dpr-rag",
    "narrativeqa": "DinoStackAI/narrativeqa-rag",
}

DEFAULT_WANDB_PROJECT = "qwen3-generative-finetuning"
DEFAULT_HUB_ORG = "DinoStackAI"

DEFAULT_METRIC_FOR_BEST_MODEL = "eval_loss"
EARLY_STOPPING_PATIENCE = 15

QWEN3_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
