"""Telco-DPR RAG dataset preprocessing for Hugging Face."""

from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.builder import (
    build_telco_dpr_rag_dataset,
    push_telco_dpr_rag_to_hub,
)
from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.processor import (
    build_corpus,
    process_telco_dpr_split,
    split_train_dev,
)

__all__ = [
    "build_corpus",
    "build_telco_dpr_rag_dataset",
    "process_telco_dpr_split",
    "push_telco_dpr_rag_to_hub",
    "split_train_dev",
]
