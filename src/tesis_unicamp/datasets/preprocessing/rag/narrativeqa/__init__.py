"""NarrativeQA RAG dataset preprocessing for Hugging Face."""

from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.builder import (
    build_narrativeqa_rag_dataset,
    push_narrativeqa_rag_to_hub,
)
from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.processor import (
    build_corpus,
    process_narrativeqa_split,
    process_narrativeqa_splits,
)

__all__ = [
    "build_corpus",
    "build_narrativeqa_rag_dataset",
    "process_narrativeqa_split",
    "process_narrativeqa_splits",
    "push_narrativeqa_rag_to_hub",
]
