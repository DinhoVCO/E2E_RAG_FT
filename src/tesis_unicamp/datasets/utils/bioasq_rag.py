from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from tesis_unicamp.datasets.utils.corpus import format_document
from tesis_unicamp.datasets.utils.indexing import index_dataset
from tesis_unicamp.embeddings.base import BaseEmbedder
from tesis_unicamp.vector_stores.base import BaseVectorStore

BIOASQ_RAG_DATASET_ID = "DinoStackAI/bioasq-rag-13b"
DEFAULT_RETRIEVAL_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def load_bioasq_rag_corpus(*, split: str = "train") -> Dataset:
    dataset = load_dataset(BIOASQ_RAG_DATASET_ID, "corpus")
    if isinstance(dataset, DatasetDict):
        return dataset[split]
    return dataset


def load_bioasq_rag_subset(name: str, *, split: str | None = None) -> Dataset:
    dataset = load_dataset(BIOASQ_RAG_DATASET_ID, name)
    if isinstance(dataset, DatasetDict):
        if split is None:
            raise ValueError(f"Subset {name!r} has splits; pass split= explicitly")
        return dataset[split]
    return dataset


def corpus_row_to_text(row: dict[str, Any]) -> str:
    return format_document(row.get("title", ""), row.get("text", ""))


def corpus_row_to_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "corpus_id": row["id"],
        "title": row.get("title", ""),
        "text": row.get("text", ""),
    }


def corpus_row_to_point_id(row: dict[str, Any]) -> int:
    return int(row["id"])


def query_to_instruct_text(
    query: str,
    *,
    task: str = DEFAULT_RETRIEVAL_TASK,
) -> str:
    return f"Instruct: {task}\nQuery:{query}"


def index_bioasq_corpus(
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    *,
    split: str = "train",
    batch_size: int | None = None,
    recreate_collection: bool = False,
    show_progress: bool = True,
) -> int:
    corpus = load_bioasq_rag_corpus(split=split)
    return index_dataset(
        corpus,
        embedder,
        store,
        text_fn=corpus_row_to_text,
        id_fn=corpus_row_to_point_id,
        payload_fn=corpus_row_to_payload,
        batch_size=batch_size,
        recreate_collection=recreate_collection,
        show_progress=show_progress,
    )
