from __future__ import annotations

from pathlib import Path

from tesis_unicamp.datasets.preprocessing.rag.retrieval.schemas import RetrievedDocRecord
from tesis_unicamp.datasets.preprocessing.rag.bioasq.constants import PROJECT_ROOT
from tesis_unicamp.datasets.preprocessing.rag.retrieval.hub import push_retrieved_docs_to_hub
from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import (
    RAG_SPLITS,
    build_retrieved_docs_dataset_dict,
    save_retrieved_docs_all_splits,
    save_retrieved_docs_to_hf_disk,
)
from tesis_unicamp.datasets.utils.bioasq_rag import (
    BIOASQ_RAG_DATASET_ID,
    load_bioasq_rag_subset,
    query_to_instruct_text,
)
from tesis_unicamp.datasets.utils.retrieval import retrieve_all_splits
from tesis_unicamp.embeddings.base import BaseEmbedder
from tesis_unicamp.vector_stores.base import BaseVectorStore

DEFAULT_QDRANT_COLLECTION = "bioasq-rag-13b-corpus"
DEFAULT_RETRIEVED_DOCS_DIR = PROJECT_ROOT / "datasets" / "retrieved" / "bioasq_rag"


def retrieve_bioasq_top_k(
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    *,
    top_k: int = 10,
    splits: tuple[str, ...] = RAG_SPLITS,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> dict[str, list[RetrievedDocRecord]]:
    return retrieve_all_splits(
        embedder,
        store,
        load_subset=load_bioasq_rag_subset,
        top_k=top_k,
        query_to_text=query_to_instruct_text,
        splits=splits,
        batch_size=batch_size,
        show_progress=show_progress,
    )


def save_bioasq_retrieved_docs(
    splits: dict[str, list[RetrievedDocRecord]],
    output_dir: Path = DEFAULT_RETRIEVED_DOCS_DIR,
) -> Path:
    save_retrieved_docs_all_splits(output_dir, splits)
    dataset_dict = build_retrieved_docs_dataset_dict(output_dir)
    save_retrieved_docs_to_hf_disk(output_dir, dataset_dict)
    return output_dir


def push_bioasq_retrieved_docs_to_hub(
    repo_id: str = BIOASQ_RAG_DATASET_ID,
    output_dir: Path = DEFAULT_RETRIEVED_DOCS_DIR,
    *,
    token: str | None = None,
    private: bool = False,
) -> None:
    push_retrieved_docs_to_hub(
        repo_id,
        output_dir,
        token=token,
        private=private,
    )
