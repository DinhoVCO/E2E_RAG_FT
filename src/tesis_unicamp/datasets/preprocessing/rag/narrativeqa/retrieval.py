from __future__ import annotations

from pathlib import Path

from tesis_unicamp.datasets.preprocessing.rag.retrieval.schemas import RetrievedDocRecord
from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.constants import PROJECT_ROOT
from tesis_unicamp.datasets.preprocessing.rag.retrieval.hub import push_retrieved_docs_to_hub
from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import (
    RAG_SPLITS,
    save_retrieved_docs_bundle,
)
from tesis_unicamp.datasets.utils.bioasq_rag import query_to_instruct_text
from tesis_unicamp.datasets.utils.narrativeqa_rag import (
    NARRATIVEQA_RAG_DATASET_ID,
    load_narrativeqa_rag_subset,
)
from tesis_unicamp.datasets.utils.retrieval import retrieve_all_splits
from tesis_unicamp.embeddings.base import BaseEmbedder
from tesis_unicamp.vector_stores.base import BaseVectorStore

DEFAULT_QDRANT_COLLECTION = "narrativeqa-rag-corpus"
DEFAULT_RETRIEVED_DOCS_DIR = PROJECT_ROOT / "datasets" / "retrieved" / "narrativeqa_rag"


def retrieve_narrativeqa_top_k(
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
        load_subset=load_narrativeqa_rag_subset,
        top_k=top_k,
        query_to_text=query_to_instruct_text,
        splits=splits,
        batch_size=batch_size,
        show_progress=show_progress,
    )


def save_narrativeqa_retrieved_docs(
    splits: dict[str, list[RetrievedDocRecord]],
    output_dir: Path = DEFAULT_RETRIEVED_DOCS_DIR,
) -> Path:
    return save_retrieved_docs_bundle(output_dir, splits)


def push_narrativeqa_retrieved_docs_to_hub(
    repo_id: str = NARRATIVEQA_RAG_DATASET_ID,
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
