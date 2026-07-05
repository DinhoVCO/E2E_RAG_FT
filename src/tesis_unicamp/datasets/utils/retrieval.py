from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

from tesis_unicamp.datasets.preprocessing.rag.retrieval.schemas import RetrievedDocRecord
from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import RAG_SPLITS
from tesis_unicamp.datasets.utils.corpus import iter_batches
from tesis_unicamp.embeddings.base import BaseEmbedder
from tesis_unicamp.vector_stores.base import BaseVectorStore


def build_relevant_corpus_ids(qrels: Dataset) -> dict[str, set[str]]:
    """Map each query_id to the set of relevant corpus_id values from qrels."""
    relevant: dict[str, set[str]] = defaultdict(set)
    for i in range(len(qrels)):
        row = qrels[i]
        relevant[row["query_id"]].add(str(row["corpus_id"]))
    return relevant


def retrieve_top_k_for_queries(
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    queries: Dataset,
    qrels: Dataset,
    *,
    top_k: int = 10,
    query_to_text: Callable[[str], str],
    batch_size: int | None = None,
    show_progress: bool = True,
) -> list[RetrievedDocRecord]:
    """Retrieve top-k corpus documents for each query and label relevance from qrels."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    relevant = build_relevant_corpus_ids(qrels)
    batch_size = batch_size or embedder.batch_size
    query_rows = [queries[i] for i in range(len(queries))]
    records: list[RetrievedDocRecord] = []

    batch_iterator = iter_batches(query_rows, batch_size)
    if show_progress:
        total_batches = (len(query_rows) + batch_size - 1) // batch_size
        batch_iterator = tqdm(
            batch_iterator,
            desc="Retrieving",
            unit="batch",
            total=total_batches,
        )

    for batch in batch_iterator:
        texts = [query_to_text(row["text"]) for row in batch]
        vectors = embedder.embed_texts(texts)
        for row, vector in zip(batch, vectors, strict=True):
            hits = store.search(vector, limit=top_k)
            query_id = row["id"]
            gold = relevant.get(query_id, set())
            for rank, hit in enumerate(hits, start=1):
                corpus_id = str((hit.get("payload") or {}).get("corpus_id", ""))
                records.append(
                    {
                        "query_id": query_id,
                        "corpus_id": corpus_id,
                        "rank": rank,
                        "retrieval_score": float(hit["score"]),
                        "is_relevant": corpus_id in gold,
                    }
                )

    return records


def retrieve_split(
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    *,
    split: str,
    load_subset: Callable[..., Dataset],
    top_k: int = 10,
    query_to_text: Callable[[str], str],
    batch_size: int | None = None,
    show_progress: bool = True,
) -> list[RetrievedDocRecord]:
    queries = load_subset("queries", split=split)
    qrels = load_subset("qrels", split=split)
    return retrieve_top_k_for_queries(
        embedder,
        store,
        queries,
        qrels,
        top_k=top_k,
        query_to_text=query_to_text,
        batch_size=batch_size,
        show_progress=show_progress,
    )


def retrieve_all_splits(
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    *,
    load_subset: Callable[..., Dataset],
    top_k: int = 10,
    query_to_text: Callable[[str], str],
    splits: tuple[str, ...] = RAG_SPLITS,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> dict[str, list[RetrievedDocRecord]]:
    return {
        split: retrieve_split(
            embedder,
            store,
            split=split,
            load_subset=load_subset,
            top_k=top_k,
            query_to_text=query_to_text,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        for split in splits
    }


@dataclass(frozen=True)
class RagRetrievalConfig:
    name: str
    hub_repo_id: str
    qdrant_collection: str
    output_dir: Path
    load_subset: Callable[..., Dataset]
    query_to_text: Callable[[str], str]
