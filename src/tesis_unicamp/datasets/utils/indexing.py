from __future__ import annotations

from collections.abc import Callable
from typing import Any

from datasets import Dataset
from tqdm import tqdm

from tesis_unicamp.datasets.utils.corpus import iter_batches
from tesis_unicamp.embeddings.base import BaseEmbedder
from tesis_unicamp.vector_stores.base import BaseVectorStore


def index_dataset(
    dataset: Dataset,
    embedder: BaseEmbedder,
    store: BaseVectorStore,
    *,
    text_fn: Callable[[dict[str, Any]], str],
    id_fn: Callable[[dict[str, Any]], int | str],
    payload_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    batch_size: int | None = None,
    recreate_collection: bool = False,
    show_progress: bool = True,
) -> int:
    """Embed rows from a Hugging Face dataset and upsert them into a vector store."""
    if len(dataset) == 0:
        return 0

    batch_size = batch_size or embedder.batch_size
    rows = [dataset[i] for i in range(len(dataset))]

    probe_vector = embedder.embed_texts([text_fn(rows[0])])[0]
    store.ensure_collection(len(probe_vector), recreate=recreate_collection)

    indexed = 0
    batch_iterator = iter_batches(rows, batch_size)
    if show_progress:
        total_batches = (len(rows) + batch_size - 1) // batch_size
        batch_iterator = tqdm(batch_iterator, desc="Indexing", unit="batch", total=total_batches)

    for batch in batch_iterator:
        texts = [text_fn(row) for row in batch]
        vectors = embedder.embed_texts(texts)
        ids = [id_fn(row) for row in batch]
        payloads = [payload_fn(row) for row in batch] if payload_fn is not None else None
        store.upsert(ids, vectors, payloads)
        indexed += len(batch)

    return indexed
