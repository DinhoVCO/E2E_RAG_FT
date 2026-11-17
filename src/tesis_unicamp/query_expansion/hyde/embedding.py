from __future__ import annotations

import numpy as np

from tesis_unicamp.datasets.utils.bioasq_rag import query_to_instruct_text
from tesis_unicamp.embeddings.base import BaseEmbedder


def compute_hyde_vector(
    embedder: BaseEmbedder,
    *,
    query: str,
    pseudo_passages: list[str],
    include_query: bool = True,
    query_to_text=query_to_instruct_text,
) -> np.ndarray:
    """Average embeddings of the instruct query and pseudo-passages."""
    texts: list[str] = []
    if include_query:
        texts.append(query_to_text(query))
    texts.extend(passage.strip() for passage in pseudo_passages if passage.strip())
    if not texts:
        raise ValueError("Cannot compute HyDE vector from empty query and passages.")

    vectors = np.asarray(embedder.embed_texts(texts), dtype=np.float32)
    return vectors.mean(axis=0)


def compute_hyde_vectors_batch(
    embedder: BaseEmbedder,
    *,
    queries: list[str],
    pseudo_passages_batch: list[list[str]],
    include_query: bool = True,
    query_to_text=query_to_instruct_text,
) -> np.ndarray:
    if len(queries) != len(pseudo_passages_batch):
        raise ValueError("queries and pseudo_passages_batch must have the same length.")

    vectors: list[np.ndarray] = []
    for query, passages in zip(queries, pseudo_passages_batch, strict=True):
        vectors.append(
            compute_hyde_vector(
                embedder,
                query=query,
                pseudo_passages=passages,
                include_query=include_query,
                query_to_text=query_to_text,
            )
        )
    return np.stack(vectors, axis=0)
