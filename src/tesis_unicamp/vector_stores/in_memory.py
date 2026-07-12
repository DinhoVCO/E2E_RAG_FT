from __future__ import annotations

from typing import Any

import faiss
import numpy as np

from tesis_unicamp.vector_stores.base import BaseVectorStore


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors in place for cosine similarity via inner product."""
    faiss.normalize_L2(vectors)
    return vectors


class InMemoryVectorStore(BaseVectorStore):
    """FAISS-backed in-memory vector store for corpus indexing and retrieval.

    Uses ``IndexFlatIP`` on L2-normalized vectors, which is equivalent to cosine
    similarity and matches the Qdrant ``Distance.COSINE`` configuration.
    """

    def __init__(self, collection_name: str = "in-memory") -> None:
        self.collection_name = collection_name
        self._vector_size: int | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._ids: list[int | str] = []
        self._payloads: list[dict[str, Any]] = []
        self._corpus_id_to_index: dict[str, int] = {}

    def _reset(self) -> None:
        self._vector_size = None
        self._index = None
        self._ids = []
        self._payloads = []
        self._corpus_id_to_index = {}

    def ensure_collection(
        self,
        vector_size: int,
        *,
        recreate: bool = False,
    ) -> None:
        if recreate:
            self._reset()

        if self._index is not None and self._vector_size != vector_size:
            raise ValueError(
                f"Vector size mismatch: store has {self._vector_size}, got {vector_size}"
            )

        if self._index is None:
            self._vector_size = vector_size
            self._index = faiss.IndexFlatIP(vector_size)

    def upsert(
        self,
        ids: list[int | str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        if payloads is not None and len(payloads) != len(ids):
            raise ValueError("payloads must match ids length when provided")
        if not vectors:
            return

        vector_size = len(vectors[0])
        self.ensure_collection(vector_size)
        assert self._index is not None

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != vector_size:
            raise ValueError("All vectors must share the same dimension")

        _normalize(matrix)
        start_idx = self.count()
        self._index.add(matrix)
        self._ids.extend(ids)
        if payloads is not None:
            self._payloads.extend(payloads)
            for offset, payload in enumerate(payloads):
                corpus_id = str(payload.get("corpus_id", ids[offset]))
                self._corpus_id_to_index[corpus_id] = start_idx + offset
        else:
            self._payloads.extend({} for _ in ids)

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del payload_filter  # Post-filtering is not needed for the current RAG datasets.

        if self._index is None or self.count() == 0:
            return []

        query = np.asarray([query_vector], dtype=np.float32)
        _normalize(query)

        k = min(limit, self.count())
        scores, indices = self._index.search(query, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            results.append(
                {
                    "id": self._ids[idx],
                    "score": float(score),
                    "payload": self._payloads[idx],
                }
            )
        return results

    def search_candidates(
        self,
        query_vector: list[float],
        candidate_corpus_ids: list[str],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return top matches restricted to a subset of corpus ids."""
        if self._index is None or self.count() == 0 or not candidate_corpus_ids:
            return []

        indices: list[int] = []
        seen: set[int] = set()
        for corpus_id in candidate_corpus_ids:
            idx = self._corpus_id_to_index.get(str(corpus_id))
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)

        if not indices:
            return []

        query = np.asarray([query_vector], dtype=np.float32)
        _normalize(query)

        candidate_vectors = np.vstack(
            [self._index.reconstruct(idx) for idx in indices]
        )
        scores = (candidate_vectors @ query.T).ravel()
        order = np.argsort(-scores)
        k = min(limit, len(order))

        results: list[dict[str, Any]] = []
        for pos in order[:k]:
            idx = indices[int(pos)]
            results.append(
                {
                    "id": self._ids[idx],
                    "score": float(scores[int(pos)]),
                    "payload": self._payloads[idx],
                }
            )
        return results

    def count(self) -> int:
        return len(self._ids)
