from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from tesis_unicamp.vector_stores.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed vector store for corpus indexing and retrieval."""

    def __init__(
        self,
        collection_name: str,
        *,
        url: str | None = None,
        api_key: str | None = None,
        client: QdrantClient | None = None,
        distance: Distance = Distance.COSINE,
    ) -> None:
        self.collection_name = collection_name
        self.distance = distance
        self._client = client or QdrantClient(
            url=url or os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=api_key or os.getenv("QDRANT_API_KEY"),
        )

    def ensure_collection(
        self,
        vector_size: int,
        *,
        recreate: bool = False,
    ) -> None:
        exists = self._client.collection_exists(self.collection_name)
        if exists and recreate:
            self._client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=self.distance),
            )

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

        points = [
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payloads[i] if payloads is not None else None,
            )
            for i, (point_id, vector) in enumerate(zip(ids, vectors, strict=True))
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if payload_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in payload_filter.items()
                ]
            )

        hits = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {},
            }
            for hit in hits
        ]

    def count(self) -> int:
        info = self._client.get_collection(self.collection_name)
        return info.points_count or 0
