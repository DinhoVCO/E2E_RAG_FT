from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """Base class for vector database backends."""

    @abstractmethod
    def ensure_collection(
        self,
        vector_size: int,
        *,
        recreate: bool = False,
    ) -> None:
        """Create the collection if it does not exist."""

    @abstractmethod
    def upsert(
        self,
        ids: list[int | str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update points in the collection."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the nearest neighbors for a query vector."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of points stored in the collection."""
