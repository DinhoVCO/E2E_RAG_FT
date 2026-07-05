from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from typing import Any

# Stable namespace for deterministic Qdrant point IDs from corpus_id strings.
_CORPUS_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def format_document(title: str, text: str, *, separator: str = "\n\n") -> str:
    title = title.strip()
    text = text.strip()
    if title and text:
        return f"{title}{separator}{text}"
    return title or text


def corpus_id_to_point_id(corpus_id: str | int) -> str:
    """Map a corpus_id to a Qdrant-compatible UUID point id."""
    return str(uuid.uuid5(_CORPUS_ID_NAMESPACE, str(corpus_id)))


def iter_batches(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def batched_map(
    items: list[Any],
    batch_size: int,
    fn: Callable[[list[Any]], list[Any]],
) -> list[Any]:
    results: list[Any] = []
    for batch in iter_batches(items, batch_size):
        results.extend(fn(batch))
    return results
