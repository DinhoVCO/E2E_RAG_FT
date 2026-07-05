from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


def format_document(title: str, text: str, *, separator: str = "\n\n") -> str:
    title = title.strip()
    text = text.strip()
    if title and text:
        return f"{title}{separator}{text}"
    return title or text


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
