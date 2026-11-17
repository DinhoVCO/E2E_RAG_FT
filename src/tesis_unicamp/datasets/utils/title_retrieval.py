from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from datasets import Dataset

from tesis_unicamp.datasets.utils.bioasq_rag import (
    DEFAULT_RETRIEVAL_TASK,
    query_to_instruct_text_with_title,
)


def corpus_row_context_title(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    section = str(row.get("section_name") or "").strip()
    if title and section:
        return f"{title} | {section}"
    return title or section


def build_corpus_title_lookup(corpus: Dataset) -> dict[str, str]:
    titles: dict[str, str] = {}
    for row in corpus:
        title = corpus_row_context_title(row)
        if title:
            titles[str(row["id"])] = title
    return titles


def build_corpus_body_lookup(corpus: Dataset) -> dict[str, str]:
    return {
        str(row["id"]): str(row.get("text") or "").strip()
        for row in corpus
    }


def build_query_document_title_lookup(
    qrels: Dataset,
    corpus: Dataset,
) -> dict[str, str]:
    """Map query id to gold document title(s) from qrels."""
    corpus_titles = build_corpus_title_lookup(corpus)
    query_titles: dict[str, list[str]] = defaultdict(list)

    for row in qrels:
        query_id = str(row["query_id"])
        corpus_id = str(row["corpus_id"])
        title = corpus_titles.get(corpus_id, "")
        if title and title not in query_titles[query_id]:
            query_titles[query_id].append(title)

    return {
        query_id: titles[0] if len(titles) == 1 else "; ".join(titles)
        for query_id, titles in query_titles.items()
    }


def make_title_aware_query_row_to_text(
    title_lookup: dict[str, str],
    *,
    task: str = DEFAULT_RETRIEVAL_TASK,
) -> Callable[[dict[str, Any]], str]:
    def query_row_to_text(row: dict[str, Any]) -> str:
        query_id = str(row["id"])
        query_text = str(row["text"])
        return query_to_instruct_text_with_title(
            query_text,
            title=title_lookup.get(query_id, ""),
            task=task,
        )

    return query_row_to_text


def build_query_title_lookup_for_split(
    *,
    load_subset: Callable[..., Dataset],
    load_corpus: Callable[..., Dataset],
    split: str,
    corpus_split: str = "train",
) -> dict[str, str]:
    qrels = load_subset("qrels", split=split)
    corpus = load_corpus(split=corpus_split)
    return build_query_document_title_lookup(qrels, corpus)
