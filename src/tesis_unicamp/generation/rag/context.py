from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from datasets import Dataset

from tesis_unicamp.datasets.preprocessing.rag.retrieval.schemas import RetrievedDocRecord
from tesis_unicamp.finetuning.generative.formatting import build_user_content
from tesis_unicamp.generation.rag.prompts import build_rag_user_prompt

DEFAULT_MIN_TOKENS_PER_CHUNK = 64
DEFAULT_MAX_TOKENS_PER_CHUNK = 512
DOC_PREFIX_TOKEN_RESERVE = 12


def build_corpus_lookup(
    corpus: Dataset,
    corpus_text_fn: Callable[[dict[str, Any]], str],
) -> dict[str, str]:
    return {str(row["id"]): corpus_text_fn(row) for row in corpus}


def build_corpus_title_lookup(corpus: Dataset) -> dict[str, str]:
    """Map corpus id to document title."""
    titles: dict[str, str] = {}
    for row in corpus:
        title = corpus_row_context_title(row)
        if title:
            titles[str(row["id"])] = title
    return titles


def corpus_row_context_title(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    section = str(row.get("section_name") or "").strip()
    if title and section:
        return f"{title} | {section}"
    return title or section


def build_corpus_body_lookup(corpus: Dataset) -> dict[str, str]:
    return {
        str(row["id"]): str(row.get("text") or "").strip()
        for row in corpus
    }


def build_query_document_title_lookup(
    qrels: Dataset,
    corpus: Dataset,
) -> dict[str, str]:
    """Map query id to the gold document title(s) from qrels."""
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


def group_retrieved_by_query(
    records: list[RetrievedDocRecord],
) -> dict[str, list[RetrievedDocRecord]]:
    grouped: dict[str, list[RetrievedDocRecord]] = defaultdict(list)
    for record in records:
        grouped[str(record["query_id"])].append(record)
    for hits in grouped.values():
        hits.sort(key=lambda row: int(row["rank"]))
    return dict(grouped)


def build_context_from_hits(
    hits: list[RetrievedDocRecord],
    corpus_lookup: dict[str, str],
    *,
    max_docs: int | None = None,
) -> str:
    parts: list[str] = []
    for index, hit in enumerate(hits[:max_docs], start=1):
        corpus_id = str(hit["corpus_id"])
        text = corpus_lookup.get(corpus_id, "").strip()
        if text:
            parts.append(f"[{index}] {text}")
    return "\n\n".join(parts)


def estimate_tokens_per_chunk(
    question: str,
    *,
    max_docs: int,
    count_prompt_tokens: Callable[[str], int],
    max_prompt_tokens: int,
    max_tokens_per_chunk: int | None = DEFAULT_MAX_TOKENS_PER_CHUNK,
) -> int:
    """Resolve the per-chunk token budget."""
    if max_tokens_per_chunk is not None and max_tokens_per_chunk > 0:
        return max_tokens_per_chunk

    question = question.strip()
    overhead = count_prompt_tokens(build_user_content(query=question, doc_texts=[]))
    context_budget = max(0, max_prompt_tokens - overhead)
    separator_reserve = max_docs * DOC_PREFIX_TOKEN_RESERVE
    body_budget = max(0, context_budget - separator_reserve)
    if max_docs <= 0:
        return body_budget
    return max(DEFAULT_MIN_TOKENS_PER_CHUNK, body_budget // max_docs)


def build_rag_user_prompt_with_budget(
    question: str,
    hits: list[RetrievedDocRecord],
    corpus_lookup: dict[str, str],
    *,
    max_docs: int,
    count_prompt_tokens: Callable[[str], int],
    max_prompt_tokens: int,
    max_tokens_per_chunk: int | None = DEFAULT_MAX_TOKENS_PER_CHUNK,
    truncate_text_to_tokens: Callable[[str, int], str] | None = None,
) -> str:
    """Build a RAG prompt by truncating each retrieved chunk to a fair token budget."""
    question = question.strip()
    if count_prompt_tokens(build_rag_user_prompt(question, "")) >= max_prompt_tokens:
        truncated_question = question
        while truncated_question and count_prompt_tokens(
            build_rag_user_prompt(truncated_question, "")
        ) >= max_prompt_tokens:
            truncated_question = truncated_question[:-200].strip()
        question = truncated_question or question[:200]

    per_chunk_tokens = estimate_tokens_per_chunk(
        question,
        max_docs=max_docs,
        count_prompt_tokens=count_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
        max_tokens_per_chunk=max_tokens_per_chunk,
    )

    while per_chunk_tokens >= DEFAULT_MIN_TOKENS_PER_CHUNK:
        prompt = _build_prompt_with_per_chunk_truncation(
            question,
            hits,
            corpus_lookup,
            max_docs=max_docs,
            per_chunk_tokens=per_chunk_tokens,
            truncate_text_to_tokens=truncate_text_to_tokens,
        )
        if count_prompt_tokens(prompt) <= max_prompt_tokens:
            return prompt
        per_chunk_tokens = max(DEFAULT_MIN_TOKENS_PER_CHUNK, int(per_chunk_tokens * 0.85))

    return _build_prompt_with_per_chunk_truncation(
        question,
        hits,
        corpus_lookup,
        max_docs=max_docs,
        per_chunk_tokens=DEFAULT_MIN_TOKENS_PER_CHUNK,
        truncate_text_to_tokens=truncate_text_to_tokens,
    )


def _build_prompt_with_per_chunk_truncation(
    question: str,
    hits: list[RetrievedDocRecord],
    corpus_lookup: dict[str, str],
    *,
    max_docs: int,
    per_chunk_tokens: int,
    truncate_text_to_tokens: Callable[[str, int], str] | None,
) -> str:
    parts: list[str] = []
    for index, hit in enumerate(hits[:max_docs], start=1):
        corpus_id = str(hit["corpus_id"])
        text = corpus_lookup.get(corpus_id, "").strip()
        if not text:
            continue
        text = _truncate_chunk_text(
            text,
            per_chunk_tokens,
            truncate_text_to_tokens=truncate_text_to_tokens,
        )
        if text:
            parts.append(f"[{index}] {text}")
    return build_rag_user_prompt(question, "\n\n".join(parts))


def _truncate_chunk_text(
    text: str,
    max_tokens: int,
    *,
    truncate_text_to_tokens: Callable[[str, int], str] | None,
) -> str:
    text = text.strip()
    if not text:
        return ""
    if truncate_text_to_tokens is not None:
        return truncate_text_to_tokens(text, max_tokens).strip()
    # Character fallback when no tokenizer-backed truncator is available.
    return text[: max(1, max_tokens * 4)].strip()
