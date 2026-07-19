from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import _load_split_records
from tesis_unicamp.datasets.utils.corpus import iter_batches
from tesis_unicamp.generation.base import BaseGenerator
from tesis_unicamp.generation.rag.context import (
    build_corpus_lookup,
    group_retrieved_by_query,
)
from tesis_unicamp.generation.rag.datasets import (
    RagGenerationDatasetConfig,
    load_answers_subset,
    load_corpus_subset,
    load_queries_subset,
)
from tesis_unicamp.query_expansion.prompts import build_rfg_expansion_prompt
from tesis_unicamp.query_expansion.schemas import ExpandedQueryRecord

DEFAULT_STAGE1_TOP_K = 5
DEFAULT_RETRIEVAL_TOP_K = 10
DEFAULT_EXPANSION_K_VALUES = (1, 3, 5, 7, 10)
DEFAULT_MAX_TOKENS_PER_CHUNK = 2048
DEFAULT_EXPANSION_MAX_TOKENS = 2048


def validate_expansion_k_values(
    values: tuple[int, ...],
    *,
    retrieval_top_k: int,
) -> tuple[int, ...]:
    if not values:
        raise ValueError("At least one expansion k value is required.")
    normalized = tuple(sorted(set(values)))
    for value in normalized:
        if value <= 0:
            raise ValueError(f"expansion k must be positive, got {value}.")
        if value > retrieval_top_k:
            raise ValueError(
                f"expansion k ({value}) cannot exceed retrieval_top_k ({retrieval_top_k})."
            )
    return normalized


def generate_expansions_for_split(
    generator: BaseGenerator,
    config: RagGenerationDatasetConfig,
    *,
    retrieved_dir: Path,
    split: str,
    max_context_docs: int = DEFAULT_STAGE1_TOP_K,
    max_prompt_tokens: int | None = None,
    max_tokens_per_chunk: int = DEFAULT_MAX_TOKENS_PER_CHUNK,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> list[ExpandedQueryRecord]:
    """Generate long-form RFG expanded queries from stage-1 retrieved documents."""
    if max_prompt_tokens is None:
        if hasattr(generator, "get_default_max_prompt_tokens"):
            max_prompt_tokens = generator.get_default_max_prompt_tokens()  # type: ignore[attr-defined]
        else:
            max_prompt_tokens = 32000

    truncate_text_to_tokens = _resolve_text_truncator(generator)
    retrieved_records = _load_split_records(retrieved_dir, split)
    grouped = group_retrieved_by_query(retrieved_records)

    corpus = load_corpus_subset(config)
    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)

    queries = load_queries_subset(config, split=split)
    answers = load_answers_subset(config, split=split)
    query_lookup = {str(row["id"]): str(row["text"]) for row in queries}
    answer_lookup = {str(row["query_id"]): str(row["answer"]) for row in answers}

    query_ids = [str(row["id"]) for row in queries if str(row["id"]) in grouped]
    if not query_ids:
        return []

    batch_size = batch_size or generator.batch_size
    records: list[ExpandedQueryRecord] = []
    batch_iterator = iter_batches(query_ids, batch_size)
    if show_progress:
        from tqdm import tqdm

        total_batches = (len(query_ids) + batch_size - 1) // batch_size
        batch_iterator = tqdm(
            batch_iterator,
            desc=f"RFG expansion ({split})",
            unit="batch",
            total=total_batches,
        )

    for batch in batch_iterator:
        user_prompts = [
            _build_expansion_prompt(
                query_lookup[query_id],
                grouped[query_id],
                corpus_lookup=corpus_lookup,
                max_docs=max_context_docs,
                max_tokens_per_chunk=max_tokens_per_chunk,
                truncate_text_to_tokens=truncate_text_to_tokens,
            )
            for query_id in batch
        ]
        expanded_texts = generator.generate_texts(user_prompts)
        for query_id, expanded_query in zip(batch, expanded_texts, strict=True):
            records.append(
                {
                    "query_id": query_id,
                    "question": query_lookup[query_id],
                    "expanded_query": expanded_query,
                    "reference_answer": answer_lookup.get(query_id, ""),
                }
            )

    return records


def _build_expansion_prompt(
    question: str,
    hits: list,
    *,
    corpus_lookup: dict[str, str],
    max_docs: int,
    max_tokens_per_chunk: int,
    truncate_text_to_tokens: Callable[[str, int], str] | None,
) -> str:
    doc_texts: list[str] = []
    for hit in hits[:max_docs]:
        corpus_id = str(hit["corpus_id"])
        text = corpus_lookup.get(corpus_id, "").strip()
        if not text:
            continue
        if truncate_text_to_tokens is not None:
            text = truncate_text_to_tokens(text, max_tokens_per_chunk)
        doc_texts.append(text)
    return build_rfg_expansion_prompt(query=question, doc_texts=doc_texts)


def _resolve_text_truncator(
    generator: BaseGenerator,
) -> Callable[[str, int], str] | None:
    if hasattr(generator, "truncate_text_to_tokens"):
        return generator.truncate_text_to_tokens  # type: ignore[attr-defined]
    return None
