from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import _load_split_records
from tesis_unicamp.datasets.utils.corpus import iter_batches
from tesis_unicamp.generation.base import BaseGenerator
from tesis_unicamp.generation.rag.context import (
    DEFAULT_MAX_TOKENS_PER_CHUNK,
    build_corpus_lookup,
    build_rag_user_prompt_with_budget,
    group_retrieved_by_query,
)
from tesis_unicamp.generation.rag.datasets import (
    RagGenerationDatasetConfig,
    load_answers_subset,
    load_corpus_subset,
    load_queries_subset,
)
from tesis_unicamp.generation.rag.schemas import GeneratedAnswerRecord


def generate_answers_for_split(
    generator: BaseGenerator,
    config: RagGenerationDatasetConfig,
    *,
    retrieved_dir: Path,
    split: str,
    max_context_docs: int = 10,
    max_prompt_tokens: int | None = None,
    max_tokens_per_chunk: int = DEFAULT_MAX_TOKENS_PER_CHUNK,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> list[GeneratedAnswerRecord]:
    """Generate RAG answers for one split using local retrieved_docs and Hub subsets."""
    if max_prompt_tokens is None:
        if hasattr(generator, "get_default_max_prompt_tokens"):
            max_prompt_tokens = generator.get_default_max_prompt_tokens()  # type: ignore[attr-defined]
        else:
            max_prompt_tokens = 32000

    count_prompt_tokens = _resolve_prompt_token_counter(generator)
    truncate_text_to_tokens = _resolve_text_truncator(generator)

    retrieved_records = _load_split_records(retrieved_dir, split)
    grouped = group_retrieved_by_query(retrieved_records)

    corpus = load_corpus_subset(config)
    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)

    queries = load_queries_subset(config, split=split)
    answers = load_answers_subset(config, split=split)
    query_lookup = {str(row["id"]): str(row["text"]) for row in queries}
    answer_lookup = {str(row["query_id"]): str(row["answer"]) for row in answers}

    query_ids = [
        str(row["id"])
        for row in queries
        if str(row["id"]) in grouped
    ]
    if not query_ids:
        return []

    batch_size = batch_size or generator.batch_size
    records: list[GeneratedAnswerRecord] = []

    batch_iterator = iter_batches(query_ids, batch_size)
    if show_progress:
        from tqdm import tqdm

        total_batches = (len(query_ids) + batch_size - 1) // batch_size
        batch_iterator = tqdm(
            batch_iterator,
            desc=f"Generating ({split})",
            unit="batch",
            total=total_batches,
        )

    for batch in batch_iterator:
        user_prompts = [
            build_rag_user_prompt_with_budget(
                query_lookup[query_id],
                grouped[query_id],
                corpus_lookup,
                max_docs=max_context_docs,
                count_prompt_tokens=count_prompt_tokens,
                max_prompt_tokens=max_prompt_tokens,
                max_tokens_per_chunk=max_tokens_per_chunk,
                truncate_text_to_tokens=truncate_text_to_tokens,
            )
            for query_id in batch
        ]
        generated_answers = generator.generate_texts(user_prompts)
        for query_id, generated_answer in zip(batch, generated_answers, strict=True):
            records.append(
                {
                    "query_id": query_id,
                    "question": query_lookup[query_id],
                    "generated_answer": generated_answer,
                    "reference_answer": answer_lookup.get(query_id, ""),
                }
            )

    return records


def _resolve_prompt_token_counter(
    generator: BaseGenerator,
) -> Callable[[str], int]:
    if hasattr(generator, "count_formatted_prompt_tokens"):
        return generator.count_formatted_prompt_tokens  # type: ignore[attr-defined]

    def _approximate_counter(prompt: str) -> int:
        return max(1, len(prompt) // 4)

    return _approximate_counter


def _resolve_text_truncator(
    generator: BaseGenerator,
) -> Callable[[str, int], str] | None:
    if hasattr(generator, "truncate_text_to_tokens"):
        return generator.truncate_text_to_tokens  # type: ignore[attr-defined]
    return None
