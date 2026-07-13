from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import _load_split_records
from tesis_unicamp.datasets.utils.corpus import iter_batches
from tesis_unicamp.finetuning.generative.formatting import (
    build_qa_user_content,
    build_user_content,
)
from tesis_unicamp.generation.base import BaseGenerator
from tesis_unicamp.generation.rag.context import (
    DEFAULT_MAX_TOKENS_PER_CHUNK,
    build_corpus_lookup,
    group_retrieved_by_query,
)
from tesis_unicamp.generation.rag.datasets import (
    RagGenerationDatasetConfig,
    load_answers_subset,
    load_corpus_subset,
    load_queries_subset,
)
from tesis_unicamp.generation.rag.schemas import GeneratedAnswerRecord

PromptMode = Literal["qa", "rag-finetune"]


def generate_answers_for_split(
    generator: BaseGenerator,
    config: RagGenerationDatasetConfig,
    *,
    retrieved_dir: Path | None,
    split: str,
    max_context_docs: int = 10,
    max_prompt_tokens: int | None = None,
    max_tokens_per_chunk: int = DEFAULT_MAX_TOKENS_PER_CHUNK,
    batch_size: int | None = None,
    show_progress: bool = True,
    prompt_mode: PromptMode = "rag-finetune",
) -> list[GeneratedAnswerRecord]:
    """Generate answers for one split."""
    if retrieved_dir is None:
        return _generate_answers_without_retrieval(
            generator,
            config,
            split=split,
            batch_size=batch_size,
            show_progress=show_progress,
            prompt_mode=prompt_mode,
        )

    if prompt_mode != "rag-finetune":
        raise ValueError(
            f"prompt_mode {prompt_mode!r} requires --no-retrieval. "
            "Use prompt_mode='rag-finetune' when retrieved_dir is set."
        )

    return _generate_answers_with_finetune_rag_prompt(
        generator,
        config,
        retrieved_dir=retrieved_dir,
        split=split,
        max_context_docs=max_context_docs,
        max_prompt_tokens=max_prompt_tokens,
        max_tokens_per_chunk=max_tokens_per_chunk,
        batch_size=batch_size,
        show_progress=show_progress,
    )


def _generate_answers_with_finetune_rag_prompt(
    generator: BaseGenerator,
    config: RagGenerationDatasetConfig,
    *,
    retrieved_dir: Path,
    split: str,
    max_context_docs: int,
    max_prompt_tokens: int | None,
    max_tokens_per_chunk: int,
    batch_size: int | None,
    show_progress: bool,
) -> list[GeneratedAnswerRecord]:
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
    records: list[GeneratedAnswerRecord] = []
    batch_iterator = iter_batches(query_ids, batch_size)
    if show_progress:
        from tqdm import tqdm

        total_batches = (len(query_ids) + batch_size - 1) // batch_size
        batch_iterator = tqdm(
            batch_iterator,
            desc=f"Generating ({split}, rag-finetune)",
            unit="batch",
            total=total_batches,
        )

    for batch in batch_iterator:
        user_prompts = [
            _build_finetune_rag_user_prompt(
                query_lookup[query_id],
                grouped[query_id],
                corpus_lookup,
                max_docs=max_context_docs,
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


def _generate_answers_without_retrieval(
    generator: BaseGenerator,
    config: RagGenerationDatasetConfig,
    *,
    split: str,
    batch_size: int | None,
    show_progress: bool,
    prompt_mode: PromptMode,
) -> list[GeneratedAnswerRecord]:
    if prompt_mode not in {"qa", "rag-finetune"}:
        raise ValueError(
            f"Unsupported prompt_mode {prompt_mode!r} for no-retrieval generation."
        )

    queries = load_queries_subset(config, split=split)
    answers = load_answers_subset(config, split=split)
    query_lookup = {str(row["id"]): str(row["text"]) for row in queries}
    answer_lookup = {str(row["query_id"]): str(row["answer"]) for row in answers}

    query_ids = list(query_lookup)
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
            desc=f"Generating ({split}, {prompt_mode}, no-retrieval)",
            unit="batch",
            total=total_batches,
        )

    for batch in batch_iterator:
        user_prompts = []
        for query_id in batch:
            question = query_lookup[query_id]
            if prompt_mode == "qa":
                user_prompts.append(build_qa_user_content(query=question))
            else:
                user_prompts.append(build_user_content(query=question, doc_texts=[]))

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


def _build_finetune_rag_user_prompt(
    question: str,
    hits: list,
    corpus_lookup: dict[str, str],
    *,
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
    return build_user_content(query=question, doc_texts=doc_texts)


def _resolve_text_truncator(
    generator: BaseGenerator,
) -> Callable[[str, int], str] | None:
    if hasattr(generator, "truncate_text_to_tokens"):
        return generator.truncate_text_to_tokens  # type: ignore[attr-defined]
    return None
