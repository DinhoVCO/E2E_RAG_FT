from __future__ import annotations

import random
from collections.abc import Callable

from tesis_unicamp.datasets.utils.corpus import iter_batches
from tesis_unicamp.generation.base import BaseGenerator
from tesis_unicamp.generation.rag.datasets import (
    RagGenerationDatasetConfig,
    load_answers_subset,
    load_queries_subset,
)
from tesis_unicamp.query_expansion.q2d.few_shot import (
    DEFAULT_FEW_SHOT_SPLIT,
    DEFAULT_MAX_PASSAGE_TOKENS,
    DEFAULT_NUM_FEW_SHOT,
    build_few_shot_pool,
    sample_few_shot_examples,
)
from tesis_unicamp.query_expansion.q2d.prompts import build_q2d_prompt
from tesis_unicamp.query_expansion.q2d.schemas import Q2dRecord

DEFAULT_Q2D_MAX_TOKENS = 512
DEFAULT_Q2D_TEMPERATURE = 0.7


def _build_expanded_query(question: str, generated_passage: str) -> str:
    question = question.strip()
    passage = generated_passage.strip()
    if not passage:
        return question
    return f"{question} {passage}"


def _resolve_text_truncator(
    generator: BaseGenerator,
) -> Callable[[str, int], str] | None:
    if hasattr(generator, "truncate_text_to_tokens"):
        return generator.truncate_text_to_tokens  # type: ignore[attr-defined]
    return None


def generate_q2d_for_split(
    generator: BaseGenerator,
    config: RagGenerationDatasetConfig,
    *,
    split: str,
    few_shot_pool: list | None = None,
    num_few_shot: int = DEFAULT_NUM_FEW_SHOT,
    few_shot_split: str = DEFAULT_FEW_SHOT_SPLIT,
    max_passage_tokens: int | None = DEFAULT_MAX_PASSAGE_TOKENS,
    seed: int = 42,
    per_query_sampling: bool = True,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> list[Q2dRecord]:
    """Generate Query2Doc passages and query+passage expansions for a split."""
    truncate_text_to_tokens = _resolve_text_truncator(generator)
    pool = few_shot_pool or build_few_shot_pool(
        config,
        split=few_shot_split,
        max_passage_tokens=max_passage_tokens,
        truncate_text_to_tokens=truncate_text_to_tokens,
    )
    if not pool:
        raise ValueError(f"No few-shot examples available from split {few_shot_split!r}.")

    queries = load_queries_subset(config, split=split)
    answers = load_answers_subset(config, split=split)
    answer_lookup = {str(row["query_id"]): str(row["answer"]) for row in answers}

    query_ids = [str(row["id"]) for row in queries]
    query_lookup = {str(row["id"]): str(row["text"]) for row in queries}
    if not query_ids:
        return []

    batch_size = batch_size or generator.batch_size
    records: list[Q2dRecord] = []
    batch_iterator = iter_batches(query_ids, batch_size)
    if show_progress:
        from tqdm import tqdm

        total_batches = (len(query_ids) + batch_size - 1) // batch_size
        batch_iterator = tqdm(
            batch_iterator,
            desc=f"Query2Doc generation ({split})",
            unit="batch",
            total=total_batches,
        )

    for batch in batch_iterator:
        prompts: list[str] = []
        batch_examples: list[list] = []
        for query_id in batch:
            if per_query_sampling:
                rng = random.Random(f"{seed}:{query_id}")
            else:
                rng = random.Random(seed)
            examples = sample_few_shot_examples(
                pool,
                num_examples=num_few_shot,
                rng=rng,
                exclude_query_ids={query_id},
            )
            batch_examples.append(examples)
            prompts.append(build_q2d_prompt(query_lookup[query_id], examples))

        passages = generator.generate_texts(prompts)
        for query_id, generated_passage, examples in zip(
            batch, passages, batch_examples, strict=True
        ):
            question = query_lookup[query_id]
            records.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "generated_passage": generated_passage,
                    "expanded_query": _build_expanded_query(question, generated_passage),
                    "reference_answer": answer_lookup.get(query_id, ""),
                    "few_shot_examples": examples,
                }
            )

    return records
