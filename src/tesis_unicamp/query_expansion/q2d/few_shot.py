from __future__ import annotations

import random
from collections.abc import Callable

from tesis_unicamp.datasets.utils.retrieval import build_relevant_corpus_ids
from tesis_unicamp.generation.rag.context import build_corpus_lookup
from tesis_unicamp.generation.rag.datasets import (
    RagGenerationDatasetConfig,
    load_corpus_subset,
    load_qrels_subset,
    load_queries_subset,
)
from tesis_unicamp.query_expansion.q2d.schemas import FewShotExample

DEFAULT_NUM_FEW_SHOT = 4
DEFAULT_FEW_SHOT_SPLIT = "train"
DEFAULT_MAX_PASSAGE_TOKENS = 2048


def build_few_shot_pool(
    config: RagGenerationDatasetConfig,
    *,
    split: str = DEFAULT_FEW_SHOT_SPLIT,
    max_passage_tokens: int | None = DEFAULT_MAX_PASSAGE_TOKENS,
    truncate_text_to_tokens: Callable[[str, int], str] | None = None,
) -> list[FewShotExample]:
    """Build (query, passage) pairs from train qrels and corpus for few-shot demos."""
    if max_passage_tokens is not None and max_passage_tokens > 0 and truncate_text_to_tokens is None:
        raise ValueError(
            "truncate_text_to_tokens is required when max_passage_tokens is set."
        )

    queries = load_queries_subset(config, split=split)
    qrels = load_qrels_subset(config, split=split)
    corpus = load_corpus_subset(config)
    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)
    relevant = build_relevant_corpus_ids(qrels)

    pool: list[FewShotExample] = []
    for row in queries:
        query_id = str(row["id"])
        query_text = str(row["text"]).strip()
        if not query_text:
            continue

        corpus_ids = sorted(relevant.get(query_id, ()))
        passage = ""
        for corpus_id in corpus_ids:
            candidate = corpus_lookup.get(corpus_id, "").strip()
            if candidate:
                passage = candidate
                break
        if not passage:
            continue

        if (
            max_passage_tokens is not None
            and max_passage_tokens > 0
            and truncate_text_to_tokens is not None
        ):
            passage = truncate_text_to_tokens(passage, max_passage_tokens).strip()
            if not passage:
                continue

        pool.append(
            {
                "query_id": query_id,
                "query": query_text,
                "passage": passage,
            }
        )
    return pool


def sample_few_shot_examples(
    pool: list[FewShotExample],
    *,
    num_examples: int = DEFAULT_NUM_FEW_SHOT,
    rng: random.Random,
    exclude_query_ids: set[str] | None = None,
) -> list[FewShotExample]:
    """Sample ``num_examples`` demos from the pool, optionally excluding query ids."""
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")

    candidates = pool
    if exclude_query_ids:
        candidates = [item for item in pool if item["query_id"] not in exclude_query_ids]
    if len(candidates) < num_examples:
        raise ValueError(
            f"Need at least {num_examples} few-shot examples, found {len(candidates)}."
        )
    return rng.sample(candidates, num_examples)
