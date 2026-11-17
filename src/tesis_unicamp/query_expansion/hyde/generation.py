from __future__ import annotations

from tesis_unicamp.datasets.utils.corpus import iter_batches
from tesis_unicamp.generation.base import BaseGenerator
from tesis_unicamp.generation.rag.datasets import (
    RagGenerationDatasetConfig,
    load_answers_subset,
    load_queries_subset,
)
from tesis_unicamp.query_expansion.hyde.prompts import build_hyde_prompt
from tesis_unicamp.query_expansion.hyde.schemas import HydeRecord

DEFAULT_NUM_PASSAGES = 8
DEFAULT_HYDE_MAX_TOKENS = 512
DEFAULT_HYDE_TEMPERATURE = 0.7


def generate_hyde_passages_for_split(
    generator: BaseGenerator,
    config: RagGenerationDatasetConfig,
    *,
    split: str,
    num_passages: int = DEFAULT_NUM_PASSAGES,
    batch_size: int | None = None,
    show_progress: bool = True,
) -> list[HydeRecord]:
    """Generate ``num_passages`` pseudo-documents per query for HyDE."""
    if num_passages <= 0:
        raise ValueError("num_passages must be positive.")
    if not hasattr(generator, "generate_texts_multi"):
        raise TypeError(
            "HyDE generation requires a generator with generate_texts_multi(); "
            "use VLLMOfflineGenerator."
        )

    queries = load_queries_subset(config, split=split)
    answers = load_answers_subset(config, split=split)
    answer_lookup = {str(row["query_id"]): str(row["answer"]) for row in answers}

    query_ids = [str(row["id"]) for row in queries]
    query_lookup = {str(row["id"]): str(row["text"]) for row in queries}
    if not query_ids:
        return []

    batch_size = batch_size or generator.batch_size
    records: list[HydeRecord] = []
    batch_iterator = iter_batches(query_ids, batch_size)
    if show_progress:
        from tqdm import tqdm

        total_batches = (len(query_ids) + batch_size - 1) // batch_size
        batch_iterator = tqdm(
            batch_iterator,
            desc=f"HyDE generation ({split})",
            unit="batch",
            total=total_batches,
        )

    for batch in batch_iterator:
        prompts = [build_hyde_prompt(query_lookup[query_id]) for query_id in batch]
        multi_outputs = generator.generate_texts_multi(prompts, n=num_passages)
        for query_id, passages in zip(batch, multi_outputs, strict=True):
            records.append(
                {
                    "query_id": query_id,
                    "question": query_lookup[query_id],
                    "pseudo_passages": passages,
                    "reference_answer": answer_lookup.get(query_id, ""),
                }
            )

    return records
