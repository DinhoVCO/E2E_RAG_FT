from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ragas import EvaluationDataset

from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import _load_split_records
from tesis_unicamp.generation.rag.context import (
    DEFAULT_MAX_TOKENS_PER_CHUNK,
    build_corpus_lookup,
    estimate_tokens_per_chunk,
    group_retrieved_by_query,
)
from tesis_unicamp.generation.rag.datasets import (
    RagGenerationDatasetConfig,
    load_corpus_subset,
)
from tesis_unicamp.generation.rag.io import load_generated_answers
from tesis_unicamp.generation.rag.schemas import GeneratedAnswerRecord


def load_generation_run_settings(generation_dir: Path) -> dict[str, Any]:
    path = generation_dir / "run_settings.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing generation run settings: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_context_chunks_from_hits(
    hits: list[dict[str, Any]],
    corpus_lookup: dict[str, str],
    *,
    max_docs: int,
    per_chunk_tokens: int,
    truncate_text_to_tokens: Callable[[str, int], str] | None = None,
) -> list[str]:
    """Build the retrieved context chunks used for RAGAS faithfulness."""
    chunks: list[str] = []
    for hit in hits[:max_docs]:
        corpus_id = str(hit["corpus_id"])
        text = corpus_lookup.get(corpus_id, "").strip()
        if not text:
            continue
        if truncate_text_to_tokens is not None:
            text = truncate_text_to_tokens(text, per_chunk_tokens).strip()
        else:
            text = text[: max(1, per_chunk_tokens * 4)].strip()
        if text:
            chunks.append(text)
    return chunks


def build_ragas_evaluation_dataset(
    config: RagGenerationDatasetConfig,
    *,
    generation_dir: Path,
    retrieved_dir: Path,
    split: str,
    top_k: int,
    max_tokens_per_chunk: int | None = DEFAULT_MAX_TOKENS_PER_CHUNK,
    max_prompt_tokens: int = 32000,
    count_prompt_tokens: Callable[[str], int] | None = None,
    truncate_text_to_tokens: Callable[[str, int], str] | None = None,
) -> EvaluationDataset:
    """Build a RAGAS dataset from generated answers and retrieved documents."""
    generated_records = load_generated_answers(generation_dir, split)
    if not generated_records:
        raise ValueError(
            f"No generated answers found in {generation_dir / split / 'generated_answers.json'}"
        )

    retrieved_records = _load_split_records(retrieved_dir, split)
    grouped = group_retrieved_by_query(retrieved_records)

    corpus = load_corpus_subset(config)
    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)

    rows: list[dict[str, Any]] = []
    missing_context = 0

    for record in generated_records:
        query_id = str(record["query_id"])
        hits = grouped.get(query_id, [])
        if not hits:
            missing_context += 1

        per_chunk_tokens = estimate_tokens_per_chunk(
            str(record["question"]),
            max_docs=top_k,
            count_prompt_tokens=count_prompt_tokens or _approximate_token_counter,
            max_prompt_tokens=max_prompt_tokens,
            max_tokens_per_chunk=max_tokens_per_chunk,
        )
        contexts = build_context_chunks_from_hits(
            hits,
            corpus_lookup,
            max_docs=top_k,
            per_chunk_tokens=per_chunk_tokens,
            truncate_text_to_tokens=truncate_text_to_tokens,
        )
        rows.append(
            {
                "query_id": query_id,
                "user_input": record["question"],
                "response": record["generated_answer"],
                "reference": record.get("reference_answer", "") or "",
                "retrieved_contexts": contexts,
            }
        )

    if missing_context:
        print(
            f"Warning: {missing_context} generated answers have no retrieved docs "
            f"for split {split!r}."
        )

    return EvaluationDataset.from_list(rows)


def _approximate_token_counter(text: str) -> int:
    return max(1, len(text) // 4)


def resolve_generation_paths(
    generation_dir: Path,
    *,
    retrieved_dir: Path | None = None,
    split: str | None = None,
    top_k: int | None = None,
    max_tokens_per_chunk: int | None = None,
    max_prompt_tokens: int | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Resolve evaluation inputs from a generation run directory."""
    settings = load_generation_run_settings(generation_dir)
    return {
        "dataset": dataset or str(settings["dataset"]),
        "retrieved_dir": Path(retrieved_dir or settings["retrieved_dir"]),
        "split": split or str(settings.get("split", "test")),
        "top_k": top_k if top_k is not None else int(settings.get("top_k", 10)),
        "max_tokens_per_chunk": (
            max_tokens_per_chunk
            if max_tokens_per_chunk is not None
            else settings.get("max_tokens_per_chunk", DEFAULT_MAX_TOKENS_PER_CHUNK)
        ),
        "max_prompt_tokens": (
            max_prompt_tokens
            if max_prompt_tokens is not None
            else int(settings.get("max_prompt_tokens", 32000))
        ),
        "generation_model": settings.get("model"),
        "generation_run_label": settings.get("run_label"),
        "retrieval_run_label": settings.get("retrieval_run_label"),
        "use_chat_template": bool(settings.get("use_chat_template", True)),
        "num_answers": int(settings.get("num_answers", 0)),
    }


def records_from_evaluation_dataset(
    dataset: EvaluationDataset,
) -> list[GeneratedAnswerRecord]:
    """Convert a RAGAS dataset back to generated-answer records."""
    records: list[GeneratedAnswerRecord] = []
    for sample in dataset.samples:
        row = sample.model_dump()
        records.append(
            {
                "query_id": str(row.get("query_id", "")),
                "question": str(row["user_input"]),
                "generated_answer": str(row["response"]),
                "reference_answer": str(row.get("reference") or ""),
            }
        )
    return records
