from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from tesis_unicamp.datasets.preprocessing.rag.retrieval.schemas import RetrievedDocRecord
from tesis_unicamp.datasets.utils.bioasq_rag import (
    BIOASQ_RAG_RESPLIT_DATASET_ID,
    corpus_row_to_text as bioasq_corpus_row_to_text,
    load_bioasq_rag_resplit_corpus,
    load_bioasq_rag_resplit_subset,
)
from tesis_unicamp.datasets.utils.narrativeqa_rag import (
    NARRATIVEQA_RAG_DATASET_ID,
    corpus_row_to_text as narrativeqa_corpus_row_to_text,
    load_narrativeqa_rag_corpus,
    load_narrativeqa_rag_subset,
)
from tesis_unicamp.datasets.utils.qasper_rag import (
    QASPER_RAG_DATASET_ID,
    corpus_row_to_text as qasper_corpus_row_to_text,
    load_qasper_rag_corpus,
    load_qasper_rag_subset,
)
from tesis_unicamp.datasets.utils.telco_dpr_rag import (
    TELCO_DPR_RAG_DATASET_ID,
    corpus_row_to_text as telco_corpus_row_to_text,
    load_telco_dpr_rag_corpus,
    load_telco_dpr_rag_subset,
)
from tesis_unicamp.finetuning.embeddings.datasets import (
    build_corpus_lookup,
    build_relevant_docs,
)
from tesis_unicamp.finetuning.generative.config import (
    DEFAULT_DATASET_SEED,
    DEFAULT_INSTRUCTION,
    MAX_ANSWER_TOKENS,
    MAX_CONTEXT_DOCS,
    MAX_DOC_TOKENS,
    MAX_QUERY_TOKENS,
    MAX_SEQ_LENGTH,
    RELEVANT_DOC_RATIO,
)
from tesis_unicamp.finetuning.generative.formatting import (
    build_training_messages,
    build_training_text,
)
from tesis_unicamp.generation.rag.context import group_retrieved_by_query

LoadSubsetFn = Callable[..., Dataset]
CorpusTextFn = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class GenerativeFinetuningDatasetConfig:
    name: str
    hub_repo_id: str
    load_corpus: LoadSubsetFn
    load_subset: LoadSubsetFn
    corpus_text_fn: CorpusTextFn
    train_split: str = "train"
    eval_split: str = "dev"


GENERATIVE_FINETUNING_DATASET_CONFIGS: dict[str, GenerativeFinetuningDatasetConfig] = {
    "bioasq-resplit": GenerativeFinetuningDatasetConfig(
        name="bioasq-resplit",
        hub_repo_id=BIOASQ_RAG_RESPLIT_DATASET_ID,
        load_corpus=load_bioasq_rag_resplit_corpus,
        load_subset=load_bioasq_rag_resplit_subset,
        corpus_text_fn=bioasq_corpus_row_to_text,
    ),
    "qasper": GenerativeFinetuningDatasetConfig(
        name="qasper",
        hub_repo_id=QASPER_RAG_DATASET_ID,
        load_corpus=load_qasper_rag_corpus,
        load_subset=load_qasper_rag_subset,
        corpus_text_fn=qasper_corpus_row_to_text,
    ),
    "telco-dpr": GenerativeFinetuningDatasetConfig(
        name="telco-dpr",
        hub_repo_id=TELCO_DPR_RAG_DATASET_ID,
        load_corpus=load_telco_dpr_rag_corpus,
        load_subset=load_telco_dpr_rag_subset,
        corpus_text_fn=telco_corpus_row_to_text,
    ),
    "narrativeqa": GenerativeFinetuningDatasetConfig(
        name="narrativeqa",
        hub_repo_id=NARRATIVEQA_RAG_DATASET_ID,
        load_corpus=load_narrativeqa_rag_corpus,
        load_subset=load_narrativeqa_rag_subset,
        corpus_text_fn=narrativeqa_corpus_row_to_text,
    ),
}


def get_generative_finetuning_config(dataset: str) -> GenerativeFinetuningDatasetConfig:
    try:
        return GENERATIVE_FINETUNING_DATASET_CONFIGS[dataset]
    except KeyError as exc:
        available = ", ".join(sorted(GENERATIVE_FINETUNING_DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {available}") from exc


def _retrieved_rows_to_records(rows: Dataset) -> list[RetrievedDocRecord]:
    return [
        {
            "query_id": str(row["query_id"]),
            "corpus_id": str(row["corpus_id"]),
            "rank": int(row["rank"]),
            "retrieval_score": float(row["retrieval_score"]),
            "is_relevant": bool(row["is_relevant"]),
        }
        for row in rows
    ]


def _has_relevant_doc(
    hits: list[RetrievedDocRecord],
    *,
    relevant_ids: set[str],
) -> bool:
    return any(
        hit["is_relevant"] or str(hit["corpus_id"]) in relevant_ids
        for hit in hits
    )


def _ensure_relevant_doc_in_context(
    hits: list[RetrievedDocRecord],
    *,
    relevant_ids: set[str],
    rng: random.Random,
) -> list[RetrievedDocRecord]:
    if not hits or not relevant_ids or _has_relevant_doc(hits, relevant_ids=relevant_ids):
        return hits

    relevant_corpus_id = rng.choice(sorted(relevant_ids))
    replacement = {
        "query_id": hits[0]["query_id"],
        "corpus_id": relevant_corpus_id,
        "rank": 0,
        "retrieval_score": 1.0,
        "is_relevant": True,
    }
    mutable_hits = list(hits)
    insert_at = rng.randrange(len(mutable_hits))
    mutable_hits[insert_at] = replacement  # type: ignore[assignment]
    return mutable_hits


def _select_context_hits(
    hits: list[RetrievedDocRecord],
    *,
    num_docs: int,
    relevant_ids: set[str],
    ensure_relevant: bool,
    rng: random.Random,
) -> list[RetrievedDocRecord]:
    if num_docs <= 0:
        return []

    selected = hits[:num_docs]
    if ensure_relevant:
        selected = _ensure_relevant_doc_in_context(
            selected,
            relevant_ids=relevant_ids,
            rng=rng,
        )
    return selected


def _hits_to_doc_texts(
    hits: list[RetrievedDocRecord],
    corpus_lookup: dict[str, str],
) -> list[str]:
    doc_texts: list[str] = []
    for hit in hits:
        text = corpus_lookup.get(str(hit["corpus_id"]), "").strip()
        if text:
            doc_texts.append(text)
    return doc_texts


def build_generative_training_examples(
    *,
    config: GenerativeFinetuningDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    instruction: str = DEFAULT_INSTRUCTION,
    max_context_docs: int = MAX_CONTEXT_DOCS,
    relevant_doc_ratio: float = RELEVANT_DOC_RATIO,
    seed: int = DEFAULT_DATASET_SEED,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_answer_tokens: int = MAX_ANSWER_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> list[dict[str, object]]:
    corpus = config.load_corpus()
    queries = config.load_subset("queries", split=split)
    answers = config.load_subset("answers", split=split)
    qrels = config.load_subset("qrels", split=split)
    retrieved = config.load_subset("retrieved_docs", split=split)

    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)
    relevant_docs = build_relevant_docs(qrels)
    grouped_retrieved = group_retrieved_by_query(_retrieved_rows_to_records(retrieved))

    query_lookup = {str(row["id"]): str(row["text"]) for row in queries}
    answer_lookup = {str(row["query_id"]): str(row["answer"]) for row in answers}

    rng = random.Random(seed)
    examples: list[dict[str, object]] = []

    for query_id, answer in answer_lookup.items():
        query_text = query_lookup.get(query_id)
        if not query_text or not answer.strip():
            continue

        num_docs = rng.randint(0, max_context_docs)
        hits = grouped_retrieved.get(query_id, [])
        relevant_ids = relevant_docs.get(query_id, set())
        ensure_relevant = num_docs > 0 and rng.random() < relevant_doc_ratio
        selected_hits = _select_context_hits(
            hits,
            num_docs=num_docs,
            relevant_ids=relevant_ids,
            ensure_relevant=ensure_relevant,
            rng=rng,
        )
        doc_texts = _hits_to_doc_texts(selected_hits, corpus_lookup)

        example = build_training_messages(
            tokenizer,
            query=query_text,
            doc_texts=doc_texts,
            answer=answer,
            instruction=instruction,
            max_query_tokens=max_query_tokens,
            max_doc_tokens=max_doc_tokens,
            max_answer_tokens=max_answer_tokens,
            max_seq_length=max_seq_length,
        )
        if example is not None:
            examples.append(example)

    if not examples:
        raise ValueError(
            f"No generative training examples could be built for split {split!r} "
            f"on dataset {config.name!r}."
        )
    return examples


def prepare_training_dataset(
    config: GenerativeFinetuningDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str | None = None,
    seed: int = DEFAULT_DATASET_SEED,
) -> Dataset:
    split = split or config.train_split
    examples = build_generative_training_examples(
        config=config,
        tokenizer=tokenizer,
        split=split,
        seed=seed,
    )
    return Dataset.from_list(examples)


def summarize_training_dataset(
    config: GenerativeFinetuningDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str | None = None,
    seed: int = DEFAULT_DATASET_SEED,
) -> dict[str, int | float]:
    split = split or config.train_split
    examples = build_generative_training_examples(
        config=config,
        tokenizer=tokenizer,
        split=split,
        seed=seed,
    )
    token_counts = [
        len(
            tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        for example in examples
    ]
    with_context = sum(
        1
        for example in examples
        if "## Context:" in example["messages"][0]["content"]
    )
    return {
        "num_examples": len(examples),
        "num_with_context": with_context,
        "num_without_context": len(examples) - with_context,
        "avg_tokens": sum(token_counts) / len(token_counts),
        "max_tokens": max(token_counts),
    }
