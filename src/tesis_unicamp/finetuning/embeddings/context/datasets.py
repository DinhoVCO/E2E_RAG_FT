from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
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
from tesis_unicamp.finetuning.embeddings.context.config import (
    DEFAULT_DATASET_SEED,
    MAX_CONTEXT_DOCS,
    MAX_DOC_TOKENS,
    MAX_POSITIVE_TOKENS,
    MAX_QUERY_TOKENS,
    MAX_SEQ_LENGTH,
    RELEVANT_DOC_RATIO,
)
from tesis_unicamp.finetuning.embeddings.context.formatting import (
    build_context_anchor_text,
    build_positive_text,
    count_tokens,
)
from tesis_unicamp.finetuning.embeddings.datasets import (
    build_corpus_lookup,
    build_relevant_docs,
)
from tesis_unicamp.generation.rag.context import group_retrieved_by_query

LoadSubsetFn = Callable[..., Dataset]
CorpusTextFn = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class ContextEmbeddingFinetuningDatasetConfig:
    name: str
    hub_repo_id: str
    load_corpus: LoadSubsetFn
    load_subset: LoadSubsetFn
    corpus_text_fn: CorpusTextFn
    train_split: str = "train"
    eval_split: str = "dev"


CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS: dict[str, ContextEmbeddingFinetuningDatasetConfig] = {
    "bioasq-resplit": ContextEmbeddingFinetuningDatasetConfig(
        name="bioasq-resplit",
        hub_repo_id=BIOASQ_RAG_RESPLIT_DATASET_ID,
        load_corpus=load_bioasq_rag_resplit_corpus,
        load_subset=load_bioasq_rag_resplit_subset,
        corpus_text_fn=bioasq_corpus_row_to_text,
    ),
    "qasper": ContextEmbeddingFinetuningDatasetConfig(
        name="qasper",
        hub_repo_id=QASPER_RAG_DATASET_ID,
        load_corpus=load_qasper_rag_corpus,
        load_subset=load_qasper_rag_subset,
        corpus_text_fn=qasper_corpus_row_to_text,
    ),
    "telco-dpr": ContextEmbeddingFinetuningDatasetConfig(
        name="telco-dpr",
        hub_repo_id=TELCO_DPR_RAG_DATASET_ID,
        load_corpus=load_telco_dpr_rag_corpus,
        load_subset=load_telco_dpr_rag_subset,
        corpus_text_fn=telco_corpus_row_to_text,
    ),
    "narrativeqa": ContextEmbeddingFinetuningDatasetConfig(
        name="narrativeqa",
        hub_repo_id=NARRATIVEQA_RAG_DATASET_ID,
        load_corpus=load_narrativeqa_rag_corpus,
        load_subset=load_narrativeqa_rag_subset,
        corpus_text_fn=narrativeqa_corpus_row_to_text,
    ),
}


def get_context_embedding_finetuning_config(
    dataset: str,
) -> ContextEmbeddingFinetuningDatasetConfig:
    try:
        return CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS[dataset]
    except KeyError as exc:
        available = ", ".join(sorted(CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS))
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


def _group_qrels_by_query(qrels: Dataset) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in qrels:
        query_id = str(row.get("query_id") or row.get("query-id"))
        corpus_id = str(row.get("corpus_id") or row.get("corpus-id"))
        grouped[query_id].append(corpus_id)
    return dict(grouped)


def _sample_context_for_query(
    *,
    query_id: str,
    hits: list[RetrievedDocRecord],
    relevant_ids: set[str],
    corpus_lookup: dict[str, str],
    rng: random.Random,
    max_context_docs: int,
    relevant_doc_ratio: float,
) -> list[str]:
    num_docs = rng.randint(0, max_context_docs)
    ensure_relevant = num_docs > 0 and rng.random() < relevant_doc_ratio
    selected_hits = _select_context_hits(
        hits,
        num_docs=num_docs,
        relevant_ids=relevant_ids,
        ensure_relevant=ensure_relevant,
        rng=rng,
    )
    return _hits_to_doc_texts(selected_hits, corpus_lookup)


def build_context_training_pairs(
    *,
    config: ContextEmbeddingFinetuningDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    max_context_docs: int = MAX_CONTEXT_DOCS,
    relevant_doc_ratio: float = RELEVANT_DOC_RATIO,
    seed: int = DEFAULT_DATASET_SEED,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_positive_tokens: int = MAX_POSITIVE_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> Dataset:
    corpus = config.load_corpus()
    queries = config.load_subset("queries", split=split)
    qrels = config.load_subset("qrels", split=split)
    retrieved = config.load_subset("retrieved_docs", split=split)

    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)
    relevant_docs = build_relevant_docs(qrels)
    grouped_retrieved = group_retrieved_by_query(_retrieved_rows_to_records(retrieved))
    grouped_qrels = _group_qrels_by_query(qrels)

    query_lookup = {str(row["id"]): str(row["text"]) for row in queries}

    rng = random.Random(seed)
    anchors: list[str] = []
    positives: list[str] = []

    for query_id, corpus_ids in grouped_qrels.items():
        query_text = query_lookup.get(query_id)
        if not query_text:
            continue

        doc_texts = _sample_context_for_query(
            query_id=query_id,
            hits=grouped_retrieved.get(query_id, []),
            relevant_ids=relevant_docs.get(query_id, set()),
            corpus_lookup=corpus_lookup,
            rng=rng,
            max_context_docs=max_context_docs,
            relevant_doc_ratio=relevant_doc_ratio,
        )
        anchor = build_context_anchor_text(
            tokenizer,
            query=query_text,
            doc_texts=doc_texts,
            max_query_tokens=max_query_tokens,
            max_doc_tokens=max_doc_tokens,
            max_seq_length=max_seq_length,
        )
        if anchor is None:
            continue

        for corpus_id in corpus_ids:
            positive_raw = corpus_lookup.get(corpus_id, "").strip()
            if not positive_raw:
                continue
            positive = build_positive_text(
                tokenizer,
                text=positive_raw,
                max_positive_tokens=max_positive_tokens,
            )
            if not positive:
                continue
            anchors.append(anchor)
            positives.append(positive)

    if not anchors:
        raise ValueError(
            f"No context embedding training pairs could be built for split {split!r} "
            f"on dataset {config.name!r}."
        )

    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def prepare_training_dataset(
    config: ContextEmbeddingFinetuningDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str | None = None,
    seed: int = DEFAULT_DATASET_SEED,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_positive_tokens: int = MAX_POSITIVE_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> Dataset:
    split = split or config.train_split
    return build_context_training_pairs(
        config=config,
        tokenizer=tokenizer,
        split=split,
        seed=seed,
        max_doc_tokens=max_doc_tokens,
        max_query_tokens=max_query_tokens,
        max_positive_tokens=max_positive_tokens,
        max_seq_length=max_seq_length,
    )


def _build_context_query_lookup(
    *,
    config: ContextEmbeddingFinetuningDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    seed: int,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, str]:
    queries = config.load_subset("queries", split=split)
    qrels = config.load_subset("qrels", split=split)
    retrieved = config.load_subset("retrieved_docs", split=split)
    corpus = config.load_corpus()

    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)
    relevant_docs = build_relevant_docs(qrels)
    grouped_retrieved = group_retrieved_by_query(_retrieved_rows_to_records(retrieved))

    rng = random.Random(seed)
    query_lookup: dict[str, str] = {}

    for row in queries:
        query_id = str(row["id"])
        query_text = str(row["text"])
        if query_id not in relevant_docs or not relevant_docs[query_id]:
            continue

        doc_texts = _sample_context_for_query(
            query_id=query_id,
            hits=grouped_retrieved.get(query_id, []),
            relevant_ids=relevant_docs[query_id],
            corpus_lookup=corpus_lookup,
            rng=rng,
            max_context_docs=MAX_CONTEXT_DOCS,
            relevant_doc_ratio=RELEVANT_DOC_RATIO,
        )
        anchor = build_context_anchor_text(
            tokenizer,
            query=query_text,
            doc_texts=doc_texts,
            max_query_tokens=max_query_tokens,
            max_doc_tokens=max_doc_tokens,
            max_seq_length=max_seq_length,
        )
        if anchor is not None:
            query_lookup[query_id] = anchor

    return query_lookup


def prepare_ir_eval_inputs(
    config: ContextEmbeddingFinetuningDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str | None = None,
    seed: int = DEFAULT_DATASET_SEED,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    split = split or config.eval_split
    corpus = config.load_corpus()
    qrels = config.load_subset("qrels", split=split)

    corpus_dict = build_corpus_lookup(corpus, config.corpus_text_fn)
    relevant_docs = build_relevant_docs(qrels)
    query_dict = _build_context_query_lookup(
        config=config,
        tokenizer=tokenizer,
        split=split,
        seed=seed,
        max_doc_tokens=max_doc_tokens,
        max_query_tokens=max_query_tokens,
        max_seq_length=max_seq_length,
    )

    if not query_dict:
        raise ValueError(f"No evaluation queries with positives in split {split!r}.")

    return query_dict, corpus_dict, relevant_docs


def ir_evaluator_name(
    config: ContextEmbeddingFinetuningDatasetConfig,
    *,
    split: str | None = None,
) -> str:
    return f"{config.name}-ctx-{split or config.eval_split}"


def default_ir_metric_for_best_model(
    config: ContextEmbeddingFinetuningDatasetConfig,
    *,
    split: str | None = None,
    ndcg_k: int = 10,
    score_function: str = "cosine",
) -> str:
    evaluator_name = ir_evaluator_name(config, split=split)
    return f"eval_{evaluator_name}_{score_function}_ndcg@{ndcg_k}"


def build_ir_evaluator(
    config: ContextEmbeddingFinetuningDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str | None = None,
    seed: int = DEFAULT_DATASET_SEED,
    batch_size: int = 32,
    name: str | None = None,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> InformationRetrievalEvaluator:
    queries, corpus, relevant_docs = prepare_ir_eval_inputs(
        config,
        tokenizer,
        split=split,
        seed=seed,
        max_doc_tokens=max_doc_tokens,
        max_query_tokens=max_query_tokens,
        max_seq_length=max_seq_length,
    )
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        batch_size=batch_size,
        name=name or ir_evaluator_name(config, split=split),
    )


def summarize_training_dataset(
    config: ContextEmbeddingFinetuningDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    split: str | None = None,
    seed: int = DEFAULT_DATASET_SEED,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_positive_tokens: int = MAX_POSITIVE_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, int | float]:
    split = split or config.train_split
    dataset = build_context_training_pairs(
        config=config,
        tokenizer=tokenizer,
        split=split,
        seed=seed,
        max_doc_tokens=max_doc_tokens,
        max_query_tokens=max_query_tokens,
        max_positive_tokens=max_positive_tokens,
        max_seq_length=max_seq_length,
    )
    anchor_tokens = [count_tokens(tokenizer, text) for text in dataset["anchor"]]
    with_context = sum(1 for text in dataset["anchor"] if "## Context:" in text)
    return {
        "num_pairs": len(dataset),
        "num_with_context": with_context,
        "num_without_context": len(dataset) - with_context,
        "avg_anchor_tokens": sum(anchor_tokens) / len(anchor_tokens),
        "max_anchor_tokens": max(anchor_tokens),
    }
