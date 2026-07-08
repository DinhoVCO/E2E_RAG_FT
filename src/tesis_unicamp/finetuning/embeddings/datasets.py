from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

from tesis_unicamp.datasets.utils.bioasq_rag import (
    BIOASQ_RAG_RESPLIT_DATASET_ID,
    corpus_row_to_text as bioasq_corpus_row_to_text,
    load_bioasq_rag_resplit_subset,
    query_to_instruct_text,
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
from tesis_unicamp.finetuning.embeddings.config import EMBEDDING_FINETUNING_DATASET_IDS

LoadSubsetFn = Callable[..., Dataset]
CorpusTextFn = Callable[[dict[str, Any]], str]
QueryTextFn = Callable[[str], str]


@dataclass(frozen=True)
class EmbeddingFinetuningDatasetConfig:
    name: str
    hub_repo_id: str
    load_corpus: LoadSubsetFn
    load_subset: LoadSubsetFn
    corpus_text_fn: CorpusTextFn
    query_text_fn: QueryTextFn = query_to_instruct_text
    train_split: str = "train"
    eval_split: str = "dev"


def _load_resplit_corpus(*, split: str = "train") -> Dataset:
    dataset = load_dataset(BIOASQ_RAG_RESPLIT_DATASET_ID, "corpus")
    if isinstance(dataset, DatasetDict):
        return dataset[split]
    return dataset


EMBEDDING_FINETUNING_DATASET_CONFIGS: dict[str, EmbeddingFinetuningDatasetConfig] = {
    "bioasq-resplit": EmbeddingFinetuningDatasetConfig(
        name="bioasq-resplit",
        hub_repo_id=EMBEDDING_FINETUNING_DATASET_IDS["bioasq-resplit"],
        load_corpus=_load_resplit_corpus,
        load_subset=load_bioasq_rag_resplit_subset,
        corpus_text_fn=bioasq_corpus_row_to_text,
    ),
    "qasper": EmbeddingFinetuningDatasetConfig(
        name="qasper",
        hub_repo_id=QASPER_RAG_DATASET_ID,
        load_corpus=load_qasper_rag_corpus,
        load_subset=load_qasper_rag_subset,
        corpus_text_fn=qasper_corpus_row_to_text,
    ),
    "telco-dpr": EmbeddingFinetuningDatasetConfig(
        name="telco-dpr",
        hub_repo_id=TELCO_DPR_RAG_DATASET_ID,
        load_corpus=load_telco_dpr_rag_corpus,
        load_subset=load_telco_dpr_rag_subset,
        corpus_text_fn=telco_corpus_row_to_text,
    ),
    "narrativeqa": EmbeddingFinetuningDatasetConfig(
        name="narrativeqa",
        hub_repo_id=NARRATIVEQA_RAG_DATASET_ID,
        load_corpus=load_narrativeqa_rag_corpus,
        load_subset=load_narrativeqa_rag_subset,
        corpus_text_fn=narrativeqa_corpus_row_to_text,
    ),
}


def get_embedding_finetuning_config(dataset: str) -> EmbeddingFinetuningDatasetConfig:
    try:
        return EMBEDDING_FINETUNING_DATASET_CONFIGS[dataset]
    except KeyError as exc:
        available = ", ".join(sorted(EMBEDDING_FINETUNING_DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {available}") from exc


def build_corpus_lookup(
    corpus: Dataset,
    corpus_text_fn: CorpusTextFn,
) -> dict[str, str]:
    return {str(row["id"]): corpus_text_fn(row) for row in corpus}


def build_query_lookup(
    queries: Dataset,
    query_text_fn: QueryTextFn | None,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in queries:
        query_id = str(row["id"])
        raw_text = str(row["text"])
        lookup[query_id] = query_text_fn(raw_text) if query_text_fn else raw_text
    return lookup


def build_relevant_docs(qrels: Dataset) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        query_id = str(row.get("query_id") or row.get("query-id"))
        corpus_id = str(row.get("corpus_id") or row.get("corpus-id"))
        relevant[query_id].add(corpus_id)
    return dict(relevant)


def _filter_queries_with_positives(
    queries: dict[str, str],
    relevant_docs: dict[str, set[str]],
) -> dict[str, str]:
    return {
        query_id: text
        for query_id, text in queries.items()
        if query_id in relevant_docs and relevant_docs[query_id]
    }


def qrels_to_training_pairs(
    qrels: Dataset,
    *,
    query_lookup: dict[str, str],
    corpus_lookup: dict[str, str],
) -> Dataset:
    anchors: list[str] = []
    positives: list[str] = []

    for row in qrels:
        query_id = str(row.get("query_id") or row.get("query-id"))
        corpus_id = str(row.get("corpus_id") or row.get("corpus-id"))
        anchor = query_lookup.get(query_id)
        positive = corpus_lookup.get(corpus_id)
        if not anchor or not positive:
            continue
        anchors.append(anchor)
        positives.append(positive)

    if not anchors:
        raise ValueError("No training pairs could be built from qrels.")

    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def prepare_training_dataset(
    config: EmbeddingFinetuningDatasetConfig,
    *,
    split: str | None = None,
) -> Dataset:
    """Build (anchor, positive) pairs from queries, qrels, and corpus."""
    split = split or config.train_split
    corpus = config.load_corpus()
    queries = config.load_subset("queries", split=split)
    qrels = config.load_subset("qrels", split=split)

    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)
    query_lookup = build_query_lookup(queries, config.query_text_fn)
    return qrels_to_training_pairs(
        qrels,
        query_lookup=query_lookup,
        corpus_lookup=corpus_lookup,
    )


def prepare_ir_eval_inputs(
    config: EmbeddingFinetuningDatasetConfig,
    *,
    split: str | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Prepare queries, corpus, and relevant_docs for InformationRetrievalEvaluator."""
    split = split or config.eval_split
    corpus = config.load_corpus()
    queries = config.load_subset("queries", split=split)
    qrels = config.load_subset("qrels", split=split)

    corpus_dict = build_corpus_lookup(corpus, config.corpus_text_fn)
    query_dict = build_query_lookup(queries, config.query_text_fn)
    relevant_docs = build_relevant_docs(qrels)
    query_dict = _filter_queries_with_positives(query_dict, relevant_docs)

    if not query_dict:
        raise ValueError(f"No evaluation queries with positives in split {split!r}.")

    return query_dict, corpus_dict, relevant_docs


def ir_evaluator_name(
    config: EmbeddingFinetuningDatasetConfig,
    *,
    split: str | None = None,
) -> str:
    return f"{config.name}-{split or config.eval_split}"


def default_ir_metric_for_best_model(
    config: EmbeddingFinetuningDatasetConfig,
    *,
    split: str | None = None,
    ndcg_k: int = 10,
    score_function: str = "cosine",
) -> str:
    """W&B / Trainer metric key for IR-based checkpoint selection."""
    evaluator_name = ir_evaluator_name(config, split=split)
    return f"eval_{evaluator_name}_{score_function}_ndcg@{ndcg_k}"


def build_ir_evaluator(
    config: EmbeddingFinetuningDatasetConfig,
    *,
    split: str | None = None,
    batch_size: int = 32,
    name: str | None = None,
) -> InformationRetrievalEvaluator:
    queries, corpus, relevant_docs = prepare_ir_eval_inputs(config, split=split)
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        batch_size=batch_size,
        name=name or ir_evaluator_name(config, split=split),
    )
