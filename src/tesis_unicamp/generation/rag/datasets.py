from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets import Dataset

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

LoadSubsetFn = Callable[..., Dataset]
CorpusTextFn = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class RagGenerationDatasetConfig:
    name: str
    hub_repo_id: str
    load_corpus: LoadSubsetFn
    load_subset: LoadSubsetFn
    corpus_text_fn: CorpusTextFn
    corpus_split: str = "train"


RAG_GENERATION_DATASET_CONFIGS: dict[str, RagGenerationDatasetConfig] = {
    "bioasq-resplit": RagGenerationDatasetConfig(
        name="bioasq-resplit",
        hub_repo_id=BIOASQ_RAG_RESPLIT_DATASET_ID,
        load_corpus=load_bioasq_rag_resplit_corpus,
        load_subset=load_bioasq_rag_resplit_subset,
        corpus_text_fn=bioasq_corpus_row_to_text,
    ),
    "qasper": RagGenerationDatasetConfig(
        name="qasper",
        hub_repo_id=QASPER_RAG_DATASET_ID,
        load_corpus=load_qasper_rag_corpus,
        load_subset=load_qasper_rag_subset,
        corpus_text_fn=qasper_corpus_row_to_text,
    ),
    "telco-dpr": RagGenerationDatasetConfig(
        name="telco-dpr",
        hub_repo_id=TELCO_DPR_RAG_DATASET_ID,
        load_corpus=load_telco_dpr_rag_corpus,
        load_subset=load_telco_dpr_rag_subset,
        corpus_text_fn=telco_corpus_row_to_text,
    ),
    "narrativeqa": RagGenerationDatasetConfig(
        name="narrativeqa",
        hub_repo_id=NARRATIVEQA_RAG_DATASET_ID,
        load_corpus=load_narrativeqa_rag_corpus,
        load_subset=load_narrativeqa_rag_subset,
        corpus_text_fn=narrativeqa_corpus_row_to_text,
    ),
}


def get_rag_generation_config(dataset: str) -> RagGenerationDatasetConfig:
    try:
        return RAG_GENERATION_DATASET_CONFIGS[dataset]
    except KeyError as exc:
        available = ", ".join(sorted(RAG_GENERATION_DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {available}") from exc


def load_answers_subset(
    config: RagGenerationDatasetConfig,
    *,
    split: str,
) -> Dataset:
    return config.load_subset("answers", split=split)


def load_queries_subset(
    config: RagGenerationDatasetConfig,
    *,
    split: str,
) -> Dataset:
    return config.load_subset("queries", split=split)


def load_corpus_subset(
    config: RagGenerationDatasetConfig,
    *,
    split: str | None = None,
) -> Dataset:
    corpus_split = split or config.corpus_split
    return config.load_corpus(split=corpus_split)


def load_qrels_subset(
    config: RagGenerationDatasetConfig,
    *,
    split: str,
) -> Dataset:
    return config.load_subset("qrels", split=split)
