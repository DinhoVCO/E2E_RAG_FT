from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from mteb.abstasks.retrieval import AbsTaskRetrieval, _filter_queries_without_positives
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata

from tesis_unicamp.datasets.utils.bioasq_rag import (
    BIOASQ_RAG_DATASET_ID,
    corpus_row_to_text as bioasq_corpus_row_to_text,
    load_bioasq_rag_corpus,
    load_bioasq_rag_subset,
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

LoadSubsetFn = Callable[..., Dataset]
CorpusTextFn = Callable[[dict[str, Any]], str]
QueryTextFn = Callable[[str], str]


@dataclass(frozen=True)
class RagRetrievalTaskConfig:
    """Configuration for a custom MTEB retrieval task backed by a RAG dataset."""

    name: str
    hf_repo_id: str
    description: str
    load_corpus: LoadSubsetFn
    load_subset: LoadSubsetFn
    corpus_text_fn: CorpusTextFn
    eval_splits: tuple[str, ...] = ("test",)
    revision: str = "main"
    query_text_fn: QueryTextFn | None = query_to_instruct_text
    reference: str = ""
    task_subtypes: tuple[str, ...] = ("Question answering",)
    domains: tuple[str, ...] = ("Written",)
    bibtex_citation: str = ""


def _build_relevant_docs(qrels: Dataset) -> dict[str, dict[str, int]]:
    relevant: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels:
        query_id = str(row.get("query_id") or row.get("query-id"))
        corpus_id = str(row.get("corpus_id") or row.get("corpus-id"))
        score = int(row.get("score", 1))
        relevant[query_id][corpus_id] = score
    return dict(relevant)


def _prepare_corpus(
    corpus: Dataset,
    corpus_text_fn: CorpusTextFn,
    *,
    num_proc: int | None,
) -> Dataset:
    def _map_row(row: dict[str, Any]) -> dict[str, str]:
        return {
            "id": str(row["id"]),
            "text": corpus_text_fn(row),
        }

    return corpus.map(
        _map_row,
        remove_columns=corpus.column_names,
        num_proc=num_proc,
    )


def _prepare_queries(
    queries: Dataset,
    query_text_fn: QueryTextFn | None,
    *,
    num_proc: int | None,
) -> Dataset:
    if query_text_fn is None:
        return queries.map(
            lambda row: {"id": str(row["id"]), "text": str(row["text"])},
            remove_columns=queries.column_names,
            num_proc=num_proc,
        )

    return queries.map(
        lambda row: {
            "id": str(row["id"]),
            "text": query_text_fn(str(row["text"])),
        },
        remove_columns=queries.column_names,
        num_proc=num_proc,
    )


def load_rag_retrieval_split(
    config: RagRetrievalTaskConfig,
    split: str,
    *,
    num_proc: int | None = None,
) -> RetrievalSplitData:
    """Load one split from a RAG dataset and convert it to MTEB v2 retrieval format."""
    corpus = config.load_corpus()
    queries = config.load_subset("queries", split=split)
    qrels = config.load_subset("qrels", split=split)

    corpus = _prepare_corpus(corpus, config.corpus_text_fn, num_proc=num_proc)
    queries = _prepare_queries(queries, config.query_text_fn, num_proc=num_proc)
    relevant_docs = _build_relevant_docs(qrels)
    relevant_docs, queries = _filter_queries_without_positives(relevant_docs, queries)

    return RetrievalSplitData(
        corpus=corpus,
        queries=queries,
        relevant_docs=relevant_docs,
        top_ranked=None,
    )


def create_rag_retrieval_task(
    config: RagRetrievalTaskConfig,
) -> AbsTaskRetrieval:
    """Create an MTEB retrieval task for a custom or project RAG dataset."""

    def load_data(
        self: AbsTaskRetrieval,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> None:
        del num_proc, kwargs
        if self.data_loaded:
            return

        self.dataset = {"default": {}}
        for split in self.eval_splits:
            self.dataset["default"][split] = load_rag_retrieval_split(
                config,
                split,
                num_proc=None,
            )
        self.data_loaded = True

    task_cls = type(
        f"{config.name.replace('-', '_').replace(' ', '_')}Task",
        (AbsTaskRetrieval,),
        {
            "metadata": TaskMetadata(
                name=config.name,
                description=config.description,
                reference=config.reference or f"https://huggingface.co/datasets/{config.hf_repo_id}",
                dataset={
                    "path": config.hf_repo_id,
                    "revision": config.revision,
                },
                type="Retrieval",
                category="t2t",
                modalities=["text"],
                eval_splits=list(config.eval_splits),
                eval_langs=["eng-Latn"],
                main_score="ndcg_at_10",
                date=None,
                domains=list(config.domains),
                task_subtypes=list(config.task_subtypes),
                license="not specified",
                annotations_creators="human-annotated",
                dialect=[],
                sample_creation="found",
                bibtex_citation=config.bibtex_citation,
            ),
            "ignore_identical_ids": True,
            "load_data": load_data,
        },
    )
    return task_cls()


RAG_RETRIEVAL_TASK_CONFIGS: dict[str, RagRetrievalTaskConfig] = {
    "bioasq": RagRetrievalTaskConfig(
        name="BioASQ-RAG",
        hf_repo_id=BIOASQ_RAG_DATASET_ID,
        description="BioASQ RAG retrieval over PubMed passages.",
        load_corpus=load_bioasq_rag_corpus,
        load_subset=load_bioasq_rag_subset,
        corpus_text_fn=bioasq_corpus_row_to_text,
        eval_splits=("train", "dev", "test"),
        domains=("Medical", "Written"),
        task_subtypes=("Question answering",),
    ),
    "qasper": RagRetrievalTaskConfig(
        name="QASPER-RAG",
        hf_repo_id=QASPER_RAG_DATASET_ID,
        description="QASPER RAG retrieval over paper paragraph chunks.",
        load_corpus=load_qasper_rag_corpus,
        load_subset=load_qasper_rag_subset,
        corpus_text_fn=qasper_corpus_row_to_text,
        eval_splits=("train", "dev", "test"),
        domains=("Academic", "Written"),
        task_subtypes=("Question answering",),
    ),
    "telco-dpr": RagRetrievalTaskConfig(
        name="TelcoDPR-RAG",
        hf_repo_id=TELCO_DPR_RAG_DATASET_ID,
        description="Telco-DPR RAG retrieval over telecom passages.",
        load_corpus=load_telco_dpr_rag_corpus,
        load_subset=load_telco_dpr_rag_subset,
        corpus_text_fn=telco_corpus_row_to_text,
        eval_splits=("train", "dev", "test"),
        domains=("Written",),
        task_subtypes=("Question answering",),
    ),
    "narrativeqa": RagRetrievalTaskConfig(
        name="NarrativeQA-RAG",
        hf_repo_id=NARRATIVEQA_RAG_DATASET_ID,
        description="NarrativeQA RAG retrieval over story passages.",
        load_corpus=load_narrativeqa_rag_corpus,
        load_subset=load_narrativeqa_rag_subset,
        corpus_text_fn=narrativeqa_corpus_row_to_text,
        eval_splits=("train", "dev", "test"),
        domains=("Fiction", "Written"),
        task_subtypes=("Question answering",),
    ),
}


def get_rag_retrieval_task(
    dataset: str,
    *,
    eval_splits: tuple[str, ...] | None = None,
) -> AbsTaskRetrieval:
    """Return a predefined project RAG task by dataset key."""
    try:
        config = RAG_RETRIEVAL_TASK_CONFIGS[dataset]
    except KeyError as exc:
        available = ", ".join(sorted(RAG_RETRIEVAL_TASK_CONFIGS))
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {available}"
        ) from exc

    if eval_splits is not None:
        config = replace(config, eval_splits=eval_splits)
    return create_rag_retrieval_task(config)


def create_custom_rag_retrieval_task(
    *,
    name: str,
    hf_repo_id: str,
    description: str,
    eval_splits: tuple[str, ...] = ("test",),
    revision: str = "main",
    corpus_text_fn: CorpusTextFn | None = None,
    query_text_fn: QueryTextFn | None = query_to_instruct_text,
    corpus_config: str = "corpus",
    corpus_split: str | None = None,
) -> AbsTaskRetrieval:
    """Create a retrieval task from any Hugging Face repo with corpus/queries/qrels configs."""

    def load_corpus(*, split: str = "train") -> Dataset:
        dataset = load_dataset(hf_repo_id, corpus_config, revision=revision)
        if isinstance(dataset, DatasetDict):
            if corpus_split is not None:
                return dataset[corpus_split]
            if split in dataset:
                return dataset[split]
            return dataset[next(iter(dataset))]
        return dataset

    def load_subset(subset_name: str, *, split: str | None = None) -> Dataset:
        dataset = load_dataset(hf_repo_id, subset_name, revision=revision)
        if isinstance(dataset, DatasetDict):
            if split is None:
                raise ValueError(f"Subset {subset_name!r} has splits; pass split= explicitly")
            return dataset[split]
        return dataset

    default_corpus_text_fn: CorpusTextFn
    if corpus_text_fn is not None:
        default_corpus_text_fn = corpus_text_fn
    else:

        def default_corpus_text_fn(row: dict[str, Any]) -> str:
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            if title and text:
                return f"{title}\n\n{text}"
            return title or text

    config = RagRetrievalTaskConfig(
        name=name,
        hf_repo_id=hf_repo_id,
        description=description,
        load_corpus=load_corpus,
        load_subset=load_subset,
        corpus_text_fn=default_corpus_text_fn,
        eval_splits=eval_splits,
        revision=revision,
        query_text_fn=query_text_fn,
    )
    return create_rag_retrieval_task(config)
