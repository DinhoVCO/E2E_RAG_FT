from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from datasets import Dataset
from mteb.abstasks.retrieval import AbsTaskRetrieval, _filter_queries_without_positives
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tesis_unicamp.datasets.preprocessing.rag.retrieval.schemas import RetrievedDocRecord
from tesis_unicamp.datasets.utils.bioasq_rag import (
    corpus_row_to_payload as bioasq_corpus_row_to_payload,
    corpus_row_to_point_id as bioasq_corpus_row_to_point_id,
    corpus_row_to_text as bioasq_corpus_row_to_text,
    load_bioasq_rag_resplit_corpus,
)
from tesis_unicamp.datasets.utils.indexing import index_dataset
from tesis_unicamp.datasets.utils import (
    index_narrativeqa_corpus,
    index_qasper_corpus,
    index_telco_dpr_corpus,
)
from tesis_unicamp.datasets.utils.qasper_top_ranked import load_top_ranked_for_split
from tesis_unicamp.datasets.utils.retrieval import (
    retrieve_all_splits,
    retrieve_all_splits_scoped,
)
from tesis_unicamp.embeddings.base import BaseEmbedder
from tesis_unicamp.evaluation.mteb.runner import (
    build_tesis_embedder,
    configure_vllm_multiprocessing,
    evaluate_retrieval,
    resolve_model,
)
from tesis_unicamp.evaluation.mteb.tasks import (
    RAG_RETRIEVAL_TASK_CONFIGS,
    RagRetrievalTaskConfig,
    _build_relevant_docs,
    _prepare_corpus,
    create_rag_retrieval_task,
)
from tesis_unicamp.finetuning.embeddings.config import DEFAULT_BASE_MODEL
from tesis_unicamp.finetuning.embeddings.context.config import (
    MAX_DOC_TOKENS,
    MAX_QUERY_TOKENS,
    MAX_SEQ_LENGTH as CONTEXT_MAX_SEQ_LENGTH,
)
from tesis_unicamp.finetuning.embeddings.context.datasets import (
    CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS,
)
from tesis_unicamp.datasets.utils.title_retrieval import (
    build_corpus_body_lookup,
    build_corpus_title_lookup,
    build_query_document_title_lookup,
)
from tesis_unicamp.finetuning.embeddings.context.formatting import (
    build_anchor_text,
    build_context_anchor_text,
    count_tokens,
    truncate_to_tokens,
)
from tesis_unicamp.finetuning.generative.formatting import (
    ContextDocument,
    build_user_content,
)
from tesis_unicamp.generation.rag.context import build_corpus_lookup, group_retrieved_by_query
from tesis_unicamp.vector_stores import InMemoryVectorStore

CONTEXT_DATASET_IDS = tuple(CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS)
DEFAULT_STAGE1_TOP_K = 10
DEFAULT_CONTEXT_K_VALUES = (1, 3, 5, 7, 10)
DEFAULT_STAGE1_ONLY_MODEL_REVISION_TEMPLATE = "ctx-lora-{dataset}-stage1"
DEFAULT_TITLE_STAGE1_ONLY_MODEL_REVISION_TEMPLATE = "ctx-lora-{dataset}-stage1-title"
DEFAULT_TITLE_MODEL_REVISION_TEMPLATE = "ctx-lora-{dataset}-k{k}-title"
DEFAULT_TITLE_CONTEXT_K_VALUES = (1, 3, 5)
DEFAULT_TITLE_MAX_TOKENS_PER_CHUNK = 2048
DEFAULT_TITLE_MAX_SEQ_LENGTH = 14336
DEFAULT_TITLE_OUTPUT_ROOT = Path("results") / "mteb" / "context_title"


def stage1_query_to_text(query: str) -> str:
    """Query format for stage-1 retrieval (Instruct + Query, no context docs)."""
    return build_anchor_text(query=query, doc_texts=[])


@dataclass(frozen=True)
class DatasetStage1Spec:
    index_fn: Callable[..., int]
    load_subset: Callable[..., Dataset]


def index_bioasq_resplit_corpus(
    embedder: BaseEmbedder,
    store: InMemoryVectorStore,
    *,
    split: str = "train",
    batch_size: int | None = None,
    recreate_collection: bool = False,
    show_progress: bool = True,
) -> int:
    corpus = load_bioasq_rag_resplit_corpus(split=split)
    return index_dataset(
        corpus,
        embedder,
        store,
        text_fn=bioasq_corpus_row_to_text,
        id_fn=bioasq_corpus_row_to_point_id,
        payload_fn=bioasq_corpus_row_to_payload,
        batch_size=batch_size,
        recreate_collection=recreate_collection,
        show_progress=show_progress,
    )


DATASET_STAGE1_SPECS: dict[str, DatasetStage1Spec] = {
    "bioasq-resplit": DatasetStage1Spec(
        index_fn=index_bioasq_resplit_corpus,
        load_subset=CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS[
            "bioasq-resplit"
        ].load_subset,
    ),
    "qasper": DatasetStage1Spec(
        index_fn=index_qasper_corpus,
        load_subset=CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS["qasper"].load_subset,
    ),
    "telco-dpr": DatasetStage1Spec(
        index_fn=index_telco_dpr_corpus,
        load_subset=CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS["telco-dpr"].load_subset,
    ),
    "narrativeqa": DatasetStage1Spec(
        index_fn=index_narrativeqa_corpus,
        load_subset=CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS["narrativeqa"].load_subset,
    ),
}


@dataclass(frozen=True)
class ContextRetrievalEvalConfig:
    dataset: str
    lora_path: str
    context_k_values: tuple[int, ...] = DEFAULT_CONTEXT_K_VALUES
    model: str = DEFAULT_BASE_MODEL
    stage1_top_k: int = DEFAULT_STAGE1_TOP_K
    splits: tuple[str, ...] = ("test",)
    backend: str = "offline"
    batch_size: int = 128
    model_revision_template: str = "ctx-lora-{dataset}-k{k}"
    stage1_only_model_revision_template: str = DEFAULT_STAGE1_ONLY_MODEL_REVISION_TEMPLATE
    stage1_only: bool = False
    corpus_split: str = "train"
    paper_scoped: bool | None = None
    output_dir: Path | None = None
    overwrite: str = "always"
    max_lora_rank: int = 16
    max_seq_length: int = CONTEXT_MAX_SEQ_LENGTH
    max_query_tokens: int = MAX_QUERY_TOKENS
    max_doc_tokens: int = MAX_DOC_TOKENS
    run_label: str | None = None
    include_query_title: bool = False
    max_tokens_per_chunk: int = DEFAULT_TITLE_MAX_TOKENS_PER_CHUNK


def _validate_context_k_values(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        raise ValueError("At least one context k value is required.")
    normalized = tuple(sorted(set(values)))
    for value in normalized:
        if value <= 0:
            raise ValueError(f"context k must be positive, got {value}.")
    return normalized


def _prepare_queries_from_id_map(
    queries: Dataset,
    query_id_to_text: dict[str, str],
) -> Dataset:
    def _map_row(row: dict[str, Any]) -> dict[str, str]:
        query_id = str(row["id"])
        return {
            "id": query_id,
            "text": query_id_to_text.get(query_id, str(row["text"])),
        }

    return queries.map(_map_row, remove_columns=queries.column_names)


def load_rag_retrieval_split_with_query_map(
    config: RagRetrievalTaskConfig,
    split: str,
    query_id_to_text: dict[str, str],
) -> RetrievalSplitData:
    raw_corpus = config.load_corpus()
    queries = config.load_subset("queries", split=split)
    qrels = config.load_subset("qrels", split=split)

    corpus = _prepare_corpus(raw_corpus, config.corpus_text_fn, num_proc=None)
    queries = _prepare_queries_from_id_map(queries, query_id_to_text)
    relevant_docs = _build_relevant_docs(qrels)
    relevant_docs, queries = _filter_queries_without_positives(relevant_docs, queries)

    return RetrievalSplitData(
        corpus=corpus,
        queries=queries,
        relevant_docs=relevant_docs,
        top_ranked=None,
    )


def create_rag_retrieval_task_with_query_maps(
    config: RagRetrievalTaskConfig,
    query_maps: dict[str, dict[str, str]],
    *,
    task_name_suffix: str,
    include_query_title: bool = False,
) -> AbsTaskRetrieval:
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
            if split not in query_maps:
                raise ValueError(
                    f"Missing context query map for split {split!r} in task {task_name_suffix!r}."
                )
            self.dataset["default"][split] = load_rag_retrieval_split_with_query_map(
                config,
                split,
                query_maps[split],
            )
        self.data_loaded = True

    task_cls = type(
        f"{config.name.replace('-', '_').replace(' ', '_')}_{task_name_suffix}_Task",
        (AbsTaskRetrieval,),
        {
            "metadata": TaskMetadata(
                name=(
                    f"{config.name}-ctx-{task_name_suffix}-Title"
                    if include_query_title
                    else f"{config.name}-ctx-{task_name_suffix}"
                ),
                description=(
                    f"{config.description} "
                    f"(context-augmented queries, stage-2 top_k={task_name_suffix}"
                    + (
                        ", ## Title: query + title/body context docs, "
                        f"{DEFAULT_TITLE_MAX_TOKENS_PER_CHUNK}-token chunk truncation)."
                        if include_query_title
                        else ")."
                    )
                ),
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


def run_stage1_retrieval(
    *,
    dataset: str,
    embedder: BaseEmbedder,
    splits: tuple[str, ...],
    stage1_top_k: int,
    corpus_split: str,
    batch_size: int,
    paper_scoped: bool,
    run_label: str,
    include_query_title: bool = False,
) -> dict[str, list[RetrievedDocRecord]]:
    if dataset not in DATASET_STAGE1_SPECS:
        valid = ", ".join(sorted(DATASET_STAGE1_SPECS))
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {valid}")

    spec = DATASET_STAGE1_SPECS[dataset]
    ctx_config = CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS[dataset]
    store = InMemoryVectorStore(
        collection_name=f"{dataset}-ctx-stage1{'-title' if include_query_title else ''}-{run_label}"
    )

    indexed = spec.index_fn(
        embedder,
        store,
        split=corpus_split,
        batch_size=batch_size,
        recreate_collection=True,
        show_progress=True,
    )
    print(f"Stage 1 indexed {indexed} corpus documents ({store.count()} vectors total)")
    if store.count() == 0:
        raise RuntimeError("Corpus index is empty; cannot run stage-1 retrieval.")

    if paper_scoped:
        if dataset != "qasper":
            raise ValueError(f"Dataset {dataset!r} does not support paper-scoped retrieval.")
        retrieved = retrieve_all_splits_scoped(
            embedder,
            store,
            load_subset=spec.load_subset,
            load_top_ranked_for_split=load_top_ranked_for_split,
            top_k=stage1_top_k,
            query_to_text=None if include_query_title else stage1_query_to_text,
            load_corpus=ctx_config.load_corpus,
            include_query_title=include_query_title,
            corpus_split=corpus_split,
            splits=splits,
            batch_size=batch_size,
            show_progress=True,
        )
    else:
        retrieved = retrieve_all_splits(
            embedder,
            store,
            load_subset=spec.load_subset,
            top_k=stage1_top_k,
            query_to_text=None if include_query_title else stage1_query_to_text,
            load_corpus=ctx_config.load_corpus,
            include_query_title=include_query_title,
            corpus_split=corpus_split,
            splits=splits,
            batch_size=batch_size,
            show_progress=True,
        )

    for split in splits:
        grouped = group_retrieved_by_query(retrieved[split])
        print(
            f"Stage 1 {split}: {len(grouped)} queries, "
            f"top_k={stage1_top_k}, avg_hits={len(retrieved[split]) / max(len(grouped), 1):.1f}"
        )
    return retrieved


def build_context_query_maps(
    *,
    dataset: str,
    splits: tuple[str, ...],
    stage1_retrieved: dict[str, list[RetrievedDocRecord]],
    context_k: int,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    max_query_tokens: int,
    max_doc_tokens: int,
) -> dict[str, dict[str, str]]:
    if context_k <= 0:
        raise ValueError("context_k must be positive.")

    ctx_config = CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS[dataset]
    raw_corpus = ctx_config.load_corpus()
    corpus_lookup = build_corpus_lookup(raw_corpus, ctx_config.corpus_text_fn)
    query_maps: dict[str, dict[str, str]] = {}

    for split in splits:
        grouped = group_retrieved_by_query(stage1_retrieved[split])
        queries = ctx_config.load_subset("queries", split=split)
        raw_query_by_id = {str(row["id"]): str(row["text"]) for row in queries}
        split_map: dict[str, str] = {}
        fallback_count = 0

        for query_id, hits in grouped.items():
            raw_query = raw_query_by_id.get(query_id, "")
            doc_texts = [
                corpus_lookup[str(hit["corpus_id"])]
                for hit in hits[:context_k]
                if str(hit["corpus_id"]) in corpus_lookup
                and corpus_lookup[str(hit["corpus_id"])].strip()
            ]
            anchor = build_context_anchor_text(
                tokenizer,
                query=raw_query,
                doc_texts=doc_texts,
                max_query_tokens=max_query_tokens,
                max_doc_tokens=max_doc_tokens,
                max_seq_length=max_seq_length,
            )
            if anchor is None:
                anchor = stage1_query_to_text(raw_query)
                fallback_count += 1
            split_map[query_id] = anchor

        if fallback_count:
            print(
                f"Stage 2 split={split}, k={context_k}: "
                f"{fallback_count} queries fell back to stage-1 format (no context)."
            )
        query_maps[split] = split_map

    return query_maps


def build_title_context_query_text(
    tokenizer: PreTrainedTokenizerBase,
    *,
    query: str,
    hits: list[RetrievedDocRecord],
    context_k: int,
    corpus_title_lookup: dict[str, str],
    corpus_body_lookup: dict[str, str],
    query_title: str | None,
    max_tokens_per_chunk: int,
    max_seq_length: int,
) -> str | None:
    """Build a stage-2 query using the generative title RAG prompt format."""
    context_docs: list[ContextDocument] = []
    for hit in hits[:context_k]:
        corpus_id = str(hit["corpus_id"])
        title = corpus_title_lookup.get(corpus_id, "").strip()
        body = corpus_body_lookup.get(corpus_id, "").strip()
        if not title and not body:
            continue
        if body:
            body = truncate_to_tokens(tokenizer, body, max_tokens_per_chunk)
        context_docs.append(ContextDocument(title=title, text=body))

    truncated_query = query.strip()
    docs = list(context_docs)

    while True:
        anchor = build_user_content(
            query=truncated_query,
            context_docs=docs,
            query_title=query_title,
        )
        if count_tokens(tokenizer, anchor) <= max_seq_length:
            return anchor
        if docs:
            docs.pop()
            continue
        if len(truncated_query) > 1:
            token_ids = tokenizer.encode(truncated_query, add_special_tokens=False)
            truncated_query = tokenizer.decode(
                token_ids[: max(1, len(token_ids) // 2)],
                skip_special_tokens=True,
            )
            continue
        return None


def build_title_context_query_maps(
    *,
    dataset: str,
    splits: tuple[str, ...],
    stage1_retrieved: dict[str, list[RetrievedDocRecord]],
    context_k: int,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens_per_chunk: int,
    max_seq_length: int,
) -> dict[str, dict[str, str]]:
    if context_k <= 0:
        raise ValueError("context_k must be positive.")

    ctx_config = CONTEXT_EMBEDDING_FINETUNING_DATASET_CONFIGS[dataset]
    raw_corpus = ctx_config.load_corpus()
    corpus_title_lookup = build_corpus_title_lookup(raw_corpus)
    corpus_body_lookup = build_corpus_body_lookup(raw_corpus)
    query_maps: dict[str, dict[str, str]] = {}

    for split in splits:
        grouped = group_retrieved_by_query(stage1_retrieved[split])
        queries = ctx_config.load_subset("queries", split=split)
        qrels = ctx_config.load_subset("qrels", split=split)
        query_title_lookup = build_query_document_title_lookup(qrels, raw_corpus)
        raw_query_by_id = {str(row["id"]): str(row["text"]) for row in queries}
        split_map: dict[str, str] = {}
        fallback_count = 0

        for query_id, hits in grouped.items():
            raw_query = raw_query_by_id.get(query_id, "")
            anchor = build_title_context_query_text(
                tokenizer,
                query=raw_query,
                hits=hits,
                context_k=context_k,
                corpus_title_lookup=corpus_title_lookup,
                corpus_body_lookup=corpus_body_lookup,
                query_title=query_title_lookup.get(query_id),
                max_tokens_per_chunk=max_tokens_per_chunk,
                max_seq_length=max_seq_length,
            )
            if anchor is None:
                anchor = build_user_content(
                    query=raw_query,
                    query_title=query_title_lookup.get(query_id),
                )
                fallback_count += 1
            split_map[query_id] = anchor

        if fallback_count:
            print(
                f"Stage 2 split={split}, k={context_k}: "
                f"{fallback_count} queries fell back to query-only title format."
            )
        query_maps[split] = split_map

    return query_maps


def save_stage1_retrieved(
    stage1_retrieved: dict[str, list[RetrievedDocRecord]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        split: [dict(record) for record in records]
        for split, records in stage1_retrieved.items()
    }
    output_path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")


def resolve_paper_scoped(dataset: str, paper_scoped: bool | None) -> bool:
    if paper_scoped is not None:
        return paper_scoped
    return dataset == "qasper"


def default_model_revision(config: ContextRetrievalEvalConfig, *, context_k: int) -> str:
    template = config.model_revision_template
    if config.include_query_title and template == "ctx-lora-{dataset}-k{k}":
        template = DEFAULT_TITLE_MODEL_REVISION_TEMPLATE
    return template.format(
        dataset=config.dataset,
        k=context_k,
    )


def default_stage1_only_model_revision(config: ContextRetrievalEvalConfig) -> str:
    template = config.stage1_only_model_revision_template
    if config.include_query_title and template == DEFAULT_STAGE1_ONLY_MODEL_REVISION_TEMPLATE:
        template = DEFAULT_TITLE_STAGE1_ONLY_MODEL_REVISION_TEMPLATE
    return template.format(dataset=config.dataset)


def _default_output_dir(config: ContextRetrievalEvalConfig, run_label: str) -> Path:
    root = DEFAULT_TITLE_OUTPUT_ROOT if config.include_query_title else Path("results") / "mteb" / "context"
    return root / config.dataset / run_label


def _evaluate_stage1_only_mteb(
    config: ContextRetrievalEvalConfig,
    *,
    paper_scoped: bool,
    run_label: str,
    output_dir: Path,
) -> list[Any]:
    print("=" * 72)
    if config.include_query_title:
        print("Stage 1 only: MTEB with Instruct + ## Title: + Query (no context)")
    else:
        print("Stage 1 only: MTEB with Instruct + Query (no context documents)")
    print("=" * 72)

    if config.backend == "offline":
        configure_vllm_multiprocessing()
    embedder = build_tesis_embedder(
        mode=config.backend,
        model=config.model,
        batch_size=config.batch_size,
        lora_path=config.lora_path,
        max_lora_rank=config.max_lora_rank,
    )
    if config.backend == "offline":
        from tesis_unicamp.embeddings.vllm_offline import VLLMOfflineEmbedder

        if isinstance(embedder, VLLMOfflineEmbedder):
            print("Loading vLLM embedder (warmup)...")
            embedder.warmup()

    rag_config = RAG_RETRIEVAL_TASK_CONFIGS[config.dataset]
    rag_config = replace(
        rag_config,
        eval_splits=config.splits,
        query_text_fn=stage1_query_to_text,
        use_top_ranked=paper_scoped if config.dataset == "qasper" else False,
        include_query_title=config.include_query_title,
    )
    task = create_rag_retrieval_task(rag_config)

    model_revision = default_stage1_only_model_revision(config)
    print(f"model_revision: {model_revision}")

    model = resolve_model(
        backend=config.backend,
        model=config.model,
        batch_size=config.batch_size,
        model_revision=model_revision,
        lora_path=config.lora_path,
        max_lora_rank=config.max_lora_rank,
        embedder=embedder,
    )
    results = evaluate_retrieval(
        model,
        [task],
        output_folder=output_dir,
        overwrite_strategy=config.overwrite,
        encode_kwargs={"batch_size": config.batch_size},
    )
    for task_result in results:
        print(task_result)
    return results


def evaluate_context_retrieval(config: ContextRetrievalEvalConfig) -> list[Any]:
    if config.dataset not in CONTEXT_DATASET_IDS:
        valid = ", ".join(CONTEXT_DATASET_IDS)
        raise ValueError(f"Unknown dataset {config.dataset!r}. Available: {valid}")
    if config.lora_path.strip() == "":
        raise ValueError("lora_path is required for context embedding evaluation.")

    paper_scoped = resolve_paper_scoped(config.dataset, config.paper_scoped)
    run_label = config.run_label or config.lora_path.rsplit("/", maxsplit=1)[-1]
    output_dir = config.output_dir or _default_output_dir(config, run_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.stage1_only:
        return _evaluate_stage1_only_mteb(
            config,
            paper_scoped=paper_scoped,
            run_label=run_label,
            output_dir=output_dir,
        )

    config = replace(
        config,
        context_k_values=_validate_context_k_values(config.context_k_values),
    )
    if config.stage1_top_k < max(config.context_k_values):
        raise ValueError(
            f"stage1_top_k ({config.stage1_top_k}) must be >= max(context_k_values) "
            f"({max(config.context_k_values)})."
        )
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if config.backend == "offline":
        configure_vllm_multiprocessing()
    embedder = build_tesis_embedder(
        mode=config.backend,
        model=config.model,
        batch_size=config.batch_size,
        lora_path=config.lora_path,
        max_lora_rank=config.max_lora_rank,
    )
    if config.backend == "offline":
        from tesis_unicamp.embeddings.vllm_offline import VLLMOfflineEmbedder

        if isinstance(embedder, VLLMOfflineEmbedder):
            print("Loading vLLM embedder (warmup)...")
            embedder.warmup()

    print("=" * 72)
    if config.include_query_title:
        print("Stage 1: retrieve candidates (Instruct + ## Title: + Query, no context)")
    else:
        print("Stage 1: retrieve candidate documents (Instruct + Query, no context)")
    print("=" * 72)
    stage1_retrieved = run_stage1_retrieval(
        dataset=config.dataset,
        embedder=embedder,
        splits=config.splits,
        stage1_top_k=config.stage1_top_k,
        corpus_split=config.corpus_split,
        batch_size=config.batch_size,
        paper_scoped=paper_scoped,
        run_label=run_label,
        include_query_title=config.include_query_title,
    )
    save_stage1_retrieved(stage1_retrieved, output_dir / "stage1_retrieved.json")

    rag_config = RAG_RETRIEVAL_TASK_CONFIGS[config.dataset]
    rag_config = replace(
        rag_config,
        eval_splits=config.splits,
        use_top_ranked=paper_scoped if config.dataset == "qasper" else False,
    )

    stage2_model = None
    all_results: list[Any] = []
    for context_k in config.context_k_values:
        model_revision = default_model_revision(config, context_k=context_k)
        print()
        print("=" * 72)
        print(f"Stage 2: MTEB evaluation with context top_k={context_k}")
        print(f"model_revision: {model_revision}")
        print("=" * 72)

        query_maps = (
            build_title_context_query_maps(
                dataset=config.dataset,
                splits=config.splits,
                stage1_retrieved=stage1_retrieved,
                context_k=context_k,
                tokenizer=tokenizer,
                max_tokens_per_chunk=config.max_tokens_per_chunk,
                max_seq_length=config.max_seq_length,
            )
            if config.include_query_title
            else build_context_query_maps(
                dataset=config.dataset,
                splits=config.splits,
                stage1_retrieved=stage1_retrieved,
                context_k=context_k,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
                max_query_tokens=config.max_query_tokens,
                max_doc_tokens=config.max_doc_tokens,
            )
        )
        task = create_rag_retrieval_task_with_query_maps(
            rag_config,
            query_maps,
            task_name_suffix=f"k{context_k}",
            include_query_title=config.include_query_title,
        )

        stage2_model = resolve_model(
            backend=config.backend,
            model=config.model,
            batch_size=config.batch_size,
            model_revision=model_revision,
            lora_path=config.lora_path,
            max_lora_rank=config.max_lora_rank,
            embedder=embedder,
        )
        results = evaluate_retrieval(
            stage2_model,
            [task],
            output_folder=output_dir,
            overwrite_strategy=config.overwrite,
            encode_kwargs={"batch_size": config.batch_size},
        )
        all_results.extend(results)
        for task_result in results:
            print(task_result)

    return all_results
