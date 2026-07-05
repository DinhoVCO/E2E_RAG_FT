from tesis_unicamp.datasets.utils.bioasq_rag import (
    BIOASQ_RAG_DATASET_ID,
    DEFAULT_RETRIEVAL_TASK,
    corpus_row_to_payload,
    corpus_row_to_point_id,
    corpus_row_to_text,
    load_bioasq_rag_corpus,
    load_bioasq_rag_subset,
    query_to_instruct_text,
    index_bioasq_corpus,
)
from tesis_unicamp.datasets.utils.corpus import batched_map, corpus_id_to_point_id, format_document, iter_batches
from tesis_unicamp.datasets.utils.indexing import index_dataset
from tesis_unicamp.datasets.utils.retrieval import (
    build_relevant_corpus_ids,
    retrieve_all_splits,
    retrieve_split,
    retrieve_top_k_for_queries,
)
from tesis_unicamp.datasets.utils.narrativeqa_rag import (
    NARRATIVEQA_RAG_DATASET_ID,
    index_narrativeqa_corpus,
    load_narrativeqa_rag_corpus,
    load_narrativeqa_rag_subset,
)
from tesis_unicamp.datasets.utils.qasper_rag import (
    QASPER_RAG_DATASET_ID,
    index_qasper_corpus,
    load_qasper_rag_corpus,
    load_qasper_rag_subset,
)
from tesis_unicamp.datasets.utils.telco_dpr_rag import (
    TELCO_DPR_RAG_DATASET_ID,
    index_telco_dpr_corpus,
    load_telco_dpr_rag_corpus,
    load_telco_dpr_rag_subset,
)

__all__ = [
    "BIOASQ_RAG_DATASET_ID",
    "DEFAULT_RETRIEVAL_TASK",
    "NARRATIVEQA_RAG_DATASET_ID",
    "QASPER_RAG_DATASET_ID",
    "TELCO_DPR_RAG_DATASET_ID",
    "batched_map",
    "corpus_id_to_point_id",
    "corpus_row_to_payload",
    "corpus_row_to_point_id",
    "corpus_row_to_text",
    "format_document",
    "index_bioasq_corpus",
    "index_dataset",
    "index_narrativeqa_corpus",
    "index_qasper_corpus",
    "index_telco_dpr_corpus",
    "iter_batches",
    "build_relevant_corpus_ids",
    "retrieve_all_splits",
    "retrieve_split",
    "retrieve_top_k_for_queries",
    "load_bioasq_rag_corpus",
    "load_bioasq_rag_subset",
    "load_narrativeqa_rag_corpus",
    "load_narrativeqa_rag_subset",
    "load_qasper_rag_corpus",
    "load_qasper_rag_subset",
    "load_telco_dpr_rag_corpus",
    "load_telco_dpr_rag_subset",
    "query_to_instruct_text",
]
