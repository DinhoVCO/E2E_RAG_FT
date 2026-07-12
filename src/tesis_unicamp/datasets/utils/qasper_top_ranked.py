from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from datasets import Dataset

from tesis_unicamp.datasets.utils.qasper_rag import (
    load_qasper_rag_corpus,
    load_qasper_rag_subset,
)


def corpus_id_to_paper_id(corpus_id: str) -> str:
    """Return the paper id prefix from a chunk corpus id (e.g. paper_00000_00001 -> paper_00000)."""
    return corpus_id.rsplit("_", 1)[0]


def build_paper_to_chunk_ids(corpus: Dataset) -> dict[str, list[str]]:
    paper_to_chunk_ids: dict[str, list[str]] = defaultdict(list)
    for row in corpus:
        corpus_id = str(row["id"])
        paper_id = corpus_id_to_paper_id(corpus_id)
        paper_to_chunk_ids[paper_id].append(corpus_id)
    return {
        paper_id: sorted(chunk_ids)
        for paper_id, chunk_ids in paper_to_chunk_ids.items()
    }


def build_top_ranked_from_dataset(top_ranked: Dataset) -> dict[str, list[str]]:
    ranked: dict[str, list[str]] = {}
    for row in top_ranked:
        query_id = str(row.get("query_id") or row.get("query-id"))
        corpus_ids = row.get("corpus_ids") or row.get("corpus-ids") or []
        ranked[query_id] = [str(corpus_id) for corpus_id in corpus_ids]
    return ranked


def build_top_ranked_from_qrels(
    qrels: Dataset,
    corpus: Dataset,
) -> dict[str, list[str]]:
    paper_to_chunk_ids = build_paper_to_chunk_ids(corpus)
    ranked: dict[str, list[str]] = {}
    for row in qrels:
        query_id = str(row.get("query_id") or row.get("query-id"))
        if query_id in ranked:
            continue
        corpus_id = str(row.get("corpus_id") or row.get("corpus-id"))
        paper_id = corpus_id_to_paper_id(corpus_id)
        ranked[query_id] = paper_to_chunk_ids[paper_id]
    return ranked


def load_top_ranked_for_split(
    split: str,
    *,
    load_subset: Callable[..., Dataset] = load_qasper_rag_subset,
    load_corpus: Callable[..., Dataset] = load_qasper_rag_corpus,
    corpus_split: str = "train",
) -> dict[str, list[str]]:
    """Load paper-scoped candidate pools for each query in a split.

    Uses the ``top_ranked`` Hub subset when available; otherwise derives pools
    from qrels and the corpus (all chunks of the query's paper).
    """
    qrels = load_subset("qrels", split=split)
    try:
        top_ranked = load_subset("top_ranked", split=split)
    except (ValueError, FileNotFoundError, OSError):
        corpus = load_corpus(split=corpus_split)
        return build_top_ranked_from_qrels(qrels, corpus)
    return build_top_ranked_from_dataset(top_ranked)
