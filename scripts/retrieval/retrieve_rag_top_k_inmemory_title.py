"""Retrieve top-k documents with gold document title in the embedding query.

Same as ``retrieve_rag_top_k_inmemory.py``, but:
- prepends ``## Title:`` (from qrels) before ``Query:`` in the retrieval prompt
- writes results under ``datasets/retrieved_inmemory_title/``

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory_title.py \\
        --dataset qasper --mode offline
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(RETRIEVAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_SCRIPT_DIR))

from retrieve_rag_top_k_inmemory import (  # noqa: E402
    DATASET_SPECS,
    _build_embedder,
    _build_parser,
    _load_env,
    _sanitize_run_label,
    _validate_cuda_for_offline,
)

from tesis_unicamp.datasets.utils.bioasq_rag import (
    load_bioasq_rag_resplit_corpus,
    load_bioasq_rag_resplit_subset,
    query_to_instruct_text,
)
from tesis_unicamp.datasets.utils.narrativeqa_rag import (
    load_narrativeqa_rag_corpus,
    load_narrativeqa_rag_subset,
)
from tesis_unicamp.datasets.utils.qasper_rag import load_qasper_rag_corpus, load_qasper_rag_subset
from tesis_unicamp.datasets.utils.qasper_top_ranked import load_top_ranked_for_split
from tesis_unicamp.datasets.utils.retrieval import retrieve_all_splits, retrieve_all_splits_scoped
from tesis_unicamp.datasets.utils.telco_dpr_rag import (
    load_telco_dpr_rag_corpus,
    load_telco_dpr_rag_subset,
)
from tesis_unicamp.vector_stores import InMemoryVectorStore

TITLE_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "retrieved_inmemory_title"

DATASET_LOAD_SUBSET = {
    "bioasq-resplit": load_bioasq_rag_resplit_subset,
    "qasper": load_qasper_rag_subset,
    "telco-dpr": load_telco_dpr_rag_subset,
    "narrativeqa": load_narrativeqa_rag_subset,
}

DATASET_LOAD_CORPUS = {
    "bioasq-resplit": load_bioasq_rag_resplit_corpus,
    "qasper": load_qasper_rag_corpus,
    "telco-dpr": load_telco_dpr_rag_corpus,
    "narrativeqa": load_narrativeqa_rag_corpus,
}


def _default_title_output_dir(dataset: str, run_label: str) -> Path:
    return TITLE_OUTPUT_ROOT / dataset / _sanitize_run_label(run_label)


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    if args.dataset not in DATASET_LOAD_SUBSET:
        raise SystemExit(
            f"Title retrieval is not configured for dataset {args.dataset!r}. "
            f"Supported: {', '.join(sorted(DATASET_LOAD_SUBSET))}"
        )

    spec = DATASET_SPECS[args.dataset]
    load_subset = DATASET_LOAD_SUBSET[args.dataset]
    load_corpus = DATASET_LOAD_CORPUS[args.dataset]

    paper_scoped = args.paper_scoped
    if paper_scoped is None:
        paper_scoped = args.dataset == "qasper"
    if paper_scoped and args.dataset != "qasper":
        raise SystemExit("--paper-scoped is only supported for --dataset qasper")

    run_label = _sanitize_run_label(args.run_label)
    if paper_scoped and "paper-scoped" not in run_label.lower():
        run_label = f"{run_label}-paper-scoped"
    output_dir = args.output_dir or _default_title_output_dir(args.dataset, run_label)
    splits = tuple(args.splits)

    _validate_cuda_for_offline(args.mode)
    if args.lora_path and args.mode != "offline":
        raise SystemExit("--lora-path requires --mode offline.")

    embedder = _build_embedder(
        args.mode,
        args.model,
        args.batch_size,
        lora_path=args.lora_path,
        max_lora_rank=args.max_lora_rank if args.lora_path else None,
    )
    store = InMemoryVectorStore(collection_name=f"{args.dataset}-{run_label}-title")

    print(f"dataset: {args.dataset}")
    print(f"mode: {args.mode}")
    print(f"model: {args.model}")
    if args.lora_path:
        print(f"lora_path: {args.lora_path}")
        print(f"max_lora_rank: {args.max_lora_rank}")
    print("query_format: Instruct + ## Title: + Query")
    print(f"run_label: {run_label}")
    print(f"vector_store: InMemoryVectorStore (FAISS IndexFlatIP)")
    print(f"top_k: {args.top_k}")
    print(f"corpus_split: {args.corpus_split}")
    print(f"splits: {', '.join(splits)}")
    if args.dataset == "qasper":
        print(f"paper_scoped: {paper_scoped}")
    print(f"output_dir: {output_dir}")

    indexed = spec.index_fn(
        embedder,
        store,
        split=args.corpus_split,
        batch_size=args.batch_size,
        recreate_collection=True,
        show_progress=True,
    )
    print(f"Indexed {indexed} corpus documents in memory ({store.count()} vectors total)")

    if store.count() == 0:
        raise SystemExit("Corpus index is empty; nothing to retrieve.")

    if paper_scoped:
        retrieved = retrieve_all_splits_scoped(
            embedder,
            store,
            load_subset=load_subset,
            load_top_ranked_for_split=load_top_ranked_for_split,
            top_k=args.top_k,
            query_to_text=query_to_instruct_text,
            load_corpus=load_corpus,
            include_query_title=True,
            corpus_split=args.corpus_split,
            splits=splits,
            batch_size=args.batch_size,
        )
    else:
        retrieved = retrieve_all_splits(
            embedder,
            store,
            load_subset=load_subset,
            top_k=args.top_k,
            query_to_text=query_to_instruct_text,
            load_corpus=load_corpus,
            include_query_title=True,
            corpus_split=args.corpus_split,
            splits=splits,
            batch_size=args.batch_size,
        )

    output_path = spec.save_fn(retrieved, output_dir=output_dir)

    for split in splits:
        count = len(retrieved[split])
        queries = count // args.top_k if args.top_k else 0
        print(f"  {split}: {queries} queries -> {count} retrieved_docs rows")

    print(f"Saved retrieved_docs to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
