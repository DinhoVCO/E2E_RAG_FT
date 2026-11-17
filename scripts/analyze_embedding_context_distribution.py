"""Analyze the context distribution used in embedding fine-tuning.

The script reproduces the seeded sampling, relevant-document injection,
individual token truncation, total anchor truncation, and positive-pair
expansion implemented by ``build_context_training_pairs``. Counts are reported
at pair level because each (anchor, positive document) pair is an independent
training example for the contrastive loss.

This is intentionally separate from
``scripts/analyze_generative_context_distribution.py``.

Example:
    uv run python scripts/analyze_embedding_context_distribution.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer

from tesis_unicamp.finetuning.embeddings.context.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATASET_SEED,
    MAX_CONTEXT_DOCS,
    MAX_DOC_TOKENS,
    MAX_POSITIVE_TOKENS,
    MAX_QUERY_TOKENS,
    MAX_SEQ_LENGTH,
    RELEVANT_DOC_RATIO,
)
from tesis_unicamp.finetuning.embeddings.context.datasets import (
    _group_qrels_by_query,
    _hits_to_doc_texts,
    _retrieved_rows_to_records,
    _select_context_hits,
    get_context_embedding_finetuning_config,
)
from tesis_unicamp.finetuning.embeddings.context.formatting import (
    build_context_anchor_text,
    build_positive_text,
    truncate_to_tokens,
)
from tesis_unicamp.finetuning.embeddings.datasets import (
    build_corpus_lookup,
    build_relevant_docs,
)
from tesis_unicamp.generation.rag.context import group_retrieved_by_query

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("qasper", "bioasq-resplit", "narrativeqa", "telco-dpr")
DATASET_LABELS = {
    "qasper": "QASPER",
    "bioasq-resplit": "BioASQ-Resplit",
    "narrativeqa": "NarrativeQA",
    "telco-dpr": "Telco-DPR",
}
CATEGORY_LABELS = {
    "without_context": "Sem contexto",
    "context_with_relevant": "Com contexto e passagem relevante",
    "context_without_relevant": "Com contexto sem passagem relevante",
}
CATEGORY_COLORS = {
    "without_context": "#7F8C8D",
    "context_with_relevant": "#148F77",
    "context_without_relevant": "#B9770E",
}


@dataclass(frozen=True)
class DistributionRow:
    dataset: str
    split: str
    total_pairs: int
    total_anchors: int
    without_context_count: int
    without_context_pct: float
    context_with_relevant_count: int
    context_with_relevant_pct: float
    context_without_relevant_count: int
    context_without_relevant_pct: float
    relevance_guarantee_applied_anchors: int
    relevance_guarantee_applied_pct: float


def _percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _anchor_contains_relevant_passage(
    *,
    anchor: str,
    selected_hits: list[dict],
    relevant_ids: set[str],
    corpus_lookup: dict[str, str],
    tokenizer,
    max_doc_tokens: int,
) -> bool:
    for hit in selected_hits:
        corpus_id = str(hit["corpus_id"])
        if corpus_id not in relevant_ids:
            continue
        text = corpus_lookup.get(corpus_id, "")
        truncated = truncate_to_tokens(tokenizer, text, max_doc_tokens)
        if truncated and truncated in anchor:
            return True
    return False


def analyze_dataset(
    dataset: str,
    *,
    split: str,
    seed: int,
    tokenizer,
    max_context_docs: int = MAX_CONTEXT_DOCS,
    relevant_doc_ratio: float = RELEVANT_DOC_RATIO,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_positive_tokens: int = MAX_POSITIVE_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> DistributionRow:
    config = get_context_embedding_finetuning_config(dataset)
    corpus = config.load_corpus()
    queries = config.load_subset("queries", split=split)
    qrels = config.load_subset("qrels", split=split)
    retrieved = config.load_subset("retrieved_docs", split=split)

    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)
    relevant_docs = build_relevant_docs(qrels)
    grouped_retrieved = group_retrieved_by_query(
        _retrieved_rows_to_records(retrieved)
    )
    grouped_qrels = _group_qrels_by_query(qrels)
    query_lookup = {str(row["id"]): str(row["text"]) for row in queries}

    rng = random.Random(seed)
    counts = {
        "without_context": 0,
        "context_with_relevant": 0,
        "context_without_relevant": 0,
    }
    total_anchors = 0
    guarantee_applied = 0

    for query_id, corpus_ids in grouped_qrels.items():
        query_text = query_lookup.get(query_id)
        if not query_text:
            continue

        num_docs = rng.randint(0, max_context_docs)
        ensure_relevant = num_docs > 0 and rng.random() < relevant_doc_ratio
        if ensure_relevant:
            guarantee_applied += 1

        relevant_ids = relevant_docs.get(query_id, set())
        selected_hits = _select_context_hits(
            grouped_retrieved.get(query_id, []),
            num_docs=num_docs,
            relevant_ids=relevant_ids,
            ensure_relevant=ensure_relevant,
            rng=rng,
        )
        doc_texts = _hits_to_doc_texts(selected_hits, corpus_lookup)
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

        valid_positives = 0
        for corpus_id in corpus_ids:
            positive_raw = corpus_lookup.get(corpus_id, "").strip()
            if not positive_raw:
                continue
            positive = build_positive_text(
                tokenizer,
                text=positive_raw,
                max_positive_tokens=max_positive_tokens,
            )
            if positive:
                valid_positives += 1
        if valid_positives == 0:
            continue

        total_anchors += 1
        if "## Context:" not in anchor:
            category = "without_context"
        elif _anchor_contains_relevant_passage(
            anchor=anchor,
            selected_hits=selected_hits,
            relevant_ids=relevant_ids,
            corpus_lookup=corpus_lookup,
            tokenizer=tokenizer,
            max_doc_tokens=max_doc_tokens,
        ):
            category = "context_with_relevant"
        else:
            category = "context_without_relevant"
        counts[category] += valid_positives

    total_pairs = sum(counts.values())
    return DistributionRow(
        dataset=DATASET_LABELS[dataset],
        split=split,
        total_pairs=total_pairs,
        total_anchors=total_anchors,
        without_context_count=counts["without_context"],
        without_context_pct=_percentage(
            counts["without_context"],
            total_pairs,
        ),
        context_with_relevant_count=counts["context_with_relevant"],
        context_with_relevant_pct=_percentage(
            counts["context_with_relevant"],
            total_pairs,
        ),
        context_without_relevant_count=counts["context_without_relevant"],
        context_without_relevant_pct=_percentage(
            counts["context_without_relevant"],
            total_pairs,
        ),
        relevance_guarantee_applied_anchors=guarantee_applied,
        relevance_guarantee_applied_pct=_percentage(
            guarantee_applied,
            total_anchors,
        ),
    )


def write_csv(rows: list[DistributionRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in rows]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_json(rows: list[DistributionRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def plot_distribution(rows: list[DistributionRow], output_stem: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(8.6, 4.6))
    datasets = [row.dataset for row in rows]
    bottoms = [0.0] * len(rows)
    category_fields = {
        "without_context": "without_context_pct",
        "context_with_relevant": "context_with_relevant_pct",
        "context_without_relevant": "context_without_relevant_pct",
    }

    for category, field in category_fields.items():
        values = [float(getattr(row, field)) for row in rows]
        bars = axis.bar(
            datasets,
            values,
            bottom=bottoms,
            label=CATEGORY_LABELS[category],
            color=CATEGORY_COLORS[category],
        )
        for bar, value, bottom in zip(bars, values, bottoms, strict=True):
            if value >= 3:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{value:.1f}%".replace(".", ","),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottoms = [
            bottom + value
            for bottom, value in zip(bottoms, values, strict=True)
        ]

    axis.set_xlabel("Conjunto de dados")
    axis.set_ylabel("Proporção dos pares de treinamento (%)")
    axis.set_ylim(0, 100)
    axis.legend(
        title="Composição da âncora",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the context distribution used in embedding fine-tuning."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed (default: 42 for train and 43 for dev).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "embedding_context_distribution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    seed = args.seed if args.seed is not None else (
        DEFAULT_DATASET_SEED if args.split == "train" else DEFAULT_DATASET_SEED + 1
    )
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_BASE_MODEL,
        trust_remote_code=True,
    )
    rows = [
        analyze_dataset(
            dataset,
            split=args.split,
            seed=seed,
            tokenizer=tokenizer,
        )
        for dataset in args.datasets
    ]

    stem = f"embedding_context_distribution_{args.split}"
    write_csv(rows, args.output_dir / f"{stem}.csv")
    write_json(rows, args.output_dir / f"{stem}.json")
    plot_distribution(rows, args.output_dir / stem)

    for row in rows:
        print(
            f"{row.dataset}: pairs={row.total_pairs}, anchors={row.total_anchors}, "
            f"sem_contexto={row.without_context_count} "
            f"({row.without_context_pct:.2f}%), "
            f"com_contexto_relevante={row.context_with_relevant_count} "
            f"({row.context_with_relevant_pct:.2f}%), "
            f"com_contexto_sem_relevante={row.context_without_relevant_count} "
            f"({row.context_without_relevant_pct:.2f}%)"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
