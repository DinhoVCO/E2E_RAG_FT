"""Analyze the actual RAG-context distribution used in generative fine-tuning.

The script reproduces the same seeded sampling, relevant-document injection,
token truncation, and prompt construction used by the generative trainer. It
reports mutually exclusive counts for examples without context, with context
and at least one qrel-relevant passage, and with context without such a passage.

Example:
    uv run python scripts/analyze_generative_context_distribution.py
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

from tesis_unicamp.finetuning.embeddings.datasets import (
    build_corpus_lookup,
    build_relevant_docs,
)
from tesis_unicamp.finetuning.generative.config import (
    DEFAULT_BASE_MODEL,
    MAX_ANSWER_TOKENS,
    MAX_CONTEXT_DOCS,
    MAX_DOC_TOKENS,
    MAX_QUERY_TOKENS,
    MAX_SEQ_LENGTH,
    RELEVANT_DOC_RATIO,
)
from tesis_unicamp.finetuning.generative.datasets import (
    _hits_to_doc_texts,
    _retrieved_rows_to_records,
    _select_context_hits,
    get_generative_finetuning_config,
)
from tesis_unicamp.finetuning.generative.formatting import (
    build_training_messages,
    truncate_to_tokens,
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
    "context_with_gold": "Com contexto e passagem relevante",
    "context_without_gold": "Com contexto sem passagem relevante",
}
CATEGORY_COLORS = {
    "without_context": "#7F8C8D",
    "context_with_gold": "#148F77",
    "context_without_gold": "#B9770E",
}


@dataclass(frozen=True)
class DistributionRow:
    dataset: str
    split: str
    total_examples: int
    without_context_count: int
    without_context_pct: float
    context_with_gold_count: int
    context_with_gold_pct: float
    context_without_gold_count: int
    context_without_gold_pct: float
    relevance_guarantee_applied_count: int
    relevance_guarantee_applied_pct: float


def _percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def analyze_dataset(
    dataset: str,
    *,
    split: str,
    seed: int,
    tokenizer,
) -> DistributionRow:
    config = get_generative_finetuning_config(dataset)
    corpus = config.load_corpus()
    queries = config.load_subset("queries", split=split)
    answers = config.load_subset("answers", split=split)
    qrels = config.load_subset("qrels", split=split)
    retrieved = config.load_subset("retrieved_docs", split=split)

    corpus_lookup = build_corpus_lookup(corpus, config.corpus_text_fn)
    relevant_docs = build_relevant_docs(qrels)
    grouped_retrieved = group_retrieved_by_query(
        _retrieved_rows_to_records(retrieved)
    )
    query_lookup = {str(row["id"]): str(row["text"]) for row in queries}
    answer_lookup = {
        str(row["query_id"]): str(row["answer"]) for row in answers
    }

    rng = random.Random(seed)
    counts = {
        "without_context": 0,
        "context_with_gold": 0,
        "context_without_gold": 0,
    }
    guarantee_applied = 0

    for query_id, answer in answer_lookup.items():
        query_text = query_lookup.get(query_id)
        if not query_text or not answer.strip():
            continue

        num_docs = rng.randint(0, MAX_CONTEXT_DOCS)
        relevant_ids = relevant_docs.get(query_id, set())
        ensure_relevant = num_docs > 0 and rng.random() < RELEVANT_DOC_RATIO
        if ensure_relevant:
            guarantee_applied += 1

        selected_hits = _select_context_hits(
            grouped_retrieved.get(query_id, []),
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
            max_query_tokens=MAX_QUERY_TOKENS,
            max_doc_tokens=MAX_DOC_TOKENS,
            max_answer_tokens=MAX_ANSWER_TOKENS,
            max_seq_length=MAX_SEQ_LENGTH,
        )
        if example is None:
            continue

        user_content = str(example["messages"][0]["content"])
        if "## Context:" not in user_content:
            counts["without_context"] += 1
            continue

        relevant_texts = []
        for hit in selected_hits:
            corpus_id = str(hit["corpus_id"])
            if corpus_id not in relevant_ids:
                continue
            text = corpus_lookup.get(corpus_id, "")
            truncated = truncate_to_tokens(tokenizer, text, MAX_DOC_TOKENS)
            if truncated:
                relevant_texts.append(truncated)

        if any(text in user_content for text in relevant_texts):
            counts["context_with_gold"] += 1
        else:
            counts["context_without_gold"] += 1

    total = sum(counts.values())
    return DistributionRow(
        dataset=DATASET_LABELS[dataset],
        split=split,
        total_examples=total,
        without_context_count=counts["without_context"],
        without_context_pct=_percentage(counts["without_context"], total),
        context_with_gold_count=counts["context_with_gold"],
        context_with_gold_pct=_percentage(counts["context_with_gold"], total),
        context_without_gold_count=counts["context_without_gold"],
        context_without_gold_pct=_percentage(
            counts["context_without_gold"], total
        ),
        relevance_guarantee_applied_count=guarantee_applied,
        relevance_guarantee_applied_pct=_percentage(guarantee_applied, total),
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
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    datasets = [row.dataset for row in rows]
    bottoms = [0.0] * len(rows)

    category_fields = {
        "without_context": "without_context_pct",
        "context_with_gold": "context_with_gold_pct",
        "context_without_gold": "context_without_gold_pct",
    }
    for category, field in category_fields.items():
        values = [float(getattr(row, field)) for row in rows]
        bars = ax.bar(
            datasets,
            values,
            bottom=bottoms,
            label=CATEGORY_LABELS[category],
            color=CATEGORY_COLORS[category],
        )
        for bar, value, bottom in zip(bars, values, bottoms):
            if value >= 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{value:.1f}%".replace(".", ","),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_xlabel("Conjunto de dados")
    ax.set_ylabel("Proporção dos exemplos de treinamento (%)")
    ax.set_ylim(0, 100)
    ax.legend(
        title="Composição do exemplo",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the RAG-context distribution used in SFT."
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
        default=PROJECT_ROOT / "reports" / "generative_context_distribution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    seed = args.seed if args.seed is not None else (42 if args.split == "train" else 43)
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

    stem = f"context_distribution_{args.split}"
    write_csv(rows, args.output_dir / f"{stem}.csv")
    write_json(rows, args.output_dir / f"{stem}.json")
    plot_distribution(rows, args.output_dir / stem)

    for row in rows:
        print(
            f"{row.dataset}: total={row.total_examples}, "
            f"sem_contexto={row.without_context_count} "
            f"({row.without_context_pct:.2f}%), "
            f"com_contexto_gold={row.context_with_gold_count} "
            f"({row.context_with_gold_pct:.2f}%), "
            f"com_contexto_sem_gold={row.context_without_gold_count} "
            f"({row.context_without_gold_pct:.2f}%)"
        )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
