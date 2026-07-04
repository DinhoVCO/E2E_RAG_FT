import random
from collections.abc import Callable

from datasets import Dataset

from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.models import (
    AnswerRecord,
    CorpusRecord,
    QrelRecord,
    QueryRecord,
    SplitData,
)


def corpus_title(row: dict) -> str:
    title = row["Document_title"]
    section = row.get("Section_name") or ""
    if section:
        return f"{title} | {section}"
    return title


def corpus_text(row: dict) -> str:
    parts = [row["text"]]
    table = row.get("Table") or ""
    table_summary = row.get("Table_summary") or ""
    if table:
        parts.append(table)
    if table_summary:
        parts.append(table_summary)
    return "\n\n".join(part for part in parts if part)


def build_corpus(corpus_dataset: Dataset) -> list[CorpusRecord]:
    documents: dict[str, CorpusRecord] = {}
    for row in corpus_dataset:
        doc_id = row["id"]
        documents[doc_id] = {
            "id": doc_id,
            "title": corpus_title(row),
            "text": corpus_text(row),
        }
    return [documents[doc_id] for doc_id in sorted(documents)]


def process_telco_dpr_split(
    relevant_docs: Dataset,
    queries_by_id: dict[str, dict],
) -> SplitData:
    queries: list[QueryRecord] = []
    qrels: list[QrelRecord] = []
    answers: list[AnswerRecord] = []

    for rel in relevant_docs:
        query_id = rel["query_id"]
        corpus_id = rel["corpus_id"]
        query = queries_by_id[query_id]

        queries.append({"id": query_id, "text": query["question"]})
        qrels.append(
            {
                "query_id": query_id,
                "corpus_id": corpus_id,
                "score": 1,
            }
        )
        answers.append(
            {
                "query_id": query_id,
                "answer": query["answer"],
            }
        )

    return {
        "queries": queries,
        "qrels": qrels,
        "answers": answers,
    }


def _partition_by_query_ids[T](
    records: list[T],
    dev_ids: set[str],
    key_fn: Callable[[T], str],
) -> tuple[list[T], list[T]]:
    dev_records = [record for record in records if key_fn(record) in dev_ids]
    train_records = [record for record in records if key_fn(record) not in dev_ids]
    return train_records, dev_records


def split_train_dev(
    split_data: SplitData,
    dev_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[SplitData, SplitData]:
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")

    query_ids = [query["id"] for query in split_data["queries"]]
    shuffled = list(query_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    dev_size = max(1, int(len(shuffled) * dev_ratio))
    dev_ids = set(shuffled[:dev_size])

    train_queries, dev_queries = _partition_by_query_ids(
        split_data["queries"],
        dev_ids,
        key_fn=lambda record: record["id"],
    )
    train_qrels, dev_qrels = _partition_by_query_ids(
        split_data["qrels"],
        dev_ids,
        key_fn=lambda record: record["query_id"],
    )
    train_answers, dev_answers = _partition_by_query_ids(
        split_data["answers"],
        dev_ids,
        key_fn=lambda record: record["query_id"],
    )

    return (
        {
            "queries": train_queries,
            "qrels": train_qrels,
            "answers": train_answers,
        },
        {
            "queries": dev_queries,
            "qrels": dev_qrels,
            "answers": dev_answers,
        },
    )
