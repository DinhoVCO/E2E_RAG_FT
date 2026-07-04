import hashlib
from collections.abc import Iterator
from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.models import (
    AnswerRecord,
    CorpusRecord,
    QrelRecord,
    QueryRecord,
    SplitData,
)


def make_query_id(document_id: str, question_text: str) -> str:
    digest = hashlib.blake2s(question_text.encode(), digest_size=4).hexdigest()
    return f"{document_id}_{digest}"


def longest_answer(answers: list[dict]) -> str:
    if not answers:
        return ""
    return max((answer.get("text", "") for answer in answers), key=len)


def _process_row(
    row: dict,
    documents: dict[str, CorpusRecord],
    split_data: SplitData,
) -> None:
    document = row["document"]
    question_text = row["question"]["text"]
    doc_id = document["id"]

    if doc_id not in documents:
        summary = document["summary"]
        documents[doc_id] = {
            "id": doc_id,
            "title": summary.get("title") or "",
            "text": summary["text"],
        }

    query_id = make_query_id(doc_id, question_text)
    split_data["queries"].append({"id": query_id, "text": question_text})
    split_data["qrels"].append(
        {
            "query_id": query_id,
            "corpus_id": doc_id,
            "score": 1,
        }
    )
    split_data["answers"].append(
        {
            "query_id": query_id,
            "answer": longest_answer(row["answers"]),
        }
    )


def process_narrativeqa_splits(
    splits: dict[str, Iterator[dict]],
) -> tuple[list[CorpusRecord], dict[str, SplitData]]:
    """Process all splits in one pass each, building corpus and split records."""
    documents: dict[str, CorpusRecord] = {}
    processed_splits: dict[str, SplitData] = {
        split_name: {"queries": [], "qrels": [], "answers": []}
        for split_name in splits
    }

    for split_name, dataset in splits.items():
        for row in dataset:
            _process_row(row, documents, processed_splits[split_name])

    corpus = [documents[doc_id] for doc_id in sorted(documents)]
    return corpus, processed_splits


def build_corpus(splits: dict[str, Iterator[dict]]) -> list[CorpusRecord]:
    documents: dict[str, CorpusRecord] = {}
    for dataset in splits.values():
        for row in dataset:
            document = row["document"]
            doc_id = document["id"]
            if doc_id in documents:
                continue
            summary = document["summary"]
            documents[doc_id] = {
                "id": doc_id,
                "title": summary.get("title") or "",
                "text": summary["text"],
            }
    return [documents[doc_id] for doc_id in sorted(documents)]


def process_narrativeqa_split(dataset: Iterator[dict]) -> SplitData:
    queries: list[QueryRecord] = []
    qrels: list[QrelRecord] = []
    answers: list[AnswerRecord] = []

    for row in dataset:
        document = row["document"]
        question_text = row["question"]["text"]
        doc_id = document["id"]
        query_id = make_query_id(doc_id, question_text)

        queries.append({"id": query_id, "text": question_text})
        qrels.append(
            {
                "query_id": query_id,
                "corpus_id": doc_id,
                "score": 1,
            }
        )
        answers.append(
            {
                "query_id": query_id,
                "answer": longest_answer(row["answers"]),
            }
        )

    return {
        "queries": queries,
        "qrels": qrels,
        "answers": answers,
    }
