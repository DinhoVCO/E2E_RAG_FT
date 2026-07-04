import random

from tesis_unicamp.datasets.preprocessing.rag.bioasq.models import (
    AnswerRecord,
    BioASQQuestion,
    CorpusRecord,
    QrelRecord,
    QueryRecord,
    SplitData,
)
from tesis_unicamp.datasets.preprocessing.rag.bioasq.pubmed import extract_pmid


def is_abstract_snippet(snippet: dict) -> bool:
    return (
        snippet.get("beginSection") == "abstract"
        and snippet.get("endSection") == "abstract"
    )


def longest_ideal_answer(ideal_answers: list[str]) -> str:
    if not ideal_answers:
        return ""
    return max(ideal_answers, key=len)


def collect_pmids_from_questions(questions: list[BioASQQuestion]) -> set[str]:
    pmids: set[str] = set()
    for question in questions:
        for snippet in question.get("snippets", []):
            if not is_abstract_snippet(snippet):
                continue
            pmid = extract_pmid(snippet.get("document", ""))
            if pmid:
                pmids.add(pmid)
    return pmids


def build_corpus(
    pmids: set[str],
    pubmed_records: dict[str, dict[str, str]],
) -> list[CorpusRecord]:
    corpus: list[CorpusRecord] = []
    for pmid in sorted(pmids):
        record = pubmed_records.get(pmid, {"title": "", "text": ""})
        corpus.append(
            {
                "id": pmid,
                "title": record.get("title", ""),
                "text": record.get("text", ""),
            }
        )
    return corpus


def process_bioasq_questions(questions: list[BioASQQuestion]) -> SplitData:
    queries: list[QueryRecord] = []
    qrels: list[QrelRecord] = []
    answers: list[AnswerRecord] = []
    seen_qrels: set[tuple[str, str]] = set()

    for question in questions:
        query_id = question["id"]
        queries.append({"id": query_id, "text": question["body"]})
        answers.append(
            {
                "query_id": query_id,
                "answer": longest_ideal_answer(question.get("ideal_answer", [])),
            }
        )

        for snippet in question.get("snippets", []):
            if not is_abstract_snippet(snippet):
                continue
            pmid = extract_pmid(snippet.get("document", ""))
            if not pmid:
                continue
            key = (query_id, pmid)
            if key in seen_qrels:
                continue
            seen_qrels.add(key)
            qrels.append(
                {
                    "query_id": query_id,
                    "corpus_id": pmid,
                    "score": 1,
                }
            )

    return {
        "queries": queries,
        "qrels": qrels,
        "answers": answers,
    }


def split_train_dev(
    questions: list[BioASQQuestion],
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[BioASQQuestion], list[BioASQQuestion]]:
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")

    shuffled = list(questions)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    dev_size = max(1, int(len(shuffled) * dev_ratio))
    dev = shuffled[:dev_size]
    train = shuffled[dev_size:]
    return train, dev
