from typing import TypedDict


class CorpusRecord(TypedDict):
    id: str
    title: str
    text: str


class QueryRecord(TypedDict):
    id: str
    text: str


class QrelRecord(TypedDict):
    query_id: str
    corpus_id: str
    score: int


class AnswerRecord(TypedDict):
    query_id: str
    answer: str


class SplitData(TypedDict):
    queries: list[QueryRecord]
    qrels: list[QrelRecord]
    answers: list[AnswerRecord]


class BioASQSnippet(TypedDict, total=False):
    offsetInBeginSection: int
    offsetInEndSection: int
    text: str
    beginSection: str
    endSection: str
    document: str


class BioASQQuestion(TypedDict, total=False):
    id: str
    body: str
    documents: list[str]
    ideal_answer: list[str]
    snippets: list[BioASQSnippet]
    type: str
    concepts: list[str]


class BioASQFile(TypedDict):
    questions: list[BioASQQuestion]
