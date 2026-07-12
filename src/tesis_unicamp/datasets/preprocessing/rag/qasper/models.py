from typing import TypedDict


class CorpusRecord(TypedDict):
    id: str
    title: str
    section_name: str
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


class TopRankedRecord(TypedDict):
    query_id: str
    corpus_ids: list[str]


class SplitData(TypedDict):
    queries: list[QueryRecord]
    qrels: list[QrelRecord]
    answers: list[AnswerRecord]
    top_ranked: list[TopRankedRecord]


class PaperIndex(TypedDict):
    chunks: list[CorpusRecord]
    text_to_ids: dict[str, list[str]]
    section_to_ids: dict[str, list[str]]
