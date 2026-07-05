from typing import TypedDict


class RetrievedDocRecord(TypedDict):
    query_id: str
    corpus_id: str
    rank: int
    retrieval_score: float
    is_relevant: bool
