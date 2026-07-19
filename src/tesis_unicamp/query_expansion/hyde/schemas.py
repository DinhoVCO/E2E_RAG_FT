from typing import TypedDict


class HydeRecord(TypedDict):
    query_id: str
    question: str
    pseudo_passages: list[str]
    reference_answer: str
