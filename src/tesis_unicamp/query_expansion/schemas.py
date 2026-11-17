from typing import NotRequired, TypedDict


class ExpandedQueryRecord(TypedDict):
    query_id: str
    question: str
    expanded_query: str
    reference_answer: str
    generated_answer: NotRequired[str]
