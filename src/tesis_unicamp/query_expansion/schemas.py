from typing import TypedDict


class ExpandedQueryRecord(TypedDict):
    query_id: str
    question: str
    expanded_query: str
    reference_answer: str
