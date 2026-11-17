from typing import TypedDict


class FewShotExample(TypedDict):
    query_id: str
    query: str
    passage: str


class Q2dRecord(TypedDict):
    query_id: str
    question: str
    generated_passage: str
    expanded_query: str
    reference_answer: str
    few_shot_examples: list[FewShotExample]
