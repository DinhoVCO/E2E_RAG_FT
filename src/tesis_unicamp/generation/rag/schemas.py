from typing import TypedDict


class GeneratedAnswerRecord(TypedDict):
    query_id: str
    question: str
    generated_answer: str
    reference_answer: str
