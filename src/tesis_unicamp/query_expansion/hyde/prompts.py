from __future__ import annotations

RAG_HYDE_PROMPT_TEMPLATE = """Please write a passage to answer the question.
Question: {}
Passage:"""


def build_hyde_prompt(query: str) -> str:
    """Build a HyDE pseudo-passage prompt for project RAG datasets."""
    return RAG_HYDE_PROMPT_TEMPLATE.format(query.strip())
