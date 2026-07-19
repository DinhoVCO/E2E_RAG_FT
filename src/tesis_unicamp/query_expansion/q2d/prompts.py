from __future__ import annotations

from tesis_unicamp.query_expansion.q2d.schemas import FewShotExample

Q2D_PROMPT_HEADER = "Write a passage that answers the given query:\n"


def build_q2d_prompt(
    target_query: str,
    examples: list[FewShotExample],
) -> str:
    """Build a Query2Doc few-shot prompt with train (query, passage) pairs."""
    parts = [Q2D_PROMPT_HEADER]
    for example in examples:
        query = example["query"].strip()
        passage = example["passage"].strip()
        parts.append(f"Query: {query}\nPassage: {passage}\n\n")
    parts.append(f"Query: {target_query.strip()}\nPassage:")
    return "".join(parts)
