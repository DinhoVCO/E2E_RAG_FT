from __future__ import annotations

from tesis_unicamp.finetuning.generative.formatting import build_user_content

RFG_LONG_FORM_INSTRUCTION = (
    "Provide a detailed, long-form answer to the query using the retrieved "
    "documents as context. If the documents do not contain sufficient "
    "information, supplement your answer using your own knowledge."
)


def build_rfg_expansion_prompt(
    *,
    query: str,
    doc_texts: list[str],
) -> str:
    """Build the user prompt for RFG long-form query expansion."""
    return build_user_content(
        query=query,
        doc_texts=doc_texts,
        instruction=RFG_LONG_FORM_INSTRUCTION,
    )
