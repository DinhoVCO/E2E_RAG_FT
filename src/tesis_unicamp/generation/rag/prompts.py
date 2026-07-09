from __future__ import annotations


def build_rag_user_prompt(question: str, context: str) -> str:
    """Build the user message for a single RAG generation request."""
    question = question.strip()
    context = context.strip()
    if not context:
        return (
            "No context was retrieved.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
    return (
        "Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
