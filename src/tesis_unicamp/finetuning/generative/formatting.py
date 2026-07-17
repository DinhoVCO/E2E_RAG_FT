from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from tesis_unicamp.finetuning.generative.config import (
    DEFAULT_INSTRUCTION,
    DEFAULT_QA_INSTRUCTION,
    MAX_ANSWER_TOKENS,
    MAX_DOC_TOKENS,
    MAX_QA_SEQ_LENGTH,
    MAX_QUERY_TOKENS,
    MAX_SEQ_LENGTH,
)

QUERY_TITLE_SECTION = "## Title:"


@dataclass(frozen=True)
class ContextDocument:
    title: str
    text: str


def format_context_document(doc: ContextDocument) -> str:
    title = doc.title.strip()
    text = doc.text.strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def truncate_to_tokens(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_tokens: int,
) -> str:
    text = text.strip()
    if not text or max_tokens <= 0:
        return ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_user_content(
    *,
    query: str,
    doc_texts: list[str] | None = None,
    context_docs: list[ContextDocument] | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
    query_title: str | None = None,
) -> str:
    query = query.strip()
    instruction = instruction.strip()
    parts = [instruction]
    if query_title and query_title.strip():
        parts.extend([QUERY_TITLE_SECTION, query_title.strip()])
    parts.extend(
        [
            "## Query:",
            query,
        ]
    )

    rendered_docs: list[str] = []
    if context_docs:
        rendered_docs = [
            format_context_document(doc)
            for doc in context_docs
            if doc.title.strip() or doc.text.strip()
        ]
    elif doc_texts:
        rendered_docs = [doc_text.strip() for doc_text in doc_texts if doc_text.strip()]

    if rendered_docs:
        parts.append("## Context:")
        for index, doc_text in enumerate(rendered_docs, start=1):
            parts.append(f"doc {index} :")
            parts.append(doc_text)
    parts.append("## Response:")
    return "\n".join(parts)


def apply_chat_template_for_training(
    tokenizer: PreTrainedTokenizerBase,
    *,
    user_content: str,
    answer: str,
) -> str:
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def build_training_messages(
    tokenizer: PreTrainedTokenizerBase,
    *,
    query: str,
    doc_texts: list[str],
    answer: str,
    instruction: str = DEFAULT_INSTRUCTION,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_answer_tokens: int = MAX_ANSWER_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, object] | None:
    truncated_query = truncate_to_tokens(tokenizer, query, max_query_tokens)
    truncated_answer = truncate_to_tokens(tokenizer, answer, max_answer_tokens)
    truncated_docs = [
        truncate_to_tokens(tokenizer, doc_text, max_doc_tokens)
        for doc_text in doc_texts
        if doc_text.strip()
    ]

    while True:
        user_content = build_user_content(
            query=truncated_query,
            doc_texts=truncated_docs,
            instruction=instruction,
        )
        text = apply_chat_template_for_training(
            tokenizer,
            user_content=user_content,
            answer=truncated_answer,
        )
        if count_tokens(tokenizer, text) <= max_seq_length:
            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": truncated_answer},
                ],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        if truncated_docs:
            truncated_docs.pop()
            continue
        if len(truncated_answer) > 1:
            token_ids = tokenizer.encode(truncated_answer, add_special_tokens=False)
            truncated_answer = tokenizer.decode(
                token_ids[: max(1, len(token_ids) // 2)],
                skip_special_tokens=True,
            )
            continue
        if len(truncated_query) > 1:
            token_ids = tokenizer.encode(truncated_query, add_special_tokens=False)
            truncated_query = tokenizer.decode(
                token_ids[: max(1, len(token_ids) // 2)],
                skip_special_tokens=True,
            )
            continue
        return None


def build_training_text(
    tokenizer: PreTrainedTokenizerBase,
    *,
    query: str,
    doc_texts: list[str],
    answer: str,
    instruction: str = DEFAULT_INSTRUCTION,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_doc_tokens: int = MAX_DOC_TOKENS,
    max_answer_tokens: int = MAX_ANSWER_TOKENS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> str:
    example = build_training_messages(
        tokenizer,
        query=query,
        doc_texts=doc_texts,
        answer=answer,
        instruction=instruction,
        max_query_tokens=max_query_tokens,
        max_doc_tokens=max_doc_tokens,
        max_answer_tokens=max_answer_tokens,
        max_seq_length=max_seq_length,
    )
    if example is None:
        return ""
    messages = example["messages"]
    return apply_chat_template_for_training(
        tokenizer,
        user_content=messages[0]["content"],
        answer=messages[1]["content"],
    )


def build_qa_user_content(
    *,
    query: str,
    instruction: str = DEFAULT_QA_INSTRUCTION,
    query_title: str | None = None,
) -> str:
    query = query.strip()
    instruction = instruction.strip()
    parts = [instruction]
    if query_title and query_title.strip():
        parts.extend([QUERY_TITLE_SECTION, query_title.strip()])
    parts.extend(
        [
            "## Query:",
            query,
            "## Response:",
        ]
    )
    return "\n".join(parts)


def build_qa_training_messages(
    tokenizer: PreTrainedTokenizerBase,
    *,
    query: str,
    answer: str,
    instruction: str = DEFAULT_QA_INSTRUCTION,
    max_query_tokens: int = MAX_QUERY_TOKENS,
    max_answer_tokens: int = MAX_ANSWER_TOKENS,
    max_seq_length: int = MAX_QA_SEQ_LENGTH,
) -> dict[str, object] | None:
    truncated_query = truncate_to_tokens(tokenizer, query, max_query_tokens)
    truncated_answer = truncate_to_tokens(tokenizer, answer, max_answer_tokens)

    while True:
        user_content = build_qa_user_content(
            query=truncated_query,
            instruction=instruction,
        )
        text = apply_chat_template_for_training(
            tokenizer,
            user_content=user_content,
            answer=truncated_answer,
        )
        if count_tokens(tokenizer, text) <= max_seq_length:
            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": truncated_answer},
                ],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        if len(truncated_answer) > 1:
            token_ids = tokenizer.encode(truncated_answer, add_special_tokens=False)
            truncated_answer = tokenizer.decode(
                token_ids[: max(1, len(token_ids) // 2)],
                skip_special_tokens=True,
            )
            continue
        if len(truncated_query) > 1:
            token_ids = tokenizer.encode(truncated_query, add_special_tokens=False)
            truncated_query = tokenizer.decode(
                token_ids[: max(1, len(token_ids) // 2)],
                skip_special_tokens=True,
            )
            continue
        return None
