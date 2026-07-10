from __future__ import annotations

from collections.abc import Callable
from typing import Any

from transformers import AutoTokenizer

from tesis_unicamp.generation.rag.prompts import build_rag_user_prompt

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question based only on the provided context. "
    "If the context does not contain enough information, say so briefly."
)


def _load_tokenizer(model_name: str, *, trust_remote_code: bool = True):
    kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
    if "mistral" in model_name.lower():
        kwargs["fix_mistral_regex"] = True
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def build_tokenizer_helpers(
    llm,
    *,
    use_chat_template: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[Callable[[str], int], Callable[[str, int], str]]:
    """Build tokenizer helpers from a loaded vLLM engine."""
    return build_tokenizer_helpers_from_tokenizer(
        llm.get_tokenizer(),
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )


def build_tokenizer_helpers_from_model_name(
    model_name: str,
    *,
    use_chat_template: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    trust_remote_code: bool = True,
) -> tuple[Callable[[str], int], Callable[[str, int], str]]:
    """Build tokenizer helpers without loading the full vLLM judge model."""
    tokenizer = _load_tokenizer(model_name, trust_remote_code=trust_remote_code)
    return build_tokenizer_helpers_from_tokenizer(
        tokenizer,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )


def build_tokenizer_helpers_from_tokenizer(
    tokenizer,
    *,
    use_chat_template: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[Callable[[str], int], Callable[[str, int], str]]:
    """Build tokenizer helpers aligned with the RAG generation pipeline."""

    def count_prompt_tokens(user_content: str) -> int:
        if use_chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            formatted = user_content
        return len(tokenizer.encode(formatted, add_special_tokens=False))

    def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)

    return count_prompt_tokens, truncate_text_to_tokens


def count_rag_prompt_tokens(
    question: str,
    context: str,
    *,
    count_prompt_tokens: Callable[[str], int],
) -> int:
    return count_prompt_tokens(build_rag_user_prompt(question, context))
