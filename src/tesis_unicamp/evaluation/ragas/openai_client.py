from __future__ import annotations

import os
from typing import Any

from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms.base import LangchainLLMWrapper

VLLM_FINISH_REASONS = {"stop", "STOP", "MAX_TOKENS", "max_tokens", "length", "eos_token"}

DEFAULT_OPENAI_API_KEY = "EMPTY"
DEFAULT_JUDGE_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"


def build_vllm_chat_extra_body(
    *,
    enable_thinking: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    """Extra OpenAI payload for vLLM (thinking off, no reasoning for gpt-oss)."""
    if model and "gpt-oss" in model.lower():
        return {
            "reasoning_effort": "high" if enable_thinking else "none",
        }

    if enable_thinking:
        return {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "thinking": True,
            },
        }
    return {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "thinking": False,
        },
    }


def normalize_openai_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def resolve_openai_api_key(api_key: str | None = None) -> str:
    return api_key or os.getenv("RAGAS_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_API_KEY


def vllm_is_finished(response: LLMResult) -> bool:
    """Treat vLLM OpenAI finish reasons (including 'length') as completed."""
    for generation in response.flatten():
        resp = generation.generations[0][0]
        finish_reason = None
        if resp.generation_info is not None:
            finish_reason = resp.generation_info.get("finish_reason")
        elif getattr(resp, "message", None) is not None:
            finish_reason = resp.message.response_metadata.get("finish_reason")
        if finish_reason is not None:
            return finish_reason in VLLM_FINISH_REASONS
    return True


def build_openai_client(*, base_url: str, api_key: str | None = None, timeout: float | None = None) -> OpenAI:
    kwargs: dict[str, Any] = {
        "api_key": resolve_openai_api_key(api_key),
        "base_url": normalize_openai_base_url(base_url),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAI(**kwargs)


def build_openai_async_client(
    *,
    base_url: str,
    api_key: str | None = None,
    timeout: float | None = None,
) -> AsyncOpenAI:
    kwargs: dict[str, Any] = {
        "api_key": resolve_openai_api_key(api_key),
        "base_url": normalize_openai_base_url(base_url),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return AsyncOpenAI(**kwargs)


def build_openai_judge_llm(
    model: str,
    *,
    base_url: str,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float | None = None,
    bypass_n: bool = False,
    enable_thinking: bool = False,
    **chat_openai_kwargs: Any,
) -> LangchainLLMWrapper:
    """Build a RAGAS judge LLM that calls a vLLM OpenAI-compatible chat server."""
    extra_body = build_vllm_chat_extra_body(enable_thinking=enable_thinking, model=model)
    llm = ChatOpenAI(
        model=model,
        api_key=resolve_openai_api_key(api_key),
        base_url=normalize_openai_base_url(base_url),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body=extra_body,
        **chat_openai_kwargs,
    )
    return LangchainLLMWrapper(
        llm,
        bypass_n=bypass_n,
        is_finished_parser=vllm_is_finished,
    )


def build_openai_ragas_embeddings(
    model: str,
    *,
    base_url: str,
    api_key: str | None = None,
    timeout: float | None = None,
) -> OpenAIEmbeddings:
    """Build RAGAS embeddings that call a vLLM OpenAI-compatible /embeddings server."""
    # RAGAS evaluate() runs async and calls aembed_text(); requires AsyncOpenAI.
    client = build_openai_async_client(base_url=base_url, api_key=api_key, timeout=timeout)
    return OpenAIEmbeddings(client=client, model=model)


def check_openai_server(
    label: str,
    *,
    base_url: str,
    api_key: str | None = None,
    expected_model: str | None = None,
    timeout: float = 30.0,
) -> list[str]:
    """Verify that an OpenAI-compatible vLLM server is reachable."""
    client = build_openai_client(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.models.list()
    model_ids = [item.id for item in response.data]
    print(f"{label} base_url: {normalize_openai_base_url(base_url)}")
    print(f"{label} available models: {model_ids or '(none)'}")
    if expected_model and model_ids and expected_model not in model_ids:
        print(
            f"Warning: {label} model {expected_model!r} not listed by the server. "
            "Ensure --served-model-name matches --judge-model / --embedding-model."
        )
    return model_ids
