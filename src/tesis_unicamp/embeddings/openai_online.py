from __future__ import annotations

import os

from openai import OpenAI

from tesis_unicamp.embeddings.base import BaseEmbedder, EmbeddingConfig


class OpenAIEmbedder(BaseEmbedder):
    """Embed texts via an OpenAI-compatible API (e.g. vLLM /v1/embeddings)."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        super().__init__(config)
        self._client = client or OpenAI(
            base_url=base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=api_key or os.getenv("VLLM_API_KEY", "EMPTY"),
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
