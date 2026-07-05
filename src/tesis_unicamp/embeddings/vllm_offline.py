from __future__ import annotations

from typing import TYPE_CHECKING

from tesis_unicamp.embeddings.base import BaseEmbedder, EmbeddingConfig

if TYPE_CHECKING:
    from vllm import LLM


class VLLMOfflineEmbedder(BaseEmbedder):
    """Embed texts in-process with vLLM (LLM(..., task=\"embed\"))."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        llm: LLM | None = None,
        **llm_kwargs: object,
    ) -> None:
        super().__init__(config)
        self._llm = llm
        self._llm_kwargs = llm_kwargs

    def _get_llm(self) -> LLM:
        if self._llm is None:
            from vllm import LLM

            self._llm = LLM(model=self.model_name, task="embed", **self._llm_kwargs)
        return self._llm

    def warmup(self) -> None:
        """Load the vLLM engine before any fork-based multiprocessing runs."""
        self._get_llm()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        outputs = self._get_llm().embed(texts)
        return [output.outputs.embedding for output in outputs]
