from __future__ import annotations

from typing import TYPE_CHECKING

from tesis_unicamp.embeddings.base import BaseEmbedder, EmbeddingConfig

DEFAULT_MAX_LORA_RANK = 16

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.lora.request import LoRARequest


class VLLMOfflineEmbedder(BaseEmbedder):
    """Embed texts in-process with vLLM (LLM(..., task=\"embed\"))."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        llm: LLM | None = None,
        lora_path: str | None = None,
        lora_name: str = "adapter",
        lora_int_id: int = 1,
        max_lora_rank: int = DEFAULT_MAX_LORA_RANK,
        **llm_kwargs: object,
    ) -> None:
        super().__init__(config)
        self._llm = llm
        self._lora_path = lora_path
        self._lora_name = lora_name
        self._lora_int_id = lora_int_id
        self._max_lora_rank = max_lora_rank
        self._llm_kwargs = llm_kwargs
        self._lora_request: LoRARequest | None = None

    def _get_llm(self) -> LLM:
        if self._llm is None:
            from vllm import LLM

            llm_kwargs = dict(self._llm_kwargs)
            if self._lora_path is not None:
                llm_kwargs.setdefault("enable_lora", True)
                llm_kwargs.setdefault("max_lora_rank", self._max_lora_rank)
            self._llm = LLM(model=self.model_name, task="embed", **llm_kwargs)
        return self._llm

    def _get_lora_request(self) -> LoRARequest | None:
        if self._lora_path is None:
            return None
        if self._lora_request is None:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest(
                self._lora_name,
                self._lora_int_id,
                self._lora_path,
            )
        return self._lora_request

    def warmup(self) -> None:
        """Load the vLLM engine before any fork-based multiprocessing runs."""
        self._get_llm()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        lora_request = self._get_lora_request()
        if lora_request is None:
            outputs = self._get_llm().embed(texts)
        else:
            outputs = self._get_llm().embed(texts, lora_request=lora_request)
        return [output.outputs.embedding for output in outputs]
