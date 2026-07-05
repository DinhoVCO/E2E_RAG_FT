from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tqdm import tqdm

# Qwen3-Embedding-4B on 80 GB GPUs; lower if OOM on very long documents.
DEFAULT_EMBED_BATCH_SIZE = 128


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE


class BaseEmbedder(ABC):
    """Base class for text embedding backends."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return vectors in the same order."""

    def embed_all(
        self,
        texts: list[str],
        *,
        show_progress: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        iterator = tqdm(batches, desc="Embedding", unit="batch") if show_progress else batches

        vectors: list[list[float]] = []
        for batch in iterator:
            vectors.extend(self.embed_texts(batch))
        return vectors
