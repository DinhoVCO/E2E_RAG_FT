from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tqdm import tqdm

DEFAULT_GENERATION_BATCH_SIZE = 8


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    batch_size: int = DEFAULT_GENERATION_BATCH_SIZE
    max_tokens: int = 512
    temperature: float = 0.0
    n: int = 1
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: tuple[str, ...] = ()


class BaseGenerator(ABC):
    """Base class for text generation backends."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    @property
    def model_name(self) -> str:
        return self.config.model

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @abstractmethod
    def generate_texts(self, prompts: list[str]) -> list[str]:
        """Generate one completion per prompt, in the same order."""

    def generate_all(
        self,
        prompts: list[str],
        *,
        show_progress: bool = True,
    ) -> list[str]:
        if not prompts:
            return []

        batches = [
            prompts[i : i + self.batch_size]
            for i in range(0, len(prompts), self.batch_size)
        ]
        iterator = tqdm(batches, desc="Generating", unit="batch") if show_progress else batches

        outputs: list[str] = []
        for batch in iterator:
            outputs.extend(self.generate_texts(batch))
        return outputs
