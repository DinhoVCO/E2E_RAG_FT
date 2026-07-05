from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

from tesis_unicamp.embeddings.base import BaseEmbedder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


class TesisEmbedderEncoder(AbsEncoder):
    """Wrap a project embedder so it can be evaluated with MTEB retrieval tasks."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        *,
        model_name: str | None = None,
        model_revision: str,
    ) -> None:
        self.embedder = embedder
        self.model = embedder
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={
                "name": model_name or embedder.model_name,
                "revision": model_revision,
                "similarity_fn_name": ScoringFunction.COSINE,
                "use_instructions": False,
            }
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        del task_metadata, hf_split, hf_subset, prompt_type, kwargs

        texts = [str(text) for batch in inputs for text in batch["text"]]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors = self.embedder.embed_texts(texts)
        return np.asarray(vectors, dtype=np.float32)
