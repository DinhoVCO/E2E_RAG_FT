from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

from mteb.types import PromptType

from tesis_unicamp.datasets.utils.bioasq_rag import query_to_instruct_text
from tesis_unicamp.embeddings.base import BaseEmbedder
from tesis_unicamp.query_expansion.hyde.embedding import compute_hyde_vector

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


class HydeMtebEncoder(AbsEncoder):
    """MTEB encoder that uses averaged HyDE query+passage embeddings for queries."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        *,
        passage_lookup: dict[str, list[str]],
        model_name: str | None = None,
        model_revision: str,
        include_query: bool = True,
    ) -> None:
        self.embedder = embedder
        self.passage_lookup = passage_lookup
        self.include_query = include_query
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
        del task_metadata, hf_split, hf_subset, kwargs

        texts = [str(text) for batch in inputs for text in batch["text"]]
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if prompt_type == PromptType.document:
            vectors = self.embedder.embed_texts(texts)
            return np.asarray(vectors, dtype=np.float32)

        vectors: list[np.ndarray] = []
        for instruct_query in texts:
            passages = self.passage_lookup.get(instruct_query)
            if passages is None:
                raise KeyError(
                    "Missing HyDE pseudo-passages for MTEB query text. "
                    "Ensure generation was run for this split and query formatting matches "
                    f"query_to_instruct_text. Query prefix: {instruct_query[:120]!r}..."
                )
            raw_query = _strip_instruct_query(instruct_query)
            vectors.append(
                compute_hyde_vector(
                    self.embedder,
                    query=raw_query,
                    pseudo_passages=passages,
                    include_query=self.include_query,
                    query_to_text=query_to_instruct_text,
                )
            )
        return np.stack(vectors, axis=0)


def _strip_instruct_query(instruct_query: str) -> str:
    prefix = "Instruct:"
    query_marker = "Query:"
    text = instruct_query.strip()
    if query_marker in text:
        return text.rsplit(query_marker, maxsplit=1)[-1].strip()
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


def build_instruct_passage_lookup(
    records: list[dict[str, object]],
) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for record in records:
        question = str(record["question"])
        passages = [str(item) for item in record["pseudo_passages"]]
        lookup[query_to_instruct_text(question)] = passages
    return lookup
