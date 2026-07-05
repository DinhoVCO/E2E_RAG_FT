from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mteb
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.cache import ResultCache
from mteb.models.search_wrappers import SearchEncoderWrapper
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

from tesis_unicamp.embeddings import (
    DEFAULT_EMBED_BATCH_SIZE,
    EmbeddingConfig,
    OpenAIEmbedder,
    VLLMOfflineEmbedder,
)
from tesis_unicamp.embeddings.base import BaseEmbedder
from tesis_unicamp.evaluation.mteb.embedder import TesisEmbedderEncoder


def configure_vllm_multiprocessing() -> None:
    """Use spawn for vLLM workers to avoid CUDA re-init errors after fork."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def build_tesis_embedder(
    *,
    mode: str,
    model: str,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> BaseEmbedder:
    config = EmbeddingConfig(model=model, batch_size=batch_size)
    if mode == "offline":
        return VLLMOfflineEmbedder(config)
    if mode == "online":
        return OpenAIEmbedder(config)
    raise ValueError(f"Unsupported embedder mode {mode!r}. Use 'offline' or 'online'.")


def resolve_model(
    *,
    backend: str,
    model: str,
    model_revision: str,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    embedder: BaseEmbedder | None = None,
) -> Any:
    """Build an MTEB-compatible model wrapper."""
    if backend == "sentence-transformers":
        return SentenceTransformerEncoderWrapper(
            model=model,
            revision=model_revision,
        )

    if embedder is None:
        if backend == "offline":
            configure_vllm_multiprocessing()
        embedder = build_tesis_embedder(
            mode=backend,
            model=model,
            batch_size=batch_size,
        )
    if backend == "offline" and isinstance(embedder, VLLMOfflineEmbedder):
        print("Loading vLLM embedder (warmup)...")
        embedder.warmup()
    encoder = TesisEmbedderEncoder(
        embedder,
        model_name=model,
        model_revision=model_revision,
    )
    return SearchEncoderWrapper(encoder)


def evaluate_retrieval(
    model: Any,
    tasks: list[AbsTaskRetrieval],
    *,
    output_folder: Path | str | None = None,
    overwrite_strategy: str = "always",
    encode_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Run MTEB retrieval evaluation on one or more tasks."""
    cache = (
        ResultCache(cache_path=output_folder)
        if output_folder is not None
        else ResultCache()
    )
    return mteb.evaluate(
        model,
        tasks=tasks,
        cache=cache,
        overwrite_strategy=overwrite_strategy,
        encode_kwargs=encode_kwargs,
        num_proc=1,
        **kwargs,
    )
