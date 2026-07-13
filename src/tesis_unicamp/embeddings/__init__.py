from tesis_unicamp.embeddings.base import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedder,
    EmbeddingConfig,
)

__all__ = [
    "DEFAULT_EMBED_BATCH_SIZE",
    "BaseEmbedder",
    "EmbeddingConfig",
    "OpenAIEmbedder",
    "VLLMOfflineEmbedder",
]


def __getattr__(name: str):
    if name == "OpenAIEmbedder":
        from tesis_unicamp.embeddings.openai_online import OpenAIEmbedder

        return OpenAIEmbedder
    if name == "VLLMOfflineEmbedder":
        from tesis_unicamp.embeddings.vllm_offline import VLLMOfflineEmbedder

        return VLLMOfflineEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
