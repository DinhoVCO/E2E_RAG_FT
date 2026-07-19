from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mteb.models.search_wrappers import SearchEncoderWrapper

from tesis_unicamp.evaluation.mteb.runner import (
    build_tesis_embedder,
    configure_vllm_multiprocessing,
    evaluate_retrieval,
)
from tesis_unicamp.evaluation.mteb.tasks import RAG_RETRIEVAL_TASK_CONFIGS, get_rag_retrieval_task
from tesis_unicamp.query_expansion.hyde.io import load_hyde_records
from tesis_unicamp.query_expansion.hyde.mteb_encoder import (
    HydeMtebEncoder,
    build_instruct_passage_lookup,
)

HYDE_DATASET_IDS = tuple(RAG_RETRIEVAL_TASK_CONFIGS)


@dataclass(frozen=True)
class HydeMtebEvalConfig:
    dataset: str
    hyde_dir: Path
    model: str = "Qwen/Qwen3-Embedding-4B"
    lora_path: str | None = None
    splits: tuple[str, ...] = ("test",)
    backend: str = "offline"
    batch_size: int = 128
    model_revision: str = "hyde-base"
    output_dir: Path | None = None
    overwrite: str = "always"
    max_lora_rank: int = 16
    paper_scoped: bool | None = None
    include_query: bool = True
    num_passages: int = 8


def resolve_paper_scoped(dataset: str, paper_scoped: bool | None) -> bool:
    if paper_scoped is not None:
        return paper_scoped
    return dataset == "qasper"


def evaluate_hyde_mteb(config: HydeMtebEvalConfig) -> list[Any]:
    if config.dataset not in HYDE_DATASET_IDS:
        valid = ", ".join(sorted(HYDE_DATASET_IDS))
        raise ValueError(f"Unknown dataset {config.dataset!r}. Available: {valid}")

    paper_scoped = resolve_paper_scoped(config.dataset, config.paper_scoped)
    output_dir = config.output_dir or Path("results") / "mteb" / "hyde" / config.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.backend == "offline":
        configure_vllm_multiprocessing()

    passage_lookup: dict[str, list[str]] = {}
    for split in config.splits:
        records = load_hyde_records(config.hyde_dir, split)
        passage_lookup.update(build_instruct_passage_lookup(records))

    embedder = build_tesis_embedder(
        mode=config.backend,
        model=config.model,
        batch_size=config.batch_size,
        lora_path=config.lora_path,
        max_lora_rank=config.max_lora_rank,
    )
    if config.backend == "offline":
        from tesis_unicamp.embeddings.vllm_offline import VLLMOfflineEmbedder

        if isinstance(embedder, VLLMOfflineEmbedder):
            print("Loading vLLM embedder (warmup)...")
            embedder.warmup()

    encoder = HydeMtebEncoder(
        embedder,
        passage_lookup=passage_lookup,
        model_name=config.model,
        model_revision=config.model_revision,
        include_query=config.include_query,
    )
    model = SearchEncoderWrapper(encoder)

    task = get_rag_retrieval_task(
        config.dataset,
        eval_splits=config.splits,
        use_top_ranked=paper_scoped if config.dataset == "qasper" else False,
    )

    print(f"dataset: {config.dataset}")
    print(f"model: {config.model}")
    if config.lora_path:
        print(f"lora_path: {config.lora_path}")
    print(f"hyde_dir: {config.hyde_dir}")
    print(f"num_passages: {config.num_passages}")
    print(f"include_query_in_average: {config.include_query}")
    print(f"model_revision: {config.model_revision}")
    print(f"queries_with_hyde: {len(passage_lookup)}")
    print(f"output_dir: {output_dir}")

    results = evaluate_retrieval(
        model,
        [task],
        output_folder=output_dir,
        overwrite_strategy=config.overwrite,
        encode_kwargs={"batch_size": config.batch_size},
    )
    for task_result in results:
        print(task_result)
    return results
