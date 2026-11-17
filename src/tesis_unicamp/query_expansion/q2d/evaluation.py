from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tesis_unicamp.datasets.utils.bioasq_rag import query_to_instruct_text
from tesis_unicamp.evaluation.mteb.context_retrieval import (
    create_rag_retrieval_task_with_query_maps,
)
from tesis_unicamp.evaluation.mteb.runner import (
    configure_vllm_multiprocessing,
    evaluate_retrieval,
    resolve_model,
)
from tesis_unicamp.evaluation.mteb.tasks import RAG_RETRIEVAL_TASK_CONFIGS
from tesis_unicamp.query_expansion.q2d.io import load_q2d_records

Q2D_DATASET_IDS = tuple(RAG_RETRIEVAL_TASK_CONFIGS)


@dataclass(frozen=True)
class Q2dMtebEvalConfig:
    dataset: str
    q2d_dir: Path
    model: str = "Qwen/Qwen3-Embedding-4B"
    lora_path: str | None = None
    splits: tuple[str, ...] = ("test",)
    backend: str = "offline"
    batch_size: int = 128
    model_revision: str = "q2d-base"
    output_dir: Path | None = None
    overwrite: str = "always"
    max_lora_rank: int = 16
    paper_scoped: bool | None = None
    apply_instruct_format: bool = True


def resolve_paper_scoped(dataset: str, paper_scoped: bool | None) -> bool:
    if paper_scoped is not None:
        return paper_scoped
    return dataset == "qasper"


def build_q2d_query_maps(
    *,
    q2d_dir: Path,
    splits: tuple[str, ...],
    apply_instruct_format: bool = True,
) -> dict[str, dict[str, str]]:
    query_maps: dict[str, dict[str, str]] = {}
    for split in splits:
        records = load_q2d_records(q2d_dir, split)
        split_map: dict[str, str] = {}
        for record in records:
            query_id = str(record["query_id"])
            expanded = str(record["expanded_query"]).strip()
            if apply_instruct_format:
                expanded = query_to_instruct_text(expanded)
            split_map[query_id] = expanded
        query_maps[split] = split_map
    return query_maps


def evaluate_q2d_mteb(config: Q2dMtebEvalConfig) -> list[Any]:
    """Run MTEB retrieval using query + generated passage as the query text."""
    if config.dataset not in Q2D_DATASET_IDS:
        valid = ", ".join(sorted(Q2D_DATASET_IDS))
        raise ValueError(f"Unknown dataset {config.dataset!r}. Available: {valid}")

    paper_scoped = resolve_paper_scoped(config.dataset, config.paper_scoped)
    output_dir = config.output_dir or Path("results") / "mteb" / "q2d" / config.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.backend == "offline":
        configure_vllm_multiprocessing()

    from tesis_unicamp.evaluation.mteb.runner import build_tesis_embedder

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

    query_maps = build_q2d_query_maps(
        q2d_dir=config.q2d_dir,
        splits=config.splits,
        apply_instruct_format=config.apply_instruct_format,
    )

    rag_config = RAG_RETRIEVAL_TASK_CONFIGS[config.dataset]
    rag_config = replace(
        rag_config,
        eval_splits=config.splits,
        query_text_fn=None,
        use_top_ranked=paper_scoped if config.dataset == "qasper" else False,
    )
    task = create_rag_retrieval_task_with_query_maps(
        rag_config,
        query_maps,
        task_name_suffix="q2d",
    )

    model = resolve_model(
        backend=config.backend,
        model=config.model,
        batch_size=config.batch_size,
        model_revision=config.model_revision,
        lora_path=config.lora_path,
        max_lora_rank=config.max_lora_rank,
        embedder=embedder,
    )

    print(f"dataset: {config.dataset}")
    print(f"model: {config.model}")
    if config.lora_path:
        print(f"lora_path: {config.lora_path}")
    print(f"q2d_dir: {config.q2d_dir}")
    print(f"model_revision: {config.model_revision}")
    print(f"num_expanded_queries: {sum(len(m) for m in query_maps.values())}")
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
