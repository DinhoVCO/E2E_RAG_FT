from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tesis_unicamp.generation.rag.datasets import RAG_GENERATION_DATASET_CONFIGS
from tesis_unicamp.query_expansion.generation import (
    DEFAULT_EXPANSION_K_VALUES,
    DEFAULT_RETRIEVAL_TOP_K,
    validate_expansion_k_values,
)

_EMBEDDING_MODES = frozenset({"base", "lora"})
_GENERATION_MODES = frozenset({"base", "lora"})

_EXPERIMENT_KEYS = frozenset(
    {
        "dataset",
        "embedding",
        "generation",
        "run_label",
        "retrieval_run_label",
        "embedding_model",
        "embedding_lora",
        "generation_model",
        "generation_lora",
        "stage1_top_k",
        "retrieval_top_k",
        "expansion_k_values",
        "split",
        "paper_scoped",
        "retrieval_batch_size",
        "generation_batch_size",
        "generation_max_tokens",
        "max_tokens_per_chunk",
        "max_prompt_tokens",
        "stage2_model_revision",
        "stage2_output_dir",
        "retrieved_root",
    }
)


@dataclass(frozen=True)
class ResolvedRfgExperiment:
    experiment_id: str
    dataset: str
    run_label: str
    retrieval_run_label: str
    embedding_model: str
    embedding_lora: str | None
    generation_model: str
    generation_lora: str | None
    embedding_mode: str
    generation_mode: str
    stage1_top_k: int
    retrieval_top_k: int
    expansion_k_values: tuple[int, ...]
    split: str
    paper_scoped: bool
    retrieval_batch_size: int
    generation_batch_size: int
    generation_max_tokens: int
    max_tokens_per_chunk: int
    max_prompt_tokens: int
    stage2_model_revision: str
    stage2_output_dir: Path | None
    retrieved_root: str


def default_experiments_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "query_expansion"
        / "configs"
        / "experiments.yaml"
    )


def load_experiments_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Experiments YAML must be a mapping: {path}")

    experiments = raw.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError(f"Experiments YAML must define a non-empty 'experiments' map: {path}")

    return raw


def list_experiment_ids(raw: dict[str, Any]) -> list[str]:
    experiments = raw["experiments"]
    return sorted(experiments)


def _defaults(raw: dict[str, Any]) -> dict[str, Any]:
    defaults = raw.get("defaults")
    if defaults is None:
        return {}
    if not isinstance(defaults, dict):
        raise ValueError("'defaults' must be a mapping.")
    return defaults


def _dataset_options(raw: dict[str, Any], dataset: str) -> dict[str, Any]:
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        return {}
    options = datasets.get(dataset)
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError(f"'datasets.{dataset}' must be a mapping.")
    return options


def _resolve_paper_scoped(
    *,
    dataset: str,
    explicit: bool | None,
    dataset_options: dict[str, Any],
) -> bool:
    if explicit is not None:
        return explicit
    if "paper_scoped" in dataset_options:
        return bool(dataset_options["paper_scoped"])
    return dataset == "qasper"


def _resolve_retrieval_run_label(
    *,
    dataset: str,
    embedding_mode: str,
    explicit: str | None,
    defaults: dict[str, Any],
    paper_scoped: bool,
) -> str:
    if explicit is not None:
        run_label = explicit
    elif embedding_mode == "base":
        run_label = str(defaults.get("base_retrieval_run_label", "vllm-offline-b128"))
    else:
        template = str(
            defaults.get("lora_retrieval_run_label_template", "vllm-lora-{dataset}-b128")
        )
        run_label = template.format(dataset=dataset)

    if paper_scoped and "paper-scoped" not in run_label.lower():
        return f"{run_label}-paper-scoped"
    return run_label


def _resolve_generation_lora(
    *,
    generation_mode: str,
    dataset: str,
    explicit: str | None,
    dataset_options: dict[str, Any],
) -> str | None:
    if explicit is not None:
        return explicit or None
    if generation_mode == "base":
        return None
    if "generation_lora" in dataset_options:
        value = dataset_options["generation_lora"]
        return str(value) if value else None
    raise ValueError(
        f"generation mode 'lora' requires "
        f"'datasets.{dataset}.generation_lora' or an explicit path."
    )


def _resolve_embedding_lora(
    *,
    embedding_mode: str,
    dataset: str,
    explicit: str | None,
    dataset_options: dict[str, Any],
) -> str | None:
    if explicit is not None:
        return explicit or None
    if embedding_mode == "base":
        return None
    if "embedding_lora" in dataset_options:
        value = dataset_options["embedding_lora"]
        return str(value) if value else None
    raise ValueError(
        f"embedding mode 'lora' requires "
        f"'datasets.{dataset}.embedding_lora' or an explicit path."
    )


def _resolve_expansion_k_values(spec: dict[str, Any], defaults: dict[str, Any]) -> tuple[int, ...]:
    values = spec.get("expansion_k_values", defaults.get("expansion_k_values", list(DEFAULT_EXPANSION_K_VALUES)))
    if not isinstance(values, list) or not values:
        raise ValueError("'expansion_k_values' must be a non-empty list of integers.")
    return tuple(int(value) for value in values)


def _resolve_stage2_model_revision(
    *,
    dataset: str,
    embedding_mode: str,
    generation_mode: str,
    expansion_k: int,
    explicit: str | None,
    defaults: dict[str, Any],
) -> str:
    if explicit is not None:
        return explicit.format(k=expansion_k) if "{k}" in explicit else explicit
    template = str(
        defaults.get(
            "stage2_model_revision_template",
            "rfg-{dataset}-emb-{embedding}-gen-{generation}-k{k}",
        )
    )
    return template.format(
        dataset=dataset,
        embedding=embedding_mode,
        generation=generation_mode,
        k=expansion_k,
    )


def resolve_experiment(raw: dict[str, Any], experiment_id: str) -> ResolvedRfgExperiment:
    experiments = raw["experiments"]
    if experiment_id not in experiments:
        available = ", ".join(sorted(experiments))
        raise ValueError(f"Unknown experiment {experiment_id!r}. Available: {available}")

    spec = experiments[experiment_id]
    if not isinstance(spec, dict):
        raise ValueError(f"Experiment {experiment_id!r} must be a mapping.")

    unknown = set(spec) - _EXPERIMENT_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys in experiment {experiment_id!r}: {', '.join(sorted(unknown))}"
        )

    defaults = _defaults(raw)
    dataset = spec.get("dataset")
    if not dataset:
        raise ValueError(f"Experiment {experiment_id!r} must define 'dataset'.")
    if dataset not in RAG_GENERATION_DATASET_CONFIGS:
        valid = ", ".join(sorted(RAG_GENERATION_DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset {dataset!r} in {experiment_id!r}. Expected: {valid}")

    embedding_mode = spec.get("embedding", "base")
    generation_mode = spec.get("generation")
    if embedding_mode not in _EMBEDDING_MODES:
        raise ValueError(
            f"Experiment {experiment_id!r} must define embedding as one of: "
            f"{', '.join(sorted(_EMBEDDING_MODES))}"
        )
    if generation_mode not in _GENERATION_MODES:
        raise ValueError(
            f"Experiment {experiment_id!r} must define generation as one of: "
            f"{', '.join(sorted(_GENERATION_MODES))}"
        )

    dataset_options = _dataset_options(raw, dataset)
    paper_scoped = _resolve_paper_scoped(
        dataset=dataset,
        explicit=spec.get("paper_scoped"),
        dataset_options=dataset_options,
    )

    retrieval_run_label = _resolve_retrieval_run_label(
        dataset=dataset,
        embedding_mode=embedding_mode,
        explicit=spec.get("retrieval_run_label"),
        defaults=defaults,
        paper_scoped=paper_scoped,
    )

    retrieval_top_k = int(
        spec.get(
            "retrieval_top_k",
            spec.get("stage1_top_k", defaults.get("retrieval_top_k", DEFAULT_RETRIEVAL_TOP_K)),
        )
    )
    expansion_k_values = _resolve_expansion_k_values(spec, defaults)
    expansion_k_values = validate_expansion_k_values(
        expansion_k_values,
        retrieval_top_k=retrieval_top_k,
    )
    stage1_top_k = max(expansion_k_values)

    stage2_output_dir = spec.get("stage2_output_dir")
    resolved_stage2_output_dir = Path(stage2_output_dir) if stage2_output_dir else None
    stage2_revision_template = spec.get("stage2_model_revision")

    return ResolvedRfgExperiment(
        experiment_id=experiment_id,
        dataset=dataset,
        run_label=str(spec.get("run_label", experiment_id)),
        retrieval_run_label=retrieval_run_label,
        embedding_model=str(
            spec.get("embedding_model", defaults.get("embedding_model", "Qwen/Qwen3-Embedding-4B"))
        ),
        embedding_lora=_resolve_embedding_lora(
            embedding_mode=embedding_mode,
            dataset=dataset,
            explicit=spec.get("embedding_lora"),
            dataset_options=dataset_options,
        ),
        generation_model=str(
            spec.get("generation_model", defaults.get("generation_model", "Qwen/Qwen3-8B"))
        ),
        generation_lora=_resolve_generation_lora(
            generation_mode=generation_mode,
            dataset=dataset,
            explicit=spec.get("generation_lora"),
            dataset_options=dataset_options,
        ),
        embedding_mode=embedding_mode,
        generation_mode=generation_mode,
        stage1_top_k=stage1_top_k,
        retrieval_top_k=retrieval_top_k,
        expansion_k_values=expansion_k_values,
        split=str(spec.get("split", defaults.get("split", "test"))),
        paper_scoped=paper_scoped,
        retrieval_batch_size=int(
            spec.get("retrieval_batch_size", defaults.get("retrieval_batch_size", 128))
        ),
        generation_batch_size=int(
            spec.get("generation_batch_size", defaults.get("generation_batch_size", 8))
        ),
        generation_max_tokens=int(
            spec.get("generation_max_tokens", defaults.get("generation_max_tokens", 2048))
        ),
        max_tokens_per_chunk=int(
            spec.get("max_tokens_per_chunk", defaults.get("max_tokens_per_chunk", 2048))
        ),
        max_prompt_tokens=int(
            spec.get("max_prompt_tokens", defaults.get("max_prompt_tokens", 0))
        ),
        stage2_model_revision=stage2_revision_template
        or str(
            defaults.get(
                "stage2_model_revision_template",
                "rfg-{dataset}-emb-{embedding}-gen-{generation}-k{k}",
            )
        ),
        stage2_output_dir=resolved_stage2_output_dir,
        retrieved_root=str(
            spec.get("retrieved_root", defaults.get("retrieved_root", "retrieved_inmemory"))
        ),
    )


def resolve_experiments(
    raw: dict[str, Any],
    *,
    experiment_ids: list[str] | None = None,
    dataset: str | None = None,
) -> list[ResolvedRfgExperiment]:
    ids = experiment_ids or list_experiment_ids(raw)
    resolved = [resolve_experiment(raw, experiment_id) for experiment_id in ids]
    if dataset is not None:
        resolved = [item for item in resolved if item.dataset == dataset]
    return resolved


def stage1_retrieved_dir(
    *,
    project_root: Path,
    dataset: str,
    retrieval_run_label: str,
    retrieved_root: str = "retrieved_inmemory",
) -> Path:
    return (
        project_root
        / "datasets"
        / retrieved_root
        / dataset
        / retrieval_run_label
    )


def expanded_queries_dir(
    *,
    project_root: Path,
    dataset: str,
    run_label: str,
    expansion_k: int | None = None,
) -> Path:
    base = project_root / "datasets" / "query_expansion" / dataset / run_label
    if expansion_k is None:
        return base
    return base / f"k{expansion_k}"


def stage2_model_revision_for_k(experiment: ResolvedRfgExperiment, expansion_k: int) -> str:
    template = experiment.stage2_model_revision
    return template.format(
        dataset=experiment.dataset,
        embedding=experiment.embedding_mode,
        generation=experiment.generation_mode,
        k=expansion_k,
    )


def stage1_retrieved_docs_path(
    *,
    project_root: Path,
    dataset: str,
    retrieval_run_label: str,
    split: str,
    retrieved_root: str = "retrieved_inmemory",
) -> Path:
    return stage1_retrieved_dir(
        project_root=project_root,
        dataset=dataset,
        retrieval_run_label=retrieval_run_label,
        retrieved_root=retrieved_root,
    ) / split / "retrieved_docs.json"


def expanded_queries_path(
    *,
    project_root: Path,
    dataset: str,
    run_label: str,
    split: str,
    expansion_k: int,
) -> Path:
    return expanded_queries_dir(
        project_root=project_root,
        dataset=dataset,
        run_label=run_label,
        expansion_k=expansion_k,
    ) / split / "expanded_queries.json"
