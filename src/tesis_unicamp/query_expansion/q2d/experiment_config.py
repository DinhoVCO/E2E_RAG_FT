from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tesis_unicamp.generation.rag.datasets import RAG_GENERATION_DATASET_CONFIGS
from tesis_unicamp.query_expansion.q2d.few_shot import DEFAULT_NUM_FEW_SHOT, DEFAULT_MAX_PASSAGE_TOKENS
from tesis_unicamp.query_expansion.q2d.generation import (
    DEFAULT_Q2D_MAX_TOKENS,
    DEFAULT_Q2D_TEMPERATURE,
)

_EXPERIMENT_KEYS = frozenset(
    {
        "dataset",
        "run_label",
        "generation_model",
        "generation_lora",
        "embedding_model",
        "embedding_lora",
        "num_few_shot",
        "few_shot_split",
        "max_few_shot_passage_tokens",
        "seed",
        "per_query_sampling",
        "split",
        "paper_scoped",
        "generation_batch_size",
        "generation_max_tokens",
        "temperature",
        "top_p",
        "retrieval_batch_size",
        "model_revision",
        "output_dir",
    }
)


@dataclass(frozen=True)
class ResolvedQ2dExperiment:
    experiment_id: str
    dataset: str
    run_label: str
    generation_model: str
    generation_lora: str | None
    embedding_model: str
    embedding_lora: str | None
    num_few_shot: int
    few_shot_split: str
    max_few_shot_passage_tokens: int
    seed: int
    per_query_sampling: bool
    split: str
    paper_scoped: bool
    generation_batch_size: int
    generation_max_tokens: int
    temperature: float
    top_p: float
    retrieval_batch_size: int
    model_revision: str
    output_dir: Path | None


def default_experiments_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "query_expansion"
        / "q2d"
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
    return sorted(raw["experiments"])


def _defaults(raw: dict[str, Any]) -> dict[str, Any]:
    defaults = raw.get("defaults")
    return defaults if isinstance(defaults, dict) else {}


def _dataset_options(raw: dict[str, Any], dataset: str) -> dict[str, Any]:
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        return {}
    options = datasets.get(dataset)
    return options if isinstance(options, dict) else {}


def resolve_experiment(raw: dict[str, Any], experiment_id: str) -> ResolvedQ2dExperiment:
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
    if not dataset or dataset not in RAG_GENERATION_DATASET_CONFIGS:
        valid = ", ".join(sorted(RAG_GENERATION_DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset {dataset!r}. Expected: {valid}")

    dataset_options = _dataset_options(raw, dataset)
    paper_scoped = spec.get("paper_scoped", dataset_options.get("paper_scoped", dataset == "qasper"))

    model_revision = spec.get("model_revision")
    if model_revision is None:
        model_revision = str(
            defaults.get("model_revision_template", "q2d-{dataset}").format(dataset=dataset)
        )

    output_dir = spec.get("output_dir")
    return ResolvedQ2dExperiment(
        experiment_id=experiment_id,
        dataset=dataset,
        run_label=str(spec.get("run_label", experiment_id)),
        generation_model=str(
            spec.get("generation_model", defaults.get("generation_model", "Qwen/Qwen3-8B"))
        ),
        generation_lora=spec.get("generation_lora") or dataset_options.get("generation_lora"),
        embedding_model=str(
            spec.get("embedding_model", defaults.get("embedding_model", "Qwen/Qwen3-Embedding-4B"))
        ),
        embedding_lora=spec.get("embedding_lora") or dataset_options.get("embedding_lora"),
        num_few_shot=int(spec.get("num_few_shot", defaults.get("num_few_shot", DEFAULT_NUM_FEW_SHOT))),
        few_shot_split=str(spec.get("few_shot_split", defaults.get("few_shot_split", "train"))),
        max_few_shot_passage_tokens=int(
            spec.get(
                "max_few_shot_passage_tokens",
                defaults.get("max_few_shot_passage_tokens", DEFAULT_MAX_PASSAGE_TOKENS),
            )
        ),
        seed=int(spec.get("seed", defaults.get("seed", 42))),
        per_query_sampling=bool(spec.get("per_query_sampling", defaults.get("per_query_sampling", True))),
        split=str(spec.get("split", defaults.get("split", "test"))),
        paper_scoped=bool(paper_scoped),
        generation_batch_size=int(
            spec.get("generation_batch_size", defaults.get("generation_batch_size", 8))
        ),
        generation_max_tokens=int(
            spec.get(
                "generation_max_tokens",
                defaults.get("generation_max_tokens", DEFAULT_Q2D_MAX_TOKENS),
            )
        ),
        temperature=float(spec.get("temperature", defaults.get("temperature", DEFAULT_Q2D_TEMPERATURE))),
        top_p=float(spec.get("top_p", defaults.get("top_p", 1.0))),
        retrieval_batch_size=int(
            spec.get("retrieval_batch_size", defaults.get("retrieval_batch_size", 128))
        ),
        model_revision=str(model_revision),
        output_dir=Path(output_dir) if output_dir else None,
    )


def resolve_experiments(
    raw: dict[str, Any],
    *,
    experiment_ids: list[str] | None = None,
    dataset: str | None = None,
) -> list[ResolvedQ2dExperiment]:
    ids = experiment_ids or list_experiment_ids(raw)
    resolved = [resolve_experiment(raw, experiment_id) for experiment_id in ids]
    if dataset is not None:
        resolved = [item for item in resolved if item.dataset == dataset]
    return resolved


def q2d_output_dir(*, project_root: Path, dataset: str, run_label: str) -> Path:
    return project_root / "datasets" / "query_expansion" / "q2d" / dataset / run_label


def q2d_expansions_path(
    *,
    project_root: Path,
    dataset: str,
    run_label: str,
    split: str,
) -> Path:
    return q2d_output_dir(project_root=project_root, dataset=dataset, run_label=run_label) / split / "q2d_expansions.json"
