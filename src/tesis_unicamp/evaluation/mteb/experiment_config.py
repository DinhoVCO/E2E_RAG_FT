from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tesis_unicamp.evaluation.mteb.tasks import RAG_RETRIEVAL_TASK_CONFIGS

_EMBEDDING_MODES = frozenset({"base", "lora"})
_BACKENDS = frozenset({"offline", "online", "sentence-transformers"})
_OVERWRITE_STRATEGIES = frozenset({"always", "never", "only-missing", "only-cache"})

_EXPERIMENT_KEYS = frozenset(
    {
        "dataset",
        "embedding",
        "model",
        "lora_path",
        "model_revision",
        "backend",
        "batch_size",
        "splits",
        "paper_scoped",
        "max_lora_rank",
        "overwrite",
        "output_dir",
    }
)


@dataclass(frozen=True)
class ResolvedMtebExperiment:
    experiment_id: str
    dataset: str
    embedding_mode: str
    model: str
    lora_path: str | None
    model_revision: str
    backend: str
    batch_size: int
    splits: tuple[str, ...]
    paper_scoped: bool | None
    max_lora_rank: int
    overwrite: str
    output_dir: Path | None


def default_experiments_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "evaluation"
        / "mteb"
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
) -> bool | None:
    if explicit is not None:
        return explicit
    if "paper_scoped" in dataset_options:
        return bool(dataset_options["paper_scoped"])
    return None


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
        f"'datasets.{dataset}.embedding_lora' or an explicit lora_path."
    )


def _resolve_model_revision(
    *,
    dataset: str,
    embedding_mode: str,
    explicit: str | None,
    defaults: dict[str, Any],
    dataset_options: dict[str, Any],
) -> str:
    if explicit is not None:
        return explicit
    if embedding_mode == "base":
        template = str(
            defaults.get(
                "base_model_revision_template",
                "vllm-offline-{dataset}-b128",
            )
        )
        return template.format(dataset=dataset)

    template = str(
        defaults.get(
            "lora_model_revision_template",
            "vllm-lora-{dataset}-b128-e{epochs}",
        )
    )
    epochs = int(dataset_options.get("lora_epochs", 10))
    return template.format(dataset=dataset, epochs=epochs)


def _resolve_splits(spec: dict[str, Any], defaults: dict[str, Any]) -> tuple[str, ...]:
    splits = spec.get("splits", defaults.get("splits", ["test"]))
    if not isinstance(splits, list) or not splits:
        raise ValueError("'splits' must be a non-empty list.")
    return tuple(str(split) for split in splits)


def resolve_experiment(raw: dict[str, Any], experiment_id: str) -> ResolvedMtebExperiment:
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
    if dataset not in RAG_RETRIEVAL_TASK_CONFIGS:
        valid = ", ".join(sorted(RAG_RETRIEVAL_TASK_CONFIGS))
        raise ValueError(f"Unknown dataset {dataset!r} in {experiment_id!r}. Expected: {valid}")

    embedding_mode = spec.get("embedding", "base")
    if embedding_mode not in _EMBEDDING_MODES:
        raise ValueError(
            f"Experiment {experiment_id!r} must define embedding as one of: "
            f"{', '.join(sorted(_EMBEDDING_MODES))}"
        )

    backend = str(spec.get("backend", defaults.get("backend", "offline")))
    if backend not in _BACKENDS:
        valid = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"backend must be one of: {valid}")

    overwrite = str(spec.get("overwrite", defaults.get("overwrite", "always")))
    if overwrite not in _OVERWRITE_STRATEGIES:
        valid = ", ".join(sorted(_OVERWRITE_STRATEGIES))
        raise ValueError(f"overwrite must be one of: {valid}")

    dataset_options = _dataset_options(raw, dataset)
    lora_path = _resolve_embedding_lora(
        embedding_mode=embedding_mode,
        dataset=dataset,
        explicit=spec.get("lora_path"),
        dataset_options=dataset_options,
    )
    if embedding_mode == "lora" and not lora_path:
        raise ValueError(f"Experiment {experiment_id!r}: embedding=lora requires a lora_path.")
    if embedding_mode == "base" and lora_path:
        raise ValueError(f"Experiment {experiment_id!r}: embedding=base cannot use lora_path.")

    output_dir = spec.get("output_dir")
    resolved_output_dir = Path(output_dir) if output_dir is not None else None

    return ResolvedMtebExperiment(
        experiment_id=experiment_id,
        dataset=dataset,
        embedding_mode=embedding_mode,
        model=str(spec.get("model", defaults.get("embedding_model", "Qwen/Qwen3-Embedding-4B"))),
        lora_path=lora_path,
        model_revision=_resolve_model_revision(
            dataset=dataset,
            embedding_mode=embedding_mode,
            explicit=spec.get("model_revision"),
            defaults=defaults,
            dataset_options=dataset_options,
        ),
        backend=backend,
        batch_size=int(spec.get("batch_size", defaults.get("batch_size", 128))),
        splits=_resolve_splits(spec, defaults),
        paper_scoped=_resolve_paper_scoped(
            dataset=dataset,
            explicit=spec.get("paper_scoped"),
            dataset_options=dataset_options,
        ),
        max_lora_rank=int(spec.get("max_lora_rank", defaults.get("max_lora_rank", 16))),
        overwrite=overwrite,
        output_dir=resolved_output_dir,
    )


def resolve_experiments(
    raw: dict[str, Any],
    *,
    experiment_ids: list[str] | None = None,
    dataset: str | None = None,
) -> list[ResolvedMtebExperiment]:
    ids = experiment_ids or list_experiment_ids(raw)
    resolved = [resolve_experiment(raw, experiment_id) for experiment_id in ids]
    if dataset is not None:
        resolved = [item for item in resolved if item.dataset == dataset]
    return resolved
