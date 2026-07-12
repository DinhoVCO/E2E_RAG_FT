from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tesis_unicamp.generation.rag.datasets import RAG_GENERATION_DATASET_CONFIGS
from tesis_unicamp.generation.rag.runner import PromptMode

_EMBEDDING_MODES = frozenset({"base", "lora", "none"})
_GENERATION_MODES = frozenset({"base", "lora", "qa"})
_PROMPT_MODES = frozenset({"inference", "qa", "rag-finetune"})

_EXPERIMENT_KEYS = frozenset(
    {
        "dataset",
        "embedding",
        "generation",
        "use_retrieval",
        "prompt_mode",
        "run_label",
        "retrieval_run_label",
        "embedding_model",
        "embedding_lora",
        "generation_model",
        "generation_lora",
        "top_k",
        "retrieval_top_k",
        "split",
        "paper_scoped",
        "retrieval_batch_size",
        "generation_batch_size",
    }
)


@dataclass(frozen=True)
class ResolvedExperiment:
    experiment_id: str
    dataset: str
    run_label: str
    use_retrieval: bool
    prompt_mode: PromptMode
    generation_mode: str
    retrieval_run_label: str
    embedding_model: str
    embedding_lora: str | None
    generation_model: str
    generation_lora: str | None
    top_k: int
    retrieval_top_k: int
    split: str
    paper_scoped: bool
    retrieval_batch_size: int
    generation_batch_size: int


def default_experiments_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "generation"
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
    defaults_key = "generation_lora_qa" if generation_mode == "qa" else "generation_lora"
    if defaults_key in dataset_options:
        value = dataset_options[defaults_key]
        return str(value) if value else None
    raise ValueError(
        f"generation mode {generation_mode!r} requires "
        f"'datasets.{dataset}.{defaults_key}' or an explicit generation_lora path."
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
    if embedding_mode in {"base", "none"}:
        return None
    if "embedding_lora" in dataset_options:
        value = dataset_options["embedding_lora"]
        return str(value) if value else None
    raise ValueError(
        f"embedding mode 'lora' requires "
        f"'datasets.{dataset}.embedding_lora' or an explicit path in the experiment."
    )


def _resolve_prompt_mode(
    *,
    spec: dict[str, Any],
    use_retrieval: bool,
    generation_mode: str,
) -> PromptMode:
    explicit = spec.get("prompt_mode")
    if explicit is not None:
        if explicit not in _PROMPT_MODES:
            valid = ", ".join(sorted(_PROMPT_MODES))
            raise ValueError(f"prompt_mode must be one of: {valid}")
        return explicit

    if not use_retrieval:
        if generation_mode == "qa":
            return "qa"
        if generation_mode == "base":
            return "inference"
        return "rag-finetune"
    if generation_mode == "qa":
        return "rag-finetune"
    return "inference"


def resolve_experiment(raw: dict[str, Any], experiment_id: str) -> ResolvedExperiment:
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

    use_retrieval = bool(spec.get("use_retrieval", True))
    embedding_mode = spec.get("embedding", "none" if not use_retrieval else "base")
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
    if not use_retrieval and embedding_mode not in {"none", "base", "lora"}:
        raise ValueError(
            f"Experiment {experiment_id!r}: use_retrieval=false ignores embedding mode."
        )
    if use_retrieval and embedding_mode == "none":
        raise ValueError(
            f"Experiment {experiment_id!r}: embedding cannot be 'none' when use_retrieval=true."
        )

    dataset_options = _dataset_options(raw, dataset)
    paper_scoped = _resolve_paper_scoped(
        dataset=dataset,
        explicit=spec.get("paper_scoped"),
        dataset_options=dataset_options,
    )

    top_k = int(spec.get("top_k", defaults.get("top_k", 5)))
    retrieval_top_k = int(spec.get("retrieval_top_k", defaults.get("retrieval_top_k", 10)))
    if use_retrieval and top_k > retrieval_top_k:
        raise ValueError(
            f"Experiment {experiment_id!r}: top_k ({top_k}) cannot exceed "
            f"retrieval_top_k ({retrieval_top_k})."
        )

    retrieval_run_label = _resolve_retrieval_run_label(
        dataset=dataset,
        embedding_mode=embedding_mode if use_retrieval else "base",
        explicit=spec.get("retrieval_run_label"),
        defaults=defaults,
        paper_scoped=paper_scoped,
    )
    prompt_mode = _resolve_prompt_mode(
        spec=spec,
        use_retrieval=use_retrieval,
        generation_mode=generation_mode,
    )

    return ResolvedExperiment(
        experiment_id=experiment_id,
        dataset=dataset,
        run_label=str(spec.get("run_label", experiment_id)),
        use_retrieval=use_retrieval,
        prompt_mode=prompt_mode,
        generation_mode=generation_mode,
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
        top_k=top_k,
        retrieval_top_k=retrieval_top_k,
        split=str(spec.get("split", defaults.get("split", "test"))),
        paper_scoped=paper_scoped,
        retrieval_batch_size=int(
            spec.get("retrieval_batch_size", defaults.get("retrieval_batch_size", 128))
        ),
        generation_batch_size=int(
            spec.get("generation_batch_size", defaults.get("generation_batch_size", 8))
        ),
    )


def resolve_experiments(
    raw: dict[str, Any],
    *,
    experiment_ids: list[str] | None = None,
    dataset: str | None = None,
) -> list[ResolvedExperiment]:
    ids = experiment_ids or list_experiment_ids(raw)
    resolved = [resolve_experiment(raw, experiment_id) for experiment_id in ids]
    if dataset is not None:
        resolved = [item for item in resolved if item.dataset == dataset]
    return resolved


def retrieved_docs_path(
    *,
    project_root: Path,
    dataset: str,
    retrieval_run_label: str,
    split: str,
) -> Path:
    return (
        project_root
        / "datasets"
        / "retrieved_inmemory"
        / dataset
        / retrieval_run_label
        / split
        / "retrieved_docs.json"
    )
