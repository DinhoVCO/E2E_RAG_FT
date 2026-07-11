from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tesis_unicamp.finetuning.generative.config import GENERATIVE_FINETUNING_DATASET_IDS

_CONFIG_KEYS = frozenset(
    {
        "dataset",
        "model",
        "output_dir",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_ratio",
        "eval_steps",
        "save_steps",
        "logging_steps",
        "save_total_limit",
        "max_seq_length",
        "train_split",
        "eval_split",
        "dataset_seed",
        "wandb_project",
        "run_name",
        "fp16",
        "bf16",
        "log_file",
        "load_best_model",
        "metric_for_best_model",
        "greater_is_better",
        "early_stopping",
        "early_stopping_patience",
    }
)


def default_configs_dir() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "finetuning"
        / "generative"
        / "configs"
    )


def default_config_path(dataset: str, *, configs_dir: Path | None = None) -> Path:
    root = configs_dir or default_configs_dir()
    return root / f"{dataset}.yaml"


def resolve_config_path(
    *,
    dataset: str | None,
    config: Path | None,
    configs_dir: Path | None = None,
) -> Path | None:
    if config is not None:
        return config
    if dataset is None:
        return None
    candidate = default_config_path(dataset, configs_dir=configs_dir)
    return candidate if candidate.is_file() else None


def load_finetuning_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")

    unknown = set(raw) - _CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {', '.join(sorted(unknown))}"
        )

    defaults = {key: value for key, value in raw.items() if key in _CONFIG_KEYS}

    dataset = defaults.get("dataset")
    if dataset is not None and dataset not in GENERATIVE_FINETUNING_DATASET_IDS:
        valid = ", ".join(sorted(GENERATIVE_FINETUNING_DATASET_IDS))
        raise ValueError(f"Unknown dataset {dataset!r} in {path}. Expected one of: {valid}")

    if "output_dir" in defaults and defaults["output_dir"] is not None:
        defaults["output_dir"] = Path(defaults["output_dir"])

    if "log_file" in defaults and defaults["log_file"] is not None:
        defaults["log_file"] = Path(defaults["log_file"])

    if "bf16" in defaults:
        defaults["no_bf16"] = not defaults.pop("bf16")

    if "load_best_model" in defaults:
        defaults["no_load_best_model"] = not defaults.pop("load_best_model")

    if "early_stopping" in defaults:
        defaults["no_early_stopping"] = not defaults.pop("early_stopping")

    return defaults
