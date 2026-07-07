from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tesis_unicamp.finetuning.embeddings.config import EMBEDDING_FINETUNING_DATASET_IDS

_CONFIG_KEYS = frozenset(
    {
        "dataset",
        "model",
        "output_dir",
        "epochs",
        "batch_size",
        "learning_rate",
        "warmup_ratio",
        "eval_steps",
        "save_steps",
        "logging_steps",
        "save_total_limit",
        "eval_batch_size",
        "mini_batch_size",
        "train_split",
        "eval_split",
        "wandb_project",
        "run_name",
        "fp16",
        "bf16",
    }
)


def default_configs_dir() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "finetuning"
        / "embeddings"
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
    if dataset is not None and dataset not in EMBEDDING_FINETUNING_DATASET_IDS:
        valid = ", ".join(sorted(EMBEDDING_FINETUNING_DATASET_IDS))
        raise ValueError(f"Unknown dataset {dataset!r} in {path}. Expected one of: {valid}")

    if "output_dir" in defaults and defaults["output_dir"] is not None:
        defaults["output_dir"] = Path(defaults["output_dir"])

    if "bf16" in defaults:
        defaults["no_bf16"] = not defaults.pop("bf16")

    return defaults
