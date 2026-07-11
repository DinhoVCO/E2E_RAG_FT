from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from tesis_unicamp.finetuning.generative.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_HUB_ORG,
    GENERATIVE_FINETUNING_DATASET_IDS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    QWEN3_LORA_TARGET_MODULES,
)

HUB_REPO_PREFIX = "Qwen3-8b-lora"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
OPTIONAL_RUN_METADATA = "best_model.json"


@dataclass(frozen=True)
class AdapterUploadSpec:
    dataset: str
    adapter_dir: Path
    repo_id: str
    run_dir: Path | None = None


def resolve_hf_token(explicit_token: str | None = None) -> str:
    token = explicit_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "Missing Hugging Face token. Set HF_TOKEN in .env or run `huggingface-cli login`."
        )
    return token


def resolve_hub_username(*, token: str, hub_user: str | None = None) -> str:
    if hub_user:
        return hub_user
    env_user = os.getenv("HF_USERNAME") or os.getenv("HUGGINGFACE_HUB_USERNAME")
    if env_user:
        return env_user
    env_org = os.getenv("HF_ORG") or os.getenv("HUGGINGFACE_HUB_ORG")
    if env_org:
        return env_org
    return DEFAULT_HUB_ORG


def default_hub_repo_id(*, hub_user: str, dataset: str) -> str:
    if dataset not in GENERATIVE_FINETUNING_DATASET_IDS:
        valid = ", ".join(sorted(GENERATIVE_FINETUNING_DATASET_IDS))
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of: {valid}")
    return f"{hub_user}/{HUB_REPO_PREFIX}-{dataset}"


def validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (adapter_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Adapter directory {adapter_dir} is missing required files: {', '.join(missing)}"
        )


def _matches_dataset_dir(dir_name: str, dataset: str) -> bool:
    return dir_name == dataset or dir_name.startswith(f"{dataset}-")


def find_adapter_dir(
    *,
    dataset: str,
    output_root: Path,
    run_dir: Path | None = None,
    adapter_dir: Path | None = None,
) -> tuple[Path, Path | None]:
    if adapter_dir is not None:
        adapter_dir = adapter_dir.resolve()
        validate_adapter_dir(adapter_dir)
        run_dir = adapter_dir.parent if adapter_dir.name == "final" else None
        return adapter_dir, run_dir

    if run_dir is not None:
        run_dir = run_dir.resolve()
        candidate = run_dir / "final"
        validate_adapter_dir(candidate)
        return candidate, run_dir

    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    matches = sorted(
        (
            path
            for path in output_root.iterdir()
            if path.is_dir() and _matches_dataset_dir(path.name, dataset)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No local run found for dataset {dataset!r} under {output_root}. "
            "Pass --run-dir or --adapter-dir explicitly."
        )

    run_dir = matches[0]
    if len(matches) > 1:
        others = ", ".join(path.name for path in matches[1:])
        print(
            f"Warning: multiple runs match {dataset!r}; using newest: {run_dir.name}. "
            f"Other matches: {others}"
        )

    adapter_dir = run_dir / "final"
    validate_adapter_dir(adapter_dir)
    return adapter_dir, run_dir


def build_hub_readme(
    *,
    repo_id: str,
    dataset: str,
    base_model: str = DEFAULT_BASE_MODEL,
    best_model_metadata: dict | None = None,
) -> str:
    hub_dataset = GENERATIVE_FINETUNING_DATASET_IDS[dataset]
    best_metric_line = ""
    if best_model_metadata:
        metric_name = best_model_metadata.get("metric_for_best_model")
        best_metric = best_model_metadata.get("best_metric")
        if metric_name is not None and best_metric is not None:
            best_metric_line = f"\n- **Best dev metric:** `{metric_name}` = {best_metric:.4f}"

    target_modules = ", ".join(f"`{name}`" for name in QWEN3_LORA_TARGET_MODULES)

    return f"""---
library_name: peft
base_model: {base_model}
tags:
- peft
- lora
- text-generation
- question-answering
- rag
license: apache-2.0
language:
- en
datasets:
- {hub_dataset}
---

# {repo_id.split("/", 1)[-1]}

LoRA adapter for [{base_model}](https://huggingface.co/{base_model}) fine-tuned on the **{dataset}** RAG generative dataset ([{hub_dataset}](https://huggingface.co/datasets/{hub_dataset})).{best_metric_line}

## Load with PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

## Load with vLLM (LoRA)

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(
    model="{base_model}",
    enable_lora=True,
    max_lora_rank={LORA_R},
)
outputs = llm.generate(
    prompts,
    lora_request=LoRARequest("{dataset}", 1, "{repo_id}"),
)
```

Use this adapter with `scripts/generation/run_rag_generation.py --lora-path {repo_id}`.

## Training details

- **Base model:** `{base_model}`
- **Fine-tuning dataset:** `{hub_dataset}`
- **Method:** LoRA (`r={LORA_R}`, `lora_alpha={LORA_ALPHA}`, `lora_dropout={LORA_DROPOUT}`)
- **Target modules:** {target_modules}
- **Loss:** SFT with completion-only masking (`assistant_only_loss=True`)
- **Best checkpoint selection:** dev `eval_loss`
"""


def prepare_upload_folder(
    *,
    adapter_dir: Path,
    repo_id: str,
    dataset: str,
    run_dir: Path | None = None,
) -> Path:
    staging_dir = Path(tempfile.mkdtemp(prefix="hf-adapter-upload-"))
    for path in adapter_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(adapter_dir)
        destination = staging_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    best_model_metadata = None
    if run_dir is not None:
        metadata_path = run_dir / OPTIONAL_RUN_METADATA
        if metadata_path.is_file():
            shutil.copy2(metadata_path, staging_dir / OPTIONAL_RUN_METADATA)
            best_model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    readme_path = staging_dir / "README.md"
    readme_path.write_text(
        build_hub_readme(
            repo_id=repo_id,
            dataset=dataset,
            best_model_metadata=best_model_metadata,
        ),
        encoding="utf-8",
    )
    return staging_dir


def upload_adapter_to_hub(
    spec: AdapterUploadSpec,
    *,
    token: str,
    private: bool = False,
    commit_message: str | None = None,
    dry_run: bool = False,
) -> str:
    validate_adapter_dir(spec.adapter_dir)
    commit_message = commit_message or f"Upload LoRA adapter for {spec.dataset}"

    if dry_run:
        print(f"[dry-run] repo_id: {spec.repo_id}")
        print(f"[dry-run] adapter_dir: {spec.adapter_dir}")
        if spec.run_dir is not None:
            print(f"[dry-run] run_dir: {spec.run_dir}")
        for path in sorted(spec.adapter_dir.rglob("*")):
            if path.is_file():
                print(f"[dry-run] file: {path.relative_to(spec.adapter_dir)}")
        return spec.repo_id

    staging_dir = prepare_upload_folder(
        adapter_dir=spec.adapter_dir,
        repo_id=spec.repo_id,
        dataset=spec.dataset,
        run_dir=spec.run_dir,
    )
    try:
        api = HfApi(token=token)
        api.create_repo(
            repo_id=spec.repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=spec.repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return spec.repo_id


def verify_hub_access(*, token: str, repo_id: str) -> str:
    api = HfApi(token=token)
    user = api.whoami()
    username = user.get("name") or user.get("fullname") or "unknown"
    try:
        api.repo_info(repo_id, repo_type="model")
    except HfHubHTTPError as exc:
        if exc.response.status_code == 404:
            return username
        raise ValueError(
            f"Cannot access model repo {repo_id!r}. "
            "Check that HF_TOKEN is valid and has write access. "
            f"Hub error: {exc}"
        ) from exc
    return username
