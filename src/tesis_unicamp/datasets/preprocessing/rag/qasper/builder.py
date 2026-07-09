import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from tesis_unicamp.datasets.preprocessing.rag.qasper.constants import (
    DEFAULT_HF_DATASET_ID,
    DEFAULT_HUB_README,
    DEFAULT_HUB_README_TEMPLATE,
    DEFAULT_OUTPUT_DIR,
)
from tesis_unicamp.datasets.preprocessing.rag.qasper.loader import load_qasper_splits
from tesis_unicamp.datasets.preprocessing.rag.qasper.models import SplitData
from tesis_unicamp.datasets.preprocessing.rag.qasper.processor import process_qasper_splits


def _split_to_dataset(split_data: SplitData) -> DatasetDict:
    return DatasetDict(
        {
            "queries": Dataset.from_list(split_data["queries"]),
            "qrels": Dataset.from_list(split_data["qrels"]),
            "answers": Dataset.from_list(split_data["answers"]),
            "top_ranked": Dataset.from_list(split_data["top_ranked"]),
        }
    )


def _save_json_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def build_qasper_rag_dataset(
    hf_dataset_id: str = DEFAULT_HF_DATASET_ID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    exact_text_evidence_only: bool = False,
    readme_template: Path = DEFAULT_HUB_README_TEMPLATE,
) -> DatasetDict:
    """Build the QASPER RAG dataset with train, dev and test splits."""
    splits = load_qasper_splits(hf_dataset_id=hf_dataset_id)
    corpus, processed_splits = process_qasper_splits(
        splits,
        exact_text_evidence_only=exact_text_evidence_only,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json_records(output_dir / "corpus.json", corpus)
    for split_name, split_data in processed_splits.items():
        split_dir = output_dir / split_name
        _save_json_records(split_dir / "queries.json", split_data["queries"])
        _save_json_records(split_dir / "qrels.json", split_data["qrels"])
        _save_json_records(split_dir / "answers.json", split_data["answers"])
        _save_json_records(split_dir / "top_ranked.json", split_data["top_ranked"])

    dataset_dict = DatasetDict(
        {
            "corpus": Dataset.from_list(corpus),
            "train": _split_to_dataset(processed_splits["train"]),
            "dev": _split_to_dataset(processed_splits["dev"]),
            "test": _split_to_dataset(processed_splits["test"]),
        }
    )
    hub_datasets = _build_hub_datasets(dataset_dict)
    hf_dir = output_dir / "hf_dataset"
    for name, dataset in hub_datasets.items():
        dataset.save_to_disk(str(hf_dir / name))
    _write_hub_readme(
        output_dir=output_dir,
        dataset_dict=dataset_dict,
        hf_dataset_id=hf_dataset_id,
        readme_template=readme_template,
    )
    return dataset_dict


def _write_hub_readme(
    output_dir: Path,
    dataset_dict: DatasetDict,
    hf_dataset_id: str,
    repo_id: str = "username/qasper-rag",
    readme_template: Path = DEFAULT_HUB_README_TEMPLATE,
) -> Path:
    readme_content = _render_hub_readme(
        readme_template,
        repo_id=repo_id,
        dataset_dict=dataset_dict,
        hf_dataset_id=hf_dataset_id,
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    return readme_path


def _render_hub_readme(
    template_path: Path,
    repo_id: str,
    dataset_dict: DatasetDict,
    hf_dataset_id: str,
) -> str:
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("{{repo_id}}", repo_id)
        .replace("{{hf_dataset_id}}", hf_dataset_id)
        .replace("{{corpus_size}}", str(len(dataset_dict["corpus"])))
        .replace("{{train_queries}}", str(len(dataset_dict["train"]["queries"])))
        .replace("{{dev_queries}}", str(len(dataset_dict["dev"]["queries"])))
        .replace("{{test_queries}}", str(len(dataset_dict["test"]["queries"])))
    )


def _top_ranked_to_hub_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "query-id": record["query_id"],
                "corpus-ids": record["corpus_ids"],
            }
            for record in records
        ]
    )


def _build_hub_datasets(
    dataset_dict: DatasetDict,
) -> dict[str, Dataset | DatasetDict]:
    """Organize data into Hub subsets: corpus, queries, qrels, answers, top_ranked."""
    return {
        "corpus": dataset_dict["corpus"],
        "queries": DatasetDict(
            {
                "train": dataset_dict["train"]["queries"],
                "dev": dataset_dict["dev"]["queries"],
                "test": dataset_dict["test"]["queries"],
            }
        ),
        "qrels": DatasetDict(
            {
                "train": dataset_dict["train"]["qrels"],
                "dev": dataset_dict["dev"]["qrels"],
                "test": dataset_dict["test"]["qrels"],
            }
        ),
        "answers": DatasetDict(
            {
                "train": dataset_dict["train"]["answers"],
                "dev": dataset_dict["dev"]["answers"],
                "test": dataset_dict["test"]["answers"],
            }
        ),
        "top_ranked": DatasetDict(
            {
                "train": _top_ranked_to_hub_dataset(
                    dataset_dict["train"]["top_ranked"]
                ),
                "dev": _top_ranked_to_hub_dataset(dataset_dict["dev"]["top_ranked"]),
                "test": _top_ranked_to_hub_dataset(
                    dataset_dict["test"]["top_ranked"]
                ),
            }
        ),
    }


def push_qasper_rag_to_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    readme_path: Path | None = None,
    hf_dataset_id: str = DEFAULT_HF_DATASET_ID,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    readme_template: Path = DEFAULT_HUB_README_TEMPLATE,
) -> None:
    """Upload the QASPER RAG dataset to the Hugging Face Hub."""
    hub_datasets = _build_hub_datasets(dataset_dict)
    for config_name, dataset in hub_datasets.items():
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            token=token,
            private=private,
        )

    readme_file = _write_hub_readme(
        output_dir=output_dir,
        dataset_dict=dataset_dict,
        hf_dataset_id=hf_dataset_id,
        repo_id=repo_id,
        readme_template=readme_template,
    )
    if readme_path and readme_path != readme_file:
        readme_content = readme_path.read_text(encoding="utf-8")
        if "{{" in readme_content:
            readme_content = _render_hub_readme(
                readme_path,
                repo_id=repo_id,
                dataset_dict=dataset_dict,
                hf_dataset_id=hf_dataset_id,
            )
    else:
        readme_content = readme_file.read_text(encoding="utf-8")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset README",
    )
