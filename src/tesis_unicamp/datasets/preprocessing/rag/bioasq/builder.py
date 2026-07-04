import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from tesis_unicamp.datasets.preprocessing.rag.bioasq.constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DEV_RATIO,
    DEFAULT_GOLDEN_DIR,
    DEFAULT_HUB_README,
    DEFAULT_HUB_README_TEMPLATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAINING_PATH,
)
from tesis_unicamp.datasets.preprocessing.rag.bioasq.loader import (
    load_golden_questions,
    load_training_questions,
)
from tesis_unicamp.datasets.preprocessing.rag.bioasq.models import SplitData
from tesis_unicamp.datasets.preprocessing.rag.bioasq.processor import (
    build_corpus,
    collect_pmids_from_questions,
    process_bioasq_questions,
    split_train_dev,
)
from tesis_unicamp.datasets.preprocessing.rag.bioasq.pubmed import fetch_pubmed_abstracts


def _split_to_dataset(split_data: SplitData) -> DatasetDict:
    return DatasetDict(
        {
            "queries": Dataset.from_list(split_data["queries"]),
            "qrels": Dataset.from_list(split_data["qrels"]),
            "answers": Dataset.from_list(split_data["answers"]),
        }
    )


def _save_json_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def build_bioasq_rag_dataset(
    training_path: Path = DEFAULT_TRAINING_PATH,
    golden_dir: Path = DEFAULT_GOLDEN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
    email: str | None = None,
    force_refresh_pubmed: bool = False,
) -> DatasetDict:
    """Build the BioASQ RAG dataset with train, dev and test splits."""
    training_questions = load_training_questions(training_path)
    test_questions = load_golden_questions(golden_dir)
    train_questions, dev_questions = split_train_dev(
        training_questions,
        dev_ratio=dev_ratio,
        seed=seed,
    )

    all_questions = train_questions + dev_questions + test_questions
    pmids = collect_pmids_from_questions(all_questions)
    pubmed_records = fetch_pubmed_abstracts(
        pmids,
        cache_dir=cache_dir,
        email=email,
        force_refresh=force_refresh_pubmed,
    )
    corpus = build_corpus(pmids, pubmed_records)

    splits = {
        "train": process_bioasq_questions(train_questions),
        "dev": process_bioasq_questions(dev_questions),
        "test": process_bioasq_questions(test_questions),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json_records(output_dir / "corpus.json", corpus)
    for split_name, split_data in splits.items():
        split_dir = output_dir / split_name
        _save_json_records(split_dir / "queries.json", split_data["queries"])
        _save_json_records(split_dir / "qrels.json", split_data["qrels"])
        _save_json_records(split_dir / "answers.json", split_data["answers"])

    dataset_dict = DatasetDict(
        {
            "corpus": Dataset.from_list(corpus),
            "train": _split_to_dataset(splits["train"]),
            "dev": _split_to_dataset(splits["dev"]),
            "test": _split_to_dataset(splits["test"]),
        }
    )
    hub_datasets = _build_hub_datasets(dataset_dict)
    hf_dir = output_dir / "hf_dataset"
    for name, dataset in hub_datasets.items():
        dataset.save_to_disk(str(hf_dir / name))
    _write_hub_readme(
        output_dir=output_dir,
        dataset_dict=dataset_dict,
        dev_ratio=dev_ratio,
        seed=seed,
    )
    return dataset_dict


def _write_hub_readme(
    output_dir: Path,
    dataset_dict: DatasetDict,
    dev_ratio: float,
    seed: int,
    repo_id: str = "username/bioasq-rag-13b",
) -> Path:
    template_path = DEFAULT_HUB_README_TEMPLATE

    readme_content = _render_hub_readme(
        template_path,
        repo_id=repo_id,
        dataset_dict=dataset_dict,
        dev_ratio=dev_ratio,
        seed=seed,
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    return readme_path


def _render_hub_readme(
    template_path: Path,
    repo_id: str,
    dataset_dict: DatasetDict,
    dev_ratio: float,
    seed: int,
) -> str:
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("{{repo_id}}", repo_id)
        .replace("{{corpus_size}}", str(len(dataset_dict["corpus"])))
        .replace("{{train_queries}}", str(len(dataset_dict["train"]["queries"])))
        .replace("{{dev_queries}}", str(len(dataset_dict["dev"]["queries"])))
        .replace("{{test_queries}}", str(len(dataset_dict["test"]["queries"])))
        .replace("{{dev_ratio}}", str(dev_ratio))
        .replace("{{seed}}", str(seed))
    )


def _build_hub_datasets(
    dataset_dict: DatasetDict,
) -> dict[str, Dataset | DatasetDict]:
    """Organize data into 4 Hub subsets: corpus, queries, qrels, answers."""
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
    }


def push_bioasq_rag_to_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    readme_path: Path | None = None,
    dev_ratio: float = DEFAULT_DEV_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Upload the BioASQ RAG dataset to the Hugging Face Hub."""
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
        dev_ratio=dev_ratio,
        seed=seed,
        repo_id=repo_id,
    )
    if readme_path and readme_path != readme_file:
        readme_content = readme_path.read_text(encoding="utf-8")
        if "{{" in readme_content:
            readme_content = _render_hub_readme(
                readme_path,
                repo_id=repo_id,
                dataset_dict=dataset_dict,
                dev_ratio=dev_ratio,
                seed=seed,
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
