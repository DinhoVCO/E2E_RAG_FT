import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from tesis_unicamp.datasets.preprocessing.rag.bioasq.constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DEV_RATIO,
    DEFAULT_GOLDEN_DIR,
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
    dataset_dict.save_to_disk(str(output_dir / "hf_dataset"))
    return dataset_dict


def push_bioasq_rag_to_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> None:
    """Upload the BioASQ RAG dataset to the Hugging Face Hub."""
    for config_name, dataset in dataset_dict.items():
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            token=token,
            private=private,
        )
