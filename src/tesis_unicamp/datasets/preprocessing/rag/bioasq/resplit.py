"""Reshuffle and resplit an existing BioASQ RAG Hub dataset."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from tesis_unicamp.datasets.preprocessing.rag.bioasq.builder import _build_hub_datasets
from tesis_unicamp.datasets.preprocessing.rag.bioasq.constants import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_RESPLIT_HUB_README_TEMPLATE,
    DEFAULT_RESPLIT_OUTPUT_DIR,
    DEFAULT_RESPLIT_DEV_RATIO,
    DEFAULT_TEST_RATIO,
)
from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import RAG_SPLITS
from tesis_unicamp.datasets.utils.bioasq_rag import BIOASQ_RAG_DATASET_ID


def _save_json_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def _split_to_dataset(split_data: dict[str, list[dict]]) -> DatasetDict:
    return DatasetDict(
        {
            "queries": Dataset.from_list(split_data["queries"]),
            "qrels": Dataset.from_list(split_data["qrels"]),
            "answers": Dataset.from_list(split_data["answers"]),
        }
    )


def split_shuffled_queries(
    queries: list[dict],
    *,
    test_ratio: float = DEFAULT_TEST_RATIO,
    dev_ratio: float = DEFAULT_RESPLIT_DEV_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle all queries, then assign test, dev and train splits."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")
    if not queries:
        raise ValueError("Cannot resplit an empty query set")

    shuffled = list(queries)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    test_size = max(1, int(n * test_ratio))
    test_queries = shuffled[:test_size]
    remaining = shuffled[test_size:]

    if not remaining:
        return [], [], test_queries

    dev_size = max(1, int(len(remaining) * dev_ratio))
    dev_queries = remaining[:dev_size]
    train_queries = remaining[dev_size:]
    return train_queries, dev_queries, test_queries


def _load_corpus(source_repo_id: str) -> Dataset:
    corpus = load_dataset(source_repo_id, "corpus")
    if isinstance(corpus, DatasetDict):
        return corpus["train"]
    return corpus


def _load_merged_queries(source_repo_id: str) -> list[dict]:
    queries = load_dataset(source_repo_id, "queries")
    merged: list[dict] = []
    seen_ids: set[str] = set()
    for split in RAG_SPLITS:
        for row in queries[split]:
            query_id = row["id"]
            if query_id in seen_ids:
                continue
            seen_ids.add(query_id)
            merged.append(dict(row))
    return merged


def _load_related_records(source_repo_id: str, config_name: str) -> dict[str, list[dict]]:
    dataset = load_dataset(source_repo_id, config_name)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for split in RAG_SPLITS:
        for row in dataset[split]:
            grouped[row["query_id"]].append(dict(row))
    return grouped


def _load_answers_by_query_id(source_repo_id: str) -> dict[str, dict]:
    dataset = load_dataset(source_repo_id, "answers")
    answers: dict[str, dict] = {}
    for split in RAG_SPLITS:
        for row in dataset[split]:
            answers[row["query_id"]] = dict(row)
    return answers


def _build_split_data(
    query_list: list[dict],
    *,
    qrels_by_query_id: dict[str, list[dict]],
    answers_by_query_id: dict[str, dict],
) -> dict[str, list[dict]]:
    queries: list[dict] = []
    qrels: list[dict] = []
    answers: list[dict] = []

    for query in query_list:
        query_id = query["id"]
        queries.append(query)
        qrels.extend(qrels_by_query_id.get(query_id, []))
        answer = answers_by_query_id.get(query_id)
        if answer is None:
            raise KeyError(f"Missing answer for query_id={query_id!r}")
        answers.append(answer)

    return {"queries": queries, "qrels": qrels, "answers": answers}


def _render_resplit_hub_readme(
    template_path: Path,
    *,
    repo_id: str,
    source_repo_id: str,
    dataset_dict: DatasetDict,
    test_ratio: float,
    dev_ratio: float,
    seed: int,
    total_queries: int,
) -> str:
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("{{repo_id}}", repo_id)
        .replace("{{source_repo_id}}", source_repo_id)
        .replace("{{corpus_size}}", str(len(dataset_dict["corpus"])))
        .replace("{{train_queries}}", str(len(dataset_dict["train"]["queries"])))
        .replace("{{dev_queries}}", str(len(dataset_dict["dev"]["queries"])))
        .replace("{{test_queries}}", str(len(dataset_dict["test"]["queries"])))
        .replace("{{total_queries}}", str(total_queries))
        .replace("{{test_ratio}}", str(test_ratio))
        .replace("{{dev_ratio}}", str(dev_ratio))
        .replace("{{seed}}", str(seed))
    )


def _write_resplit_hub_readme(
    output_dir: Path,
    *,
    repo_id: str,
    source_repo_id: str,
    dataset_dict: DatasetDict,
    test_ratio: float,
    dev_ratio: float,
    seed: int,
    total_queries: int,
    template_path: Path = DEFAULT_RESPLIT_HUB_README_TEMPLATE,
) -> Path:
    readme_content = _render_resplit_hub_readme(
        template_path,
        repo_id=repo_id,
        source_repo_id=source_repo_id,
        dataset_dict=dataset_dict,
        test_ratio=test_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
        total_queries=total_queries,
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    return readme_path


def resplit_bioasq_rag_from_hub(
    source_repo_id: str = BIOASQ_RAG_DATASET_ID,
    output_dir: Path = DEFAULT_RESPLIT_OUTPUT_DIR,
    test_ratio: float = DEFAULT_TEST_RATIO,
    dev_ratio: float = DEFAULT_RESPLIT_DEV_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
) -> DatasetDict:
    """Merge Hub splits, shuffle queries, and rebuild train/dev/test."""
    all_queries = _load_merged_queries(source_repo_id)
    qrels_by_query_id = _load_related_records(source_repo_id, "qrels")
    answers_by_query_id = _load_answers_by_query_id(source_repo_id)

    train_queries, dev_queries, test_queries = split_shuffled_queries(
        all_queries,
        test_ratio=test_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
    )

    splits = {
        "train": _build_split_data(
            train_queries,
            qrels_by_query_id=qrels_by_query_id,
            answers_by_query_id=answers_by_query_id,
        ),
        "dev": _build_split_data(
            dev_queries,
            qrels_by_query_id=qrels_by_query_id,
            answers_by_query_id=answers_by_query_id,
        ),
        "test": _build_split_data(
            test_queries,
            qrels_by_query_id=qrels_by_query_id,
            answers_by_query_id=answers_by_query_id,
        ),
    }

    corpus = _load_corpus(source_repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json_records(output_dir / "corpus.json", [dict(row) for row in corpus])
    for split_name, split_data in splits.items():
        split_dir = output_dir / split_name
        _save_json_records(split_dir / "queries.json", split_data["queries"])
        _save_json_records(split_dir / "qrels.json", split_data["qrels"])
        _save_json_records(split_dir / "answers.json", split_data["answers"])

    dataset_dict = DatasetDict(
        {
            "corpus": corpus,
            "train": _split_to_dataset(splits["train"]),
            "dev": _split_to_dataset(splits["dev"]),
            "test": _split_to_dataset(splits["test"]),
        }
    )

    hub_datasets = _build_hub_datasets(dataset_dict)
    hf_dir = output_dir / "hf_dataset"
    for name, dataset in hub_datasets.items():
        dataset.save_to_disk(str(hf_dir / name))

    _write_resplit_hub_readme(
        output_dir,
        repo_id=source_repo_id,
        source_repo_id=source_repo_id,
        dataset_dict=dataset_dict,
        test_ratio=test_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
        total_queries=len(all_queries),
    )
    return dataset_dict


def push_resplit_bioasq_rag_to_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    *,
    source_repo_id: str = BIOASQ_RAG_DATASET_ID,
    output_dir: Path = DEFAULT_RESPLIT_OUTPUT_DIR,
    token: str | None = None,
    private: bool = False,
    readme_path: Path | None = None,
    test_ratio: float = DEFAULT_TEST_RATIO,
    dev_ratio: float = DEFAULT_RESPLIT_DEV_RATIO,
    seed: int = DEFAULT_RANDOM_SEED,
    total_queries: int | None = None,
) -> None:
    """Upload a reshuffled BioASQ RAG dataset to the Hugging Face Hub."""
    if total_queries is None:
        total_queries = sum(
            len(dataset_dict[split]["queries"]) for split in RAG_SPLITS
        )

    hub_datasets = _build_hub_datasets(dataset_dict)
    for config_name, dataset in hub_datasets.items():
        dataset.push_to_hub(
            repo_id,
            config_name=config_name,
            token=token,
            private=private,
        )

    readme_file = _write_resplit_hub_readme(
        output_dir,
        repo_id=repo_id,
        source_repo_id=source_repo_id,
        dataset_dict=dataset_dict,
        test_ratio=test_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
        total_queries=total_queries,
        template_path=readme_path or DEFAULT_RESPLIT_HUB_README_TEMPLATE,
    )

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=readme_file.read_text(encoding="utf-8").encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add resplit dataset README",
    )
