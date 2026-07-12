from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict

from tesis_unicamp.datasets.preprocessing.rag.retrieval.schemas import RetrievedDocRecord

RAG_SPLITS = ("train", "dev", "test")


def save_retrieved_docs(
    output_dir: Path,
    split: str,
    records: list[RetrievedDocRecord],
) -> Path:
    """Save retrieved_docs rows for one split as JSON."""
    path = output_dir / split / "retrieved_docs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    return path


def save_retrieved_docs_all_splits(
    output_dir: Path,
    splits: dict[str, list[RetrievedDocRecord]],
) -> None:
    for split_name, records in splits.items():
        save_retrieved_docs(output_dir, split_name, records)


def save_retrieved_docs_bundle(
    output_dir: Path,
    splits: dict[str, list[RetrievedDocRecord]],
) -> Path:
    """Save JSON per split and export only the splits present in ``splits``."""
    save_retrieved_docs_all_splits(output_dir, splits)
    dataset_dict = build_retrieved_docs_dataset_dict(
        output_dir,
        splits=tuple(splits.keys()),
    )
    save_retrieved_docs_to_hf_disk(output_dir, dataset_dict)
    return output_dir


def build_retrieved_docs_dataset_dict(
    output_dir: Path,
    *,
    splits: tuple[str, ...] = RAG_SPLITS,
) -> DatasetDict:
    return DatasetDict(
        {
            split: Dataset.from_list(_load_split_records(output_dir, split))
            for split in splits
        }
    )


def load_retrieved_docs_from_disk(
    output_dir: Path,
    *,
    splits: tuple[str, ...] = RAG_SPLITS,
) -> DatasetDict:
    return build_retrieved_docs_dataset_dict(output_dir, splits=splits)


def save_retrieved_docs_to_hf_disk(
    output_dir: Path,
    dataset_dict: DatasetDict,
) -> Path:
    hf_dir = output_dir / "hf_dataset" / "retrieved_docs"
    dataset_dict.save_to_disk(str(hf_dir))
    return hf_dir


def _load_split_records(output_dir: Path, split: str) -> list[RetrievedDocRecord]:
    path = output_dir / split / "retrieved_docs.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing retrieved docs for split {split!r}: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
