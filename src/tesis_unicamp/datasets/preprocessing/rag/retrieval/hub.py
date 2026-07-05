from __future__ import annotations

from datasets import DatasetDict

from tesis_unicamp.datasets.preprocessing.rag.retrieval.io import (
    RAG_SPLITS,
    load_retrieved_docs_from_disk,
    save_retrieved_docs_to_hf_disk,
)


def push_retrieved_docs_to_hub(
    repo_id: str,
    output_dir,
    *,
    token: str | None = None,
    private: bool = False,
    splits: tuple[str, ...] = RAG_SPLITS,
    commit_message: str = "Add retrieved_docs subset",
) -> None:
    """Upload the retrieved_docs config (train/dev/test) to the Hugging Face Hub."""
    from pathlib import Path

    output_path = Path(output_dir)
    dataset_dict = load_retrieved_docs_from_disk(output_path, splits=splits)
    save_retrieved_docs_to_hf_disk(output_path, dataset_dict)
    dataset_dict.push_to_hub(
        repo_id,
        config_name="retrieved_docs",
        token=token,
        private=private,
        commit_message=commit_message,
    )


def push_retrieved_docs_dataset_dict_to_hub(
    repo_id: str,
    dataset_dict: DatasetDict,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Add retrieved_docs subset",
) -> None:
    dataset_dict.push_to_hub(
        repo_id,
        config_name="retrieved_docs",
        token=token,
        private=private,
        commit_message=commit_message,
    )
