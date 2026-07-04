from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from tesis_unicamp.datasets.preprocessing.rag.narrativeqa.constants import (
    DEFAULT_HF_DATASET_ID,
    HF_TO_LOCAL_SPLIT,
)

PARQUET_COLUMNS = ["document.id", "document.summary", "question", "answers"]


def clear_stale_prepare_locks(hf_dataset_id: str = DEFAULT_HF_DATASET_ID) -> list[Path]:
    """Remove stale builder.lock files left by interrupted downloads."""
    cache_name = hf_dataset_id.replace("/", "___")
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / cache_name
    removed: list[Path] = []
    if not cache_root.exists():
        return removed
    for lock_path in cache_root.rglob("*.lock"):
        try:
            lock_path.unlink(missing_ok=True)
            removed.append(lock_path)
        except PermissionError:
            print(
                f"Could not remove lock held by another process: {lock_path}\n"
                "Stop other running builds (or orphaned python/uv processes) and retry."
            )
    return removed


@lru_cache(maxsize=1)
def _list_split_parquet_files(hf_dataset_id: str) -> dict[str, list[str]]:
    files = HfApi().list_repo_files(hf_dataset_id, repo_type="dataset")
    split_files: dict[str, list[str]] = {hf_split: [] for hf_split in HF_TO_LOCAL_SPLIT}
    for file_path in files:
        if not file_path.endswith(".parquet"):
            continue
        for hf_split in split_files:
            prefix = f"data/{hf_split}-"
            if file_path.startswith(prefix):
                split_files[hf_split].append(file_path)
    for hf_split in split_files:
        split_files[hf_split].sort()
    return split_files


def _normalize_row(raw: dict) -> dict:
    return {
        "document": {
            "id": raw["id"],
            "summary": raw["summary"],
        },
        "question": raw["question"],
        "answers": raw["answers"],
    }


def iter_narrativeqa_split(
    hf_dataset_id: str,
    hf_split: str,
    *,
    show_progress: bool = True,
) -> Iterator[dict]:
    """Yield rows for one split, reading only summary fields from parquet."""
    split_files = _list_split_parquet_files(hf_dataset_id)[hf_split]
    file_iterator = split_files
    if show_progress:
        file_iterator = tqdm(
            split_files,
            desc=f"Files {hf_split}",
            unit="file",
        )
    for parquet_file in file_iterator:
        local_path = hf_hub_download(hf_dataset_id, parquet_file, repo_type="dataset")
        table = pq.read_table(local_path, columns=PARQUET_COLUMNS)
        for raw in table.to_pylist():
            yield _normalize_row(raw)


def load_narrativeqa_splits(
    hf_dataset_id: str = DEFAULT_HF_DATASET_ID,
) -> dict[str, Iterator[dict]]:
    """Load NarrativeQA rows from Hugging Face parquet files.

    Reads only document.id, document.summary, question and answers.
    Skips document.text (~3.4 GB of full stories) for much faster builds.
    """
    print(
        "Loading NarrativeQA from parquet on Hugging Face "
        "(corpus uses document.summary.text only; full stories are skipped)."
    )
    file_counts = _list_split_parquet_files(hf_dataset_id)
    for hf_split, local_split in HF_TO_LOCAL_SPLIT.items():
        print(f"  {local_split}: {len(file_counts[hf_split])} parquet files")
    return {
        local_split: iter_narrativeqa_split(hf_dataset_id, hf_split)
        for hf_split, local_split in HF_TO_LOCAL_SPLIT.items()
    }
