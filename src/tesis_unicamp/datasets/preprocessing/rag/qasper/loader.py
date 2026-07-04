from collections.abc import Iterator

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from tesis_unicamp.datasets.preprocessing.rag.qasper.constants import (
    DEFAULT_HF_DATASET_ID,
    DEFAULT_PARQUET_REVISION,
    HF_TO_LOCAL_SPLIT,
    PARQUET_COLUMNS,
    PARQUET_FILES,
)


def iter_qasper_split(
    hf_dataset_id: str,
    hf_split: str,
    *,
    revision: str = DEFAULT_PARQUET_REVISION,
    show_progress: bool = True,
) -> Iterator[dict]:
    """Yield rows for one QASPER split from Hugging Face parquet files."""
    parquet_file = PARQUET_FILES[hf_split]
    local_path = hf_hub_download(
        hf_dataset_id,
        parquet_file,
        repo_type="dataset",
        revision=revision,
    )
    table = pq.read_table(local_path, columns=PARQUET_COLUMNS)
    rows = table.to_pylist()
    row_iterator: list[dict] | tqdm = rows
    if show_progress:
        row_iterator = tqdm(rows, desc=f"Rows {hf_split}", unit="paper")
    yield from row_iterator


def load_qasper_splits(
    hf_dataset_id: str = DEFAULT_HF_DATASET_ID,
    *,
    revision: str = DEFAULT_PARQUET_REVISION,
) -> dict[str, Iterator[dict]]:
    """Load QASPER rows from Hugging Face parquet files."""
    print(
        "Loading QASPER from parquet on Hugging Face "
        f"({hf_dataset_id}, revision={revision})."
    )
    for hf_split, local_split in HF_TO_LOCAL_SPLIT.items():
        print(f"  {local_split}: {PARQUET_FILES[hf_split]}")
    return {
        local_split: iter_qasper_split(
            hf_dataset_id,
            hf_split,
            revision=revision,
        )
        for hf_split, local_split in HF_TO_LOCAL_SPLIT.items()
    }
