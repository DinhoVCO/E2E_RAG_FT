from datasets import Dataset, concatenate_datasets, load_dataset

from tesis_unicamp.datasets.preprocessing.rag.telco_dpr.constants import (
    DEFAULT_HF_DATASET_ID,
    HF_TO_LOCAL_SPLIT,
)


def load_telco_dpr_corpus(hf_dataset_id: str = DEFAULT_HF_DATASET_ID) -> Dataset:
    """Load and concatenate corpus small + extended splits."""
    small = load_dataset(hf_dataset_id, "corpus", split="small")
    extended = load_dataset(hf_dataset_id, "corpus", split="extended")
    return concatenate_datasets([small, extended])


def load_telco_dpr_queries(hf_dataset_id: str = DEFAULT_HF_DATASET_ID) -> Dataset:
    return load_dataset(hf_dataset_id, "queries", split="queries")


def load_telco_dpr_relevant_docs(
    hf_dataset_id: str = DEFAULT_HF_DATASET_ID,
) -> dict[str, Dataset]:
    return {
        local_split: load_dataset(hf_dataset_id, "relevant_docs", split=hf_split)
        for hf_split, local_split in HF_TO_LOCAL_SPLIT.items()
    }
