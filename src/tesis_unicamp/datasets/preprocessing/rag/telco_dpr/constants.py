from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]

DEFAULT_HF_DATASET_ID = "thainasaraiva/Telco-DPR"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "telco_dpr_rag"
DEFAULT_HUB_README_TEMPLATE = Path(__file__).resolve().parent / "README.template.md"
DEFAULT_HUB_README = DEFAULT_OUTPUT_DIR / "README.md"

HF_TO_LOCAL_SPLIT = {
    "train": "train",
    "test": "test",
}

DEFAULT_DEV_RATIO = 0.2
DEFAULT_RANDOM_SEED = 42
