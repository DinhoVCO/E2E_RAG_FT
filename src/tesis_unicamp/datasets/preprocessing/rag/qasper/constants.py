from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]

DEFAULT_HF_DATASET_ID = "allenai/qasper"
DEFAULT_PARQUET_REVISION = "refs/convert/parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "qasper_rag"
STRICT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "qasper_rag_strict"
DEFAULT_HUB_README_TEMPLATE = Path(__file__).resolve().parent / "README.template.md"
STRICT_HUB_README_TEMPLATE = Path(__file__).resolve().parent / "README.strict.template.md"
DEFAULT_HUB_README = DEFAULT_OUTPUT_DIR / "README.md"
STRICT_HUB_README = STRICT_OUTPUT_DIR / "README.md"

HF_TO_LOCAL_SPLIT = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}

PARQUET_FILES = {
    "train": "qasper/train/0000.parquet",
    "validation": "qasper/validation/0000.parquet",
    "test": "qasper/test/0000.parquet",
}

PARQUET_COLUMNS = ["id", "title", "abstract", "full_text", "qas"]
