from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]

DEFAULT_TRAINING_PATH = (
    PROJECT_ROOT / "datasets" / "raw" / "bioasq_13b" / "BioASQ-training13b" / "training13b.json"
)
DEFAULT_GOLDEN_DIR = PROJECT_ROOT / "datasets" / "raw" / "bioasq_13b" / "Task13BGoldenEnriched"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed" / "bioasq_rag"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "datasets" / "processed" / "bioasq_rag" / "pubmed_cache"

GOLDEN_FILE_PATTERN = "*_golden.json"

DEFAULT_DEV_RATIO = 0.1
DEFAULT_RANDOM_SEED = 42

PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_BATCH_SIZE = 100
PUBMED_REQUEST_DELAY_SECONDS = 0.34
PUBMED_MAX_RETRIES = 3
