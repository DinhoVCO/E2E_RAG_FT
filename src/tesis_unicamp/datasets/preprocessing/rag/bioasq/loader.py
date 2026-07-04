import json
from pathlib import Path

from tesis_unicamp.datasets.preprocessing.rag.bioasq.constants import GOLDEN_FILE_PATTERN
from tesis_unicamp.datasets.preprocessing.rag.bioasq.models import BioASQFile, BioASQQuestion


def load_bioasq_json(path: Path) -> BioASQFile:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_training_questions(training_path: Path) -> list[BioASQQuestion]:
    data = load_bioasq_json(training_path)
    return data["questions"]


def load_golden_questions(golden_dir: Path) -> list[BioASQQuestion]:
    questions: list[BioASQQuestion] = []
    for golden_path in sorted(golden_dir.glob(GOLDEN_FILE_PATTERN)):
        data = load_bioasq_json(golden_path)
        questions.extend(data["questions"])
    return questions
