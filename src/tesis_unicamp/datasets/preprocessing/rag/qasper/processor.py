from collections import defaultdict
from collections.abc import Iterator

from tesis_unicamp.datasets.preprocessing.rag.qasper.models import (
    CorpusRecord,
    PaperIndex,
    SplitData,
)

FLOAT_SELECTED_MARKER = "FLOAT SELECTED"
ABSTRACT_SECTION_NAME = "abstract"


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def make_chunk_id(paper_id: str, chunk_index: int) -> str:
    return f"{paper_id}_{chunk_index:05d}"


def _empty_split_data() -> SplitData:
    return {"queries": [], "qrels": [], "answers": [], "top_ranked": []}


def is_valid_answer(answer: dict) -> bool:
    if answer.get("unanswerable"):
        return False
    if (answer.get("free_form_answer") or "").strip():
        return True
    if answer.get("extractive_spans"):
        return True
    yes_no = answer.get("yes_no")
    return yes_no is True or yes_no is False


def answer_text(answer: dict) -> str:
    free_form = (answer.get("free_form_answer") or "").strip()
    if free_form:
        return free_form

    spans = [
        span.strip()
        for span in answer.get("extractive_spans") or []
        if (span or "").strip()
    ]
    if spans:
        return " ".join(spans)

    yes_no = answer.get("yes_no")
    if yes_no is True:
        return "Yes"
    if yes_no is False:
        return "No"
    return ""


def longest_answer(answers: list[dict]) -> str:
    texts = [text for text in (answer_text(answer) for answer in answers) if text]
    if not texts:
        return ""
    return max(texts, key=len)


def filter_evidence(evidence: list[str]) -> list[str]:
    return [
        item
        for item in (evidence or [])
        if item and FLOAT_SELECTED_MARKER not in item
    ]


def build_paper_index(row: dict) -> PaperIndex:
    paper_id = row["id"]
    title = row.get("title") or ""
    abstract = (row.get("abstract") or "").strip()
    full_text = row.get("full_text") or {}

    chunks: list[CorpusRecord] = []
    text_to_ids: dict[str, list[str]] = defaultdict(list)
    section_to_ids: dict[str, list[str]] = defaultdict(list)
    chunk_index = 0

    if abstract:
        chunk_id = make_chunk_id(paper_id, chunk_index)
        chunks.append(
            {
                "id": chunk_id,
                "title": title,
                "section_name": ABSTRACT_SECTION_NAME,
                "text": abstract,
            }
        )
        text_to_ids[normalize_text(abstract)].append(chunk_id)
        chunk_index += 1

    section_names = full_text.get("section_name") or []
    paragraphs = full_text.get("paragraphs") or []
    for section_name, section_paragraphs in zip(section_names, paragraphs):
        section = (section_name or "").strip()
        section_norm = normalize_text(section)
        section_chunk_ids: list[str] = []

        for paragraph in section_paragraphs or []:
            text = (paragraph or "").strip()
            if not text:
                continue
            chunk_id = make_chunk_id(paper_id, chunk_index)
            chunks.append(
                {
                    "id": chunk_id,
                    "title": title,
                    "section_name": section,
                    "text": text,
                }
            )
            text_to_ids[normalize_text(text)].append(chunk_id)
            section_chunk_ids.append(chunk_id)
            chunk_index += 1

        if section_norm and section_chunk_ids:
            section_to_ids[section_norm].extend(section_chunk_ids)

    return {
        "chunks": chunks,
        "text_to_ids": dict(text_to_ids),
        "section_to_ids": dict(section_to_ids),
    }


def resolve_evidence_to_chunk_ids(
    evidence: list[str],
    paper_index: PaperIndex,
    *,
    exact_text_evidence_only: bool = False,
) -> set[str]:
    chunk_ids: set[str] = set()
    for item in evidence:
        normalized = normalize_text(item)
        if not normalized:
            continue
        if normalized in paper_index["text_to_ids"]:
            chunk_ids.update(paper_index["text_to_ids"][normalized])
        elif (
            not exact_text_evidence_only
            and normalized in paper_index["section_to_ids"]
        ):
            chunk_ids.update(paper_index["section_to_ids"][normalized])
    return chunk_ids


def process_paper_qas(
    row: dict,
    paper_index: PaperIndex,
    split_data: SplitData,
    *,
    exact_text_evidence_only: bool = False,
) -> None:
    qas = row.get("qas") or {}
    questions = qas.get("question") or []
    question_ids = qas.get("question_id") or []
    answers_blocks = qas.get("answers") or []

    for question_index, question_text in enumerate(questions):
        if question_index >= len(answers_blocks):
            continue

        valid_answers = [
            answer
            for answer in answers_blocks[question_index].get("answer") or []
            if is_valid_answer(answer)
        ]
        if not valid_answers:
            continue

        chunk_ids: set[str] = set()
        for answer in valid_answers:
            filtered_evidence = filter_evidence(answer.get("evidence") or [])
            if not filtered_evidence:
                continue
            chunk_ids.update(
                resolve_evidence_to_chunk_ids(
                    filtered_evidence,
                    paper_index,
                    exact_text_evidence_only=exact_text_evidence_only,
                )
            )
        if not chunk_ids:
            continue

        if question_index < len(question_ids) and question_ids[question_index]:
            query_id = question_ids[question_index]
        else:
            query_id = f"{row['id']}_{question_index:04d}"

        split_data["queries"].append({"id": query_id, "text": question_text})
        split_data["top_ranked"].append(
            {
                "query_id": query_id,
                "corpus_ids": [chunk["id"] for chunk in paper_index["chunks"]],
            }
        )
        for corpus_id in sorted(chunk_ids):
            split_data["qrels"].append(
                {
                    "query_id": query_id,
                    "corpus_id": corpus_id,
                    "score": 1,
                }
            )
        split_data["answers"].append(
            {
                "query_id": query_id,
                "answer": longest_answer(valid_answers),
            }
        )


def process_qasper_splits(
    splits: dict[str, Iterator[dict]],
    *,
    exact_text_evidence_only: bool = False,
) -> tuple[list[CorpusRecord], dict[str, SplitData]]:
    """Build the shared corpus and query splits from QASPER papers."""
    paper_indexes: dict[str, PaperIndex] = {}
    processed_splits: dict[str, SplitData] = {
        split_name: _empty_split_data() for split_name in splits
    }

    for split_name, dataset in splits.items():
        for row in dataset:
            paper_id = row["id"]
            if paper_id not in paper_indexes:
                paper_indexes[paper_id] = build_paper_index(row)
            process_paper_qas(
                row,
                paper_indexes[paper_id],
                processed_splits[split_name],
                exact_text_evidence_only=exact_text_evidence_only,
            )

    corpus: list[CorpusRecord] = []
    for paper_id in sorted(paper_indexes):
        corpus.extend(paper_indexes[paper_id]["chunks"])

    return corpus, processed_splits
