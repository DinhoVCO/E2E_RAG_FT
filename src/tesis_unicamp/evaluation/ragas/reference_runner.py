from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, EvaluationResult
from ragas.embeddings import HuggingfaceEmbeddings
from ragas.metrics import (
    BleuScore,
    ChrfScore,
    ExactMatch,
    RougeScore,
    SemanticSimilarity,
    StringPresence,
)
from ragas.metrics._string import DistanceMeasure, NonLLMStringSimilarity
from ragas.run_config import RunConfig

from tesis_unicamp.evaluation.ragas.dataset import load_generation_run_settings
from tesis_unicamp.evaluation.ragas.io import (
    build_summary,
    evaluation_result_to_rows,
    save_ragas_results,
)
from tesis_unicamp.evaluation.ragas.openai_client import (
    DEFAULT_EMBEDDING_BASE_URL,
    build_openai_ragas_embeddings,
    check_openai_server,
)
from tesis_unicamp.evaluation.ragas.runner import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RAGAS_MAX_WORKERS,
    configure_ragas_runtime,
)
from tesis_unicamp.generation.rag.io import load_generated_answers

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_GENERATED_ROOT = PROJECT_ROOT / "datasets" / "generated"
DEFAULT_REFERENCE_OUTPUT_ROOT = PROJECT_ROOT / "results" / "ragas-reference"

TRADITIONAL_SCORES_FILENAME = "reference_traditional_scores.json"
TRADITIONAL_SUMMARY_FILENAME = "reference_traditional_summary.json"
SEMANTIC_SCORES_FILENAME = "reference_semantic_similarity_scores.json"
SEMANTIC_SUMMARY_FILENAME = "reference_semantic_similarity_summary.json"


def build_traditional_metrics() -> list[Any]:
    """Non-LLM reference metrics (response vs reference only)."""
    return [
        BleuScore(),
        RougeScore(),
        ExactMatch(),
        StringPresence(),
        ChrfScore(),
        NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN),
    ]


def build_semantic_similarity_metric() -> list[Any]:
    return [SemanticSimilarity()]


def build_reference_evaluation_dataset(
    generation_dir: Path,
    *,
    split: str,
) -> EvaluationDataset:
    """Build a RAGAS dataset using only question, response, and reference."""
    generated_records = load_generated_answers(generation_dir, split)
    if not generated_records:
        raise ValueError(
            f"No generated answers found in "
            f"{generation_dir / split / 'generated_answers.json'}"
        )

    rows: list[dict[str, Any]] = []
    for record in generated_records:
        rows.append(
            {
                "query_id": str(record["query_id"]),
                "user_input": record["question"],
                "response": record["generated_answer"],
                "reference": record.get("reference_answer", "") or "",
            }
        )
    return EvaluationDataset.from_list(rows)


def discover_generation_dirs(
    *,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    dataset: str | None = None,
) -> list[Path]:
    """Find generation run directories under datasets/generated."""
    if not generated_root.exists():
        return []

    dirs: list[Path] = []
    for settings_path in sorted(generated_root.glob("**/run_settings.json")):
        generation_dir = settings_path.parent.resolve()
        if dataset is not None and generation_dir.parent.name != dataset:
            continue
        try:
            settings = load_generation_run_settings(generation_dir)
        except (OSError, ValueError, KeyError):
            continue
        split = str(settings.get("split", "test"))
        answers_path = generation_dir / split / "generated_answers.json"
        if not answers_path.exists():
            continue
        dirs.append(generation_dir)
    return dirs


def default_reference_output_dir(
    generation_dir: Path,
    *,
    metric_group: str,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    output_root: Path = DEFAULT_REFERENCE_OUTPUT_ROOT,
) -> Path:
    relative = generation_dir.relative_to(generated_root)
    return output_root / metric_group / relative


def resolve_reference_evaluation_context(
    generation_dir: Path,
    *,
    split: str | None,
    dataset: str | None,
) -> dict[str, Any]:
    settings = load_generation_run_settings(generation_dir)
    return {
        "dataset": dataset or str(settings["dataset"]),
        "split": split or str(settings.get("split", "test")),
        "generation_run_label": settings.get("run_label"),
        "retrieval_run_label": settings.get("retrieval_run_label"),
    }


def _attach_generated_fields(
    per_sample_scores: list[dict[str, Any]],
    generation_dir: Path,
    split: str,
) -> None:
    generated_records = load_generated_answers(generation_dir, split)
    for index, row in enumerate(per_sample_scores):
        if index >= len(generated_records):
            break
        record = generated_records[index]
        row.setdefault("query_id", record["query_id"])
        row.setdefault("question", record["question"])
        row.setdefault("generated_answer", record["generated_answer"])
        row.setdefault("reference_answer", record["reference_answer"])


def _run_reference_evaluation(
    *,
    generation_dir: Path,
    output_dir: Path,
    split: str | None,
    dataset: str | None,
    metrics: Sequence[Any],
    embeddings: Any | None,
    run_config: RunConfig | None,
    batch_size: int | None,
    show_progress: bool,
    raise_exceptions: bool,
    metric_group: str,
    scores_filename: str,
    summary_filename: str,
    extra_run_settings: dict[str, Any] | None = None,
) -> EvaluationResult:
    context = resolve_reference_evaluation_context(
        generation_dir,
        split=split,
        dataset=dataset,
    )
    effective_split = context["split"]
    evaluation_dataset = build_reference_evaluation_dataset(
        generation_dir,
        split=effective_split,
    )

    effective_run_config = run_config or RunConfig(
        max_workers=DEFAULT_RAGAS_MAX_WORKERS,
        timeout=300,
    )
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=list(metrics),
        llm=None,
        embeddings=embeddings,
        run_config=effective_run_config,
        batch_size=batch_size,
        show_progress=show_progress,
        raise_exceptions=raise_exceptions,
    )

    per_sample_scores = evaluation_result_to_rows(result)
    _attach_generated_fields(per_sample_scores, generation_dir, effective_split)
    summary = build_summary(result)
    run_settings = {
        "metric_group": metric_group,
        "dataset": context["dataset"],
        "generation_dir": str(generation_dir.resolve()),
        "generation_run_label": context["generation_run_label"],
        "retrieval_run_label": context["retrieval_run_label"],
        "split": effective_split,
        "metrics": [metric.name for metric in metrics],
        "num_samples": len(per_sample_scores),
        "summary": summary,
    }
    if extra_run_settings:
        run_settings.update(extra_run_settings)

    save_ragas_results(
        output_dir,
        effective_split,
        per_sample_scores=per_sample_scores,
        summary=summary,
        run_settings=run_settings,
        scores_filename=scores_filename,
        summary_filename=summary_filename,
    )
    return result


def evaluate_reference_traditional_metrics(
    *,
    generation_dir: Path,
    output_dir: Path,
    split: str | None = None,
    dataset: str | None = None,
    metrics: Sequence[Any] | None = None,
    run_config: RunConfig | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
    raise_exceptions: bool = False,
) -> EvaluationResult:
    """Evaluate BLEU, ROUGE, CHRF, exact match, etc. (CPU only)."""
    configure_ragas_runtime()
    return _run_reference_evaluation(
        generation_dir=generation_dir,
        output_dir=output_dir,
        split=split,
        dataset=dataset,
        metrics=metrics or build_traditional_metrics(),
        embeddings=None,
        run_config=run_config,
        batch_size=batch_size,
        show_progress=show_progress,
        raise_exceptions=raise_exceptions,
        metric_group="traditional",
        scores_filename=TRADITIONAL_SCORES_FILENAME,
        summary_filename=TRADITIONAL_SUMMARY_FILENAME,
    )


def build_local_ragas_embeddings(
    model_name: str,
    *,
    device: str = "cuda",
    encode_batch_size: int = 32,
) -> HuggingfaceEmbeddings:
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device:
        model_kwargs["device"] = device
    return HuggingfaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"batch_size": encode_batch_size},
    )


def evaluate_reference_semantic_similarity(
    *,
    generation_dir: Path,
    output_dir: Path,
    split: str | None = None,
    dataset: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_device: str = "cuda",
    embedding_batch_size: int = 32,
    embedding_base_url: str | None = None,
    openai_api_key: str | None = None,
    api_timeout: float = 300.0,
    skip_server_check: bool = False,
    run_config: RunConfig | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
    raise_exceptions: bool = False,
) -> EvaluationResult:
    """Evaluate semantic similarity between response and reference."""
    configure_ragas_runtime()

    if embedding_base_url:
        if not skip_server_check:
            check_openai_server(
                "embedding",
                base_url=embedding_base_url,
                api_key=openai_api_key,
                expected_model=embedding_model,
                timeout=api_timeout,
            )
        embeddings = build_openai_ragas_embeddings(
            embedding_model,
            base_url=embedding_base_url,
            api_key=openai_api_key,
            timeout=api_timeout,
        )
        embedding_backend = "openai"
    else:
        effective_device = embedding_device
        if effective_device == "cuda":
            import torch

            if not torch.cuda.is_available():
                effective_device = "cpu"
        embeddings = build_local_ragas_embeddings(
            embedding_model,
            device=effective_device,
            encode_batch_size=embedding_batch_size,
        )
        embedding_backend = "local"

    return _run_reference_evaluation(
        generation_dir=generation_dir,
        output_dir=output_dir,
        split=split,
        dataset=dataset,
        metrics=build_semantic_similarity_metric(),
        embeddings=embeddings,
        run_config=run_config,
        batch_size=batch_size,
        show_progress=show_progress,
        raise_exceptions=raise_exceptions,
        metric_group="semantic-similarity",
        scores_filename=SEMANTIC_SCORES_FILENAME,
        summary_filename=SEMANTIC_SUMMARY_FILENAME,
        extra_run_settings={
            "embedding_model": embedding_model,
            "embedding_backend": embedding_backend,
            "embedding_device": embedding_device if embedding_backend == "local" else None,
            "embedding_base_url": embedding_base_url or DEFAULT_EMBEDDING_BASE_URL,
            "embedding_batch_size": embedding_batch_size,
            "api_timeout": api_timeout,
        },
    )


def summary_path_for_run(
    output_dir: Path,
    split: str,
    *,
    metric_group: str,
) -> Path:
    filename = (
        TRADITIONAL_SUMMARY_FILENAME
        if metric_group == "traditional"
        else SEMANTIC_SUMMARY_FILENAME
    )
    return output_dir / split / filename
