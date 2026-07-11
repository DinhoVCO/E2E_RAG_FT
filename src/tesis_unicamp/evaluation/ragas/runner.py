from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import AnswerCorrectness, SemanticSimilarity
from ragas.run_config import RunConfig

from tesis_unicamp.evaluation.ragas.dataset import (
    build_ragas_evaluation_dataset,
    resolve_generation_paths,
)
from tesis_unicamp.evaluation.ragas.io import (
    build_summary,
    evaluation_result_to_rows,
    save_ragas_results,
)
from tesis_unicamp.evaluation.ragas.openai_client import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_JUDGE_BASE_URL,
    build_openai_judge_llm,
    build_openai_ragas_embeddings,
    check_openai_server,
)
from tesis_unicamp.evaluation.ragas.tokenizer import build_tokenizer_helpers_from_model_name
from tesis_unicamp.generation.rag.io import load_generated_answers


DEFAULT_JUDGE_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_JUDGE_MAX_TOKENS = 2048
DEFAULT_RAGAS_MAX_WORKERS = 64
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_API_TIMEOUT = 300.0
DEFAULT_METRICS = (
    AnswerCorrectness(),
    SemanticSimilarity(),
)


def configure_ragas_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_default_metrics() -> list[Any]:
    return list(DEFAULT_METRICS)


def evaluate_generated_answers(
    *,
    generation_dir: Path,
    output_dir: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    tokenizer_model: str | None = None,
    judge_base_url: str = DEFAULT_JUDGE_BASE_URL,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    openai_api_key: str | None = None,
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    judge_temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    api_timeout: float = DEFAULT_API_TIMEOUT,
    enable_judge_thinking: bool = False,
    use_chat_template: bool | None = None,
    embedding_batch_size: int = 32,
    retrieved_dir: Path | None = None,
    split: str | None = None,
    top_k: int | None = None,
    max_tokens_per_chunk: int | None = None,
    max_prompt_tokens: int | None = None,
    dataset: str | None = None,
    metrics: Sequence[Any] | None = None,
    run_config: RunConfig | None = None,
    batch_size: int | None = None,
    show_progress: bool = True,
    raise_exceptions: bool = False,
    skip_server_check: bool = False,
) -> EvaluationResult:
    """Evaluate generated RAG answers with RAGAS via OpenAI-compatible vLLM APIs."""
    configure_ragas_runtime()

    if not skip_server_check:
        check_openai_server(
            "judge",
            base_url=judge_base_url,
            api_key=openai_api_key,
            expected_model=judge_model,
            timeout=api_timeout,
        )
        check_openai_server(
            "embedding",
            base_url=embedding_base_url,
            api_key=openai_api_key,
            expected_model=embedding_model,
            timeout=api_timeout,
        )

    resolved = resolve_generation_paths(
        generation_dir,
        retrieved_dir=retrieved_dir,
        split=split,
        top_k=top_k,
        max_tokens_per_chunk=max_tokens_per_chunk,
        max_prompt_tokens=max_prompt_tokens,
        dataset=dataset,
    )
    dataset_name = str(resolved["dataset"])
    from tesis_unicamp.generation import get_rag_generation_config

    config = get_rag_generation_config(dataset_name)
    effective_split = str(resolved["split"])
    effective_top_k = int(resolved["top_k"])
    effective_max_tokens_per_chunk = resolved["max_tokens_per_chunk"]
    effective_max_prompt_tokens = int(resolved["max_prompt_tokens"])
    effective_use_chat_template = (
        use_chat_template
        if use_chat_template is not None
        else bool(resolved["use_chat_template"])
    )
    effective_tokenizer_model = tokenizer_model or judge_model

    count_prompt_tokens, truncate_text_to_tokens = build_tokenizer_helpers_from_model_name(
        effective_tokenizer_model,
        use_chat_template=effective_use_chat_template,
    )

    evaluation_dataset = build_ragas_evaluation_dataset(
        config,
        generation_dir=generation_dir,
        retrieved_dir=Path(resolved["retrieved_dir"]),
        split=effective_split,
        top_k=effective_top_k,
        max_tokens_per_chunk=effective_max_tokens_per_chunk,
        max_prompt_tokens=effective_max_prompt_tokens,
        count_prompt_tokens=count_prompt_tokens,
        truncate_text_to_tokens=truncate_text_to_tokens,
    )

    bypass_n = any(
        token in judge_model.lower()
        for token in ("deepseek", "r1", "reasoning", "gpt-oss")
    )
    judge_llm = build_openai_judge_llm(
        judge_model,
        base_url=judge_base_url,
        api_key=openai_api_key,
        max_tokens=judge_max_tokens,
        temperature=judge_temperature,
        timeout=api_timeout,
        bypass_n=bypass_n,
        enable_thinking=enable_judge_thinking,
    )
    embeddings = build_openai_ragas_embeddings(
        embedding_model,
        base_url=embedding_base_url,
        api_key=openai_api_key,
        timeout=api_timeout,
    )

    selected_metrics = list(metrics or build_default_metrics())
    effective_run_config = run_config or RunConfig(
        max_workers=DEFAULT_RAGAS_MAX_WORKERS,
        timeout=int(api_timeout),
    )

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=selected_metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=effective_run_config,
        batch_size=batch_size,
        show_progress=show_progress,
        raise_exceptions=raise_exceptions,
    )

    per_sample_scores = evaluation_result_to_rows(result)
    generated_records = load_generated_answers(generation_dir, effective_split)
    for index, row in enumerate(per_sample_scores):
        if index < len(generated_records):
            row["query_id"] = generated_records[index]["query_id"]
            row["question"] = generated_records[index]["question"]
            row["generated_answer"] = generated_records[index]["generated_answer"]
            row["reference_answer"] = generated_records[index]["reference_answer"]
    summary = build_summary(result)
    run_settings = {
        "dataset": dataset_name,
        "generation_dir": str(generation_dir),
        "generation_run_label": resolved.get("generation_run_label"),
        "retrieval_run_label": resolved.get("retrieval_run_label"),
        "retrieved_dir": str(resolved["retrieved_dir"]),
        "split": effective_split,
        "top_k": effective_top_k,
        "max_tokens_per_chunk": effective_max_tokens_per_chunk,
        "max_prompt_tokens": effective_max_prompt_tokens,
        "use_chat_template": effective_use_chat_template,
        "judge_model": judge_model,
        "embedding_model": embedding_model,
        "tokenizer_model": effective_tokenizer_model,
        "judge_base_url": judge_base_url,
        "embedding_base_url": embedding_base_url,
        "judge_max_tokens": judge_max_tokens,
        "judge_temperature": judge_temperature,
        "enable_judge_thinking": enable_judge_thinking,
        "api_timeout": api_timeout,
        "embedding_batch_size": embedding_batch_size,
        "metrics": [metric.name for metric in selected_metrics],
        "num_samples": len(per_sample_scores),
        "summary": summary,
    }
    save_ragas_results(
        output_dir,
        effective_split,
        per_sample_scores=per_sample_scores,
        summary=summary,
        run_settings=run_settings,
    )
    return result
