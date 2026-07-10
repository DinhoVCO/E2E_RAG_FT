"""Evaluate generated RAG answers with RAGAS via OpenAI-compatible vLLM servers.

Prerequisites:
    1) vLLM judge server (OpenAI chat API), e.g.:
       vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \\
         --host 0.0.0.0 --port 8000 \\
         --tensor-parallel-size 1 \\
         --served-model-name mistralai/Mistral-Small-3.1-24B-Instruct-2503

    2) vLLM embedding server (OpenAI embeddings API), e.g.:
       vllm serve Qwen/Qwen3-Embedding-8B \\
         --host 0.0.0.0 --port 8001 \\
         --task embed \\
         --served-model-name Qwen/Qwen3-Embedding-8B

    3) Run evaluation (no local GPU required on this machine):
       python scripts/evaluation/ragas/run_rag_ragas_evaluation.py \\
         --generation-dir datasets/generated/telco-dpr/generation-b128-vllm-offline-b128
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.evaluation.ragas.openai_client import (
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_JUDGE_BASE_URL,
)
from tesis_unicamp.evaluation.ragas.runner import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RAGAS_MAX_WORKERS,
    configure_ragas_runtime,
    evaluate_generated_answers,
)
from tesis_unicamp.generation.rag.datasets import RAG_GENERATION_DATASET_CONFIGS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "ragas"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _default_output_dir(generation_dir: Path) -> Path:
    relative = generation_dir.relative_to(PROJECT_ROOT / "datasets" / "generated")
    return DEFAULT_OUTPUT_ROOT / relative


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generated RAG answers with RAGAS (OpenAI API -> vLLM).",
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing run_settings.json and "
            "<split>/generated_answers.json from a generation run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write RAGAS scores (default: results/ragas/<dataset>/<run_label>).",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(RAG_GENERATION_DATASET_CONFIGS),
        default=None,
        help="Override dataset name from generation run_settings.json.",
    )
    parser.add_argument(
        "--retrieved-dir",
        type=Path,
        default=None,
        help="Override retrieved docs directory from run_settings.json.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split to evaluate (default: value from run_settings.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of retrieved chunks used as context (default: run_settings.json).",
    )
    parser.add_argument(
        "--max-tokens-per-chunk",
        type=int,
        default=None,
        help="Per-chunk token budget for context reconstruction.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Prompt token budget used when reconstructing contexts.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help=(
            "Model id exposed by the judge vLLM server "
            f"(default: {DEFAULT_JUDGE_MODEL})."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("RAGAS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help=(
            "Model id exposed by the embedding vLLM server "
            f"(default: {DEFAULT_EMBEDDING_MODEL})."
        ),
    )
    parser.add_argument(
        "--tokenizer-model",
        default=os.getenv("RAGAS_TOKENIZER_MODEL"),
        help=(
            "Hugging Face model id used only to rebuild/truncate contexts. "
            "Defaults to --judge-model when unset."
        ),
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("RAGAS_JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL),
        help=f"OpenAI-compatible base URL for the judge server (default: {DEFAULT_JUDGE_BASE_URL}).",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("RAGAS_EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL),
        help=(
            "OpenAI-compatible base URL for the embedding server "
            f"(default: {DEFAULT_EMBEDDING_BASE_URL})."
        ),
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("RAGAS_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="API key sent to vLLM OpenAI servers (default: EMPTY).",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=int(os.getenv("RAGAS_JUDGE_MAX_TOKENS", "1024")),
        help="Maximum tokens for each RAGAS judge completion (default: 1024).",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=float(os.getenv("RAGAS_JUDGE_TEMPERATURE", "0.0")),
        help="Sampling temperature for the judge model (default: 0.0).",
    )
    parser.add_argument(
        "--enable-judge-thinking",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("RAGAS_ENABLE_JUDGE_THINKING", "0").lower() in {"1", "true", "yes"},
        help=(
            "Enable chain-of-thought / thinking in judge requests (default: off). "
            "When off, sends chat_template_kwargs enable_thinking=false to vLLM."
        ),
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=float(os.getenv("RAGAS_API_TIMEOUT", "300")),
        help="HTTP timeout in seconds for judge/embedding API calls (default: 300).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=int(os.getenv("RAGAS_EMBEDDING_BATCH_SIZE", "32")),
        help="Embedding batch size for semantic similarity (default: 32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional RAGAS evaluation batch size.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("RAGAS_MAX_WORKERS", str(DEFAULT_RAGAS_MAX_WORKERS))),
        help=f"Concurrent RAGAS workers (default: {DEFAULT_RAGAS_MAX_WORKERS} for API mode).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("RAGAS_TIMEOUT", "300")),
        help="Per-metric timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply the model chat template when reconstructing contexts. "
            "Defaults to the value stored in generation run_settings.json."
        ),
    )
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="Skip the startup OpenAI /v1/models health check.",
    )
    parser.add_argument(
        "--raise-exceptions",
        action="store_true",
        help="Stop immediately when a metric fails for a sample.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)

    generation_dir = args.generation_dir.resolve()
    if not generation_dir.exists():
        raise SystemExit(f"Generation directory not found: {generation_dir}")

    output_dir = args.output_dir or _default_output_dir(generation_dir)
    output_dir = output_dir.resolve()

    configure_ragas_runtime()

    from ragas.run_config import RunConfig

    print(f"generation_dir: {generation_dir}")
    print(f"output_dir: {output_dir}")
    print(f"judge_model: {args.judge_model}")
    print(f"embedding_model: {args.embedding_model}")
    print(f"judge_base_url: {args.judge_base_url}")
    print(f"embedding_base_url: {args.embedding_base_url}")
    if args.tokenizer_model:
        print(f"tokenizer_model: {args.tokenizer_model}")

    result = evaluate_generated_answers(
        generation_dir=generation_dir,
        output_dir=output_dir,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        tokenizer_model=args.tokenizer_model,
        judge_base_url=args.judge_base_url,
        embedding_base_url=args.embedding_base_url,
        openai_api_key=args.openai_api_key,
        judge_max_tokens=args.judge_max_tokens,
        judge_temperature=args.judge_temperature,
        api_timeout=args.api_timeout,
        enable_judge_thinking=args.enable_judge_thinking,
        use_chat_template=args.use_chat_template,
        embedding_batch_size=args.embedding_batch_size,
        retrieved_dir=args.retrieved_dir,
        split=args.split,
        top_k=args.top_k,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        max_prompt_tokens=args.max_prompt_tokens,
        dataset=args.dataset,
        batch_size=args.batch_size,
        run_config=RunConfig(
            max_workers=args.max_workers,
            timeout=args.timeout,
        ),
        skip_server_check=args.skip_server_check,
        raise_exceptions=args.raise_exceptions,
    )

    print(result)
    print(f"Saved RAGAS results to {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
