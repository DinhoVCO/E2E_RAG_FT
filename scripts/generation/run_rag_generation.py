"""Generate RAG answers from retrieved context using vLLM offline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/generation/run_rag_generation.py \\
        --dataset qasper \\
        --retrieval-run-label vllm-offline-b128

    CUDA_VISIBLE_DEVICES=0 python scripts/generation/run_rag_generation.py \\
        --dataset telco-dpr \\
        --model Qwen/Qwen3-8B \\
        --lora-path path/to/adapter \\
        --retrieval-run-label vllm-lora-telco-dpr-b128 \\
        --run-label vllm-lora-telco-dpr-gen-b128
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.generation import (
    DEFAULT_GENERATION_BATCH_SIZE,
    GenerationConfig,
    RAG_GENERATION_DATASET_CONFIGS,
    VLLMOfflineGenerator,
    configure_vllm_multiprocessing,
    generate_answers_for_split,
    get_rag_generation_config,
)
from tesis_unicamp.generation.rag.io import save_generated_answers_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_RETRIEVAL_RUN_LABEL = "vllm-offline-b128"
DEFAULT_RUN_LABEL = "generation-default"
DEFAULT_RETRIEVED_ROOT = PROJECT_ROOT / "datasets" / "retrieved_inmemory"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "generated"
DEFAULT_SPLIT = "test"
DEFAULT_TOP_K = 10
DEFAULT_MAX_TOKENS_PER_CHUNK = 512


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _sanitize_label(label: str, *, fallback: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip())
    return sanitized or fallback


def _default_retrieved_dir(dataset: str, retrieval_run_label: str) -> Path:
    return DEFAULT_RETRIEVED_ROOT / dataset / _sanitize_label(
        retrieval_run_label,
        fallback=DEFAULT_RETRIEVAL_RUN_LABEL,
    )


def _default_output_dir(dataset: str, run_label: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / dataset / _sanitize_label(run_label, fallback=DEFAULT_RUN_LABEL)


def _validate_cuda() -> None:
    import torch

    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()

    print(f"CUDA_VISIBLE_DEVICES: {visible or '(unset)'}")
    print(f"torch.cuda.device_count(): {count}")

    if count == 0:
        raise SystemExit(
            "No CUDA device is visible. Request a GPU and set CUDA_VISIBLE_DEVICES "
            "before running offline generation."
        )

    try:
        print(f"cuda:0 -> {torch.cuda.get_device_name(0)}")
    except RuntimeError as exc:
        raise SystemExit(f"Cannot access cuda:0: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RAG answers from retrieved context with vLLM offline.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(RAG_GENERATION_DATASET_CONFIGS),
        required=True,
        help="RAG dataset to generate answers for",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_LLM_MODEL", DEFAULT_MODEL),
        help=f"Generation model (default: {DEFAULT_MODEL} or VLLM_LLM_MODEL env var)",
    )
    parser.add_argument(
        "--lora-path",
        default=os.getenv("LORA_PATH"),
        help="LoRA adapter path or Hugging Face repo id (offline only)",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=int(os.getenv("MAX_LORA_RANK", "16")),
        help="max_lora_rank passed to vLLM when --lora-path is set (default: 16)",
    )
    parser.add_argument(
        "--retrieval-run-label",
        default=os.getenv("RETRIEVAL_RUN_LABEL", DEFAULT_RETRIEVAL_RUN_LABEL),
        help="Subfolder under datasets/retrieved_inmemory/<dataset>/ (default: vllm-offline-b128)",
    )
    parser.add_argument(
        "--retrieved-dir",
        type=Path,
        default=None,
        help="Directory containing <split>/retrieved_docs.json",
    )
    parser.add_argument(
        "--run-label",
        default=os.getenv("GENERATION_RUN_LABEL", DEFAULT_RUN_LABEL),
        help="Subfolder under datasets/generated/<dataset>/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write generated_answers.json",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Query split to generate answers for (default: test)",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=int(os.getenv("GENERATION_TOP_K", DEFAULT_TOP_K)),
        help="Number of retrieved documents to include in the prompt (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("GENERATION_BATCH_SIZE", DEFAULT_GENERATION_BATCH_SIZE)),
        help=f"Prompts per generation batch (default: {DEFAULT_GENERATION_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("GENERATION_MAX_TOKENS", "512")),
        help="Maximum tokens to generate per answer (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("GENERATION_TEMPERATURE", "0.0")),
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens-per-chunk",
        type=int,
        default=int(os.getenv("GENERATION_MAX_TOKENS_PER_CHUNK", DEFAULT_MAX_TOKENS_PER_CHUNK)),
        help=f"Maximum tokens per retrieved chunk (default: {DEFAULT_MAX_TOKENS_PER_CHUNK})",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=int(os.getenv("GENERATION_MAX_PROMPT_TOKENS", "0")),
        help=(
            "Maximum prompt tokens including context (0 = auto: max_model_len - max_tokens - 256)"
        ),
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Send raw prompts without applying the model chat template",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)

    retrieval_run_label = _sanitize_label(
        args.retrieval_run_label,
        fallback=DEFAULT_RETRIEVAL_RUN_LABEL,
    )
    run_label = _sanitize_label(args.run_label, fallback=DEFAULT_RUN_LABEL)
    retrieved_dir = args.retrieved_dir or _default_retrieved_dir(args.dataset, retrieval_run_label)
    output_dir = args.output_dir or _default_output_dir(args.dataset, run_label)

    _validate_cuda()
    configure_vllm_multiprocessing()

    config = get_rag_generation_config(args.dataset)
    generator = VLLMOfflineGenerator(
        GenerationConfig(
            model=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        ),
        lora_path=args.lora_path,
        max_lora_rank=args.max_lora_rank if args.lora_path else None,
        use_chat_template=not args.no_chat_template,
    )

    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    if args.lora_path:
        print(f"lora_path: {args.lora_path}")
        print(f"max_lora_rank: {args.max_lora_rank}")
    print(f"retrieved_dir: {retrieved_dir}")
    print(f"split: {args.split}")
    print(f"top_k: {args.top_k}")
    print(f"max_tokens_per_chunk: {args.max_tokens_per_chunk}")
    print(f"output_dir: {output_dir}")

    generator.warmup()

    max_prompt_tokens = args.max_prompt_tokens
    if max_prompt_tokens <= 0:
        max_prompt_tokens = generator.get_default_max_prompt_tokens()

    max_tokens_per_chunk = args.max_tokens_per_chunk

    print(f"max_prompt_tokens: {max_prompt_tokens}")

    records = generate_answers_for_split(
        generator,
        config,
        retrieved_dir=retrieved_dir,
        split=args.split,
        max_context_docs=args.top_k,
        max_prompt_tokens=max_prompt_tokens,
        max_tokens_per_chunk=max_tokens_per_chunk,
        batch_size=args.batch_size,
    )
    run_settings = {
        "dataset": args.dataset,
        "model": args.model,
        "lora_path": args.lora_path,
        "retrieval_run_label": retrieval_run_label,
        "retrieved_dir": str(retrieved_dir),
        "run_label": run_label,
        "split": args.split,
        "top_k": args.top_k,
        "max_tokens_per_chunk": max_tokens_per_chunk,
        "max_prompt_tokens": max_prompt_tokens,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "use_chat_template": not args.no_chat_template,
        "num_answers": len(records),
    }
    save_generated_answers_bundle(
        output_dir,
        {args.split: records},
        run_settings=run_settings,
    )

    print(f"Generated {len(records)} answers")
    print(f"Saved generated answers to {output_dir / args.split / 'generated_answers.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
