"""Generate long-form RFG expanded queries from stage-1 retrieved documents.

Uses qwen3-8B (base or LoRA) to produce a detailed answer from the top-k
retrieved documents. The answer is saved as the expanded query for stage-2
retrieval evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_generate_expansion.py \\
        --dataset qasper \\
        --retrieval-run-label vllm-offline-b128-paper-scoped \\
        --run-label qasper-rfg-emb-base-gen-base

    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_generate_expansion.py \\
        --dataset telco-dpr \\
        --model Qwen/Qwen3-8B \\
        --lora-path DinoStackAI/Qwen3-8b-lora-telco-dpr \\
        --retrieval-run-label vllm-offline-b128 \\
        --run-label telco-dpr-rfg-emb-base-gen-lora
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
    VLLMOfflineGenerator,
    configure_vllm_multiprocessing,
    get_rag_generation_config,
)
from tesis_unicamp.query_expansion.generation import (
    DEFAULT_EXPANSION_K_VALUES,
    DEFAULT_EXPANSION_MAX_TOKENS,
    DEFAULT_MAX_TOKENS_PER_CHUNK,
    generate_expansions_for_split,
)
from tesis_unicamp.query_expansion.io import save_expanded_queries_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_RETRIEVAL_RUN_LABEL = "vllm-offline-b128"
DEFAULT_RUN_LABEL = "rfg-expansion-default"
DEFAULT_RETRIEVED_ROOT = "retrieved_inmemory"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "query_expansion"
DEFAULT_SPLIT = "test"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _sanitize_label(label: str, *, fallback: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip())
    return sanitized or fallback


def _default_retrieved_dir(
    dataset: str,
    retrieval_run_label: str,
    *,
    retrieved_root: str,
) -> Path:
    return (
        PROJECT_ROOT
        / "datasets"
        / retrieved_root
        / dataset
        / _sanitize_label(retrieval_run_label, fallback=DEFAULT_RETRIEVAL_RUN_LABEL)
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
        description="Generate RFG long-form expanded queries with vLLM offline.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(
            {
                "bioasq-resplit",
                "qasper",
                "telco-dpr",
                "narrativeqa",
            }
        ),
        required=True,
        help="RAG dataset",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_LLM_MODEL", DEFAULT_MODEL),
        help=f"Generation model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--lora-path",
        default=os.getenv("LORA_PATH"),
        help="LoRA adapter path or Hugging Face repo id",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=int(os.getenv("MAX_LORA_RANK", "16")),
        help="max_lora_rank for vLLM when --lora-path is set (default: 16)",
    )
    parser.add_argument(
        "--retrieval-run-label",
        default=os.getenv("RFG_RETRIEVAL_RUN_LABEL", DEFAULT_RETRIEVAL_RUN_LABEL),
        help="Subfolder under datasets/retrieved_inmemory_rfg/<dataset>/",
    )
    parser.add_argument(
        "--retrieved-root",
        default=os.getenv("RETRIEVED_ROOT", DEFAULT_RETRIEVED_ROOT),
        help=(
            "Root folder under datasets/ for stage-1 retrieved_docs "
            f"(default: {DEFAULT_RETRIEVED_ROOT})"
        ),
    )
    parser.add_argument(
        "--retrieved-dir",
        type=Path,
        default=None,
        help="Directory containing <split>/retrieved_docs.json",
    )
    parser.add_argument(
        "--run-label",
        default=os.getenv("RFG_RUN_LABEL", DEFAULT_RUN_LABEL),
        help="Subfolder under datasets/query_expansion/<dataset>/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write expanded_queries.json",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Query split (default: test)",
    )
    parser.add_argument(
        "--expansion-k",
        type=int,
        nargs="+",
        default=[int(value) for value in os.getenv("RFG_EXPANSION_K", "1 3 5 7 10").split()],
        metavar="K",
        help=(
            "One or more top-k values for expansion. Each k generates expanded queries "
            f"using the first k retrieved documents (default: {' '.join(map(str, DEFAULT_EXPANSION_K_VALUES))})."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Deprecated alias for a single --expansion-k value.",
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
        default=int(os.getenv("RFG_EXPANSION_MAX_TOKENS", DEFAULT_EXPANSION_MAX_TOKENS)),
        help=f"Max tokens for long-form expansion (default: {DEFAULT_EXPANSION_MAX_TOKENS})",
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
        default=int(
            os.getenv("RFG_MAX_TOKENS_PER_CHUNK", DEFAULT_MAX_TOKENS_PER_CHUNK)
        ),
        help=f"Max tokens per retrieved document (default: {DEFAULT_MAX_TOKENS_PER_CHUNK})",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=int(os.getenv("GENERATION_MAX_PROMPT_TOKENS", "0")),
        help="Max prompt tokens (0 = auto)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate expanded queries even when output files already exist",
    )
    parser.add_argument(
        "--paper-scoped",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="QASPER only: append -paper-scoped to retrieval-run-label if missing",
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

    paper_scoped = args.paper_scoped
    if paper_scoped is None:
        paper_scoped = args.dataset == "qasper"
    if (
        paper_scoped
        and args.dataset == "qasper"
        and "paper-scoped" not in retrieval_run_label.lower()
    ):
        retrieval_run_label = f"{retrieval_run_label}-paper-scoped"

    expansion_k_values = sorted(set(args.expansion_k))
    if args.top_k is not None:
        expansion_k_values = sorted(set([args.top_k, *expansion_k_values]))

    retrieved_dir = args.retrieved_dir or _default_retrieved_dir(
        args.dataset,
        retrieval_run_label,
        retrieved_root=args.retrieved_root,
    )
    if not (retrieved_dir / args.split / "retrieved_docs.json").is_file():
        raise FileNotFoundError(
            f"Missing stage-1 retrieved docs: {retrieved_dir / args.split / 'retrieved_docs.json'}. "
            "Run stage-1 retrieval first."
        )

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
    )

    print("RFG query expansion generation")
    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    if args.lora_path:
        print(f"lora_path: {args.lora_path}")
    print(f"retrieved_dir: {retrieved_dir}")
    print(f"split: {args.split}")
    print(f"expansion_k: {expansion_k_values}")
    print(f"max_tokens_per_chunk: {args.max_tokens_per_chunk}")
    print(f"max_tokens: {args.max_tokens}")

    base_output_dir = args.output_dir or _default_output_dir(args.dataset, run_label)
    print(f"output_dir: {base_output_dir}")

    generator.warmup()

    max_prompt_tokens = args.max_prompt_tokens
    if max_prompt_tokens <= 0:
        max_prompt_tokens = generator.get_default_max_prompt_tokens()
    print(f"max_prompt_tokens: {max_prompt_tokens}")

    for expansion_k in expansion_k_values:
        output_dir = base_output_dir / f"k{expansion_k}"
        output_path = output_dir / args.split / "expanded_queries.json"
        if output_path.is_file() and not args.force:
            print(f"Skipping k={expansion_k} (exists: {output_path})")
            continue

        print(f">>> generating expansions for k={expansion_k}")
        records = generate_expansions_for_split(
            generator,
            config,
            retrieved_dir=retrieved_dir,
            split=args.split,
            max_context_docs=expansion_k,
            max_prompt_tokens=max_prompt_tokens,
            max_tokens_per_chunk=args.max_tokens_per_chunk,
            batch_size=args.batch_size,
        )

        run_settings = {
            "framework": "RFG",
            "dataset": args.dataset,
            "model": args.model,
            "lora_path": args.lora_path,
            "retrieved_root": args.retrieved_root,
            "retrieval_run_label": retrieval_run_label,
            "retrieved_dir": str(retrieved_dir),
            "run_label": run_label,
            "split": args.split,
            "expansion_k": expansion_k,
            "max_tokens_per_chunk": args.max_tokens_per_chunk,
            "max_prompt_tokens": max_prompt_tokens,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "num_expansions": len(records),
        }
        save_expanded_queries_bundle(
            output_dir,
            {args.split: records},
            run_settings=run_settings,
        )

        print(f"Generated {len(records)} expanded queries for k={expansion_k}")
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
