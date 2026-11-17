"""Generate Query2Doc expansions with vLLM offline (4 few-shot train examples).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/q2d/run_q2d_generate.py \\
        --dataset bioasq-resplit \\
        --run-label bioasq-resplit-q2d

    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/q2d/run_q2d_generate.py \\
        --dataset qasper \\
        --num-few-shot 4 \\
        --seed 42
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
from tesis_unicamp.query_expansion.q2d.few_shot import (
    DEFAULT_FEW_SHOT_SPLIT,
    DEFAULT_MAX_PASSAGE_TOKENS,
    DEFAULT_NUM_FEW_SHOT,
    build_few_shot_pool,
)
from tesis_unicamp.query_expansion.q2d.generation import (
    DEFAULT_Q2D_MAX_TOKENS,
    DEFAULT_Q2D_TEMPERATURE,
    generate_q2d_for_split,
)
from tesis_unicamp.query_expansion.q2d.io import save_q2d_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_RUN_LABEL = "q2d-default"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "query_expansion" / "q2d"
DEFAULT_SPLIT = "test"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip())
    return sanitized or DEFAULT_RUN_LABEL


def _validate_cuda() -> None:
    import torch

    if torch.cuda.device_count() == 0:
        raise SystemExit("No CUDA device visible.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Query2Doc expansions with vLLM offline.")
    parser.add_argument(
        "--dataset",
        choices=("bioasq-resplit", "qasper", "telco-dpr", "narrativeqa"),
        required=True,
    )
    parser.add_argument("--model", default=os.getenv("VLLM_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--lora-path", default=os.getenv("LORA_PATH"))
    parser.add_argument("--max-lora-rank", type=int, default=int(os.getenv("MAX_LORA_RANK", "16")))
    parser.add_argument("--run-label", default=os.getenv("Q2D_RUN_LABEL", DEFAULT_RUN_LABEL))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--num-few-shot",
        type=int,
        default=int(os.getenv("Q2D_NUM_FEW_SHOT", DEFAULT_NUM_FEW_SHOT)),
        help=f"Few-shot train examples per query (default: {DEFAULT_NUM_FEW_SHOT})",
    )
    parser.add_argument("--few-shot-split", default=os.getenv("Q2D_FEW_SHOT_SPLIT", DEFAULT_FEW_SHOT_SPLIT))
    parser.add_argument(
        "--max-few-shot-passage-tokens",
        type=int,
        default=int(os.getenv("Q2D_MAX_FEW_SHOT_PASSAGE_TOKENS", DEFAULT_MAX_PASSAGE_TOKENS)),
        help=f"Truncate each few-shot passage to this many tokens (default: {DEFAULT_MAX_PASSAGE_TOKENS})",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("Q2D_SEED", "42")))
    parser.add_argument(
        "--fixed-few-shot",
        action="store_true",
        help="Use the same 4 train examples for every query (default: sample per query)",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("GENERATION_BATCH_SIZE", DEFAULT_GENERATION_BATCH_SIZE)))
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("Q2D_MAX_TOKENS", DEFAULT_Q2D_MAX_TOKENS)),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("Q2D_TEMPERATURE", DEFAULT_Q2D_TEMPERATURE)),
    )
    parser.add_argument("--top-p", type=float, default=float(os.getenv("Q2D_TOP_P", "1.0")))
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the model chat template (default: off, plain Query2Doc prompt)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    run_label = _sanitize_label(args.run_label)

    _validate_cuda()
    configure_vllm_multiprocessing()

    generator = VLLMOfflineGenerator(
        GenerationConfig(
            model=args.model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=("\n\nQuery:", "\nQuery:"),
        ),
        lora_path=args.lora_path,
        max_lora_rank=args.max_lora_rank if args.lora_path else None,
        use_chat_template=args.use_chat_template,
    )

    config = get_rag_generation_config(args.dataset)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.dataset / run_label)

    print("Query2Doc expansion generation")
    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    print(f"num_few_shot: {args.num_few_shot}")
    print(f"few_shot_split: {args.few_shot_split}")
    print(f"max_few_shot_passage_tokens: {args.max_few_shot_passage_tokens}")
    print(f"per_query_sampling: {not args.fixed_few_shot}")
    print(f"seed: {args.seed}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"temperature: {args.temperature}")
    print(f"output_dir: {output_dir}")

    generator.warmup()
    few_shot_pool = build_few_shot_pool(
        config,
        split=args.few_shot_split,
        max_passage_tokens=args.max_few_shot_passage_tokens,
        truncate_text_to_tokens=generator.truncate_text_to_tokens,
    )
    print(f"few_shot_pool_size: {len(few_shot_pool)}")
    if len(few_shot_pool) < args.num_few_shot:
        raise SystemExit(
            f"Need at least {args.num_few_shot} train examples, found {len(few_shot_pool)}."
        )

    records = generate_q2d_for_split(
        generator,
        config,
        split=args.split,
        few_shot_pool=few_shot_pool,
        num_few_shot=args.num_few_shot,
        few_shot_split=args.few_shot_split,
        max_passage_tokens=args.max_few_shot_passage_tokens,
        seed=args.seed,
        per_query_sampling=not args.fixed_few_shot,
        batch_size=args.batch_size,
    )

    save_q2d_bundle(
        output_dir,
        {args.split: records},
        run_settings={
            "framework": "Query2Doc",
            "dataset": args.dataset,
            "model": args.model,
            "lora_path": args.lora_path,
            "run_label": run_label,
            "split": args.split,
            "num_few_shot": args.num_few_shot,
            "few_shot_split": args.few_shot_split,
            "max_few_shot_passage_tokens": args.max_few_shot_passage_tokens,
            "seed": args.seed,
            "per_query_sampling": not args.fixed_few_shot,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_chat_template": args.use_chat_template,
            "few_shot_pool_size": len(few_shot_pool),
            "num_queries": len(records),
        },
    )
    print(f"Generated Query2Doc expansions for {len(records)} queries")
    print(f"Saved to {output_dir / args.split / 'q2d_expansions.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
