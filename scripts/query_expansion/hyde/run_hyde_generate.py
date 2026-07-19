"""Generate HyDE pseudo-passages with vLLM offline (n completions per query).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/hyde/run_hyde_generate.py \\
        --dataset bioasq-resplit \\
        --run-label bioasq-resplit-hyde

    CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/hyde/run_hyde_generate.py \\
        --dataset qasper \\
        --num-passages 8 \\
        --temperature 0.7 \\
        --max-tokens 512
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
from tesis_unicamp.query_expansion.hyde.generation import (
    DEFAULT_HYDE_MAX_TOKENS,
    DEFAULT_HYDE_TEMPERATURE,
    DEFAULT_NUM_PASSAGES,
    generate_hyde_passages_for_split,
)
from tesis_unicamp.query_expansion.hyde.io import save_hyde_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_RUN_LABEL = "hyde-default"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "query_expansion" / "hyde"
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
    parser = argparse.ArgumentParser(description="Generate HyDE pseudo-passages with vLLM offline.")
    parser.add_argument(
        "--dataset",
        choices=("bioasq-resplit", "qasper", "telco-dpr", "narrativeqa"),
        required=True,
    )
    parser.add_argument("--model", default=os.getenv("VLLM_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--lora-path", default=os.getenv("LORA_PATH"))
    parser.add_argument("--max-lora-rank", type=int, default=int(os.getenv("MAX_LORA_RANK", "16")))
    parser.add_argument("--run-label", default=os.getenv("HYDE_RUN_LABEL", DEFAULT_RUN_LABEL))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--num-passages",
        type=int,
        default=int(os.getenv("HYDE_NUM_PASSAGES", DEFAULT_NUM_PASSAGES)),
        help=f"Completions per query (default: {DEFAULT_NUM_PASSAGES})",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("GENERATION_BATCH_SIZE", DEFAULT_GENERATION_BATCH_SIZE)))
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("HYDE_MAX_TOKENS", DEFAULT_HYDE_MAX_TOKENS)),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("HYDE_TEMPERATURE", DEFAULT_HYDE_TEMPERATURE)),
    )
    parser.add_argument("--top-p", type=float, default=float(os.getenv("HYDE_TOP_P", "1.0")))
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the model chat template (default: off, plain HyDE prompt)",
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
            n=args.num_passages,
            top_p=args.top_p,
            stop=("\n\n\n",),
        ),
        lora_path=args.lora_path,
        max_lora_rank=args.max_lora_rank if args.lora_path else None,
        use_chat_template=args.use_chat_template,
    )

    config = get_rag_generation_config(args.dataset)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.dataset / run_label)

    print("HyDE pseudo-passage generation")
    print(f"dataset: {args.dataset}")
    print(f"model: {args.model}")
    print(f"num_passages: {args.num_passages}")
    print(f"temperature: {args.temperature}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"output_dir: {output_dir}")

    generator.warmup()
    records = generate_hyde_passages_for_split(
        generator,
        config,
        split=args.split,
        num_passages=args.num_passages,
        batch_size=args.batch_size,
    )

    save_hyde_bundle(
        output_dir,
        {args.split: records},
        run_settings={
            "framework": "HyDE",
            "dataset": args.dataset,
            "model": args.model,
            "lora_path": args.lora_path,
            "run_label": run_label,
            "split": args.split,
            "num_passages": args.num_passages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_chat_template": args.use_chat_template,
            "num_queries": len(records),
        },
    )
    print(f"Generated HyDE passages for {len(records)} queries")
    print(f"Saved to {output_dir / args.split / 'hyde_passages.json'}")


if __name__ == "__main__":
    main(sys.argv[1:])
