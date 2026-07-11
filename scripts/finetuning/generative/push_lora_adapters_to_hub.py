"""Upload Qwen3-8B LoRA adapters to Hugging Face Hub.

Usage:
    # Upload one dataset (auto-detect newest local run):
    python scripts/finetuning/generative/push_lora_adapters_to_hub.py --dataset telco-dpr

    # Upload a specific run directory:
    python scripts/finetuning/generative/push_lora_adapters_to_hub.py \
        --dataset telco-dpr \
        --run-dir models/qwen3-8b-lora/telco-dpr-b4-e6

    # Upload all four datasets:
    python scripts/finetuning/generative/push_lora_adapters_to_hub.py --all

    # Dry run:
    python scripts/finetuning/generative/push_lora_adapters_to_hub.py --dataset qasper --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.finetuning.generative.config import GENERATIVE_FINETUNING_DATASET_IDS
from tesis_unicamp.finetuning.generative.hub_upload import (
    AdapterUploadSpec,
    default_hub_repo_id,
    find_adapter_dir,
    resolve_hf_token,
    resolve_hub_username,
    upload_adapter_to_hub,
    verify_hub_access,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "qwen3-8b-lora"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload Qwen3-8B LoRA adapters to Hugging Face Hub.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        choices=sorted(GENERATIVE_FINETUNING_DATASET_IDS),
        help="Dataset whose LoRA adapter to upload.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Upload adapters for all supported datasets.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Local training run directory containing final/ (default: newest match under output-root).",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Path to adapter directory (e.g. .../final). Overrides --run-dir.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory with fine-tuned runs (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hub model repo (default: DinoStackAI/Qwen3-8b-lora-<dataset>).",
    )
    parser.add_argument(
        "--hub-user",
        default=None,
        help="Hugging Face org or username (default: HF_USERNAME/HF_ORG from .env, else DinoStackAI).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (default: HF_TOKEN from .env).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the model repo as private.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message for the Hub upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print upload plan without pushing to the Hub.",
    )
    return parser


def _upload_one(
    *,
    dataset: str,
    args: argparse.Namespace,
    token: str,
    hub_user: str,
) -> str:
    adapter_dir, run_dir = find_adapter_dir(
        dataset=dataset,
        output_root=args.output_root,
        run_dir=args.run_dir,
        adapter_dir=args.adapter_dir,
    )
    repo_id = args.repo_id or default_hub_repo_id(hub_user=hub_user, dataset=dataset)

    print(f"dataset: {dataset}")
    print(f"repo_id: {repo_id}")
    print(f"adapter_dir: {adapter_dir}")
    if run_dir is not None:
        print(f"run_dir: {run_dir}")

    if not args.dry_run:
        verify_hub_access(token=token, repo_id=repo_id)

    spec = AdapterUploadSpec(
        dataset=dataset,
        adapter_dir=adapter_dir,
        repo_id=repo_id,
        run_dir=run_dir,
    )
    uploaded_repo = upload_adapter_to_hub(
        spec,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"[dry-run] would upload to https://huggingface.co/{uploaded_repo}")
    else:
        print(f"Uploaded adapter to https://huggingface.co/{uploaded_repo}")
    return uploaded_repo


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    token = resolve_hf_token(args.token)
    hub_user = resolve_hub_username(token=token, hub_user=args.hub_user)

    datasets = sorted(GENERATIVE_FINETUNING_DATASET_IDS) if args.all else [args.dataset]
    if args.all and (args.run_dir is not None or args.adapter_dir is not None):
        raise SystemExit("--run-dir and --adapter-dir cannot be used with --all.")
    if args.all and args.repo_id is not None:
        raise SystemExit("--repo-id cannot be used with --all.")

    print(f"hub_user: {hub_user}")
    uploaded: list[str] = []
    for dataset in datasets:
        uploaded.append(
            _upload_one(
                dataset=dataset,
                args=args,
                token=token,
                hub_user=hub_user,
            )
        )
        print()

    if len(uploaded) > 1:
        print("Uploaded repos:")
        for repo_id in uploaded:
            print(f"  - https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
