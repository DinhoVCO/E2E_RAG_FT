"""Run a RAG experiment defined in YAML (retrieval + generation).

Each experiment configures embedding model, generation model, dataset, run label,
and top-k. Retrieval and generation run as separate subprocesses so vLLM reloads
cleanly between the embedding and generation models.

Usage:
    # List experiments
    python scripts/generation/run_rag_experiment.py --list

    # One experiment
    python scripts/generation/run_rag_experiment.py \\
        --experiment telco-dpr-emb-lora-gen-lora-top5

    # All experiments for one dataset
    python scripts/generation/run_rag_experiment.py --dataset telco-dpr

    # All 16 experiments
    python scripts/generation/run_rag_experiment.py --all

    # Skip retrieval if retrieved_docs already exist
    python scripts/generation/run_rag_experiment.py \\
        --experiment telco-dpr-emb-base-gen-lora-top5 --skip-retrieval
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.generation.experiment_config import (
    ResolvedExperiment,
    default_experiments_path,
    list_experiment_ids,
    load_experiments_yaml,
    resolve_experiment,
    resolve_experiments,
    retrieved_docs_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_SCRIPT = PROJECT_ROOT / "scripts" / "retrieval" / "retrieve_rag_top_k_inmemory.py"
GENERATION_SCRIPT = PROJECT_ROOT / "scripts" / "generation" / "run_rag_generation.py"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAG experiments from YAML (retrieval + generation).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_experiments_path(),
        help="Path to experiments YAML (default: scripts/generation/configs/experiments.yaml)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        metavar="ID",
        help="Experiment id from YAML (repeatable)",
    )
    group.add_argument(
        "--dataset",
        help="Run all experiments for this dataset",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run every experiment in the YAML",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List experiment ids and exit",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip retrieval even if retrieved_docs are missing",
    )
    parser.add_argument(
        "--force-retrieval",
        action="store_true",
        help="Always re-run retrieval before generation",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run retrieval step only",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Run generation step only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing",
    )
    return parser


def _format_lora(value: str | None) -> str:
    return value or "(base)"


def _print_experiment_summary(experiment: ResolvedExperiment) -> None:
    print(f"experiment: {experiment.experiment_id}")
    print(f"dataset: {experiment.dataset}")
    print(f"run_label: {experiment.run_label}")
    print(f"retrieval_run_label: {experiment.retrieval_run_label}")
    print(f"embedding_model: {experiment.embedding_model}")
    print(f"embedding_lora: {_format_lora(experiment.embedding_lora)}")
    print(f"generation_model: {experiment.generation_model}")
    print(f"generation_lora: {_format_lora(experiment.generation_lora)}")
    print(f"top_k: {experiment.top_k}")
    print(f"retrieval_top_k: {experiment.retrieval_top_k}")
    print(f"split: {experiment.split}")
    if experiment.dataset == "qasper":
        print(f"paper_scoped: {experiment.paper_scoped}")


def _list_experiments(raw: dict) -> None:
    for experiment_id in list_experiment_ids(raw):
        experiment = resolve_experiment(raw, experiment_id)
        print(
            f"{experiment_id}\t"
            f"dataset={experiment.dataset}\t"
            f"emb={'lora' if experiment.embedding_lora else 'base'}\t"
            f"gen={'lora' if experiment.generation_lora else 'base'}\t"
            f"retrieval={experiment.retrieval_run_label}\t"
            f"run={experiment.run_label}"
        )


def _retrieval_command(experiment: ResolvedExperiment) -> list[str]:
    command = [
        sys.executable,
        str(RETRIEVAL_SCRIPT),
        "--dataset",
        experiment.dataset,
        "--mode",
        "offline",
        "--model",
        experiment.embedding_model,
        "--top-k",
        str(experiment.retrieval_top_k),
        "--batch-size",
        str(experiment.retrieval_batch_size),
        "--run-label",
        experiment.retrieval_run_label.removesuffix("-paper-scoped")
        if experiment.paper_scoped
        else experiment.retrieval_run_label,
        "--splits",
        experiment.split,
    ]
    if experiment.embedding_lora:
        command.extend(["--lora-path", experiment.embedding_lora])
    if experiment.dataset == "qasper":
        command.append("--paper-scoped" if experiment.paper_scoped else "--no-paper-scoped")
    return command


def _generation_command(experiment: ResolvedExperiment) -> list[str]:
    command = [
        sys.executable,
        str(GENERATION_SCRIPT),
        "--dataset",
        experiment.dataset,
        "--model",
        experiment.generation_model,
        "--retrieval-run-label",
        experiment.retrieval_run_label,
        "--run-label",
        experiment.run_label,
        "--split",
        experiment.split,
        "--top-k",
        str(experiment.top_k),
        "--batch-size",
        str(experiment.generation_batch_size),
    ]
    if experiment.generation_lora:
        command.extend(["--lora-path", experiment.generation_lora])
    return command


def _needs_retrieval(
    experiment: ResolvedExperiment,
    *,
    force_retrieval: bool,
    skip_retrieval: bool,
) -> bool:
    if skip_retrieval:
        return False
    if force_retrieval:
        return True
    path = retrieved_docs_path(
        project_root=PROJECT_ROOT,
        dataset=experiment.dataset,
        retrieval_run_label=experiment.retrieval_run_label,
        split=experiment.split,
    )
    return not path.is_file()


def _run_command(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"$ {printable}")
    if dry_run:
        return
    env = os.environ.copy()
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _run_experiment(
    experiment: ResolvedExperiment,
    *,
    skip_retrieval: bool,
    force_retrieval: bool,
    retrieval_only: bool,
    generation_only: bool,
    dry_run: bool,
) -> None:
    print("=" * 72)
    _print_experiment_summary(experiment)
    print("=" * 72)

    retrieved_path = retrieved_docs_path(
        project_root=PROJECT_ROOT,
        dataset=experiment.dataset,
        retrieval_run_label=experiment.retrieval_run_label,
        split=experiment.split,
    )

    run_retrieval = not generation_only and _needs_retrieval(
        experiment,
        force_retrieval=force_retrieval,
        skip_retrieval=skip_retrieval,
    )
    run_generation = not retrieval_only

    if run_retrieval:
        print(">>> retrieval")
        _run_command(_retrieval_command(experiment), dry_run=dry_run)
    else:
        print(f">>> skipping retrieval (using {retrieved_path})")

    if run_generation:
        if not run_retrieval and not retrieved_path.is_file():
            raise FileNotFoundError(
                f"Missing retrieved docs for generation: {retrieved_path}. "
                "Run retrieval first or drop --skip-retrieval."
            )
        print(">>> generation")
        _run_command(_generation_command(experiment), dry_run=dry_run)
    print()


def _select_experiments(args: argparse.Namespace, raw: dict) -> list[ResolvedExperiment]:
    if args.experiments:
        return resolve_experiments(raw, experiment_ids=args.experiments)
    if args.dataset:
        return resolve_experiments(raw, dataset=args.dataset)
    if args.all:
        return resolve_experiments(raw)
    raise SystemExit(
        "Specify --experiment, --dataset, or --all. Use --list to see available experiments."
    )


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    raw = load_experiments_yaml(args.config)

    if args.list:
        _list_experiments(raw)
        return

    experiments = _select_experiments(args, raw)
    print(f"config: {args.config}")
    print(f"experiments_to_run: {len(experiments)}")
    print()

    for experiment in experiments:
        _run_experiment(
            experiment,
            skip_retrieval=args.skip_retrieval,
            force_retrieval=args.force_retrieval,
            retrieval_only=args.retrieval_only,
            generation_only=args.generation_only,
            dry_run=args.dry_run,
        )

    print(f"Completed {len(experiments)} experiment(s).")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
