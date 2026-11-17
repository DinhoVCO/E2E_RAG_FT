"""Run MTEB retrieval experiments with gold document title in queries.

Each experiment evaluates one embedding setup (base or LoRA) on one project RAG
dataset. Queries are formatted like retrieve_rag_top_k_inmemory_title.py
(Instruct + ## Title: + Query:). Experiments run as separate subprocesses so
vLLM reloads cleanly.

Usage:
    # List experiments
    python scripts/evaluation/mteb/run_mteb_experiment_title.py --list

    # One experiment
    python scripts/evaluation/mteb/run_mteb_experiment_title.py \\
        --experiment telco-dpr-emb-lora-title

    # All experiments for one dataset
    python scripts/evaluation/mteb/run_mteb_experiment_title.py --dataset telco-dpr

    # All 8 title experiments
    python scripts/evaluation/mteb/run_mteb_experiment_title.py --all

    # Dry run
    python scripts/evaluation/mteb/run_mteb_experiment_title.py \\
        --experiment qasper-emb-base-title --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.evaluation.mteb.experiment_config import (
    ResolvedMtebExperiment,
    default_experiments_title_path,
    list_experiment_ids,
    load_experiments_yaml,
    resolve_experiment,
    resolve_experiments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MTEB_SCRIPT = PROJECT_ROOT / "scripts" / "evaluation" / "mteb" / "run_mteb_retrieval.py"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MTEB retrieval experiments with query title from YAML.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_experiments_title_path(),
        help=(
            "Path to experiments YAML "
            "(default: scripts/evaluation/mteb/configs/experiments_title.yaml)"
        ),
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
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing",
    )
    return parser


def _format_lora(value: str | None) -> str:
    return value or "(base)"


def _print_experiment_summary(experiment: ResolvedMtebExperiment) -> None:
    print(f"experiment: {experiment.experiment_id}")
    print(f"dataset: {experiment.dataset}")
    print(f"embedding: {experiment.embedding_mode}")
    print(f"backend: {experiment.backend}")
    print(f"model: {experiment.model}")
    print(f"lora_path: {_format_lora(experiment.lora_path)}")
    print(f"model_revision: {experiment.model_revision}")
    print(f"batch_size: {experiment.batch_size}")
    print(f"splits: {', '.join(experiment.splits)}")
    print(f"include_query_title: {experiment.include_query_title}")
    if experiment.paper_scoped is True:
        print("paper_scoped: true")
    elif experiment.paper_scoped is False:
        print("paper_scoped: false")
    print(f"overwrite: {experiment.overwrite}")
    if experiment.output_dir is not None:
        print(f"output_dir: {experiment.output_dir}")


def _list_experiments(raw: dict) -> None:
    for experiment_id in list_experiment_ids(raw):
        experiment = resolve_experiment(raw, experiment_id)
        print(
            f"{experiment_id}\t"
            f"dataset={experiment.dataset}\t"
            f"emb={experiment.embedding_mode}\t"
            f"revision={experiment.model_revision}\t"
            f"title={experiment.include_query_title}\t"
            f"lora={_format_lora(experiment.lora_path)}"
        )


def _mteb_command(experiment: ResolvedMtebExperiment) -> list[str]:
    command = [
        sys.executable,
        str(MTEB_SCRIPT),
        "--dataset",
        experiment.dataset,
        "--backend",
        experiment.backend,
        "--model",
        experiment.model,
        "--model-revision",
        experiment.model_revision,
        "--batch-size",
        str(experiment.batch_size),
        "--splits",
        *experiment.splits,
        "--overwrite",
        experiment.overwrite,
    ]
    if experiment.lora_path:
        command.extend(
            [
                "--lora-path",
                experiment.lora_path,
                "--max-lora-rank",
                str(experiment.max_lora_rank),
            ]
        )
    if experiment.paper_scoped is True:
        command.append("--paper-scoped")
    elif experiment.paper_scoped is False:
        command.append("--full-corpus")
    if experiment.include_query_title:
        command.append("--include-query-title")
    if experiment.output_dir is not None:
        command.extend(["--output-dir", str(experiment.output_dir)])
    return command


def _run_command(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"$ {printable}")
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _run_experiment(experiment: ResolvedMtebExperiment, *, dry_run: bool) -> None:
    print("=" * 72)
    _print_experiment_summary(experiment)
    print("=" * 72)
    _run_command(_mteb_command(experiment), dry_run=dry_run)
    print()


def _select_experiments(args: argparse.Namespace, raw: dict) -> list[ResolvedMtebExperiment]:
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
        _run_experiment(experiment, dry_run=args.dry_run)

    print(f"Completed {len(experiments)} experiment(s).")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
