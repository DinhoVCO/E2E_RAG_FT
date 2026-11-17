"""Run HyDE experiments from YAML (generation + MTEB).

Usage:
    python scripts/query_expansion/hyde/run_hyde_experiment.py --list
    python scripts/query_expansion/hyde/run_hyde_experiment.py --experiment bioasq-resplit-hyde
    python scripts/query_expansion/hyde/run_hyde_experiment.py --all
    python scripts/query_expansion/hyde/run_hyde_experiment.py --experiment qasper-hyde --generation-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.query_expansion.hyde.experiment_config import (
    ResolvedHydeExperiment,
    default_experiments_path,
    hyde_passages_path,
    list_experiment_ids,
    load_experiments_yaml,
    resolve_experiment,
    resolve_experiments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GENERATE_SCRIPT = PROJECT_ROOT / "scripts" / "query_expansion" / "hyde" / "run_hyde_generate.py"
MTEB_SCRIPT = PROJECT_ROOT / "scripts" / "query_expansion" / "hyde" / "run_hyde_mteb.py"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HyDE experiments from YAML.")
    parser.add_argument("--config", type=Path, default=default_experiments_path())
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--experiment", action="append", dest="experiments", metavar="ID")
    group.add_argument("--dataset")
    group.add_argument("--all", action="store_true")
    group.add_argument("--list", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--skip-mteb", action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--mteb-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _print_summary(experiment: ResolvedHydeExperiment) -> None:
    print(f"experiment: {experiment.experiment_id}")
    print(f"dataset: {experiment.dataset}")
    print(f"run_label: {experiment.run_label}")
    print(f"generation_model: {experiment.generation_model}")
    print(f"embedding_model: {experiment.embedding_model}")
    print(f"num_passages: {experiment.num_passages}")
    print(f"model_revision: {experiment.model_revision}")


def _generation_command(experiment: ResolvedHydeExperiment) -> list[str]:
    command = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--dataset",
        experiment.dataset,
        "--model",
        experiment.generation_model,
        "--run-label",
        experiment.run_label,
        "--split",
        experiment.split,
        "--num-passages",
        str(experiment.num_passages),
        "--batch-size",
        str(experiment.generation_batch_size),
        "--max-tokens",
        str(experiment.generation_max_tokens),
        "--temperature",
        str(experiment.temperature),
        "--top-p",
        str(experiment.top_p),
        "--no-use-chat-template",
    ]
    if experiment.generation_lora:
        command.extend(["--lora-path", experiment.generation_lora])
    return command


def _mteb_command(experiment: ResolvedHydeExperiment) -> list[str]:
    hyde_dir = PROJECT_ROOT / "datasets" / "query_expansion" / "hyde" / experiment.dataset / experiment.run_label
    command = [
        sys.executable,
        str(MTEB_SCRIPT),
        "--dataset",
        experiment.dataset,
        "--hyde-dir",
        str(hyde_dir),
        "--model",
        experiment.embedding_model,
        "--model-revision",
        experiment.model_revision,
        "--batch-size",
        str(experiment.retrieval_batch_size),
        "--splits",
        experiment.split,
        "--num-passages",
        str(experiment.num_passages),
    ]
    if experiment.embedding_lora:
        command.extend(["--lora-path", experiment.embedding_lora])
    if experiment.dataset == "qasper":
        command.append("--paper-scoped" if experiment.paper_scoped else "--full-corpus")
    if experiment.output_dir is not None:
        command.extend(["--output-dir", str(experiment.output_dir)])
    return command


def _needs_generation(experiment: ResolvedHydeExperiment, *, force: bool, skip: bool) -> bool:
    if skip:
        return False
    if force:
        return True
    path = hyde_passages_path(
        project_root=PROJECT_ROOT,
        dataset=experiment.dataset,
        run_label=experiment.run_label,
        split=experiment.split,
    )
    return not path.is_file()


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print(f"$ {' '.join(command)}")
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _run_experiment(
    experiment: ResolvedHydeExperiment,
    *,
    skip_generation: bool,
    force_generation: bool,
    skip_mteb: bool,
    generation_only: bool,
    mteb_only: bool,
    dry_run: bool,
) -> None:
    print("=" * 72)
    _print_summary(experiment)
    print("=" * 72)

    hyde_path = hyde_passages_path(
        project_root=PROJECT_ROOT,
        dataset=experiment.dataset,
        run_label=experiment.run_label,
        split=experiment.split,
    )

    run_generation = (
        not mteb_only
        and _needs_generation(experiment, force=force_generation, skip=skip_generation)
    )
    run_mteb = not generation_only and not skip_mteb

    if run_generation:
        print(">>> HyDE generation (n pseudo-passages per query)")
        _run_command(_generation_command(experiment), dry_run=dry_run)
    elif not mteb_only:
        print(f">>> skipping generation (using {hyde_path})")

    if run_mteb:
        if not run_generation and not hyde_path.is_file() and not dry_run:
            raise FileNotFoundError(f"Missing HyDE passages: {hyde_path}")
        print(">>> HyDE MTEB evaluation")
        _run_command(_mteb_command(experiment), dry_run=dry_run)
    elif not generation_only:
        print(">>> skipping MTEB")

    print()


def main(argv: list[str] | None = None) -> None:
    _load_env()
    args = _build_parser().parse_args(argv)
    raw = load_experiments_yaml(args.config)

    if args.list:
        for experiment_id in list_experiment_ids(raw):
            experiment = resolve_experiment(raw, experiment_id)
            print(f"{experiment_id}\tdataset={experiment.dataset}\trun={experiment.run_label}")
        return

    if args.experiments:
        experiments = resolve_experiments(raw, experiment_ids=args.experiments)
    elif args.dataset:
        experiments = resolve_experiments(raw, dataset=args.dataset)
    elif args.all:
        experiments = resolve_experiments(raw)
    else:
        raise SystemExit("Specify --experiment, --dataset, or --all.")

    for experiment in experiments:
        _run_experiment(
            experiment,
            skip_generation=args.skip_generation,
            force_generation=args.force_generation,
            skip_mteb=args.skip_mteb,
            generation_only=args.generation_only,
            mteb_only=args.mteb_only,
            dry_run=args.dry_run,
        )

    print(f"Completed {len(experiments)} HyDE experiment(s).")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
