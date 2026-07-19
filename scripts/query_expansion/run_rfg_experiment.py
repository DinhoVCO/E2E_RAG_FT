"""Run RFG (Retrieval-Feedback-Grounded Multi-Query Expansion) experiments.

Each experiment runs three stages as separate subprocesses so vLLM reloads
cleanly between embedding and generation models:

  1. Stage 1 retrieval  — embed corpus, retrieve top-5 docs (embedding model)
  2. Query expansion    — generate long-form answer from top-5 (qwen3-8B)
  3. Stage 2 MTEB       — embed expanded query, evaluate retrieval (embedding model)

Usage:
    # List experiments
    python scripts/query_expansion/run_rfg_experiment.py --list

    # One experiment
    python scripts/query_expansion/run_rfg_experiment.py \\
        --experiment telco-dpr-rfg-emb-base-gen-lora

    # All experiments for one dataset
    python scripts/query_expansion/run_rfg_experiment.py --dataset telco-dpr

    # All 16 experiments
    python scripts/query_expansion/run_rfg_experiment.py --all

    # Skip stage 1 if retrieved_docs already exist
    python scripts/query_expansion/run_rfg_experiment.py \\
        --experiment qasper-rfg-emb-lora-gen-lora --skip-stage1

    # Run only expansion (stage 2 skipped)
    python scripts/query_expansion/run_rfg_experiment.py \\
        --experiment qasper-rfg-emb-base-gen-base --expansion-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from tesis_unicamp.query_expansion.experiment_config import (
    ResolvedRfgExperiment,
    default_experiments_path,
    expanded_queries_path,
    list_experiment_ids,
    load_experiments_yaml,
    resolve_experiment,
    resolve_experiments,
    stage1_retrieved_docs_path,
    stage2_model_revision_for_k,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_SCRIPT = PROJECT_ROOT / "scripts" / "query_expansion" / "run_rfg_stage1_retrieval.py"
EXPANSION_SCRIPT = PROJECT_ROOT / "scripts" / "query_expansion" / "run_rfg_generate_expansion.py"
STAGE2_SCRIPT = PROJECT_ROOT / "scripts" / "query_expansion" / "run_rfg_stage2_mteb.py"


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RFG query expansion experiments from YAML.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_experiments_path(),
        help="Path to experiments YAML",
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
        "--skip-stage1",
        action="store_true",
        help="Skip stage-1 retrieval if retrieved_docs exist",
    )
    parser.add_argument(
        "--force-stage1",
        action="store_true",
        help="Always re-run stage-1 retrieval",
    )
    parser.add_argument(
        "--skip-expansion",
        action="store_true",
        help="Skip query expansion if expanded_queries exist",
    )
    parser.add_argument(
        "--force-expansion",
        action="store_true",
        help="Always re-run query expansion",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip stage-2 MTEB evaluation",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run stage-1 retrieval only",
    )
    parser.add_argument(
        "--expansion-only",
        action="store_true",
        help="Run query expansion only (requires existing stage-1 results)",
    )
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Run stage-2 MTEB only (requires existing expanded queries)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing",
    )
    return parser


def _format_lora(value: str | None) -> str:
    return value or "(base)"


def _print_experiment_summary(experiment: ResolvedRfgExperiment) -> None:
    print(f"experiment: {experiment.experiment_id}")
    print(f"dataset: {experiment.dataset}")
    print(f"run_label: {experiment.run_label}")
    print(f"retrieval_run_label: {experiment.retrieval_run_label}")
    print(f"retrieved_root: {experiment.retrieved_root}")
    print(f"embedding: {experiment.embedding_mode} ({experiment.embedding_model})")
    print(f"embedding_lora: {_format_lora(experiment.embedding_lora)}")
    print(f"generation: {experiment.generation_mode} ({experiment.generation_model})")
    print(f"generation_lora: {_format_lora(experiment.generation_lora)}")
    print(f"retrieval_top_k: {experiment.retrieval_top_k}")
    print(f"expansion_k: {list(experiment.expansion_k_values)}")
    print(f"max_tokens_per_chunk: {experiment.max_tokens_per_chunk}")
    print(f"generation_max_tokens: {experiment.generation_max_tokens}")
    print(f"split: {experiment.split}")
    if experiment.dataset == "qasper":
        print(f"paper_scoped: {experiment.paper_scoped}")
    print(f"stage2_model_revision_template: {experiment.stage2_model_revision}")


def _list_experiments(raw: dict) -> None:
    for experiment_id in list_experiment_ids(raw):
        experiment = resolve_experiment(raw, experiment_id)
        print(
            f"{experiment_id}\t"
            f"dataset={experiment.dataset}\t"
            f"emb={experiment.embedding_mode}\t"
            f"gen={experiment.generation_mode}\t"
            f"retrieval={experiment.retrieval_run_label}\t"
            f"run={experiment.run_label}"
        )


def _stage1_command(experiment: ResolvedRfgExperiment) -> list[str]:
    command = [
        sys.executable,
        str(STAGE1_SCRIPT),
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


def _expansion_command(experiment: ResolvedRfgExperiment) -> list[str]:
    command = [
        sys.executable,
        str(EXPANSION_SCRIPT),
        "--dataset",
        experiment.dataset,
        "--model",
        experiment.generation_model,
        "--retrieval-run-label",
        experiment.retrieval_run_label,
        "--retrieved-root",
        experiment.retrieved_root,
        "--run-label",
        experiment.run_label,
        "--split",
        experiment.split,
        "--expansion-k",
        *[str(k) for k in experiment.expansion_k_values],
        "--batch-size",
        str(experiment.generation_batch_size),
        "--max-tokens",
        str(experiment.generation_max_tokens),
        "--max-tokens-per-chunk",
        str(experiment.max_tokens_per_chunk),
        "--max-prompt-tokens",
        str(experiment.max_prompt_tokens),
    ]
    if experiment.generation_lora:
        command.extend(["--lora-path", experiment.generation_lora])
    if experiment.dataset == "qasper":
        command.append("--paper-scoped" if experiment.paper_scoped else "--no-paper-scoped")
    return command


def _stage2_command(experiment: ResolvedRfgExperiment, *, expansion_k: int) -> list[str]:
    expanded_dir = (
        PROJECT_ROOT
        / "datasets"
        / "query_expansion"
        / experiment.dataset
        / experiment.run_label
        / f"k{expansion_k}"
    )
    command = [
        sys.executable,
        str(STAGE2_SCRIPT),
        "--dataset",
        experiment.dataset,
        "--expanded-queries-dir",
        str(expanded_dir),
        "--model",
        experiment.embedding_model,
        "--model-revision",
        stage2_model_revision_for_k(experiment, expansion_k),
        "--batch-size",
        str(experiment.retrieval_batch_size),
        "--splits",
        experiment.split,
    ]
    if experiment.embedding_lora:
        command.extend(["--lora-path", experiment.embedding_lora])
    if experiment.dataset == "qasper":
        command.append("--paper-scoped" if experiment.paper_scoped else "--full-corpus")
    if experiment.stage2_output_dir is not None:
        command.extend(["--output-dir", str(experiment.stage2_output_dir)])
    return command


def _needs_stage1(
    experiment: ResolvedRfgExperiment,
    *,
    force: bool,
    skip: bool,
) -> bool:
    if skip:
        return False
    if force:
        return True
    path = stage1_retrieved_docs_path(
        project_root=PROJECT_ROOT,
        dataset=experiment.dataset,
        retrieval_run_label=experiment.retrieval_run_label,
        split=experiment.split,
        retrieved_root=experiment.retrieved_root,
    )
    return not path.is_file()


def _needs_expansion(
    experiment: ResolvedRfgExperiment,
    *,
    force: bool,
    skip: bool,
) -> bool:
    if skip:
        return False
    if force:
        return True
    for expansion_k in experiment.expansion_k_values:
        path = expanded_queries_path(
            project_root=PROJECT_ROOT,
            dataset=experiment.dataset,
            run_label=experiment.run_label,
            split=experiment.split,
            expansion_k=expansion_k,
        )
        if not path.is_file():
            return True
    return False


def _missing_expansion_k_values(experiment: ResolvedRfgExperiment) -> list[int]:
    missing: list[int] = []
    for expansion_k in experiment.expansion_k_values:
        path = expanded_queries_path(
            project_root=PROJECT_ROOT,
            dataset=experiment.dataset,
            run_label=experiment.run_label,
            split=experiment.split,
            expansion_k=expansion_k,
        )
        if not path.is_file():
            missing.append(expansion_k)
    return missing


def _run_command(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"$ {printable}")
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _run_experiment(
    experiment: ResolvedRfgExperiment,
    *,
    skip_stage1: bool,
    force_stage1: bool,
    skip_expansion: bool,
    force_expansion: bool,
    skip_stage2: bool,
    stage1_only: bool,
    expansion_only: bool,
    stage2_only: bool,
    dry_run: bool,
) -> None:
    print("=" * 72)
    _print_experiment_summary(experiment)
    print("=" * 72)

    run_stage1 = (
        not expansion_only
        and not stage2_only
        and _needs_stage1(experiment, force=force_stage1, skip=skip_stage1)
    )
    run_expansion = not stage1_only and not stage2_only and _needs_expansion(
        experiment,
        force=force_expansion,
        skip=skip_expansion,
    )
    run_stage2 = not stage1_only and not expansion_only and not skip_stage2

    stage1_path = stage1_retrieved_docs_path(
        project_root=PROJECT_ROOT,
        dataset=experiment.dataset,
        retrieval_run_label=experiment.retrieval_run_label,
        split=experiment.split,
        retrieved_root=experiment.retrieved_root,
    )
    expansion_path = expanded_queries_path(
        project_root=PROJECT_ROOT,
        dataset=experiment.dataset,
        run_label=experiment.run_label,
        split=experiment.split,
        expansion_k=experiment.expansion_k_values[0],
    )
    missing_k = _missing_expansion_k_values(experiment)

    if run_stage1:
        print(">>> stage 1: retrieval (top-k for expansion)")
        _run_command(_stage1_command(experiment), dry_run=dry_run)
    elif not stage2_only and not expansion_only:
        print(f">>> skipping stage 1 (using {stage1_path})")

    if run_expansion:
        if not run_stage1 and not stage1_path.is_file() and not dry_run:
            raise FileNotFoundError(
                f"Missing stage-1 retrieved docs: {stage1_path}. Run stage 1 first."
            )
        if missing_k:
            print(
                ">>> query expansion (long-form generation) "
                f"for k={', '.join(map(str, missing_k))}"
            )
        else:
            print(">>> query expansion (all k values already present)")
        _run_command(_expansion_command(experiment), dry_run=dry_run)
    elif not stage2_only and not stage1_only:
        print(f">>> skipping expansion (using {expansion_path.parent.parent}/k*/...)")

    if run_stage2:
        for expansion_k in experiment.expansion_k_values:
            k_path = expanded_queries_path(
                project_root=PROJECT_ROOT,
                dataset=experiment.dataset,
                run_label=experiment.run_label,
                split=experiment.split,
                expansion_k=expansion_k,
            )
            if not run_expansion and not k_path.is_file() and not dry_run:
                raise FileNotFoundError(
                    f"Missing expanded queries for k={expansion_k}: {k_path}. "
                    "Run expansion first."
                )
            print(f">>> stage 2: MTEB retrieval (expanded queries, k={expansion_k})")
            _run_command(_stage2_command(experiment, expansion_k=expansion_k), dry_run=dry_run)
    elif not stage1_only and not expansion_only:
        print(">>> skipping stage 2 MTEB")

    print()


def _select_experiments(args: argparse.Namespace, raw: dict) -> list[ResolvedRfgExperiment]:
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
            skip_stage1=args.skip_stage1,
            force_stage1=args.force_stage1,
            skip_expansion=args.skip_expansion,
            force_expansion=args.force_expansion,
            skip_stage2=args.skip_stage2,
            stage1_only=args.stage1_only,
            expansion_only=args.expansion_only,
            stage2_only=args.stage2_only,
            dry_run=args.dry_run,
        )

    print(f"Completed {len(experiments)} RFG experiment(s).")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
