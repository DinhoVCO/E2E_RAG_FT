"""Shared CLI helpers for reference-based RAGAS metrics."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ragas.run_config import RunConfig

from tesis_unicamp.evaluation.ragas.reference_runner import (
    DEFAULT_GENERATED_ROOT,
    DEFAULT_REFERENCE_OUTPUT_ROOT,
    default_reference_output_dir,
    discover_generation_dirs,
    resolve_reference_evaluation_context,
    summary_path_for_run,
)


def add_generation_selection_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generation-dir",
        type=Path,
        action="append",
        dest="generation_dirs",
        help=(
            "Directory containing run_settings.json and "
            "<split>/generated_answers.json (repeatable)."
        ),
    )
    group.add_argument(
        "--dataset",
        help="Evaluate all generation runs under datasets/generated/<dataset>/",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all generation runs under datasets/generated/",
    )


def add_common_reference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=DEFAULT_GENERATED_ROOT,
        help="Root directory containing generated RAG answers.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_REFERENCE_OUTPUT_ROOT,
        help="Root directory for reference metric outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (only valid with a single --generation-dir).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split to evaluate (default: value from run_settings.json).",
    )
    parser.add_argument(
        "--dataset-override",
        default=None,
        help="Override dataset name from generation run_settings.json.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose summary JSON already exists.",
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
        default=8,
        help="Concurrent RAGAS workers (default: 8).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-metric timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--raise-exceptions",
        action="store_true",
        help="Stop immediately when a metric fails for a sample.",
    )


def resolve_generation_targets(args: argparse.Namespace) -> list[Path]:
    if args.generation_dirs:
        return [path.resolve() for path in args.generation_dirs]
    if args.all:
        return discover_generation_dirs(generated_root=args.generated_root.resolve())
    if args.dataset:
        return discover_generation_dirs(
            generated_root=args.generated_root.resolve(),
            dataset=args.dataset,
        )
    raise ValueError("No generation targets selected.")


def resolve_output_dir(
    *,
    generation_dir: Path,
    metric_group: str,
    args: argparse.Namespace,
) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return default_reference_output_dir(
        generation_dir,
        metric_group=metric_group,
        generated_root=args.generated_root.resolve(),
        output_root=args.output_root.resolve(),
    )


def build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        max_workers=args.max_workers,
        timeout=args.timeout,
    )


def run_reference_metric_batch(
    *,
    args: argparse.Namespace,
    metric_group: str,
    evaluate_one: Callable[..., Any],
    evaluate_kwargs: dict[str, Any],
) -> int:
    if args.output_dir is not None:
        if not args.generation_dirs or len(args.generation_dirs) != 1:
            raise SystemExit("--output-dir requires exactly one --generation-dir.")

    generation_dirs = resolve_generation_targets(args)
    if not generation_dirs:
        print(f"No generation runs found under {args.generated_root}")
        return 1

    run_config = build_run_config(args)
    failures = 0

    for generation_dir in generation_dirs:
        context = resolve_reference_evaluation_context(
            generation_dir,
            split=args.split,
            dataset=args.dataset_override,
        )
        output_dir = resolve_output_dir(
            generation_dir=generation_dir,
            metric_group=metric_group,
            args=args,
        )
        summary_path = summary_path_for_run(
            output_dir,
            context["split"],
            metric_group=metric_group,
        )
        if args.skip_existing and summary_path.exists():
            print(f"Skipping existing results: {summary_path}")
            continue

        print(f"generation_dir: {generation_dir}")
        print(f"output_dir: {output_dir}")
        try:
            result = evaluate_one(
                generation_dir=generation_dir,
                output_dir=output_dir,
                split=args.split,
                dataset=args.dataset_override,
                run_config=run_config,
                batch_size=args.batch_size,
                raise_exceptions=args.raise_exceptions,
                **evaluate_kwargs,
            )
        except Exception as exc:
            failures += 1
            print(f"Failed {generation_dir}: {exc}")
            continue

        print(result)
        print(f"Saved reference metrics to {output_dir}")

    return 1 if failures else 0
