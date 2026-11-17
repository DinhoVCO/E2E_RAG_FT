"""Run RFG experiments with query concatenated to the generated expansion.

Each expanded query is stored as ``{original_query} + {generated_answer}``,
following the same pattern as Query2Doc (Q2D).

Uses ``configs/experiments_concat.yaml`` by default (16 runs, -concat suffix).

Usage:
    python scripts/query_expansion/run_rfg_concat_experiment.py --list

    python scripts/query_expansion/run_rfg_concat_experiment.py \\
        --experiment bioasq-resplit-rfg-emb-base-gen-base-concat

    python scripts/query_expansion/run_rfg_concat_experiment.py --dataset qasper

    python scripts/query_expansion/run_rfg_concat_experiment.py --all
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "experiments_concat.yaml"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_rfg_experiment import main as run_rfg_main  # noqa: E402


def _inject_default_config(argv: list[str]) -> list[str]:
    if any(arg == "--config" or arg.startswith("--config=") for arg in argv):
        return argv
    return ["--config", str(DEFAULT_CONFIG), *argv]


def main(argv: list[str] | None = None) -> None:
    run_rfg_main(_inject_default_config(argv or sys.argv[1:]))


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        raise SystemExit(exc.code) from exc
