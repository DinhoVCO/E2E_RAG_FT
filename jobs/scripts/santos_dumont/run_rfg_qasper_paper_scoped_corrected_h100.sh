#!/usr/bin/env bash
# Re-evaluate QASPER RFG with paper-scoped retrieval in both stages.
#
# Stage 1 and the existing query expansions were produced from the chunks of
# each query's associated paper. This script reruns stage 2 with the same
# paper-scoped candidate pools and writes to a new result root.
#
# The corrected experiment uses the main RFG comparison configuration:
# embedding base + generator base + query concatenation, for k=1,3,5,7,10.
#
# Usage:
#   bash jobs/scripts/santos_dumont/run_rfg_qasper_paper_scoped_corrected_h100.sh
#
# Optional:
#   K_VALUES="5 7" TIME=02:00:00 \
#     bash jobs/scripts/santos_dumont/run_rfg_qasper_paper_scoped_corrected_h100.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
K_VALUES="${K_VALUES:-1 3 5 7 10}"
TIME="${TIME:-02:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-65536M}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/slurm}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"

EXPANSION_ROOT="${ROOT}/datasets/query_expansion/qasper/qasper-rfg-emb-base-gen-base-concat"
OUTPUT_ROOT="${ROOT}/results/mteb/rfg_qasper_paper_scoped_corrected/qasper"

mkdir -p "$LOG_DIR"

read -r -a k_list <<< "$K_VALUES"
for k in "${k_list[@]}"; do
  expanded_dir="${EXPANSION_ROOT}/k${k}"
  expanded_file="${expanded_dir}/test/expanded_queries.json"
  if [[ ! -f "$expanded_file" ]]; then
    echo "Missing expanded queries: ${expanded_file}" >&2
    exit 1
  fi

  job_name="rfg-qasper-paper-scoped-k${k}"
  model_revision="rfg-qasper-emb-base-gen-base-concat-k${k}-paper-scoped-corrected"
  job_id=$(sbatch --parsable \
    --job-name="$job_name" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --gres=gpu:1 \
    --chdir="$ROOT" \
    --output="${LOG_DIR}/${job_name}-%j.out" \
    --error="${LOG_DIR}/${job_name}-%j.err" \
    --wrap="set -euo pipefail
cd '${ROOT}'
export CUDA_VISIBLE_DEVICES=0
echo SLURM_JOB_ID=\$SLURM_JOB_ID
echo k=${k}
echo Stage-1 scope: paper
echo Stage-2 scope: paper
echo Started: \$(date -Is)
exec ${PYTHON_RUNNER} scripts/query_expansion/run_rfg_stage2_mteb.py \
  --dataset qasper \
  --expanded-queries-dir '${expanded_dir}' \
  --model-revision '${model_revision}' \
  --paper-scoped \
  --output-dir '${OUTPUT_ROOT}'")

  echo "Submitted batch job ${job_id} (${job_name})"
  echo "Monitor: squeue -j ${job_id}"
  echo "Results: ${OUTPUT_ROOT}"
  echo
done
