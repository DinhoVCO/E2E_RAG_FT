# Qwen3-8B LoRA Generative Fine-Tuning

Fine-tune `Qwen/Qwen3-8B` with LoRA on project RAG generative datasets using [TRL](https://github.com/huggingface/trl) `SFTTrainer`. Each dataset is trained in a separate run with **thinking disabled** (`enable_thinking=False`).

Training uses:

- **LoRA** with `task_type=TaskType.CAUSAL_LM`, `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `bias="none"`
- **Max sequence length**: 3712 tokens
- **Token budgets**: query 512, each document 512, answer 512
- **Context variation**: 0–5 retrieved documents per example
- **Relevant context**: in 70% of examples with context, at least one relevant passage (from qrels if missing)
- **Trainer**: `SFTTrainer` with `assistant_only_loss=True` (completion-only masking; replaces `DataCollatorForCompletionOnlyLM` in TRL >= 1.8)
- **Best checkpoint**: `load_best_model_at_end=True` on `eval_loss` (lower is better)
- **Early stopping**: `EarlyStoppingCallback` with `patience=15` eval steps
- **Logging**: Weights & Biases (`wandb`)
- **Epochs**: 3 (6 for `telco-dpr`)

Implementation lives in `src/tesis_unicamp/finetuning/generative/`.

## Prerequisites

1. Install dependencies from the repo root:

```bash
uv sync
```

2. Configure environment variables in `.env`:

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT_STEP2=qwen3-generative-finetuning
```

3. (Recommended) Download datasets and the base model before GPU jobs:

```bash
python scripts/download_hf.py --preset rag-full --datasets qasper
python scripts/download_hf.py --snapshot --repo Qwen/Qwen3-8B --repo-type model
```

## Supported Datasets

| CLI key           | Hugging Face repo                         | Epochs |
|-------------------|-------------------------------------------|--------|
| `bioasq-resplit`  | `DinoStackAI/bioasq-rag-13b-resplit`      | 3      |
| `qasper`          | `DinoStackAI/qasper-rag`                  | 3      |
| `telco-dpr`       | `DinoStackAI/telco-dpr-rag`               | 6      |
| `narrativeqa`     | `DinoStackAI/narrativeqa-rag`             | 3      |

### Training data format

Examples are built from the **train** split of:

- `retrieved_docs` — top-k retrieval hits
- `queries` — question text
- `answers` — reference answers
- `qrels` — relevant passages (used to inject a relevant doc when needed)
- `corpus` — document text lookup

User message (inside chat template):

```
{instruction}
## Query:
{query}
## Context:
doc 1 :
{doc1}
...
## Response:
```

Assistant message: `{answer}`

The full example is rendered with `tokenizer.apply_chat_template(..., enable_thinking=False)`.

## Quick Start

From the repo root:

```bash
# Single dataset
CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/generative/finetune_qwen3_generative.py \
  --dataset qasper
```

Or use the shell wrapper:

```bash
bash jobs/scripts/finetuning/finetune_qwen3_generative.sh qasper
```

Train all four datasets sequentially:

```bash
bash jobs/scripts/finetuning/finetune_qwen3_generative_all.sh
```

## Santos Dumont (SLURM batch)

On the Santos Dumont cluster (`ict-h100`), submit a **non-interactive** batch job with 1 GPU so training keeps running after you disconnect. The helper uses `sbatch` (SLURM runs the command on the allocated node via `srun` internally).

From the repo root:

```bash
bash jobs/scripts/santos_dumont/run_finetune_generative_h100.sh narrativeqa
```

Defaults: partition `ict-h100`, 1 GPU, 8 CPUs, 128 GB RAM, 12 h wall time, account `smartassistant`. SLURM stdout/stderr go to `logs/slurm/ft-<dataset>-<job_id>.out` (and `.err`).

Monitor and manage the job:

```bash
squeue -u "$USER"
tail -f logs/slurm/ft-narrativeqa-<JOB_ID>.out
scancel <JOB_ID>
```

Variants:

```bash
# Longer wall time
TIME=24:00:00 bash jobs/scripts/santos_dumont/run_finetune_generative_h100.sh narrativeqa

# Custom job name in squeue
JOB_NAME=narrativeqa-v2 bash jobs/scripts/santos_dumont/run_finetune_generative_h100.sh narrativeqa

# Extra CLI flags forwarded to finetune_qwen3_generative.py
bash jobs/scripts/santos_dumont/run_finetune_generative_h100.sh narrativeqa --epochs 3
```

For an **interactive** GPU shell on the same partition (e.g. debugging), use `bash jobs/scripts/santos_dumont/run_ict_h100.sh` instead.

## YAML configs

Per-dataset defaults live in `scripts/finetuning/generative/configs/`:

- `narrativeqa.yaml`
- `qasper.yaml`
- `telco-dpr.yaml`
- `bioasq-resplit.yaml`

Override any value via CLI flags.

## Output

Checkpoints and the final LoRA adapter are saved under:

```
models/qwen3-8b-lora/<run_name>/
models/qwen3-8b-lora/<run_name>/checkpoint-*/
models/qwen3-8b-lora/<run_name>/final/
models/qwen3-8b-lora/<run_name>/best_model.json
```

- **Checkpoints**: saved every `save_steps` (max `save_total_limit=3` kept).
- **`final/`**: best checkpoint by `eval_loss` when `load_best_model` is enabled (default).
- **`best_model.json`**: records the winning checkpoint path and metric value.

Keep `eval_steps` and `save_steps` aligned (same value) so every saved checkpoint has eval metrics.

Logs are written to `logs/<run_name>.log` unless `--no-log-file` is passed.
