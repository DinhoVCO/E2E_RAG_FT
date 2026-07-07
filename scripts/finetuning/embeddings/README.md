# Qwen3-Embedding-4B LoRA Fine-Tuning

Fine-tune `Qwen/Qwen3-Embedding-4B` with LoRA on project RAG retrieval datasets using [sentence-transformers](https://www.sbert.net/). Each dataset is trained in a separate run.

Training uses:

- **LoRA** with `task_type=TaskType.FEATURE_EXTRACTION`, `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
- **Max sequence length**: 512 tokens
- **Loss**: `CachedMultipleNegativesRankingLoss` (effective batch 128, encode mini-batch 32)
- **Batch sampler**: `BatchSamplers.NO_DUPLICATES`
- **Evaluator**: `InformationRetrievalEvaluator` (NDCG, MRR, Recall@k on the dev split)
- **Logging**: Weights & Biases (`wandb`)
- **Checkpoints**: saved every `--save-steps` steps

Implementation lives in `src/tesis_unicamp/finetuning/embeddings/`.

## Prerequisites

1. Install dependencies from the repo root:

```bash
uv sync
```

2. Configure environment variables in `.env`:

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=qwen3-embedding-finetuning
```

3. (Recommended) Download datasets and the base model on a login node before GPU jobs:

```bash
python scripts/download_hf.py --preset rag-full --datasets qasper
python scripts/download_hf.py --snapshot --repo Qwen/Qwen3-Embedding-4B --repo-type model
```

## Supported Datasets

Each CLI `--dataset` key maps to one Hugging Face repo. Runs are independent (one dataset per training job).

| CLI key           | Hugging Face repo                         |
|-------------------|-------------------------------------------|
| `bioasq-resplit`  | `dinho1597/bioasq-rag-13b-resplit`        |
| `qasper`          | `DinoStackAI/qasper-rag`                  |
| `telco-dpr`       | `DinoStackAI/telco-dpr-rag`               |
| `narrativeqa`     | `DinoStackAI/narrativeqa-rag`             |

### Training data format

Training pairs are built from the **train** split:

- **anchor**: query text with the Qwen3 instruct prefix (`Instruct: …\nQuery:…`)
- **positive**: relevant document text from the corpus

If a query has multiple relevant documents in `qrels`, one `(anchor, positive)` pair is created per qrel row. During evaluation, all relevant documents for a query are grouped into a single set for IR metrics.

## Quick Start

From the repo root:

```bash
# Single dataset
CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset qasper
```

Or use the shell wrapper:

```bash
bash jobs/scripts/finetuning/finetune_qwen3_embedding.sh qasper
```

Train all four datasets sequentially:

```bash
bash jobs/scripts/finetuning/finetune_qwen3_embedding_all.sh
```

## YAML configs (per dataset)

Each dataset has a YAML file under `scripts/finetuning/embeddings/configs/` with recommended hyperparameters. When you pass `--dataset`, the matching file is loaded automatically if it exists (for example `configs/telco-dpr.yaml`).

| Dataset | Config file |
|---------|-------------|
| `telco-dpr` | `configs/telco-dpr.yaml` |
| `qasper` | `configs/qasper.yaml` |
| `narrativeqa` | `configs/narrativeqa.yaml` |
| `bioasq-resplit` | `configs/bioasq-resplit.yaml` |

Example (`configs/telco-dpr.yaml`):

```yaml
dataset: telco-dpr
batch_size: 128
epochs: 10
logging_steps: 1
eval_steps: 8
save_steps: 8
```

Run with the YAML defaults (CLI flags override YAML values):

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset telco-dpr
```

Or point to a custom config explicitly:

```bash
python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --config scripts/finetuning/embeddings/configs/qasper.yaml \
  --dataset qasper \
  --epochs 5
```

Supported YAML keys match the CLI options (snake_case): `model`, `output_dir`, `epochs`, `batch_size`, `learning_rate`, `warmup_ratio`, `eval_steps`, `save_steps`, `logging_steps`, `save_total_limit`, `eval_batch_size`, `mini_batch_size`, `train_split`, `eval_split`, `wandb_project`, `run_name`, `fp16`, `bf16`.

### Parallel training (4 GPUs, one dataset per GPU)

On a node with 4 GPUs (e.g. Santos Dumont `ict-h100`), fine-tune all four datasets at once by pinning one job to each GPU. See **[Recommended settings (W&B loss curve + IR eval)](#recommended-settings-wb-loss-curve--ir-eval)** below for the full commands with logging, eval, and checkpoints.

Quick version (defaults only — may not show W&B curves on small datasets):

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset telco-dpr --batch-size 128 > logs/finetune-telco-dpr.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset qasper --batch-size 128 > logs/finetune-qasper.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset narrativeqa --batch-size 128 > logs/finetune-narrativeqa.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset bioasq-resplit --batch-size 128 > logs/finetune-bioasq-resplit.log 2>&1 &

wait
```

Each line ends with `> logs/....log 2>&1 &`:

- `> logs/....log` — writes stdout to a separate log file per dataset
- `2>&1` — redirects stderr to the same file (errors included)
- `&` — runs the job in the background so all four can start together

**Important:** each job must pin exactly one GPU with `CUDA_VISIBLE_DEVICES=0`, `=1`, etc. Do **not** use `CUDA_VISIBLE_DEVICES=0,1` for a single job — that enables DataParallel and can OOM on H100 80GB.

Monitor a run live:

```bash
tail -f logs/finetune-qasper.log
```

W&B still logs metrics independently per dataset. The log files are a local copy of console output only.

When requesting the node via SLURM, increase system RAM and CPUs for four concurrent jobs (for example `--mem=131072M`, `--cpus-per-task=32` in `jobs/scripts/santos_dumont/run_ict_h100.sh`).

### Recommended settings (W&B loss curve + IR eval)

Default `--logging-steps 50` and `--eval-steps 500` skip most metrics on small datasets (e.g. telco-dpr finishes in ~4 steps with 1 epoch). Use **`--epochs 3`**, **`--logging-steps 1`**, **`--eval-steps 2`**, and **`--save-steps 2`** so train loss and `InformationRetrievalEvaluator` metrics appear in W&B.

Pin **one GPU per job** (`CUDA_VISIBLE_DEVICES=0` for a single run). Use `2>&1 | tee logs/....log` to print to the terminal and save a log file.

#### telco-dpr

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset telco-dpr \
  2>&1 | tee logs/finetune-telco-dpr.log
```

Settings come from `configs/telco-dpr.yaml` (`epochs: 10`, `eval_steps: 8`, etc.).

#### qasper

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset qasper \
  2>&1 | tee logs/finetune-qasper.log
```

#### narrativeqa

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset narrativeqa \
  2>&1 | tee logs/finetune-narrativeqa.log
```

#### bioasq-resplit

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset bioasq-resplit \
  2>&1 | tee logs/finetune-bioasq-resplit.log
```

For larger datasets (qasper, narrativeqa, bioasq-resplit), you may increase `--eval-steps` and `--save-steps` (e.g. 50–100) to reduce evaluation overhead.

#### All four in parallel (4 GPUs)

Same training flags, one GPU per dataset, background jobs:

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset telco-dpr \
  > logs/finetune-telco-dpr.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset qasper \
  > logs/finetune-qasper.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset narrativeqa \
  > logs/finetune-narrativeqa.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset bioasq-resplit \
  > logs/finetune-bioasq-resplit.log 2>&1 &

wait
```

Monitor any run:

```bash
tail -f logs/finetune-qasper.log
```

## Custom Training Run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset bioasq-resplit \
  --output-dir models/qwen3-embedding-4b-lora/bioasq-resplit \
  --epochs 3 \
  --batch-size 128 \
  --learning-rate 2e-5 \
  --eval-steps 250 \
  --save-steps 250 \
  --run-name qwen3-lora-bioasq-resplit-e3
```

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/<dataset>.yaml` if present | YAML file with training hyperparameters |
| `--dataset` | *(required)* | One of: `bioasq-resplit`, `qasper`, `telco-dpr`, `narrativeqa` |
| `--model` | `Qwen/Qwen3-Embedding-4B` | Base embedding model |
| `--output-dir` | `models/qwen3-embedding-4b-lora/<dataset>` | Checkpoints and final adapter output |
| `--epochs` | `1` | Training epochs |
| `--batch-size` | `128` | Per-device train/eval batch size |
| `--learning-rate` | `2e-5` | Optimizer learning rate |
| `--warmup-ratio` | `0.1` | LR warmup ratio |
| `--eval-steps` | `500` | Evaluate every N steps |
| `--save-steps` | `500` | Save checkpoint every N steps |
| `--save-total-limit` | `3` | Max checkpoints to keep |
| `--logging-steps` | `50` | Log metrics every N steps |
| `--eval-batch-size` | `32` | Batch size for `InformationRetrievalEvaluator` |
| `--mini-batch-size` | `32` | Encode chunk size inside `CachedMultipleNegativesRankingLoss` |
| `--train-split` | `train` | Split for building training pairs |
| `--eval-split` | `dev` | Split for IR evaluation |
| `--wandb-project` | `WANDB_PROJECT` from `.env` | W&B project name |
| `--run-name` | `qwen3-embedding-4b-lora-<dataset>-b<batch>-e<epochs>` | W&B run name |
| `--fp16` | off | Enable FP16 training |
| `--no-bf16` | off | Disable BF16 (BF16 is on by default) |

## Weights & Biases

- **Project**: `qwen3-embedding-finetuning` (or `WANDB_PROJECT` in `.env`)
- **Run name** (default): `qwen3-embedding-4b-lora-<dataset>-b<batch_size>-e<epochs>`

The default run name is applied automatically when you do not pass `--run-name`. At startup, the script prints the resolved name to the console:

```
wandb_run_name: qwen3-embedding-4b-lora-telco-dpr-b128-e10
```

Examples:

| Dataset | Command flags | Default W&B run name |
|---------|---------------|----------------------|
| `telco-dpr` | `--batch-size 128 --epochs 10` | `qwen3-embedding-4b-lora-telco-dpr-b128-e10` |
| `qasper` | `--batch-size 128 --epochs 3` | `qwen3-embedding-4b-lora-qasper-b128-e3` |
| `narrativeqa` | default batch/epochs | `qwen3-embedding-4b-lora-narrativeqa-b128-e1` |
| `bioasq-resplit` | `--batch-size 64 --epochs 5` | `qwen3-embedding-4b-lora-bioasq-resplit-b64-e5` |

To use a custom run name, pass `--run-name`:

```bash
python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset telco-dpr --batch-size 128 --epochs 10 \
  --run-name mi-experimento-custom
```

Override the W&B project with `--wandb-project` or `WANDB_PROJECT` in `.env`.

## Outputs

For dataset `qasper`, artifacts are written under:

```
models/qwen3-embedding-4b-lora/qasper/
├── checkpoint-500/
├── checkpoint-1000/
├── ...
└── final/          # final LoRA adapter + SentenceTransformer config
```

Load the fine-tuned model:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/qwen3-embedding-4b-lora/qasper/final")
embeddings = model.encode(["Instruct: ...\nQuery:What is X?", "Document text..."])
```

## Memory Notes

Default settings target **1× H100 80GB per job**:

- Model loads in **bf16** (not float32)
- **Effective batch size** 128 via `CachedMultipleNegativesRankingLoss`
- **Encode mini-batch** 32 (lower with `--mini-batch-size 16` if OOM persists)

If you see `CUDA out of memory` or `Currently using DataParallel`:

1. Set **one GPU per job**: `CUDA_VISIBLE_DEVICES=0` (not all four visible)
2. Lower mini-batch: `--mini-batch-size 16`
3. Lower effective batch: `--batch-size 64`
4. Lower eval load: `--eval-batch-size 16`

Parallel runs (4 datasets) need **4 separate processes**, each with its own `CUDA_VISIBLE_DEVICES`.

## Related Scripts

| Script | Purpose |
|--------|---------|
| `scripts/finetuning/embeddings/finetune_qwen3_embedding.py` | Main training CLI |
| `jobs/scripts/finetuning/finetune_qwen3_embedding.sh` | Wrapper for one dataset |
| `jobs/scripts/finetuning/finetune_qwen3_embedding_all.sh` | Train all four datasets |
| `scripts/download_hf.py` | Download datasets and base model from Hugging Face |
