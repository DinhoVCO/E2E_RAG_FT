# Qwen3-Embedding-4B LoRA Fine-Tuning

Fine-tune `Qwen/Qwen3-Embedding-4B` with LoRA on project RAG retrieval datasets using [sentence-transformers](https://www.sbert.net/). Each dataset is trained in a separate run.

Training uses:

- **LoRA** with `task_type=TaskType.FEATURE_EXTRACTION`, `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
- **Max sequence length**: 512 tokens
- **Loss**: `MultipleNegativesRankingLoss`
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
| `--train-split` | `train` | Split for building training pairs |
| `--eval-split` | `dev` | Split for IR evaluation |
| `--wandb-project` | `WANDB_PROJECT` from `.env` | W&B project name |
| `--run-name` | `qwen3-embedding-4b-lora-<dataset>` | W&B run name |
| `--fp16` | off | Enable FP16 training |
| `--no-bf16` | off | Disable BF16 (BF16 is on by default) |

## Weights & Biases

- **Project**: `qwen3-embedding-finetuning` (or `WANDB_PROJECT` in `.env`)
- **Run name** (default): `qwen3-embedding-4b-lora-<dataset>`

Examples:

| Dataset | Default W&B run name |
|---------|----------------------|
| `bioasq-resplit` | `qwen3-embedding-4b-lora-bioasq-resplit` |
| `qasper` | `qwen3-embedding-4b-lora-qasper` |
| `telco-dpr` | `qwen3-embedding-4b-lora-telco-dpr` |
| `narrativeqa` | `qwen3-embedding-4b-lora-narrativeqa` |

Override with `--run-name` or `--wandb-project`.

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

Default batch size is **128 per device**. With a 4B model this may require multiple GPUs or a smaller `--batch-size` if you hit OOM. Reduce batch size before disabling BF16.

## Related Scripts

| Script | Purpose |
|--------|---------|
| `scripts/finetuning/embeddings/finetune_qwen3_embedding.py` | Main training CLI |
| `jobs/scripts/finetuning/finetune_qwen3_embedding.sh` | Wrapper for one dataset |
| `jobs/scripts/finetuning/finetune_qwen3_embedding_all.sh` | Train all four datasets |
| `scripts/download_hf.py` | Download datasets and base model from Hugging Face |
