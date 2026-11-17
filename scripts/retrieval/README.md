# Retrieval scripts

Command-line utilities to retrieve top-k documents for project RAG datasets, label relevance from `qrels`, and save or upload `retrieved_docs` for downstream answer-generation evaluation.

Run commands from the **repository root** with the virtual environment activated:

```bash
source .venv/bin/activate
```

---

## Overview

| Pipeline | Script | Vector index | Default splits | Best for |
|----------|--------|--------------|----------------|----------|
| **In-memory (FAISS)** | `retrieve_rag_top_k_inmemory.py` | FAISS `IndexFlatIP` in RAM | `test` | Quick test-split runs, multiple embedding models, no Qdrant |
| **Qdrant** | `retrieve_rag_top_k.py` | Persistent Qdrant collection | `train dev test` | Production pipeline, reusing a pre-indexed corpus |
| **Hub upload** | `push_retrieved_docs_to_hub.py` | — | — | Publishing `retrieved_docs` to Hugging Face |

Both retrieval scripts:

- Embed queries with the same backends as indexing (`offline` vLLM or `online` API).
- Use the instruct query format by default (same as MTEB and indexing scripts).
- Compare each hit against `qrels` and set `is_relevant`.
- Write JSON plus a local Hugging Face `DatasetDict` export.

Default embedding model: `Qwen/Qwen3-Embedding-4B`.

---

## Prerequisites

- Python environment with project dependencies installed (`uv sync`).
- For `--mode offline`: a CUDA GPU and `CUDA_VISIBLE_DEVICES` set.
- For `--mode online`: a running OpenAI-compatible embedding server (e.g. vLLM via `jobs/scripts/vllm/serve_embedding_4b.sh`).
- Hugging Face access to the dataset repos (public Hub datasets listed below).
- For **Qdrant retrieval only**: a running Qdrant instance and an indexed corpus (see [jobs/qdrant.md](../../jobs/qdrant.md) and `scripts/embeddings/index_*_corpus.py`).

No Qdrant instance is required for the in-memory script.

---

## Which script should I use?

| Goal | Script |
|------|--------|
| Evaluate answer generation on the `test` split across several embedding models | `retrieve_rag_top_k_inmemory.py` |
| Avoid managing multiple Qdrant collections per model | `retrieve_rag_top_k_inmemory.py` |
| Reuse a corpus already indexed in Qdrant | `retrieve_rag_top_k.py` |
| Retrieve all splits (`train`, `dev`, `test`) for Hub publication | `retrieve_rag_top_k.py` |
| Upload saved `retrieved_docs` to Hugging Face | `push_retrieved_docs_to_hub.py` |

For IR metrics (nDCG, Recall@k) without saving per-query hits, use `scripts/evaluation/mteb/run_mteb_retrieval.py` instead.

---

## Quick start (in-memory, recommended for generation eval)

Retrieve top-10 documents on the `test` split with the base embedding model:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory.py \
  --dataset qasper \
  --mode offline
```

Retrieve with a fine-tuned LoRA adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory.py \
  --dataset telco-dpr \
  --mode offline \
  --model Qwen/Qwen3-Embedding-4B \
  --lora-path DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr \
  --run-label vllm-lora-telco-dpr-b128
```

Run all four main datasets with distinct run labels:

```bash
for ds in qasper bioasq telco-dpr narrativeqa; do
  CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory.py \
    --dataset "$ds" \
    --mode offline \
    --run-label vllm-offline-b128
done
```

Run **base model + all four LoRA adapters** in one batch (8 runs total):

```bash
bash scripts/retrieval/run_all_inmemory_retrieval.sh
```

This executes retrieval for `telco-dpr`, `qasper`, `narrativeqa`, and `bioasq-resplit` with:

- Base model → `--run-label vllm-offline-b128`
- LoRA adapters → `--run-label vllm-lora-<dataset>-b128`

Optional overrides: `SKIP_BASE=1`, `SKIP_LORA=1`, `CUDA_VISIBLE_DEVICES`, `TOP_K`, `BATCH_SIZE`.

---

## Script reference

### `run_all_inmemory_retrieval.sh`

Batch runner for all four datasets (`telco-dpr`, `qasper`, `narrativeqa`, `bioasq-resplit`):

1. Base model (`vllm-offline-b128`) — 4 runs
2. LoRA adapters (`vllm-lora-<dataset>-b128`) — 4 runs

```bash
bash scripts/retrieval/run_all_inmemory_retrieval.sh
```

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device |
| `BASE_MODEL` | `Qwen/Qwen3-Embedding-4B` | Base embedding model |
| `TOP_K` | `10` | Documents per query |
| `BATCH_SIZE` | `128` | Embedding batch size |
| `BASE_RUN_LABEL` | `vllm-offline-b128` | Output subfolder for base runs |
| `LORA_RUN_LABEL_PREFIX` | `vllm-lora` | Prefix for LoRA run labels |
| `SKIP_BASE` | `0` | Set to `1` to skip base-model runs |
| `SKIP_LORA` | `0` | Set to `1` to skip LoRA runs |

---

### `retrieve_rag_top_k_inmemory.py`

Embeds the full corpus into an in-memory FAISS index, retrieves top-k documents, and saves `retrieved_docs`. No Qdrant required.

**Offline mode (recommended):**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory.py \
  --dataset qasper \
  --mode offline \
  --top-k 10 \
  --run-label vllm-offline-b128
```

**Online mode:**

```bash
bash jobs/scripts/vllm/serve_embedding_4b.sh

python scripts/retrieval/retrieve_rag_top_k_inmemory.py \
  --dataset narrativeqa \
  --mode online \
  --run-label vllm-online-b128
```

#### Available datasets

Use `--dataset` with one of:

| Key | Hugging Face repo |
|-----|-------------------|
| `bioasq` | `DinoStackAI/bioasq-rag-13b` |
| `bioasq-resplit` | `DinoStackAI/bioasq-rag-13b-resplit` |
| `qasper` | `DinoStackAI/qasper-rag` |
| `telco-dpr` | `DinoStackAI/telco-dpr-rag` |
| `narrativeqa` | `DinoStackAI/narrativeqa-rag` |

#### Fine-tuned LoRA adapters

Use `--lora-path` with `--mode offline`. `--model` must be the base model (`Qwen/Qwen3-Embedding-4B`).

| Dataset | `--lora-path` |
|---------|---------------|
| `telco-dpr` | `DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr` |
| `qasper` | `DinoStackAI/Qwen3-Emb-4b-lora-qasper` |
| `narrativeqa` | `DinoStackAI/Qwen3-Emb-4b-lora-narrativeqa` |
| `bioasq-resplit` | `DinoStackAI/Qwen3-Emb-4b-lora-bioasq-resplit` |

Use a distinct `--run-label` per embedding model so results do not overwrite each other.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | RAG dataset key (see table above). |
| `--mode` | *(required)* | `offline` (in-process vLLM) or `online` (API server). |
| `--model` | `Qwen/Qwen3-Embedding-4B` | Embedding model name or Hub repo id. |
| `--lora-path` | — | LoRA adapter path or Hub repo id (`offline` only). |
| `--max-lora-rank` | `16` | `max_lora_rank` passed to vLLM when `--lora-path` is set. |
| `--run-label` | `inmemory-default` | Subfolder name under `datasets/retrieved_inmemory/<dataset>/`. |
| `--output-dir` | auto | Override output directory. |
| `--top-k` | `10` | Documents retrieved per query. |
| `--batch-size` | `128` | Embedding batch size. |
| `--corpus-split` | `train` | Corpus split to embed and index. |
| `--splits` | `test` | Query splits to retrieve (`train`, `dev`, `test`). |
| `--paper-scoped` / `--no-paper-scoped` | on for `qasper` | Restrict QASPER retrieval to each query's paper via `top_ranked`. Appends `-paper-scoped` to `--run-label` when enabled. |

#### QASPER paper-scoped retrieval

QASPER questions belong to a single paper. By default, `--dataset qasper` uses **paper-scoped** retrieval: each query searches only the chunks listed in the Hub `top_ranked` subset (all paragraphs of that paper), matching MTEB evaluation with `--paper-scoped`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory.py \
  --dataset qasper \
  --mode offline \
  --run-label vllm-offline-b128
# Output: datasets/retrieved_inmemory/qasper/vllm-offline-b128-paper-scoped/test/retrieved_docs.json
```

To use full-corpus retrieval (legacy behavior):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory.py \
  --dataset qasper \
  --mode offline \
  --no-paper-scoped \
  --run-label vllm-offline-b128-full-corpus
```

#### Title-aware retrieval (`retrieve_rag_top_k_inmemory_title.py`)

For generation experiments with `--include-title-prompt`, embed queries with the gold
document title before `Query:`:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
## Title:
<gold title from qrels>
Query:<question>
```

Results are saved under `datasets/retrieved_inmemory_title/` (same run-label layout as
above). Title generation experiments in `experiments_title.yaml` use this path
automatically; `run_rag_generation.py` can also launch it when retrieved docs are
missing (`--run-title-retrieval-if-missing`).

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory_title.py \
  --dataset telco-dpr \
  --mode offline \
  --run-label vllm-offline-b128
```

#### Output layout

```
datasets/retrieved_inmemory/
  qasper/
    vllm-offline-b128/
      test/retrieved_docs.json
      hf_dataset/retrieved_docs/
  telco-dpr/
    vllm-lora-telco-dpr-b128/
      test/retrieved_docs.json
      hf_dataset/retrieved_docs/
```

---

### `retrieve_rag_top_k.py`

Embeds queries and searches a **pre-indexed** Qdrant collection. Only queries are embedded at retrieval time; corpus vectors must already be in Qdrant.

**Prerequisites:** start Qdrant and index the corpus first:

```bash
bash jobs/scripts/santos_dumont/run_qdrant.sh

CUDA_VISIBLE_DEVICES=0 python scripts/embeddings/index_qasper_corpus.py --mode offline
```

**Offline retrieval:**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k.py \
  --dataset qasper \
  --mode offline \
  --top-k 10
```

**Test split only:**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k.py \
  --dataset qasper \
  --mode offline \
  --splits test
```

#### Default Qdrant collections

| Dataset | Default collection |
|---------|------------------|
| BioASQ | `bioasq-rag-13b-corpus` |
| QASPER | `qasper-rag-corpus` |
| Telco-DPR | `telco-dpr-rag-corpus` |
| NarrativeQA | `narrativeqa-rag-corpus` |

`bioasq-resplit` reuses the BioASQ collection.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | RAG dataset key. |
| `--mode` | *(required)* | `offline` or `online`. |
| `--model` | `Qwen/Qwen3-Embedding-4B` | Embedding model for query encoding. |
| `--collection` | dataset-specific | Qdrant collection name. |
| `--qdrant-url` | `http://localhost:6333` | Qdrant REST URL. |
| `--output-dir` | `datasets/retrieved/<dataset>_rag/` | Output directory. |
| `--top-k` | `10` | Documents retrieved per query. |
| `--batch-size` | `128` | Query embedding batch size. |
| `--splits` | `train dev test` | Query splits to retrieve. |

#### Output layout

```
datasets/retrieved/
  qasper_rag/
    train/retrieved_docs.json
    dev/retrieved_docs.json
    test/retrieved_docs.json
    hf_dataset/retrieved_docs/
```

---

### `push_retrieved_docs_to_hub.py`

Uploads the local `retrieved_docs` config (train/dev/test) to the dataset repo on the Hub.

```bash
python scripts/retrieval/push_retrieved_docs_to_hub.py --dataset qasper
```

By default this reads from `datasets/retrieved/<dataset>_rag/`. Point to in-memory results with `--output-dir`:

```bash
python scripts/retrieval/push_retrieved_docs_to_hub.py \
  --dataset qasper \
  --output-dir datasets/retrieved_inmemory/qasper/vllm-offline-b128
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | RAG dataset key. |
| `--repo-id` | dataset-specific Hub repo | Target Hugging Face dataset repo. |
| `--output-dir` | `datasets/retrieved/<dataset>_rag/` | Local `retrieved_docs` directory. |
| `--token` | `HF_TOKEN` env var | Hugging Face write token. |
| `--private` | off | Upload as a private dataset. |

#### Default Hub repos

| Dataset | Repo id |
|---------|---------|
| BioASQ | `DinoStackAI/bioasq-rag-13b` |
| BioASQ resplit | `DinoStackAI/bioasq-rag-13b-resplit` |
| QASPER | `DinoStackAI/qasper-rag` |
| Telco-DPR | `DinoStackAI/telco-dpr-rag` |
| NarrativeQA | `DinoStackAI/narrativeqa-rag` |

Load after upload:

```python
from datasets import load_dataset

retrieved = load_dataset("DinoStackAI/qasper-rag", "retrieved_docs")
test_hits = retrieved["test"]
```

---

## `retrieved_docs` schema

Each row describes one retrieved document for one query:

```json
{
  "query_id": "q1",
  "corpus_id": "doc17",
  "rank": 1,
  "retrieval_score": 0.92,
  "is_relevant": true
}
```

| Field | Description |
|-------|-------------|
| `query_id` | Query identifier from the `queries` subset |
| `corpus_id` | Document id from the corpus payload |
| `rank` | Rank in the retrieval list (1 = best match) |
| `retrieval_score` | Cosine similarity score |
| `is_relevant` | `true` if `corpus_id` appears in `qrels` for that query and split |

To build RAG prompts for answer generation, join `corpus_id` with the `corpus` subset to recover document text.

---

## Query formatting

Queries use the instruct format (consistent with indexing and MTEB evaluation):

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:<question text>
```

---

## Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | both retrieval scripts | Default `--model` value |
| `LORA_PATH` | in-memory script | Default `--lora-path` value |
| `MAX_LORA_RANK` | in-memory script | Default `--max-lora-rank` value |
| `RETRIEVAL_RUN_LABEL` | in-memory script | Default `--run-label` value |
| `RETRIEVAL_TOP_K` | both retrieval scripts | Default `--top-k` value |
| `EMBED_BATCH_SIZE` | both retrieval scripts | Default `--batch-size` value |
| `QDRANT_URL` | Qdrant script | Default `--qdrant-url` value |
| `VLLM_BASE_URL` | online mode | OpenAI-compatible API base URL |
| `VLLM_API_KEY` | online mode | API key (default: `EMPTY`) |
| `HF_TOKEN` | push script | Hugging Face write token |

---

## Performance notes

| Dataset | Approx. test queries | Suggestion |
|---------|-------------------|------------|
| NarrativeQA | moderate | Full in-memory run is reasonable |
| BioASQ | moderate | Full in-memory run is reasonable |
| Telco-DPR | moderate | Full in-memory run is reasonable |
| QASPER | ~1.3k test queries; ~81k corpus chunks | Paper-scoped by default (`top_ranked`); start with `--splits test` |

**In-memory script:** embeds the full corpus on every run. Use distinct `--run-label` values to keep results from different models separate.

**Qdrant script:** embeds queries only; corpus vectors are reused from Qdrant. Better when you run retrieval many times against the same indexed model.

---

## Troubleshooting

### `Collection '...' is empty` (Qdrant script)

Index the corpus first:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/embeddings/index_<dataset>_corpus.py --mode offline
```

### `No CUDA device is visible` (offline mode)

Request a GPU in your SLURM session and set `CUDA_VISIBLE_DEVICES=0` before launching the script.

### `--lora-path requires --mode offline`

LoRA adapters are only supported with in-process vLLM (`--mode offline`), not the online API backend.

### Missing Hugging Face token (push script)

Set `HF_TOKEN` in `.env` or run `huggingface-cli login`.

---

## Related scripts

| Path | Purpose |
|------|---------|
| `../embeddings/index_*_corpus.py` | Index corpus vectors into Qdrant |
| `../evaluation/mteb/run_mteb_retrieval.py` | IR metrics (nDCG, Recall@k) without saving per-query hits |
| `../evaluation/README.md` | MTEB evaluation documentation |
| `../README.md` | End-to-end RAG pipeline overview |
| `../../jobs/qdrant.md` | Qdrant setup on the cluster |
