# Evaluation scripts

This folder contains scripts to evaluate **information retrieval** on project RAG datasets using [MTEB](https://github.com/embeddings-benchmark/mteb).

## Overview

| Pipeline | Script | Index | Metrics |
|----------|--------|-------|---------|
| **MTEB evaluation** | `mteb/run_mteb_retrieval.py` | In-memory (dense search) | nDCG@k, MAP@k, Recall@k, MRR, … |
| **Qdrant retrieval** | `../retrieval/retrieve_rag_top_k.py` | Qdrant | `retrieved_docs` + `is_relevant` |

MTEB evaluation does **not** use Qdrant. It embeds the full corpus in memory and ranks documents with cosine similarity, then computes standard IR metrics against `qrels`.

Both pipelines use the same embedding backends, corpus text formatting, and instruct query format by default.

---

## Prerequisites

- Python environment with project dependencies installed (`uv sync` or equivalent).
- For `--backend offline`: a CUDA GPU and `CUDA_VISIBLE_DEVICES` set.
- For `--backend online`: a running OpenAI-compatible embedding server (e.g. vLLM via `jobs/scripts/vllm/serve_embedding_4b.sh`).
- Hugging Face access to the dataset repos (public Hub datasets listed below).

No Qdrant instance is required for MTEB evaluation.

---

## Quick start

Evaluate QASPER on the `test` split with vLLM (recommended first run):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset qasper \
  --backend offline \
  --splits test \
  --model-revision vllm-offline-b128
```

Evaluate BioASQ on all splits:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset bioasq \
  --backend offline \
  --splits train dev test \
  --model-revision vllm-offline-b128
```

---

## Script

```
scripts/evaluation/mteb/run_mteb_retrieval.py
```

Library code lives under `src/tesis_unicamp/evaluation/mteb/`.

---

## Embedding backends

Use `--backend` to choose how the model is loaded:

| Backend | Engine | Default model | Notes |
|---------|--------|---------------|-------|
| `offline` (default) | vLLM in-process | `Qwen/Qwen3-Embedding-4B` | Same as indexing/retrieval scripts. Requires GPU. |
| `online` | OpenAI-compatible API | `Qwen/Qwen3-Embedding-4B` | Point to vLLM or another server via `.env`. |
| `sentence-transformers` | Sentence Transformers | User-defined (`--model`) | Useful for baselines or built-in MTEB leaderboard tasks. |

The offline backend sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` and preloads the model before evaluation to avoid CUDA fork errors with MTEB.

---

## Predefined project datasets

Pass one of these values to `--dataset`:

| Key | MTEB task name | Hugging Face repo |
|-----|----------------|-------------------|
| `bioasq` | BioASQ-RAG | `DinoStackAI/bioasq-rag-13b` |
| `bioasq-resplit` | BioASQ-RAG-Resplit | `DinoStackAI/bioasq-rag-13b-resplit` |
| `qasper` | QASPER-RAG | `DinoStackAI/qasper-rag` |
| `telco-dpr` | TelcoDPR-RAG | `DinoStackAI/telco-dpr-rag` |
| `narrativeqa` | NarrativeQA-RAG | `DinoStackAI/narrativeqa-rag` |

Each dataset exposes Hub configs: `corpus`, `queries`, `qrels`, and `answers`. Evaluation uses `corpus` + `queries` + `qrels`.

### Corpus and query formatting

- **Corpus text** matches the indexing scripts (e.g. QASPER includes `title`, `section_name`, and `text`).
- **Queries** use the instruct format by default (same as `retrieve_rag_top_k.py`):

  ```
  Instruct: Given a web search query, retrieve relevant passages that answer the query
  Query:<question text>
  ```

  Pass `--raw-queries` to disable the instruct wrapper.

---

## Fine-tuned LoRA models (Qwen3-Emb-4b-lora)

Evaluate each **dataset-specific LoRA adapter** on its matching RAG task with the `sentence-transformers` backend. Use one GPU per run (`CUDA_VISIBLE_DEVICES=0`).

Hub repos (default):

| Dataset | `--dataset` | Hub model |
|---------|-------------|-----------|
| telco-dpr | `telco-dpr` | `DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr` |
| qasper | `qasper` | `DinoStackAI/Qwen3-Emb-4b-lora-qasper` |
| narrativeqa | `narrativeqa` | `DinoStackAI/Qwen3-Emb-4b-lora-narrativeqa` |
| bioasq-resplit | `bioasq-resplit` | `DinoStackAI/Qwen3-Emb-4b-lora-bioasq-resplit` |

`--model-revision` is the **MTEB results subfolder** (your run label). `--hf-revision` (default: `main`) is only the Hub git revision when loading from Hugging Face.

### telco-dpr

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset telco-dpr \
  --backend sentence-transformers \
  --model DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr \
  --model-revision telco-dpr-b128-e20 \
  --splits test \
  --batch-size 128
```

### qasper

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset qasper \
  --backend sentence-transformers \
  --model DinoStackAI/Qwen3-Emb-4b-lora-qasper \
  --model-revision qasper-b128-e10 \
  --splits test \
  --batch-size 128
```

### narrativeqa

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset narrativeqa \
  --backend sentence-transformers \
  --model DinoStackAI/Qwen3-Emb-4b-lora-narrativeqa \
  --model-revision narrativeqa-b128-e10 \
  --splits test \
  --batch-size 128
```

### bioasq-resplit

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset bioasq-resplit \
  --backend sentence-transformers \
  --model DinoStackAI/Qwen3-Emb-4b-lora-bioasq-resplit \
  --model-revision bioasq-resplit-b128-e10 \
  --splits test \
  --batch-size 128
```

Results are written under:

```
results/mteb/results/DinoStackAI__Qwen3-Emb-4b-lora-<dataset>/<model-revision>/<TaskName>.json
```

Example:

```
results/mteb/results/DinoStackAI__Qwen3-Emb-4b-lora-qasper/qasper-b128-e10/QASPER-RAG.json
```

### Local adapters (without Hub)

Replace `--model` with the path to `final/`:

```bash
--model models/qwen3-embedding-4b-lora/qasper-b128-e10/final
```

For faster evaluation, see [Fine-tuned LoRA models with vLLM (offline)](#fine-tuned-lora-models-with-vllm-offline).

---

## Fine-tuned LoRA models with vLLM (offline)

Evaluate each **dataset-specific LoRA adapter** on its matching RAG task with the `offline` backend (vLLM in-process). Use one GPU per run (`CUDA_VISIBLE_DEVICES=0`).

`--model` must be the **base model** (`Qwen/Qwen3-Embedding-4B`). `--lora-path` is the Hub repo or local path to the adapter (`final/`). `--model-revision` is the **MTEB results subfolder** (your run label).

| Dataset | `--dataset` | `--lora-path` |
|---------|-------------|---------------|
| telco-dpr | `telco-dpr` | `DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr` |
| qasper | `qasper` | `DinoStackAI/Qwen3-Emb-4b-lora-qasper` |
| narrativeqa | `narrativeqa` | `DinoStackAI/Qwen3-Emb-4b-lora-narrativeqa` |
| bioasq-resplit | `bioasq-resplit` | `DinoStackAI/Qwen3-Emb-4b-lora-bioasq-resplit` |

### telco-dpr

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset telco-dpr \
  --backend offline \
  --model Qwen/Qwen3-Embedding-4B \
  --lora-path DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr \
  --model-revision vllm-lora-telco-dpr-b128-e20 \
  --splits test \
  --batch-size 128
```

### qasper

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset qasper \
  --backend offline \
  --model Qwen/Qwen3-Embedding-4B \
  --lora-path DinoStackAI/Qwen3-Emb-4b-lora-qasper \
  --model-revision vllm-lora-qasper-b128-e10 \
  --splits test \
  --batch-size 128
```

### narrativeqa

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset narrativeqa \
  --backend offline \
  --model Qwen/Qwen3-Embedding-4B \
  --lora-path DinoStackAI/Qwen3-Emb-4b-lora-narrativeqa \
  --model-revision vllm-lora-narrativeqa-b128-e10 \
  --splits test \
  --batch-size 128
```

### bioasq-resplit

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset bioasq-resplit \
  --backend offline \
  --model Qwen/Qwen3-Embedding-4B \
  --lora-path DinoStackAI/Qwen3-Emb-4b-lora-bioasq-resplit \
  --model-revision vllm-lora-bioasq-resplit-b128-e10 \
  --splits test \
  --batch-size 128
```

### Local adapters (without Hub)

Replace `--lora-path` with the path to `final/`:

```bash
--lora-path models/qwen3-embedding-4b-lora/qasper-b128-e10/final
```

Results are written under:

```
results/mteb/results/Qwen__Qwen3-Embedding-4B/<model-revision>/<TaskName>.json
```

Example:

```
results/mteb/results/Qwen__Qwen3-Embedding-4B/vllm-lora-qasper-b128-e10/QASPER-RAG.json
```

The first run includes vLLM warmup and CUDA graph capture (~1–2 min on H100); inference is faster than the `sentence-transformers` backend.

---

## CLI options

```bash
python scripts/evaluation/mteb/run_mteb_retrieval.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | — | Project RAG dataset key (see table above). |
| `--mteb-task` | — | Built-in MTEB retrieval task, e.g. `NFCorpus`. |
| `--hf-repo-id` | — | Custom Hub repo with `corpus` / `queries` / `qrels` configs. |
| `--task-name` | repo suffix | Task name when using `--hf-repo-id`. |
| `--task-description` | generic text | Task description for custom Hub datasets. |
| `--backend` | `offline` | `offline`, `online`, or `sentence-transformers`. |
| `--model` | `Qwen/Qwen3-Embedding-4B` | Model name or Hugging Face repo id. |
| `--model-revision` | *(required)* | MTEB results subfolder name (e.g. `lora-bioasq-resplit-b128-e10`). |
| `--lora-path` | — | LoRA adapter Hub repo or local path (`offline` backend only). |
| `--max-lora-rank` | `16` | `max_lora_rank` for vLLM when `--lora-path` is set. |
| `--hf-revision` | `main` | Hub git revision when loading `--model` from Hugging Face (ST backend). |
| `--batch-size` | `128` | Embedding batch size (project backends). |
| `--splits` | `test` | One or more of: `train`, `dev`, `test`. |
| `--output-dir` | `results/mteb` | Directory for MTEB result cache. |
| `--raw-queries` | off | Skip instruct formatting on queries. |
| `--overwrite` | `always` | MTEB cache strategy: `always`, `never`, `only-missing`, `only-cache`. |

Exactly one of `--dataset`, `--mteb-task`, or `--hf-repo-id` is required.

---

## Results

Results are written under `--output-dir` (default: `results/mteb/`):

```
results/mteb/
  results/
    Qwen__Qwen3-Embedding-4B/
      vllm-offline-b128/              # from --model-revision
        QASPER-RAG.json       # metrics per split
        model_meta.json
        run_settings.jsonl    # batch size, MTEB version, etc.
```

Example:

```bash
python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset qasper --backend offline --splits test \
  --model-revision vllm-offline-b128
```

```
results/mteb/results/Qwen__Qwen3-Embedding-4B/vllm-offline-b128/QASPER-RAG.json
```

### Main metrics

| Metric | Meaning |
|--------|---------|
| `ndcg_at_10` | Normalized DCG at rank 10 (main score). |
| `recall_at_k` | Fraction of all relevant docs found in top-k. |
| `map_at_k` | Mean average precision at k. |
| `mrr_at_k` | Mean reciprocal rank at k. |
| `hit_rate_at_k` | Fraction of queries with ≥1 relevant doc in top-k. |

Open the task JSON file and check `scores.<split>[0].main_score` for the headline number (defaults to `ndcg_at_10`).

### Interpreting scores

- Custom tasks such as **QASPER-RAG** are **not** directly comparable to MTEB leaderboard benchmarks (NFCorpus, SciFact, etc.).
- QASPER uses a **large shared corpus** (~81k chunks) with **multiple relevant chunks per query**, which makes full-corpus dense retrieval harder than small-corpus benchmarks.
- Low `ndcg_at_10` can still be a valid baseline; compare `hit_rate_at_10` and `recall_at_k` for a fuller picture.

---

## Examples

### Built-in MTEB task with Sentence Transformers

```bash
python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --mteb-task NFCorpus \
  --backend sentence-transformers \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model-revision main
```

### Custom Hugging Face dataset

The repo must provide configs named `corpus`, `queries`, and `qrels` (with `query_id`, `corpus_id`, `score` columns in qrels):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --hf-repo-id user/my-rag-dataset \
  --task-name MyRAG \
  --backend offline \
  --splits test \
  --model-revision vllm-offline-b128
```

### Online backend (vLLM server)

```bash
# Terminal 1
bash jobs/scripts/vllm/serve_embedding_4b.sh

# Terminal 2
python scripts/evaluation/mteb/run_mteb_retrieval.py \
  --dataset qasper \
  --backend online \
  --splits test \
  --model-revision vllm-online-b128
```

### Programmatic usage

```python
from tesis_unicamp.evaluation.mteb import (
    evaluate_retrieval,
    get_rag_retrieval_task,
    resolve_model,
)

task = get_rag_retrieval_task("qasper", eval_splits=("test",))
model = resolve_model(
    backend="offline",
    model="Qwen/Qwen3-Embedding-4B",
    model_revision="vllm-offline-b128",
)

results = evaluate_retrieval(
    model,
    [task],
    output_folder="results/mteb",
    encode_kwargs={"batch_size": 128},
)

for task_result in results:
    print(task_result)
```

---

## Performance notes

| Dataset | Approx. queries (all splits) | Suggestion |
|---------|------------------------------|------------|
| NarrativeQA | ~few thousand | Full run is reasonable |
| BioASQ | moderate | Full run is reasonable |
| Telco-DPR | moderate | Full run is reasonable |
| QASPER | ~81k corpus chunks; ~1.3k test queries | Start with `--splits test`; full run is slow |

MTEB embeds the **entire corpus** on every run (no Qdrant cache). QASPER test typically takes several minutes on a single H100-class GPU.

---

## Troubleshooting

### `Cannot re-initialize CUDA in forked subprocess`

Use `--backend offline` with an up-to-date checkout (spawn + vLLM warmup are applied automatically). Ensure `CUDA_VISIBLE_DEVICES` is set before launching the script.

### `ValidationError` for `TaskMetadata`

Custom task metadata must use valid MTEB v2 fields (e.g. domain `Academic`, not `Scientific`). Predefined project tasks already follow this schema.

### Very low nDCG on QASPER

Expected for full-corpus retrieval without paper-level filtering. See [Interpreting scores](#interpreting-scores) above.

---

## Related scripts

| Path | Purpose |
|------|---------|
| `../retrieval/retrieve_rag_top_k.py` | Top-k retrieval via Qdrant + `is_relevant` labels |
| `../embeddings/index_*_corpus.py` | Index corpus vectors into Qdrant |
| `../README.md` | End-to-end RAG pipeline documentation |
