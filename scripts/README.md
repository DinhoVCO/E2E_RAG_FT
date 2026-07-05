# Scripts

Command-line utilities for embedding, indexing, retrieval, and Hugging Face dataset management.

Run commands from the **repository root** with the virtual environment activated:

```bash
source .venv/bin/activate
```

For cluster workflows (SLURM, Qdrant, GPU sessions), see [jobs/instructions.md](../jobs/instructions.md) and [jobs/qdrant.md](../jobs/qdrant.md).

---

## RAG pipeline overview

End-to-end workflow for each RAG dataset:

1. **Pre-download** corpus files on the login node (cluster only).
2. **Start Qdrant** on the compute node (inside your GPU session).
3. **Index** the corpus into Qdrant with vLLM embeddings.
4. **Retrieve** top-k documents for every query split.
5. **Push** the `retrieved_docs` subset to Hugging Face.

Default embedding model: `Qwen/Qwen3-Embedding-4B`.

---

## 0. Pre-download (login node, cluster)

Compute nodes often hang when downloading from Hugging Face. Cache datasets on the login node first:

```bash
# Corpus parquet only (for indexing)
python scripts/download_hf.py --preset rag-corpus --datasets qasper

# Full dataset repo (corpus + queries + qrels + answers + retrieved_docs)
python scripts/download_hf.py --preset rag-full --datasets qasper

# All RAG datasets, full repos
python scripts/download_hf.py --preset rag-full

python scripts/download_hf.py --snapshot --repo Qwen/Qwen3-Embedding-4B --repo-type model
```

See [download_hf.py](./download_hf.py) for custom downloads (`--repo`, `--file`, `--snapshot`).

---

## 1. Index corpus into Qdrant

Qdrant must be running before indexing (`bash jobs/scripts/santos_dumont/run_qdrant.sh`).

**Offline mode (recommended)** — loads the model in-process on the GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/embeddings/index_qasper_corpus.py --mode offline
```

**Online mode** — requires a vLLM embedding server:

```bash
bash jobs/scripts/vllm/serve_embedding_4b.sh
python scripts/embeddings/index_qasper_corpus.py --mode online
```

### Index scripts and default Qdrant collections

| Dataset | Index script | Default collection |
|---------|--------------|-------------------|
| BioASQ | `scripts/embeddings/index_bioasq_corpus.py` | `bioasq-rag-13b-corpus` |
| QASPER | `scripts/embeddings/index_qasper_corpus.py` | `qasper-rag-corpus` |
| Telco-DPR | `scripts/embeddings/index_telco_dpr_corpus.py` | `telco-dpr-rag-corpus` |
| NarrativeQA | `scripts/embeddings/index_narrativeqa_corpus.py` | `narrativeqa-rag-corpus` |

Common options:

```bash
--collection NAME          # override Qdrant collection
--qdrant-url URL           # default: http://localhost:6333
--batch-size N             # default: 128 (env: EMBED_BATCH_SIZE)
--recreate-collection      # delete and recreate the collection
```

Verify indexing:

```bash
curl -s http://localhost:6333/collections/qasper-rag-corpus | jq '.result.points_count'
```

---

## 2. Retrieve top-k documents

Embeds queries, searches Qdrant, labels relevance from `qrels`, and saves results locally.

**Offline mode (recommended):**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k.py \
  --dataset qasper \
  --mode offline \
  --top-k 10
```

**Online mode:**

```bash
bash jobs/scripts/vllm/serve_embedding_4b.sh
python scripts/retrieval/retrieve_rag_top_k.py \
  --dataset qasper \
  --mode online \
  --top-k 10
```

### Available datasets

Use `--dataset` with one of: `bioasq`, `qasper`, `telco-dpr`, `narrativeqa`.

### Useful options

```bash
--splits train dev test     # default: all three splits
--splits test               # run a single split first (recommended for QASPER)
--output-dir PATH           # default: datasets/retrieved/<dataset>_rag/
--collection NAME           # override Qdrant collection
--batch-size N              # query embedding batch size
```

### Output layout

```
datasets/retrieved/
  qasper_rag/
    train/retrieved_docs.json
    dev/retrieved_docs.json
    test/retrieved_docs.json
    hf_dataset/retrieved_docs/
```

### `retrieved_docs` schema

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
| `corpus_id` | Document id from the Qdrant payload (original corpus id) |
| `rank` | Rank in the retrieval list (1 = best match) |
| `retrieval_score` | Cosine similarity score from Qdrant |
| `is_relevant` | `true` if `corpus_id` appears in `qrels` for that query and split |

---

## 3. Push `retrieved_docs` to Hugging Face

Uploads the `retrieved_docs` config (train/dev/test) to the dataset repo on the Hub:

```bash
python scripts/retrieval/push_retrieved_docs_to_hub.py --dataset qasper
```

Options:

```bash
--repo-id dinho1597/qasper-rag   # default: dataset-specific Hub repo
--output-dir PATH                # default: datasets/retrieved/<dataset>_rag/
--token TOKEN                    # default: HF_TOKEN env var
--private                        # upload as private dataset
```

Default Hub repos:

| Dataset | Repo id |
|---------|---------|
| BioASQ | `dinho1597/bioasq-rag-13b` |
| QASPER | `dinho1597/qasper-rag` |
| Telco-DPR | `dinho1597/telco-dpr-rag` |
| NarrativeQA | `dinho1597/narrativeqa-rag` |

Load after upload:

```python
from datasets import load_dataset

retrieved = load_dataset("dinho1597/qasper-rag", "retrieved_docs")
test_hits = retrieved["test"]
```

---

## How offline retrieval works

With `--mode offline`:

1. **Embed queries** with `VLLMOfflineEmbedder` (vLLM in-process, same model used for indexing).
2. **Search Qdrant** for the top-k nearest corpus vectors per query.
3. **Set `is_relevant`** by comparing each hit's `corpus_id` against `qrels` for the same split.
4. **Save** JSON and a local Hugging Face export under `datasets/retrieved/`.

Offline mode embeds **queries only**. Vector search runs on **Qdrant** using the corpus vectors indexed in step 1.

Queries use the instruct format (`query_to_instruct_text`), consistent with BioASQ retrieval:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:<question text>
```

---

## Performance notes

| Dataset | Approx. queries (all splits) | Suggestion |
|---------|------------------------------|------------|
| NarrativeQA | ~few thousand | Full run is reasonable |
| BioASQ | moderate | Full run is reasonable |
| Telco-DPR | moderate | Full run is reasonable |
| QASPER | ~81k per split × 3 splits | Start with `--splits test`; full run may take several hours |

Each query produces `top_k` rows (default 10), so QASPER generates on the order of **millions** of `retrieved_docs` rows for all splits.

---

## Other scripts

| Path | Purpose |
|------|---------|
| `scripts/download_hf.py` | Pre-download Hub datasets and models |
| `scripts/embeddings/test_qwen3_embedding.py` | Smoke test for online/offline embeddings |
| `scripts/create_dataset/create_*_rag_dataset.py` | Build and push gold RAG subsets from raw sources |
