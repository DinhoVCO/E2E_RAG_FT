# Query expansion scripts (RFG)

Command-line utilities to run **RFG** (*Retrieval-Feedback-Grounded Multi-Query Expansion*) experiments: retrieve documents, generate long-form expanded queries with an LLM, and evaluate second-stage retrieval with MTEB.

Run commands from the **repository root** with the virtual environment activated:

```bash
source .venv/bin/activate
# or: uv run python ...
```

---

## Overview

| Script | Purpose |
|--------|---------|
| `run_rfg_experiment.py` | Orchestrate the full 3-stage pipeline from YAML config |
| `run_rfg_stage1_retrieval.py` | Stage 1: embed corpus and retrieve top-k documents |
| `run_rfg_generate_expansion.py` | Stage 2: generate long-form expanded queries (one or more k values) |
| `run_rfg_stage2_mteb.py` | Stage 3: MTEB retrieval evaluation using expanded queries |
| `configs/experiments.yaml` | Experiment matrix (16 runs: embedding × generation × dataset) |

Library code lives under `src/tesis_unicamp/query_expansion/`.

SLURM batch submission (Santos Dumont H100):

```bash
bash jobs/scripts/santos_dumont/run_rfg_experiment_h100.sh --submit-each --all
```

---

## How RFG works

RFG is a **two-stage retrieval** approach with an LLM feedback step in the middle:

1. **Stage 1 — Initial retrieval**  
   Embed the original query with `Qwen3-Embedding-4B` (base or LoRA) and retrieve the top documents from the corpus (default: top-10, stored under `datasets/retrieved_inmemory/`).

2. **Query expansion — LLM feedback**  
   Pass the first *k* retrieved documents (k ∈ {1, 3, 5, 7, 10}) to `Qwen3-8B` (base or LoRA). The model produces a **long-form answer** that serves as an expanded query. If the documents are insufficient, the model may use its own knowledge.

3. **Stage 2 — Expanded retrieval (MTEB)**  
   Embed the long-form response with the **same embedding model** used in stage 1 and run MTEB retrieval against the full corpus. This step is repeated for each expansion k.

Each stage runs as a **separate subprocess** inside `run_rfg_experiment.py` so vLLM reloads cleanly when switching between the embedding and generation models.

```
Original query
      ↓
[Stage 1] Embedding retrieval (top-10 docs)
      ↓
[Expansion] LLM long-form answer from top-k docs  (k = 1, 3, 5, 7, 10)
      ↓
[Stage 2] Embed expanded answer → MTEB retrieval evaluation
```

---

## Experiment matrix

`configs/experiments.yaml` defines **16 experiments**:

- **4 datasets:** `telco-dpr`, `qasper`, `narrativeqa`, `bioasq-resplit`
- **4 model combos per dataset:**
  - embedding base + generation base
  - embedding base + generation LoRA
  - embedding LoRA + generation base
  - embedding LoRA + generation LoRA

Example experiment id: `telco-dpr-rfg-emb-lora-gen-base`

List all experiments:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_experiment.py --list
```

---

## Prerequisites

1. Python environment with project dependencies (`uv sync`).
2. A CUDA GPU and `CUDA_VISIBLE_DEVICES` set for offline vLLM runs.
3. **Stage 1 retrieval (recommended):** existing top-10 results under `datasets/retrieved_inmemory/`.  
   If you already ran the standard RAG retrieval pipeline, stage 1 is **skipped automatically**.

Expected stage-1 layout (reused from the RAG pipeline):

```
datasets/retrieved_inmemory/
  telco-dpr/vllm-offline-b128/test/retrieved_docs.json
  telco-dpr/vllm-lora-telco-dpr-b128/test/retrieved_docs.json
  qasper/vllm-offline-b128-paper-scoped/test/retrieved_docs.json
  ...
```

If stage-1 files are missing, `run_rfg_experiment.py` runs retrieval before expansion (unless `--skip-stage1` is set).

---

## Expansion prompt and token limits

Each expansion uses the RFG instruction and the same document layout as generative RAG fine-tuning (`src/tesis_unicamp/query_expansion/prompts.py`):

```
Provide a detailed, long-form answer to the query using the retrieved
documents as context. If the documents do not contain sufficient
information, supplement your answer using your own knowledge.
## Query:
{question}
## Context:
doc 1 :
{document 1}
...
## Response:
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expansion_k_values` | `1, 3, 5, 7, 10` | Number of retrieved docs passed to the LLM |
| `max_tokens_per_chunk` | `2048` | Max tokens per document in the prompt |
| `generation_max_tokens` | `2048` | Max tokens generated for the expanded query |
| `retrieval_top_k` | `10` | Documents retrieved in stage 1 (must be ≥ max k) |

The chat template of Qwen3-8B is applied; thinking mode is disabled by default.

---

## Output layout

### Stage 1 (retrieval)

```
datasets/retrieved_inmemory/<dataset>/<retrieval-run-label>/test/retrieved_docs.json
```

Run labels match the RAG pipeline:

- Base: `vllm-offline-b128` (QASPER: `vllm-offline-b128-paper-scoped`)
- LoRA: `vllm-lora-<dataset>-b128` (QASPER: `...-paper-scoped`)

### Expansion (per k)

```
datasets/query_expansion/<dataset>/<run-label>/k1/test/expanded_queries.json
datasets/query_expansion/<dataset>/<run-label>/k3/test/expanded_queries.json
...
datasets/query_expansion/<dataset>/<run-label>/k10/test/expanded_queries.json
```

Each JSON record contains: `query_id`, `question`, `expanded_query`, `reference_answer`.

### Stage 2 (MTEB)

```
results/mteb/rfg/<dataset>/
```

MTEB result subfolder (model revision), e.g.:

`rfg-telco-dpr-emb-base-gen-lora-k5`

---

## Quick start

### Full pipeline — one experiment

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_experiment.py \
    --experiment telco-dpr-rfg-emb-base-gen-lora
```

### All experiments for one dataset

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_experiment.py --dataset qasper
```

### All 16 experiments

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_experiment.py --all
```

### Dry run (print commands only)

```bash
python scripts/query_expansion/run_rfg_experiment.py \
    --experiment bioasq-resplit-rfg-emb-base-gen-base --dry-run
```

---

## Running individual stages

### Stage 1 only

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_experiment.py \
    --experiment qasper-rfg-emb-base-gen-base --stage1-only
```

Or directly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_stage1_retrieval.py \
    --dataset qasper --mode offline --run-label vllm-offline-b128 --top-k 10
```

### Expansion only (stage 1 must exist)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_experiment.py \
    --experiment narrativeqa-rfg-emb-lora-gen-lora --expansion-only
```

Or directly (all k values in one vLLM session):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_generate_expansion.py \
    --dataset narrativeqa \
    --retrieval-run-label vllm-lora-narrativeqa-b128 \
    --run-label narrativeqa-rfg-emb-lora-gen-lora \
    --expansion-k 1 3 5 7 10
```

### Stage 2 MTEB only (expansions must exist)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_experiment.py \
    --experiment bioasq-resplit-rfg-emb-base-gen-base --stage2-only
```

Or for a single k:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/run_rfg_stage2_mteb.py \
    --dataset bioasq-resplit \
    --expanded-queries-dir datasets/query_expansion/bioasq-resplit/bioasq-resplit-rfg-emb-base-gen-base/k5 \
    --model-revision rfg-bioasq-resplit-emb-base-gen-base-k5
```

---

## Orchestrator flags

`run_rfg_experiment.py` supports the same skip/force patterns as `run_rag_experiment.py`:

| Flag | Effect |
|------|--------|
| `--skip-stage1` | Skip retrieval; require existing `retrieved_docs.json` |
| `--force-stage1` | Always re-run stage-1 retrieval |
| `--skip-expansion` | Skip expansion if all k outputs exist |
| `--force-expansion` | Regenerate all expansion k values |
| `--skip-stage2` | Stop after expansion |
| `--stage1-only` | Run retrieval only |
| `--expansion-only` | Run expansion only |
| `--stage2-only` | Run MTEB only |
| `--dry-run` | Print subprocess commands without executing |

Stage 1 and expansion are **skipped automatically** when the expected output files already exist.

---

## SLURM (Santos Dumont H100)

Submit from the repo root:

```bash
# One experiment
bash jobs/scripts/santos_dumont/run_rfg_experiment_h100.sh \
    --experiment telco-dpr-rfg-emb-base-gen-base

# One GPU job per experiment (16 parallel jobs)
bash jobs/scripts/santos_dumont/run_rfg_experiment_h100.sh --submit-each --all

# Four jobs for one dataset
bash jobs/scripts/santos_dumont/run_rfg_experiment_h100.sh --submit-each --dataset qasper

# Skip stage 1 when retrieved_inmemory already exists
bash jobs/scripts/santos_dumont/run_rfg_experiment_h100.sh \
    --experiment qasper-rfg-emb-lora-gen-lora --skip-stage1
```

Defaults: partition `ict-h100`, 1 GPU, 12 h wall time, logs under `logs/slurm/`.

Override with environment variables: `JOB_NAME`, `TIME`, `ACCOUNT`, `PARTITION`, `MEM`.

---

## Configuration

Edit `configs/experiments.yaml` to change defaults or add experiments.

Key defaults:

```yaml
defaults:
  embedding_model: Qwen/Qwen3-Embedding-4B
  generation_model: Qwen/Qwen3-8B
  retrieval_top_k: 10
  expansion_k_values: [1, 3, 5, 7, 10]
  generation_max_tokens: 2048
  max_tokens_per_chunk: 2048
  retrieved_root: retrieved_inmemory
  stage2_model_revision_template: "rfg-{dataset}-emb-{embedding}-gen-{generation}-k{k}"
```

Per-dataset LoRA paths are under the `datasets:` section (same Hub repos as the RAG experiments).

Use a custom config file:

```bash
python scripts/query_expansion/run_rfg_experiment.py \
    --config path/to/experiments.yaml --experiment my-experiment-id
```

---

## Related scripts

| Area | Location |
|------|----------|
| Standard RAG retrieval (stage-1 input) | `scripts/retrieval/retrieve_rag_top_k_inmemory.py` |
| RAG generation (short answers) | `scripts/generation/` |
| MTEB embedding evaluation | `scripts/evaluation/mteb/` |
| Context-augmented MTEB (different approach) | `scripts/evaluation/mteb/run_mteb_context_retrieval.py` |

RFG differs from context MTEB in that the **expanded query is a full LLM-generated answer**, not a structured context-augmented embedding prompt.
