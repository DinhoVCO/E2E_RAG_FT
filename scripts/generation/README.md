# Generation scripts

Command-line utilities to generate RAG answers from previously retrieved documents using vLLM offline.

Run commands from the **repository root** with the virtual environment activated:

```bash
source .venv/bin/activate
```

---

## Overview

| Script | Purpose |
|--------|---------|
| `run_rag_generation.py` | Generate answers for one dataset |
| `run_rag_experiment.py` | Run retrieval + generation from YAML experiment config |
| `run_all_rag_generation.sh` | Run base-model generation for all four datasets sequentially |

Generation is a **downstream step** after retrieval. It does **not** re-embed or re-retrieve documents. It reads local `retrieved_docs`, joins them with corpus text from Hugging Face, builds a RAG prompt, and generates answers with an LLM.

Default generation model: `Qwen/Qwen3-8B`.

Library code lives under `src/tesis_unicamp/generation/`.

---

## Prerequisites

1. Python environment with project dependencies installed (`uv sync`).
2. A CUDA GPU and `CUDA_VISIBLE_DEVICES` set.
3. Local `retrieved_docs` from a previous retrieval run (see [../retrieval/README.md](../retrieval/README.md)).

Expected input layout (default retrieval label):

```
datasets/retrieved_inmemory/
  qasper/vllm-offline-b128/test/retrieved_docs.json
  telco-dpr/vllm-offline-b128/test/retrieved_docs.json
  narrativeqa/vllm-offline-b128/test/retrieved_docs.json
  bioasq-resplit/vllm-offline-b128/test/retrieved_docs.json
```

If retrieval has not been run yet:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/retrieval/run_all_inmemory_retrieval.sh
# or base model only:
SKIP_LORA=1 bash scripts/retrieval/run_all_inmemory_retrieval.sh
```

---

## Pipeline

```
retrieved_docs (local JSON)
        +
corpus / queries / answers (Hugging Face)
        ↓
build context (top-k docs, 512 tokens/chunk)
        ↓
vLLM offline generation (Qwen3-8B)
        ↓
datasets/generated/<dataset>/<run-label>/test/generated_answers.json
```

---

## Quick start

### All four datasets (base model)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/generation/run_all_rag_generation.sh
```

### One dataset

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generation/run_rag_generation.py \
  --dataset qasper \
  --retrieval-run-label vllm-offline-b128 \
  --run-label generation-b128-vllm-offline-b128
```

### Cluster (SLURM)

```bash
# Request a GPU session
bash jobs/scripts/santos_dumont/run_ict_h100.sh

# Inside the session
cd /path/to/E2E_RAG_FT
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/generation/run_all_rag_generation.sh
```

---

## Experiment matrix (YAML)

`scripts/generation/configs/experiments.yaml` defines **32 experiments**:

| Type | Example id |
|------|------------|
| RAG matrix (emb × gen) | `telco-dpr-emb-base-gen-lora-top5` |
| Base emb + QA LoRA + top 5 | `telco-dpr-emb-base-gen-qa-top5` |
| Base gen, no retrieval | `telco-dpr-gen-base-noretrieval` |
| QA LoRA, no retrieval | `telco-dpr-gen-qa-noretrieval` |
| RAG LoRA, no retrieval | `telco-dpr-gen-lora-noretrieval` |

QA adapters are loaded from `models/qwen3-8b-lora-qa/<run>/final` (see `generation_lora_qa` per dataset in the YAML).

Each experiment configures:

- **dataset**
- **embedding** (`base` or `lora`) → retrieval model and `retrieval_run_label`
- **generation** (`base`, `lora`, or `qa`) → generation model and LoRA path
- **use_retrieval** (`true` / `false`) → whether to run retrieval and include docs in the prompt
- **prompt_mode** (auto): `inference` for legacy RAG runs, `qa` for QA-only, `rag-finetune` for fine-tune template
- **run_label** → output folder under `datasets/generated/<dataset>/`
- **top_k** → documents used in the generation prompt (default: 5)

LoRA repo ids per dataset live under `datasets:` in the YAML. Defaults (`embedding_model`, `generation_model`, `retrieval_top_k`, etc.) live under `defaults:`.

```bash
# List experiments
python scripts/generation/run_rag_experiment.py --list

# One experiment (retrieval if missing, then generation)
CUDA_VISIBLE_DEVICES=0 python scripts/generation/run_rag_experiment.py \
  --experiment telco-dpr-emb-lora-gen-lora-top5

# All 4 experiments for one dataset
CUDA_VISIBLE_DEVICES=0 python scripts/generation/run_rag_experiment.py --dataset telco-dpr

# All 16 experiments
CUDA_VISIBLE_DEVICES=0 python scripts/generation/run_rag_experiment.py --all

# Generation only (retrieval already done)
CUDA_VISIBLE_DEVICES=0 python scripts/generation/run_rag_experiment.py \
  --experiment telco-dpr-emb-base-gen-lora-top5 --skip-retrieval

# Preview commands
python scripts/generation/run_rag_experiment.py \
  --experiment qasper-emb-lora-gen-lora-top5 --dry-run
```

Retrieval and generation run as **separate subprocesses** so vLLM reloads cleanly between the embedding and generation models.

Output example:

```
datasets/generated/telco-dpr/telco-dpr-emb-lora-gen-lora-top5/test/generated_answers.json
```

Override any field per experiment in the YAML, e.g. custom `run_label`, `top_k`, `embedding_lora`, or `retrieval_run_label`.

### Cluster (SLURM batch)

Submit a non-interactive job on `ict-h100` (same pattern as fine-tuning):

```bash
bash jobs/scripts/santos_dumont/run_rag_experiment_h100.sh \
  --experiment telco-dpr-emb-lora-gen-lora-top5

bash jobs/scripts/santos_dumont/run_rag_experiment_h100.sh --dataset telco-dpr

TIME=24:00:00 bash jobs/scripts/santos_dumont/run_rag_experiment_h100.sh --all
```

Logs go to `logs/slurm/rag-<experiment>-<job_id>.out`.

---

## Script reference

### `run_all_rag_generation.sh`

Runs base-model generation for:

- `telco-dpr`
- `qasper`
- `narrativeqa`
- `bioasq-resplit`

Defaults:

| Variable | Default |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | `0` |
| `MODEL` | `Qwen/Qwen3-8B` |
| `TOP_K` | `5` |
| `RETRIEVAL_RUN_LABEL` | `vllm-offline-b128` |
| `GENERATION_RUN_LABEL` | `generation-b128` |

Example with overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
TOP_K=10 \
RETRIEVAL_RUN_LABEL=vllm-offline-b128 \
GENERATION_RUN_LABEL=generation-b128 \
bash scripts/generation/run_all_rag_generation.sh
```

This script runs datasets **sequentially on one GPU**. It does not use LoRA adapters.

---

### `run_rag_generation.py`

Generate answers for a single dataset.

**Base model (default):**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generation/run_rag_generation.py \
  --dataset telco-dpr \
  --model Qwen/Qwen3-8B \
  --retrieval-run-label vllm-offline-b128 \
  --run-label generation-b128-vllm-offline-b128 \
  --top-k 5
```

**With a generation LoRA adapter** (not embedding LoRA):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generation/run_rag_generation.py \
  --dataset qasper \
  --model Qwen/Qwen3-8B \
  --lora-path path/or/hub/repo/to/generation-lora \
  --retrieval-run-label vllm-lora-qasper-b128 \
  --run-label generation-lora-qasper
```

#### Available datasets

| Key | Hugging Face repo |
|-----|-------------------|
| `telco-dpr` | `DinoStackAI/telco-dpr-rag` |
| `qasper` | `DinoStackAI/qasper-rag` |
| `narrativeqa` | `DinoStackAI/narrativeqa-rag` |
| `bioasq-resplit` | `DinoStackAI/bioasq-rag-13b-resplit` |

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | RAG dataset key |
| `--model` | `Qwen/Qwen3-8B` | Generation model name or Hub repo id |
| `--lora-path` | — | LoRA adapter for the **generation** model |
| `--max-lora-rank` | `16` | `max_lora_rank` for vLLM when `--lora-path` is set |
| `--retrieval-run-label` | `vllm-offline-b128` | Subfolder under `datasets/retrieved_inmemory/<dataset>/` |
| `--retrieved-dir` | auto | Override path to `retrieved_docs` directory |
| `--run-label` | `generation-default` | Subfolder under `datasets/generated/<dataset>/` |
| `--output-dir` | auto | Override output directory |
| `--split` | `test` | Query split to generate answers for |
| `--top-k` | `10` | Number of retrieved documents to include in the prompt |
| `--max-tokens-per-chunk` | `512` | Token limit per retrieved chunk |
| `--max-prompt-tokens` | auto | Max prompt tokens (`max_model_len - max_tokens - 256`) |
| `--max-tokens` | `512` | Max tokens to generate per answer |
| `--temperature` | `0.0` | Sampling temperature |
| `--batch-size` | `8` | Prompts per generation batch |
| `--enable-thinking` | off | Enable Qwen3 thinking mode (disabled by default) |
| `--no-chat-template` | off | Send raw prompts without chat template |

#### Output layout

```
datasets/generated/
  qasper/generation-b128-vllm-offline-b128/
    test/generated_answers.json
    run_settings.json
```

---

## Prompt format

### System message

```
Answer the question based only on the provided context.
If the context does not contain enough information, say so briefly.
```

### User message (with retrieval, default)

```
Answer the query using the documents as support...
## Query:
<raw query text>

## Context:
doc 1 :
<title>\n\n<text chunk, truncated to 512 tokens by default>

doc 2 :
...
## Response:
```

With `--include-title-prompt` (see `experiments_title.yaml`):

```
## Title:
<gold document title from qrels>

## Query:
...

## Context:
doc 1 :
{title}
{body}
...
```

Notes:

- Queries use **raw text** from the `queries` subset (no retrieval instruct format).
- Each retrieved chunk is truncated to `--max-tokens-per-chunk` (default: 512).
- `--top-k` controls how many retrieved documents are included (by rank).
- Qwen3 **thinking is disabled by default** (`enable_thinking=False` in the chat template).
- `--include-title-prompt` (default: off) adds the gold document title before the query.
  Retrieved chunks already embed their own title in the text; the explicit `## Title:`
  section anchors the question to the source document even when retrieval is noisy.

### Title-aware experiments (`experiments_title.yaml`)

A separate 32-experiment matrix lives in `scripts/generation/configs/experiments_title.yaml`.
It enables `--include-title-prompt`, uses **title-aware retrieval** (saved under
`datasets/retrieved_inmemory_title/`), **top 5**, **2048 tokens/chunk**, **2048 max
generation tokens**, and a **14336** prompt budget.

Retrieval query format:

```
Instruct: Given a web search query, retrieve relevant passages that answer the query
## Title:
<gold document title from qrels>
Query:<question>
```

Generation context docs are rendered as:

```
doc 1 :
{title}
{body}
```

```bash
python scripts/generation/run_rag_experiment.py \
  --config scripts/generation/configs/experiments_title.yaml \
  --experiment telco-dpr-emb-base-gen-base-title-top5-2k
```

Cluster (SLURM batch) for the title matrix:

```bash
bash jobs/scripts/santos_dumont/run_rag_experiment_title_h100.sh \
  --experiment telco-dpr-emb-base-gen-base-title-top5-2k

bash jobs/scripts/santos_dumont/run_rag_experiment_title_h100.sh --dataset telco-dpr

TIME=24:00:00 bash jobs/scripts/santos_dumont/run_rag_experiment_title_h100.sh --all
```

Submit one GPU job per experiment (parallel):

```bash
bash jobs/scripts/santos_dumont/run_rag_experiment_title_h100.sh \
  --submit-each --dataset telco-dpr

bash jobs/scripts/santos_dumont/run_rag_experiment_title_h100.sh --submit-each --all
```

Logs go to `logs/slurm/rag-title-<experiment>-<job_id>.out`.

### No-retrieval prompts (`--no-retrieval`)

When generating without retrieved context, add `--include-title-prompt` to prepend the
gold document title from qrels:

```
Answer the question based only on the provided context.
## Title:
Pump Up the Volume (film)

## Query:
Where does this radio station take place?

## Response:
```

For qasper, the title is the paper name. For telco-dpr, it is the spec section path
(e.g. `NR and NG-RAN Overall Description;Stage 2 | 6 Layer 2 | ...`).

Disable with `--no-include-title-prompt` (default for the base `experiments.yaml` matrix).

The `question` field in `generated_answers.json` still stores the original query text
(without the title) for evaluation traceability.

---

## `generated_answers` schema

Each row in `generated_answers.json`:

```json
{
  "query_id": "q1",
  "question": "What method was used?",
  "generated_answer": "The model used ...",
  "reference_answer": "Gold answer from the answers subset"
}
```

`run_settings.json` stores the full run configuration (model, `top_k`, retrieval label, thinking mode, etc.).

---

## Linking retrieval and generation

The `--retrieval-run-label` must match the folder created during retrieval:

| Retrieval `--run-label` | Generation `--retrieval-run-label` |
|-------------------------|-------------------------------------|
| `vllm-offline-b128` | `vllm-offline-b128` |
| `vllm-lora-telco-dpr-b128` | `vllm-lora-telco-dpr-b128` |

`--top-k` in generation must be `<=` the `top-k` used during retrieval. If you need more documents, re-run retrieval with a higher `--top-k`.

---

## Environment variables

| Variable | Used for | Default |
|----------|----------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` in batch script |
| `VLLM_LLM_MODEL` | `--model` | `Qwen/Qwen3-8B` |
| `LORA_PATH` | `--lora-path` | — |
| `MAX_LORA_RANK` | `--max-lora-rank` | `16` |
| `RETRIEVAL_RUN_LABEL` | `--retrieval-run-label` | `vllm-offline-b128` |
| `GENERATION_RUN_LABEL` | `--run-label` | `generation-default` |
| `GENERATION_TOP_K` | `--top-k` | `10` |
| `GENERATION_MAX_TOKENS_PER_CHUNK` | `--max-tokens-per-chunk` | `512` |
| `GENERATION_MAX_PROMPT_TOKENS` | `--max-prompt-tokens` | auto |
| `GENERATION_MAX_TOKENS` | `--max-tokens` | `512` |
| `GENERATION_TEMPERATURE` | `--temperature` | `0.0` |
| `GENERATION_BATCH_SIZE` | `--batch-size` | `8` |

---

## Performance notes

| Dataset | Approx. test queries | Notes |
|---------|-------------------|-------|
| Telco-DPR | moderate | Reasonable on one H100/A100 |
| NarrativeQA | moderate | Reasonable on one GPU |
| BioASQ-resplit | moderate | Long PubMed abstracts; chunk truncation helps |
| QASPER | ~1.3k test queries | Longest run; several hours on one GPU |

- The batch script reloads vLLM for **each dataset** (4 separate Python processes).
- Default `TOP_K=5` and `512 tokens/chunk` keeps prompts well within `max_model_len`.
- Use `--top-k 10` only if retrieval was run with `--top-k 10`.

---

## Troubleshooting

### `Missing retrieved docs for split 'test'`

Run retrieval first:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/retrieval/retrieve_rag_top_k_inmemory.py \
  --dataset qasper --mode offline --run-label vllm-offline-b128
```

### `No CUDA device is visible`

Request a GPU in your SLURM session and set `CUDA_VISIBLE_DEVICES=0`.

### Prompt too long / `max_model_len` error

Reduce context size:

```bash
--top-k 3 --max-tokens-per-chunk 256
```

### Model generates thinking blocks

Thinking is disabled by default. Do **not** pass `--enable-thinking` unless you want reasoning mode.

### Used embedding LoRA as `--lora-path`

Embedding adapters (`Qwen3-Emb-4b-lora-*`) are for retrieval only. Generation `--lora-path` must be a **generation model** adapter.

---

## Related scripts

| Path | Purpose |
|------|---------|
| `../retrieval/README.md` | Top-k retrieval (input for generation) |
| `../retrieval/run_all_inmemory_retrieval.sh` | Batch retrieval for all datasets |
| `../evaluation/mteb/run_mteb_retrieval.py` | IR metrics (no answer generation) |
| `../README.md` | End-to-end RAG pipeline overview |
| `../../src/tesis_unicamp/generation/` | Generation library code |
