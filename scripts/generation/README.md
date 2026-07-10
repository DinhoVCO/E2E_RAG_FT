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

### User message

```
Context:
[1] <chunk 1, truncated to 512 tokens>

[2] <chunk 2, truncated to 512 tokens>
...

Question: <raw query text>

Answer:
```

Notes:

- Queries use **raw text** from the `queries` subset (no retrieval instruct format).
- Each retrieved chunk is truncated to `--max-tokens-per-chunk` (default: 512).
- `--top-k` controls how many retrieved documents are included (by rank).
- Qwen3 **thinking is disabled by default** (`enable_thinking=False` in the chat template).

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
