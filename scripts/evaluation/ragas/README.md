# RAGAS answer evaluation (OpenAI API → vLLM)

Evaluate **generated RAG answers** with [RAGAS](https://docs.ragas.io/) by calling **vLLM OpenAI-compatible HTTP APIs**. The evaluation script does **not** load models locally; you run judge and embedding servers separately with `vllm serve`.

---

## Architecture

```
┌─────────────────────┐     HTTP /v1/chat/completions     ┌──────────────────┐
│  run_rag_ragas_     │ ─────────────────────────────────►│ vLLM judge       │
│  evaluation.py      │                                   │ (Mistral-Small)  │
│  (CPU / login node) │     HTTP /v1/embeddings           └──────────────────┘
│                     │ ─────────────────────────────────►┌──────────────────┐
└─────────────────────┘                                   │ vLLM embeddings  │
                                                          │ (--task embed)   │
                                                          └──────────────────┘
```

| Component | Default model id | API |
|-----------|------------------|-----|
| **Judge LLM** | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Chat completions |
| **Embeddings** | `Qwen/Qwen3-Embedding-8B` | Embeddings |

### Metrics (default)

| Metric | Enabled by default | Needs LLM | Needs embeddings |
|--------|-------------------|-----------|------------------|
| `answer_correctness` | Yes | Yes | Yes |
| `semantic_similarity` | Yes | No | Yes |
| `faithfulness` | No | Yes | No |

---

## Reference metrics (response vs reference only)

Lightweight evaluation that compares `generated_answer` to `reference_answer` without retrieved context or judge LLM.

| Script | Metrics | Hardware |
|--------|---------|----------|
| `run_rag_reference_metrics_traditional.py` | BLEU, ROUGE, CHRF, exact match, string presence, non-LLM string similarity | CPU |
| `run_rag_reference_metrics_semantic.py` | `semantic_similarity` | GPU via vLLM embedding server (Qwen3-Embedding-8B) |

Traditional metrics follow the [RAGAS traditional NLP metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/traditional/) docs.

### vLLM embedding server (semantic similarity)

Start only the embedding model on 1 H100 (interactive):

```bash
bash jobs/scripts/santos_dumont/run_vllm_embedding_server_h100.sh
```

The batch job for semantic similarity starts vLLM embed on the same node automatically:

```bash
bash jobs/scripts/santos_dumont/run_rag_reference_metrics_semantic_h100.sh --all
```

Or point at an already-running server:

```bash
export RAGAS_EMBEDDING_BASE_URL=http://<embed-host>:8001/v1
export RAGAS_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
bash jobs/scripts/santos_dumont/run_rag_reference_metrics_semantic_h100.sh --all
```

### Evaluate all generated runs

```bash
# CPU metrics on ict-h100 (1 GPU requested for QOS; no GPU used by the script)
bash jobs/scripts/santos_dumont/run_rag_reference_metrics_traditional.sh --all

# GPU — semantic similarity (starts vLLM Qwen3-Embedding-8B on the same node)
bash jobs/scripts/santos_dumont/run_rag_reference_metrics_semantic_h100.sh --all
```

Or directly:

```bash
python scripts/evaluation/ragas/run_rag_reference_metrics_traditional.py --all --skip-existing
python scripts/evaluation/ragas/run_rag_reference_metrics_semantic.py --dataset telco-dpr
```

### Output layout

```
results/ragas-reference/
  traditional/<dataset>/<run_label>/test/reference_traditional_scores.json
  semantic-similarity/<dataset>/<run_label>/test/reference_semantic_similarity_scores.json
```

---

## 1. Start vLLM servers (on GPU nodes)

**Pre-download models on the login node** (compute nodes often hang on Hub downloads):

```bash
python scripts/download_hf.py --preset ragas-models
# judge only:
python scripts/download_hf.py --preset ragas-models --models judge
```

Or a single repo:

```bash
python scripts/download_hf.py --snapshot \
  --repo mistralai/Mistral-Small-3.1-24B-Instruct-2503 --repo-type model
```

**gpt-oss note:** if you use `openai/gpt-oss-20b` as judge (`JUDGE_MODEL=openai/gpt-oss-20b`), you need a vLLM build with gpt-oss support ([OpenAI cookbook](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm)). On SDumont compute nodes, also pre-download Harmony tiktoken vocabs on the login node:

```bash
bash jobs/scripts/santos_dumont/download_tiktoken_encodings.sh
# creates data/tiktoken_encodings/{o200k_base,cl100k_base}.tiktoken
```

`run_vllm_servers_h100.sh` sets `TIKTOKEN_ENCODINGS_BASE` automatically for gpt-oss judges.

**SLURM helper (Santos Dumont / ict-h100):**

```bash
bash jobs/scripts/santos_dumont/run_vllm_servers_h100.sh
```

Judge on node 1 (port 8000), embeddings on node 2 (port 8001). RAGAS only needs reachable URLs.

**Judge (example: port 8000, 1× H100):**

```bash
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --served-model-name mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

For gpt-oss judges, RAGAS sends `reasoning_effort=none` by default (no thinking channel; better JSON for RAGAS prompts).

**Embeddings (example: port 8001, node 2 / 1–4 GPUs):**

```bash
vllm serve Qwen/Qwen3-Embedding-8B \
  --host 0.0.0.0 --port 8001 \
  --task embed \
  --tensor-parallel-size 1 \
  --served-model-name Qwen/Qwen3-Embedding-8B
```

With 2 nodes you can run judge on node 1 and embeddings on node 2 — RAGAS only needs reachable URLs.

---

## 2. Run evaluation

```bash
export RAGAS_JUDGE_BASE_URL=http://<judge-host>:8000/v1
export RAGAS_EMBEDDING_BASE_URL=http://<embed-host>:8001/v1
export RAGAS_OPENAI_API_KEY=EMPTY

bash jobs/scripts/santos_dumont/run_ragas_eval.sh \
  datasets/generated/telco-dpr/generation-b128-vllm-offline-b128
```

Or directly:

```bash
python scripts/evaluation/ragas/run_rag_ragas_evaluation.py \
  --generation-dir datasets/generated/telco-dpr/generation-b128-vllm-offline-b128 \
  --judge-base-url http://127.0.0.1:8000/v1 \
  --embedding-base-url http://127.0.0.1:8001/v1
```

No `CUDA_VISIBLE_DEVICES` is required on the machine running RAGAS.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGAS_JUDGE_BASE_URL` | `http://127.0.0.1:8000/v1` | Judge vLLM OpenAI base URL |
| `RAGAS_EMBEDDING_BASE_URL` | `http://127.0.0.1:8001/v1` | Embedding vLLM OpenAI base URL |
| `RAGAS_OPENAI_API_KEY` | `EMPTY` | API key (vLLM usually ignores it) |
| `RAGAS_JUDGE_MODEL` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Must match `--served-model-name` on judge server |
| `RAGAS_EMBEDDING_MODEL` | Qwen3-Embedding-8B id | Must match embedding server |
| `RAGAS_TOKENIZER_MODEL` | same as judge | HF id for context truncation only |
| `RAGAS_MAX_WORKERS` | `64` | Concurrent RAGAS API workers |
| `RAGAS_JUDGE_MAX_TOKENS` | `2048` | Max tokens per judge completion |
| `RAGAS_API_TIMEOUT` | `300` | HTTP timeout (seconds) |

---

## Performance tuning

RAGAS is often **client-bound**: vLLM on H100 can serve many more parallel requests than the default worker count.

| Knob | Default (updated) | Effect |
|------|-------------------|--------|
| `RAGAS_MAX_WORKERS` / `--max-workers` | `64` | More concurrent HTTP calls to judge + embeddings |
| `RAGAS_JUDGE_MAX_TOKENS` / `--judge-max-tokens` | `2048` | Max tokens per judge completion (higher reduces JSON truncation) |
| `JUDGE_MAX_NUM_SEQS` (vLLM server) | `128` | More in-flight sequences on the judge GPU |
| `EMBEDDING_MAX_NUM_SEQS` | `256` | Higher embedding throughput |
| `EMBEDDING_MAX_MODEL_LEN` | `8192` | Lower than Qwen default (40k); frees KV cache for batching |

**Suggested fast run** (if vLLM servers keep up):

```bash
export RAGAS_MAX_WORKERS=64
export RAGAS_JUDGE_MAX_TOKENS=2048

bash jobs/scripts/santos_dumont/run_ragas_eval.sh <generation-dir>
```

If you see HTTP 503 / timeouts, lower `RAGAS_MAX_WORKERS` (e.g. 64 → 32) or raise `--api-timeout` and `--timeout`.

**vLLM server-side** (optional overrides when starting servers):

```bash
JUDGE_MAX_NUM_SEQS=256 EMBEDDING_MAX_NUM_SEQS=512 \
  bash jobs/scripts/santos_dumont/run_vllm_servers_h100.sh
```

**gpt-oss on H100:** your `judge.log` warns that **FlashInfer is missing** — installing it can improve MoE throughput:

```bash
uv pip install 'vllm[flashinfer]'
```

Run RAGAS from a node **close to** the vLLM hosts (same cluster, not over slow WAN) to cut network latency.

---

## CLI options

```bash
python scripts/evaluation/ragas/run_rag_ragas_evaluation.py --help
```

Key flags: `--judge-base-url`, `--embedding-base-url`, `--judge-model`, `--embedding-model`, `--tokenizer-model`, `--max-workers`, `--skip-server-check`.

---

## Outputs

```
results/ragas/<dataset>/<run_label>/
  run_settings.json
  test/
    ragas_scores.json
    ragas_summary.json
```

---

## Troubleshooting

### Mistral tokenizer warning

vLLM may log `incorrect regex pattern` for `Mistral-Small-3.1-24B-Instruct-2503`. Do **not** pass `--tokenizer-mode mistral` on vLLM 0.11 with this model — it crashes during multimodal init (`MistralTokenizer` lacks `convert_tokens_to_ids`). The warning is harmless for text-only RAGAS judge use.

RAGAS context truncation uses `fix_mistral_regex=True` in `tokenizer.py` (client side only).

### Connection refused

Ensure vLLM servers are up and URLs point to the correct host/port (use node hostname from `$SLURM_JOB_NODELIST`, not `localhost`, if eval runs on a different node).

### Model not listed

The script calls `GET /v1/models` at startup. `--judge-model` / `--embedding-model` must match `--served-model-name` on each vLLM instance.

### Timeouts

Increase `--api-timeout` and `--timeout` for large datasets or slow judge responses.

---

## Related

| Path | Purpose |
|------|---------|
| `src/tesis_unicamp/evaluation/ragas/openai_client.py` | OpenAI client helpers |
| `jobs/scripts/santos_dumont/run_ragas_eval.sh` | Env wrapper for evaluation |
| `scripts/generation/run_rag_generation.py` | Generate answers to evaluate |
