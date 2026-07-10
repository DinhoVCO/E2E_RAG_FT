# RAGAS answer evaluation (OpenAI API → vLLM)

Evaluate **generated RAG answers** with [RAGAS](https://docs.ragas.io/) by calling **vLLM OpenAI-compatible HTTP APIs**. The evaluation script does **not** load models locally; you run judge and embedding servers separately with `vllm serve`.

---

## Architecture

```
┌─────────────────────┐     HTTP /v1/chat/completions     ┌──────────────────┐
│  run_rag_ragas_     │ ─────────────────────────────────►│ vLLM judge       │
│  evaluation.py      │                                   │ (gpt-oss-20b)    │
│  (CPU / login node) │     HTTP /v1/embeddings           └──────────────────┘
│                     │ ─────────────────────────────────►┌──────────────────┐
└─────────────────────┘                                   │ vLLM embeddings  │
                                                          │ (--task embed)   │
                                                          └──────────────────┘
```

| Component | Default model id | API |
|-----------|------------------|-----|
| **Judge LLM** | `openai/gpt-oss-20b` | Chat completions |
| **Embeddings** | `Qwen/Qwen3-Embedding-8B` | Embeddings |

### Metrics (default)

| Metric | Enabled by default | Needs LLM | Needs embeddings |
|--------|-------------------|-----------|------------------|
| `answer_correctness` | Yes | Yes | Yes |
| `semantic_similarity` | Yes | No | Yes |
| `faithfulness` | No | Yes | No |

---

## 1. Start vLLM servers (on GPU nodes)

**gpt-oss note:** `openai/gpt-oss-20b` needs a vLLM build with gpt-oss support ([OpenAI cookbook](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm)). On SDumont compute nodes, also pre-download Harmony tiktoken vocabs on the login node:

```bash
bash jobs/scripts/santos_dumont/download_tiktoken_encodings.sh
# creates data/tiktoken_encodings/{o200k_base,cl100k_base}.tiktoken
```

`run_vllm_servers_h100.sh` sets `TIKTOKEN_ENCODINGS_BASE` automatically for gpt-oss judges.

**SLURM helpers (Santos Dumont / ict-h100):**

| Script | Allocation | Layout |
|--------|------------|--------|
| `run_vllm_servers_h100.sh` | 2 nodes × 4 GPUs | Judge on node 1, embeddings on node 2 |
| `run_vllm_servers_h100_1n2g.sh` | **1 node × 2 GPUs** | Judge on GPU 0, embeddings on GPU 1 (same host) |

```bash
# Compact: one node, two GPUs
bash jobs/scripts/santos_dumont/run_vllm_servers_h100_1n2g.sh

# Both URLs use the same hostname, different ports:
#   http://<node>:8000/v1  (judge)
#   http://<node>:8001/v1  (embeddings)
```

Use the `--served-model-name` values that match what you pass to RAGAS.

**Judge (example: port 8000, 1× H100):**

```bash
vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --served-model-name openai/gpt-oss-20b
```

RAGAS sends `reasoning_effort=none` by default for gpt-oss judges (no thinking channel; better JSON for RAGAS prompts).

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
| `RAGAS_JUDGE_MODEL` | `openai/gpt-oss-20b` | Must match `--served-model-name` on judge server |
| `RAGAS_EMBEDDING_MODEL` | Qwen3-Embedding-8B id | Must match embedding server |
| `RAGAS_TOKENIZER_MODEL` | same as judge | HF id for context truncation only |
| `RAGAS_MAX_WORKERS` | `16` | Concurrent RAGAS API workers |
| `RAGAS_JUDGE_MAX_TOKENS` | `1024` | Max tokens per judge completion |
| `RAGAS_API_TIMEOUT` | `300` | HTTP timeout (seconds) |

---

## Performance tuning

RAGAS is often **client-bound**: vLLM on H100 can serve many more parallel requests than the default worker count.

| Knob | Default (updated) | Effect |
|------|-------------------|--------|
| `RAGAS_MAX_WORKERS` / `--max-workers` | `16` | More concurrent HTTP calls to judge + embeddings |
| `RAGAS_JUDGE_MAX_TOKENS` / `--judge-max-tokens` | `1024` | Shorter judge outputs → faster generation (JSON metrics) |
| `JUDGE_MAX_NUM_SEQS` (vLLM server) | `128` | More in-flight sequences on the judge GPU |
| `EMBEDDING_MAX_NUM_SEQS` | `256` | Higher embedding throughput |
| `EMBEDDING_MAX_MODEL_LEN` | `8192` | Lower than Qwen default (40k); frees KV cache for batching |

**Suggested fast run** (after restarting vLLM servers with the new script defaults):

```bash
export RAGAS_MAX_WORKERS=32
export RAGAS_JUDGE_MAX_TOKENS=512

bash jobs/scripts/santos_dumont/run_ragas_eval.sh <generation-dir>
```

Increase `RAGAS_MAX_WORKERS` gradually (16 → 32 → 64). If you see HTTP 503 / timeouts, lower workers or raise `--api-timeout`.

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
