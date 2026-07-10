# RAGAS answer evaluation (OpenAI API → vLLM)

Evaluate **generated RAG answers** with [RAGAS](https://docs.ragas.io/) by calling **vLLM OpenAI-compatible HTTP APIs**. The evaluation script does **not** load models locally; you run judge and embedding servers separately with `vllm serve`.

---

## Architecture

```
┌─────────────────────┐     HTTP /v1/chat/completions     ┌──────────────────┐
│  run_rag_ragas_     │ ─────────────────────────────────►│ vLLM judge       │
│  evaluation.py      │                                   │ (70B, GPUs…)     │
│  (CPU / login node) │     HTTP /v1/embeddings           └──────────────────┘
│                     │ ─────────────────────────────────►┌──────────────────┐
└─────────────────────┘                                   │ vLLM embeddings  │
                                                          │ (--task embed)   │
                                                          └──────────────────┘
```

| Component | Default model id | API |
|-----------|------------------|-----|
| **Judge LLM** | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | Chat completions |
| **Embeddings** | `Qwen/Qwen3-Embedding-8B` | Embeddings |

### Metrics

| Metric | Needs LLM | Needs embeddings |
|--------|-----------|------------------|
| `faithfulness` | Yes | No |
| `answer_correctness` | Yes | Yes |
| `semantic_similarity` | No | Yes |

---

## 1. Start vLLM servers (on GPU nodes)

Use the `--served-model-name` values that match what you pass to RAGAS.

**Judge (example: port 8000, node 1 / 4 GPUs):**

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-70B
```

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
| `RAGAS_JUDGE_MODEL` | DeepSeek 70B id | Must match `--served-model-name` on judge server |
| `RAGAS_EMBEDDING_MODEL` | Qwen3-Embedding-8B id | Must match embedding server |
| `RAGAS_TOKENIZER_MODEL` | same as judge | HF id for context truncation only |
| `RAGAS_MAX_WORKERS` | `4` | Concurrent RAGAS API workers |
| `RAGAS_API_TIMEOUT` | `300` | HTTP timeout (seconds) |

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

Increase `--api-timeout` and `--timeout` for large datasets or slow judge (R1 reasoning).

---

## Related

| Path | Purpose |
|------|---------|
| `src/tesis_unicamp/evaluation/ragas/openai_client.py` | OpenAI client helpers |
| `jobs/scripts/santos_dumont/run_ragas_eval.sh` | Env wrapper for evaluation |
| `scripts/generation/run_rag_generation.py` | Generate answers to evaluate |
