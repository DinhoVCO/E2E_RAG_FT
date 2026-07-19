# HyDE query expansion

[HyDE](https://arxiv.org/abs/2212.10496) (*Hypothetical Document Embeddings*): generate pseudo-passages from each query, embed the query plus the passages, average the vectors, and retrieve with the mean embedding.

## Pipeline

1. **Generation** — `Qwen3-8B` samples **n=8** pseudo-passages per query (`SamplingParams.n=8`, temperature=0.7).
2. **Embedding** — `Qwen3-Embedding-4B` embeds:
   - instruct-formatted query
   - 8 raw pseudo-passages
3. **Search** — mean vector → MTEB retrieval evaluation.

## Prompt (all 4 datasets)

```
Please write a passage to answer the question.
Question: {query}
Passage:
```

## Quick start

```bash
# Full pipeline (one dataset)
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/hyde/run_hyde_experiment.py \
    --experiment bioasq-resplit-hyde

# All 4 datasets
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/hyde/run_hyde_experiment.py --all

# SLURM (1 GPU per experiment)
bash jobs/scripts/santos_dumont/run_hyde_experiment_h100.sh --submit-each --all
```

## Outputs

```
datasets/query_expansion/hyde/<dataset>/<run-label>/test/hyde_passages.json
results/mteb/hyde/<dataset>/   # model_revision e.g. hyde-bioasq-resplit
```

## Library

`src/tesis_unicamp/query_expansion/hyde/`
