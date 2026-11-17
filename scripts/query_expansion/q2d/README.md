# Query2Doc (Q2D)

Query2Doc expands each test query by generating a pseudo-passage with a few-shot
prompt built from random train `(query, passage)` pairs, then concatenates the
original query with the generated passage for dense retrieval evaluation.

## Pipeline

1. **Few-shot pool** — Build train examples from qrels + corpus (first relevant
   document per train query, truncated to 2048 tokens with the generation model tokenizer).
2. **Generation** — For each test query, sample 4 train examples and prompt
   `Qwen/Qwen3-8B` to write a passage (vLLM offline).
3. **Retrieval** — Embed `query + generated_passage` with `Qwen/Qwen3-Embedding-4B`
   (instruct format) and run MTEB on the 4 RAG datasets.

## Prompt format

```
Write a passage that answers the given query:

Query: {train_query_1}
Passage: {train_passage_1}

Query: {train_query_2}
Passage: {train_passage_2}

...

Query: {target_query}
Passage:
```

## Usage

```bash
# List experiments
python scripts/query_expansion/q2d/run_q2d_experiment.py --list

# Full pipeline (generation + MTEB)
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/q2d/run_q2d_experiment.py \
    --experiment bioasq-resplit-q2d

# Generation only
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/q2d/run_q2d_generate.py \
    --dataset bioasq-resplit \
    --run-label bioasq-resplit-q2d

# MTEB only
CUDA_VISIBLE_DEVICES=0 python scripts/query_expansion/q2d/run_q2d_mteb.py \
    --dataset bioasq-resplit \
    --q2d-dir datasets/query_expansion/q2d/bioasq-resplit/bioasq-resplit-q2d \
    --model-revision q2d-bioasq-resplit
```

## Outputs

| Artifact | Path |
|----------|------|
| Expansions | `datasets/query_expansion/q2d/{dataset}/{run_label}/test/q2d_expansions.json` |
| MTEB results | `results/mteb/q2d/{dataset}/` |

Each expansion record stores `question`, `generated_passage`, `expanded_query`
(query + passage), and the 4 few-shot examples used.

## SLURM (ict-h100)

```bash
bash jobs/scripts/santos_dumont/run_q2d_experiment_h100.sh --experiment bioasq-resplit-q2d
bash jobs/scripts/santos_dumont/run_q2d_experiment_h100.sh --submit-each --all
```
