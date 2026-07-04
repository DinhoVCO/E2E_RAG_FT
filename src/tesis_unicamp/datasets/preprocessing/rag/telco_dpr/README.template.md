---
license: apache-2.0
task_categories:
  - question-answering
  - text-retrieval
language:
  - en
tags:
  - rag
  - telco-dpr
  - 3gpp
  - telecom
  - retrieval
size_categories:
  - 10K<n<100K
configs:
  - config_name: corpus
    data_files:
      - split: train
        path: corpus/*
  - config_name: queries
    data_files:
      - split: train
        path: queries/train*
      - split: dev
        path: queries/dev*
      - split: test
        path: queries/test*
  - config_name: qrels
    data_files:
      - split: train
        path: qrels/train*
      - split: dev
        path: qrels/dev*
      - split: test
        path: qrels/test*
  - config_name: answers
    data_files:
      - split: train
        path: answers/train*
      - split: dev
        path: answers/dev*
      - split: test
        path: answers/test*
---

# Telco-DPR RAG

Dataset for Retrieval-Augmented Generation (RAG) based on [Telco-DPR](https://huggingface.co/datasets/{{hf_dataset_id}}).

## Structure

| Subset | Splits | Description |
|--------|--------|-------------|
| `corpus` | train (default) | 3GPP technical passages (text + tables) shared across all query splits |
| `queries` | train, dev, test | Synthetic telecom QA questions |
| `qrels` | train, dev, test | Relevance judgments (query ↔ passage) |
| `answers` | train, dev, test | Reference answers |

## Dataset statistics

| Split | Queries | Corpus |
|-------|--------:|-------:|
| train | {{train_queries}} | {{corpus_size}} |
| dev   | {{dev_queries}} | {{corpus_size}} |
| test  | {{test_queries}} | {{corpus_size}} |

The corpus is shared across all splits and concatenates the `small` and `extended` splits from the original `corpus` subset.

- **Corpus source:** `corpus/small` + `corpus/extended` from [{{hf_dataset_id}}](https://huggingface.co/datasets/{{hf_dataset_id}})
- **Train/Test splits:** mapped from the original `relevant_docs` subset
- **Dev split:** {{dev_ratio}} of the original train split (random seed {{seed}})

## Source

| Component | Telco-DPR resource |
|-----------|-------------------|
| Train | `relevant_docs/train` (after dev holdout) |
| Dev | `relevant_docs/train` ({{dev_ratio}} holdout, seed {{seed}}) |
| Test | `relevant_docs/test` |
| Corpus | `corpus/small` + `corpus/extended` |
| Queries | `queries` subset |
| Answers | `answer` field from `queries` |

## Schema

### corpus
```json
{"id": "...", "title": "...", "text": "..."}
```

### queries
```json
{"id": "...", "text": "..."}
```

### qrels
```json
{"query_id": "...", "corpus_id": "...", "score": 1}
```

### answers
```json
{"query_id": "...", "answer": "..."}
```

## Usage

```python
from datasets import load_dataset

corpus = load_dataset("{{repo_id}}", "corpus")["train"]
queries = load_dataset("{{repo_id}}", "queries")
qrels = load_dataset("{{repo_id}}", "qrels")
answers = load_dataset("{{repo_id}}", "answers")

train_queries = queries["train"]
test_qrels = qrels["test"]
test_answers = answers["test"]
```

## Citation

Telco-DPR is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

```bibtex
@article{telco-dpr-2024,
    title = {Telco-DPR: A Hybrid Dataset for Retrieval-Augmented Generation in the Telecom Domain},
    author = {Saraiva, Thaina and others},
    journal = {arXiv preprint arXiv:2410.19790},
    year = {2024},
}
```
