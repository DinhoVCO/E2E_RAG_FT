---
license: cc-by-2.5
task_categories:
  - question-answering
  - text-retrieval
language:
  - en
tags:
  - rag
  - bioasq
  - biomedical
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

# BioASQ RAG 13B (Resplit)

Reshuffled version of [`{{source_repo_id}}`](https://huggingface.co/datasets/{{source_repo_id}}) for Retrieval-Augmented Generation (RAG).

All original `train`, `dev` and `test` queries were merged, shuffled with seed `{{seed}}`, and reassigned using:

1. `{{test_ratio}}` of all queries → `test`
2. `{{dev_ratio}}` of the remaining queries → `dev`
3. the rest → `train`

The shared PubMed `corpus` is unchanged from the source dataset.

## Structure

| Subset | Splits | Description |
|--------|--------|-------------|
| `corpus` | train (default) | PubMed abstracts shared across all query splits |
| `queries` | train, dev, test | Biomedical questions |
| `qrels` | train, dev, test | Relevance judgments (query ↔ document) |
| `answers` | train, dev, test | Reference answers |

## Dataset statistics

| Split | Queries | Corpus |
|-------|--------:|-------:|
| train | {{train_queries}} | {{corpus_size}} |
| dev   | {{dev_queries}} | {{corpus_size}} |
| test  | {{test_queries}} | {{corpus_size}} |

- **Total queries merged:** {{total_queries}}
- **Test ratio:** {{test_ratio}}
- **Dev ratio (after test):** {{dev_ratio}}
- **Random seed:** {{seed}}

## Schema

### corpus
```json
{"id": "24323361", "title": "...", "text": "..."}
```

### queries
```json
{"id": "...", "text": "..."}
```

### qrels
```json
{"query_id": "...", "corpus_id": "24323361", "score": 1}
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
dev_qrels = qrels["dev"]
test_answers = answers["test"]
```

## Citation

BioASQ data are distributed under [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/). If you use this dataset, please cite the original BioASQ challenge papers:

> Nentidis, A., G. Katsimpras, A. Krithara, and G. Paliouras, "Overview of BioASQ Tasks 13b and Synergy13 in CLEF2025", CLEF 2025 Working Notes, 2025.

> George Tsatsaronis et al., "An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition", BMC bioinformatics, 2015.
