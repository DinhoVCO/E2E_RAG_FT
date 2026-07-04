---
license: apache-2.0
task_categories:
  - question-answering
  - text-retrieval
language:
  - en
tags:
  - rag
  - narrativeqa
  - reading-comprehension
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

# NarrativeQA RAG

Dataset for Retrieval-Augmented Generation (RAG) based on [NarrativeQA](https://huggingface.co/datasets/{{hf_dataset_id}}).

## Structure

| Subset | Splits | Description |
|--------|--------|-------------|
| `corpus` | train (default) | Wikipedia plot summaries shared across all query splits |
| `queries` | train, dev, test | Reading comprehension questions |
| `qrels` | train, dev, test | Relevance judgments (query ↔ document) |
| `answers` | train, dev, test | Reference answers (longest annotated answer) |

## Dataset statistics

| Split | Queries | Corpus |
|-------|--------:|-------:|
| train | {{train_queries}} | {{corpus_size}} |
| dev   | {{dev_queries}} | {{corpus_size}} |
| test  | {{test_queries}} | {{corpus_size}} |

The corpus is shared across all splits and contains Wikipedia plot summaries (`document.summary.text`) from the original NarrativeQA documents.

- **Dev split:** mapped from the original `validation` split
- **Corpus source:** unique documents from train, validation and test splits

## Source

| Component | NarrativeQA resource |
|-----------|----------------------|
| Train | `train` split from [{{hf_dataset_id}}](https://huggingface.co/datasets/{{hf_dataset_id}}) |
| Dev | `validation` split |
| Test | `test` split |
| Corpus | `document.summary.text` (+ `document.summary.title`) |
| Answers | Longest answer text per question |

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
dev_qrels = qrels["dev"]
test_answers = answers["test"]
```

## Citation

NarrativeQA is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

```bibtex
@article{kocisky-etal-2018-narrativeqa,
    title = "The {N}arrative{QA} Reading Comprehension Challenge",
    author = "Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
      Schwarz, Jonathan  and
      Blunsom, Phil  and
      Dyer, Chris  and
      Hermann, Karl Moritz  and
      Melis, G{\'a}bor  and
      Grefenstette, Edward",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "6",
    year = "2018",
    pages = "317--328",
}
```
