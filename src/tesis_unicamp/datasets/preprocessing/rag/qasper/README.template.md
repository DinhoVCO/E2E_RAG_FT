---
license: cc-by-4.0
task_categories:
  - question-answering
  - text-retrieval
language:
  - en
tags:
  - rag
  - qasper
  - scientific-qa
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

# QASPER RAG

Dataset for Retrieval-Augmented Generation (RAG) based on [QASPER](https://huggingface.co/datasets/{{hf_dataset_id}}).

## Structure

| Subset | Splits | Description |
|--------|--------|-------------|
| `corpus` | train (default) | Paper chunks (abstract + full-text paragraphs) shared across all query splits |
| `queries` | train, dev, test | Information-seeking questions over scientific papers |
| `qrels` | train, dev, test | Relevance judgments (query ↔ paragraph chunk) |
| `answers` | train, dev, test | Reference answers (longest valid free-form answer) |

## Dataset statistics

| Split | Queries | Corpus |
|-------|--------:|-------:|
| train | {{train_queries}} | {{corpus_size}} |
| dev   | {{dev_queries}} | {{corpus_size}} |
| test  | {{test_queries}} | {{corpus_size}} |

The corpus is shared across all splits and contains paragraph-level chunks from the abstract and full text of each paper.

- **Dev split:** mapped from the original `validation` split
- **Corpus source:** unique papers from train, validation and test splits
- **Chunking:** one chunk for the abstract (`section_name: abstract`) and one chunk per paragraph in `full_text`

## Source

| Component | QASPER resource |
|-----------|-----------------|
| Train | `train` split from [{{hf_dataset_id}}](https://huggingface.co/datasets/{{hf_dataset_id}}) |
| Dev | `validation` split |
| Test | `test` split |
| Corpus | `abstract` + `full_text.paragraphs` |
| Queries | `qas.question` |
| Qrels | `qas.answers[*].answer[*].evidence` matched to corpus chunks |
| Answers | Longest valid answer per question (`free_form_answer`, joined `extractive_spans`, or `Yes`/`No`) |

## Filtering

Questions are kept only when at least one answer satisfies all of the following:

- `unanswerable` is `false`
- the answer has `free_form_answer`, non-empty `extractive_spans`, or `yes_no`
- after removing evidence items containing `FLOAT SELECTED`, at least one answer still has evidence that matches the corpus

Evidence items containing `FLOAT SELECTED` are removed individually. Questions are omitted only when no valid answer has remaining evidence.

## Schema

### corpus
```json
{"id": "...", "title": "...", "section_name": "...", "text": "..."}
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

QASPER is released under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

```bibtex
@inproceedings{Dasigi2021ADO,
  title={A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers},
  author={Pradeep Dasigi and Kyle Lo and Iz Beltagy and Arman Cohan and Noah A. Smith and Matt Gardner},
  year={2021}
}
```
