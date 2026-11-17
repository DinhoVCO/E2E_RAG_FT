"""Registro dos conjuntos de dados do benchmark RAG."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from tesis_unicamp.datasets.utils.bioasq_rag import (
    BIOASQ_RAG_DATASET_ID,
    BIOASQ_RAG_RESPLIT_DATASET_ID,
    corpus_row_to_text as bioasq_corpus_row_to_text,
)
from tesis_unicamp.datasets.utils.narrativeqa_rag import (
    NARRATIVEQA_RAG_DATASET_ID,
    corpus_row_to_text as narrativeqa_corpus_row_to_text,
)
from tesis_unicamp.datasets.utils.qasper_rag import (
    QASPER_RAG_DATASET_ID,
    corpus_row_to_text as qasper_corpus_row_to_text,
)
from tesis_unicamp.datasets.utils.telco_dpr_rag import (
    TELCO_DPR_RAG_DATASET_ID,
    corpus_row_to_text as telco_corpus_row_to_text,
)

SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class DatasetSpec:
    """Especificação de um dataset RAG padronizado."""

    chave: str
    rotulo: str
    dominio: str
    repo_id: str
    text_fn: Callable[[dict[str, Any]], str]
    unidade_corpus: str
    # Função opcional: id do documento "pai" (ex.: paper_id no QASPER).
    doc_pai_fn: Callable[[str], str] | None = None
    # Incluir no suite principal da tese (figuras/tabelas principais).
    suite_principal: bool = True


def _qasper_paper_id(corpus_id: str) -> str:
    return str(corpus_id).rsplit("_", 1)[0]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "qasper": DatasetSpec(
        chave="qasper",
        rotulo="QASPER-RAG",
        dominio="Literatura científica (PLN)",
        repo_id=QASPER_RAG_DATASET_ID,
        text_fn=qasper_corpus_row_to_text,
        unidade_corpus="Parágrafo / abstract de artigo",
        doc_pai_fn=_qasper_paper_id,
        suite_principal=True,
    ),
    "bioasq-resplit": DatasetSpec(
        chave="bioasq-resplit",
        rotulo="BioASQ-RAG-Resplit",
        dominio="Biomedicina",
        repo_id=BIOASQ_RAG_RESPLIT_DATASET_ID,
        text_fn=bioasq_corpus_row_to_text,
        unidade_corpus="Abstract PubMed",
        suite_principal=True,
    ),
    "narrativeqa": DatasetSpec(
        chave="narrativeqa",
        rotulo="NarrativeQA-RAG",
        dominio="Narrativa / literatura",
        repo_id=NARRATIVEQA_RAG_DATASET_ID,
        text_fn=narrativeqa_corpus_row_to_text,
        unidade_corpus="Resumo (summary) do documento",
        suite_principal=True,
    ),
    "telco-dpr": DatasetSpec(
        chave="telco-dpr",
        rotulo="Telco-DPR-RAG",
        dominio="Telecomunicações",
        repo_id=TELCO_DPR_RAG_DATASET_ID,
        text_fn=telco_corpus_row_to_text,
        unidade_corpus="Passagem técnica",
        suite_principal=True,
    ),
    "bioasq": DatasetSpec(
        chave="bioasq",
        rotulo="BioASQ-RAG (original)",
        dominio="Biomedicina",
        repo_id=BIOASQ_RAG_DATASET_ID,
        text_fn=bioasq_corpus_row_to_text,
        unidade_corpus="Abstract PubMed",
        suite_principal=False,
    ),
}

SUITE_PRINCIPAL = tuple(
    chave for chave, spec in DATASET_SPECS.items() if spec.suite_principal
)


def carregar_subset(repo_id: str, name: str, split: str | None = None) -> Dataset:
    dataset = load_dataset(repo_id, name)
    if isinstance(dataset, DatasetDict):
        if split is None:
            raise ValueError(f"O subset {name!r} possui partições; informe split=.")
        return dataset[split]
    return dataset


def carregar_corpus(spec: DatasetSpec) -> Dataset:
    dataset = load_dataset(spec.repo_id, "corpus")
    if isinstance(dataset, DatasetDict):
        # Corpus único compartilhado entre partições.
        if "train" in dataset:
            return dataset["train"]
        return dataset[next(iter(dataset.keys()))]
    return dataset
