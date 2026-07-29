"""Cálculo de métricas descritivas do benchmark RAG."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tesis_unicamp.datasets.analysis.registry import (
    DATASET_SPECS,
    SPLITS,
    SUITE_PRINCIPAL,
    DatasetSpec,
    carregar_corpus,
    carregar_subset,
)

TOKENIZER_PADRAO = "Qwen/Qwen3-Embedding-4B"
PERCENTIS = (50, 75, 90, 95, 99)


@dataclass
class EstatisticasComprimento:
    n: int
    media: float
    desvio: float
    minimo: int
    maximo: int
    percentis: dict[str, float]

    @classmethod
    def from_values(cls, values: list[int] | np.ndarray) -> EstatisticasComprimento:
        arr = np.asarray(values, dtype=np.int64)
        if arr.size == 0:
            return cls(
                n=0,
                media=0.0,
                desvio=0.0,
                minimo=0,
                maximo=0,
                percentis={f"p{p}": 0.0 for p in PERCENTIS},
            )
        percentis = {
            f"p{p}": float(np.percentile(arr, p)) for p in PERCENTIS
        }
        return cls(
            n=int(arr.size),
            media=float(arr.mean()),
            desvio=float(arr.std()),
            minimo=int(arr.min()),
            maximo=int(arr.max()),
            percentis=percentis,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EstatisticasParticao:
    n_consultas: int
    n_consultas_unicas: int
    n_qrels: int
    n_respostas: int
    n_consultas_sem_positivo: int
    relevants_por_consulta: EstatisticasComprimento
    tokens_consulta: EstatisticasComprimento | None = None
    palavras_consulta: EstatisticasComprimento | None = None
    tokens_resposta: EstatisticasComprimento | None = None
    palavras_resposta: EstatisticasComprimento | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_consultas": self.n_consultas,
            "n_consultas_unicas": self.n_consultas_unicas,
            "n_qrels": self.n_qrels,
            "n_respostas": self.n_respostas,
            "n_consultas_sem_positivo": self.n_consultas_sem_positivo,
            "relevants_por_consulta": self.relevants_por_consulta.to_dict(),
        }
        if self.tokens_consulta is not None:
            out["tokens_consulta"] = self.tokens_consulta.to_dict()
        if self.palavras_consulta is not None:
            out["palavras_consulta"] = self.palavras_consulta.to_dict()
        if self.tokens_resposta is not None:
            out["tokens_resposta"] = self.tokens_resposta.to_dict()
        if self.palavras_resposta is not None:
            out["palavras_resposta"] = self.palavras_resposta.to_dict()
        return out


@dataclass
class OverlapDocumentos:
    """Sobreposição de documentos relevantes entre partições."""

    conjuntos: dict[str, int]
    intersecoes: dict[str, int]
    fracoes_em_relacao_ao_menor: dict[str, float]
    fracao_test_em_train: float
    fracao_dev_em_train: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetMetrics:
    chave: str
    rotulo: str
    dominio: str
    repo_id: str
    unidade_corpus: str
    n_corpus: int
    n_docs_relevantes_unicos: int
    fracao_corpus_relevante: float
    tokens_corpus: EstatisticasComprimento | None
    palavras_corpus: EstatisticasComprimento
    particoes: dict[str, EstatisticasParticao]
    overlap_query_ids: dict[str, int]
    overlap_docs_relevantes: OverlapDocumentos
    overlap_docs_pai: OverlapDocumentos | None = None
    # Distribuições brutas (para plots); podem ser omitidas no JSON completo.
    dist_tokens_corpus: list[int] = field(default_factory=list)
    dist_palavras_corpus: list[int] = field(default_factory=list)
    dist_tokens_consulta_por_split: dict[str, list[int]] = field(default_factory=dict)
    dist_palavras_consulta_por_split: dict[str, list[int]] = field(
        default_factory=dict
    )
    dist_relevants_por_split: dict[str, list[int]] = field(default_factory=dict)

    def to_dict(self, *, incluir_distribuicoes: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "chave": self.chave,
            "rotulo": self.rotulo,
            "dominio": self.dominio,
            "repo_id": self.repo_id,
            "unidade_corpus": self.unidade_corpus,
            "n_corpus": self.n_corpus,
            "n_docs_relevantes_unicos": self.n_docs_relevantes_unicos,
            "fracao_corpus_relevante": self.fracao_corpus_relevante,
            "tokens_corpus": (
                self.tokens_corpus.to_dict() if self.tokens_corpus else None
            ),
            "palavras_corpus": self.palavras_corpus.to_dict(),
            "particoes": {k: v.to_dict() for k, v in self.particoes.items()},
            "overlap_query_ids": self.overlap_query_ids,
            "overlap_docs_relevantes": self.overlap_docs_relevantes.to_dict(),
            "overlap_docs_pai": (
                self.overlap_docs_pai.to_dict() if self.overlap_docs_pai else None
            ),
        }
        if incluir_distribuicoes:
            out["dist_tokens_corpus"] = self.dist_tokens_corpus
            out["dist_palavras_corpus"] = self.dist_palavras_corpus
            out["dist_tokens_consulta_por_split"] = self.dist_tokens_consulta_por_split
            out["dist_palavras_consulta_por_split"] = (
                self.dist_palavras_consulta_por_split
            )
            out["dist_relevants_por_split"] = self.dist_relevants_por_split
        return out


@dataclass
class FluxoBioasqResplit:
    """Mapeamento das consultas do BioASQ original para o resplit."""

    n_consultas_originais: int
    n_consultas_resplit: int
    matriz_transicao: dict[str, dict[str, int]]
    consultas_apenas_original: int
    consultas_apenas_resplit: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    tokenizer_nome: str | None
    datasets: dict[str, DatasetMetrics]
    fluxo_bioasq_resplit: FluxoBioasqResplit | None = None

    def to_dict(self, *, incluir_distribuicoes: bool = False) -> dict[str, Any]:
        return {
            "tokenizer_nome": self.tokenizer_nome,
            "datasets": {
                k: v.to_dict(incluir_distribuicoes=incluir_distribuicoes)
                for k, v in self.datasets.items()
            },
            "fluxo_bioasq_resplit": (
                self.fluxo_bioasq_resplit.to_dict()
                if self.fluxo_bioasq_resplit
                else None
            ),
        }


def _contar_palavras(texto: str) -> int:
    return len(texto.split())


def _contar_tokens_lote(
    tokenizer: PreTrainedTokenizerBase,
    textos: list[str],
    *,
    batch_size: int = 256,
    desc: str = "Tokens",
) -> list[int]:
    comprimentos: list[int] = []
    for inicio in tqdm(range(0, len(textos), batch_size), desc=desc, leave=False):
        lote = textos[inicio : inicio + batch_size]
        encoded = tokenizer(
            lote,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        comprimentos.extend(len(ids) for ids in encoded["input_ids"])
    return comprimentos


def _overlap_conjuntos(por_split: dict[str, set[str]]) -> OverlapDocumentos:
    tamanhos = {split: len(vals) for split, vals in por_split.items()}
    intersecoes: dict[str, int] = {}
    fracoes: dict[str, float] = {}
    pares = [("train", "dev"), ("train", "test"), ("dev", "test")]
    for a, b in pares:
        chave = f"{a}∩{b}"
        inter = por_split.get(a, set()) & por_split.get(b, set())
        intersecoes[chave] = len(inter)
        menor = min(len(por_split.get(a, set())), len(por_split.get(b, set()))) or 1
        fracoes[chave] = len(inter) / menor

    train = por_split.get("train", set())
    test = por_split.get("test", set())
    dev = por_split.get("dev", set())
    fracao_test_em_train = (len(test & train) / len(test)) if test else 0.0
    fracao_dev_em_train = (len(dev & train) / len(dev)) if dev else 0.0
    return OverlapDocumentos(
        conjuntos=tamanhos,
        intersecoes=intersecoes,
        fracoes_em_relacao_ao_menor=fracoes,
        fracao_test_em_train=fracao_test_em_train,
        fracao_dev_em_train=fracao_dev_em_train,
    )


def _overlap_ids_consulta(por_split: dict[str, list[str]]) -> dict[str, int]:
    sets = {s: set(ids) for s, ids in por_split.items()}
    out: dict[str, int] = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        out[f"{a}∩{b}"] = len(sets.get(a, set()) & sets.get(b, set()))
    return out


def _mapa_query_para_split(spec: DatasetSpec) -> dict[str, str]:
    mapa: dict[str, str] = {}
    for split in SPLITS:
        queries = carregar_subset(spec.repo_id, "queries", split=split)
        for qid in queries["id"]:
            # Em caso de IDs duplicados, mantém a primeira ocorrência.
            mapa.setdefault(str(qid), split)
    return mapa


def calcular_fluxo_bioasq_resplit() -> FluxoBioasqResplit:
    original = _mapa_query_para_split(DATASET_SPECS["bioasq"])
    resplit = _mapa_query_para_split(DATASET_SPECS["bioasq-resplit"])
    matriz: dict[str, dict[str, int]] = {
        src: {dst: 0 for dst in SPLITS} for src in SPLITS
    }
    comuns = set(original) & set(resplit)
    for qid in comuns:
        matriz[original[qid]][resplit[qid]] += 1
    return FluxoBioasqResplit(
        n_consultas_originais=len(original),
        n_consultas_resplit=len(resplit),
        matriz_transicao=matriz,
        consultas_apenas_original=len(set(original) - set(resplit)),
        consultas_apenas_resplit=len(set(resplit) - set(original)),
    )


def analisar_dataset(
    chave: str,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    batch_size: int = 256,
    max_corpus: int | None = None,
    max_queries: int | None = None,
) -> DatasetMetrics:
    if chave not in DATASET_SPECS:
        raise KeyError(f"Dataset desconhecido: {chave!r}")
    spec = DATASET_SPECS[chave]

    corpus = carregar_corpus(spec)
    if max_corpus is not None and len(corpus) > max_corpus:
        corpus = corpus.select(range(max_corpus))

    textos_corpus = [spec.text_fn(row) for row in corpus]
    palavras_corpus = [_contar_palavras(t) for t in textos_corpus]
    tokens_corpus_vals: list[int] | None = None
    if tokenizer is not None:
        tokens_corpus_vals = _contar_tokens_lote(
            tokenizer,
            textos_corpus,
            batch_size=batch_size,
            desc=f"Tokens corpus [{spec.rotulo}]",
        )

    particoes: dict[str, EstatisticasParticao] = {}
    query_ids_por_split: dict[str, list[str]] = {}
    docs_relevantes_por_split: dict[str, set[str]] = {}
    docs_pai_por_split: dict[str, set[str]] = defaultdict(set)
    dist_tokens_q: dict[str, list[int]] = {}
    dist_palavras_q: dict[str, list[int]] = {}
    dist_relevants: dict[str, list[int]] = {}
    todos_docs_relevantes: set[str] = set()

    for split in SPLITS:
        queries = carregar_subset(spec.repo_id, "queries", split=split)
        qrels = carregar_subset(spec.repo_id, "qrels", split=split)
        answers = carregar_subset(spec.repo_id, "answers", split=split)

        if max_queries is not None and len(queries) > max_queries:
            keep_ids = set(str(x) for x in queries["id"][:max_queries])
            queries = queries.select(range(max_queries))
            qrel_mask = [str(qid) in keep_ids for qid in qrels["query_id"]]
            qrels = qrels.select([i for i, m in enumerate(qrel_mask) if m])
            ans_mask = [str(qid) in keep_ids for qid in answers["query_id"]]
            answers = answers.select([i for i, m in enumerate(ans_mask) if m])

        qids = [str(x) for x in queries["id"]]
        qids_unicos = set(qids)
        query_ids_por_split[split] = qids

        contagem_rel: Counter[str] = Counter(str(x) for x in qrels["query_id"])
        docs_split = {str(x) for x in qrels["corpus_id"]}
        docs_relevantes_por_split[split] = docs_split
        todos_docs_relevantes |= docs_split

        if spec.doc_pai_fn is not None:
            docs_pai_por_split[split] = {spec.doc_pai_fn(cid) for cid in docs_split}

        relevants_lista = [int(contagem_rel.get(qid, 0)) for qid in qids_unicos]
        n_sem_positivo = sum(1 for n in relevants_lista if n == 0)

        textos_q = [str(t) for t in queries["text"]]
        palavras_q = [_contar_palavras(t) for t in textos_q]
        tokens_q: list[int] | None = None
        if tokenizer is not None:
            tokens_q = _contar_tokens_lote(
                tokenizer,
                textos_q,
                batch_size=batch_size,
                desc=f"Tokens queries [{spec.rotulo}/{split}]",
            )

        textos_a = [str(t) for t in answers["answer"]]
        palavras_a = [_contar_palavras(t) for t in textos_a]
        tokens_a: list[int] | None = None
        if tokenizer is not None and textos_a:
            tokens_a = _contar_tokens_lote(
                tokenizer,
                textos_a,
                batch_size=batch_size,
                desc=f"Tokens answers [{spec.rotulo}/{split}]",
            )

        dist_palavras_q[split] = palavras_q
        dist_relevants[split] = relevants_lista
        if tokens_q is not None:
            dist_tokens_q[split] = tokens_q

        particoes[split] = EstatisticasParticao(
            n_consultas=len(qids),
            n_consultas_unicas=len(qids_unicos),
            n_qrels=len(qrels),
            n_respostas=len(answers),
            n_consultas_sem_positivo=n_sem_positivo,
            relevants_por_consulta=EstatisticasComprimento.from_values(relevants_lista),
            tokens_consulta=(
                EstatisticasComprimento.from_values(tokens_q) if tokens_q else None
            ),
            palavras_consulta=EstatisticasComprimento.from_values(palavras_q),
            tokens_resposta=(
                EstatisticasComprimento.from_values(tokens_a) if tokens_a else None
            ),
            palavras_resposta=EstatisticasComprimento.from_values(palavras_a),
        )

    n_corpus = len(textos_corpus)
    n_rel = len(todos_docs_relevantes)
    return DatasetMetrics(
        chave=spec.chave,
        rotulo=spec.rotulo,
        dominio=spec.dominio,
        repo_id=spec.repo_id,
        unidade_corpus=spec.unidade_corpus,
        n_corpus=n_corpus,
        n_docs_relevantes_unicos=n_rel,
        fracao_corpus_relevante=(n_rel / n_corpus) if n_corpus else 0.0,
        tokens_corpus=(
            EstatisticasComprimento.from_values(tokens_corpus_vals)
            if tokens_corpus_vals is not None
            else None
        ),
        palavras_corpus=EstatisticasComprimento.from_values(palavras_corpus),
        particoes=particoes,
        overlap_query_ids=_overlap_ids_consulta(query_ids_por_split),
        overlap_docs_relevantes=_overlap_conjuntos(docs_relevantes_por_split),
        overlap_docs_pai=(
            _overlap_conjuntos(dict(docs_pai_por_split))
            if spec.doc_pai_fn is not None
            else None
        ),
        dist_tokens_corpus=tokens_corpus_vals or [],
        dist_palavras_corpus=palavras_corpus,
        dist_tokens_consulta_por_split=dist_tokens_q,
        dist_palavras_consulta_por_split=dist_palavras_q,
        dist_relevants_por_split=dist_relevants,
    )


def carregar_tokenizer(
    nome: str = TOKENIZER_PADRAO,
    *,
    local_files_only: bool = False,
) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        nome,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def analisar_benchmark(
    datasets: list[str] | None = None,
    *,
    tokenizer_nome: str | None = TOKENIZER_PADRAO,
    local_files_only: bool = False,
    batch_size: int = 256,
    max_corpus: int | None = None,
    max_queries: int | None = None,
    incluir_fluxo_bioasq: bool = True,
) -> BenchmarkReport:
    chaves = list(datasets) if datasets else list(SUITE_PRINCIPAL)
    tokenizer = None
    if tokenizer_nome:
        tokenizer = carregar_tokenizer(
            tokenizer_nome, local_files_only=local_files_only
        )

    metricas: dict[str, DatasetMetrics] = {}
    for chave in chaves:
        print(f"[análise] Processando {chave}...")
        metricas[chave] = analisar_dataset(
            chave,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_corpus=max_corpus,
            max_queries=max_queries,
        )

    fluxo = None
    precisa_fluxo = incluir_fluxo_bioasq and (
        "bioasq-resplit" in chaves or "bioasq" in chaves
    )
    if precisa_fluxo:
        # Garante que o original seja carregável mesmo fora da suite.
        print("[análise] Calculando fluxo BioASQ original → resplit...")
        fluxo = calcular_fluxo_bioasq_resplit()

    return BenchmarkReport(
        tokenizer_nome=tokenizer_nome,
        datasets=metricas,
        fluxo_bioasq_resplit=fluxo,
    )
