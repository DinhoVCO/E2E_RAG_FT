"""Exportação de tabelas (CSV / LaTeX / Markdown) em português."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tesis_unicamp.datasets.analysis.metrics import BenchmarkReport
from tesis_unicamp.datasets.analysis.registry import SPLITS


def _fmt_float(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}"


def tabela_resumo(report: BenchmarkReport) -> pd.DataFrame:
    linhas: list[dict[str, Any]] = []
    for chave, m in report.datasets.items():
        linha: dict[str, Any] = {
            "dataset": m.rotulo,
            "chave": chave,
            "dominio": m.dominio,
            "unidade_corpus": m.unidade_corpus,
            "n_corpus": m.n_corpus,
            "n_docs_relevantes_unicos": m.n_docs_relevantes_unicos,
            "fracao_corpus_relevante": round(m.fracao_corpus_relevante, 4),
            "repo_id": m.repo_id,
        }
        for split in SPLITS:
            p = m.particoes[split]
            linha[f"n_consultas_{split}"] = p.n_consultas
            linha[f"n_consultas_unicas_{split}"] = p.n_consultas_unicas
            linha[f"n_qrels_{split}"] = p.n_qrels
            linha[f"n_respostas_{split}"] = p.n_respostas
            linha[f"n_sem_positivo_{split}"] = p.n_consultas_sem_positivo
            linha[f"media_relevants_{split}"] = round(
                p.relevants_por_consulta.media, 3
            )
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_resumo_latex_friendly(report: BenchmarkReport) -> pd.DataFrame:
    """Versão compacta para colar no capítulo (Tabela-resumo)."""
    linhas: list[dict[str, Any]] = []
    for m in report.datasets.values():
        linhas.append(
            {
                "Conjunto": m.rotulo,
                "Domínio": m.dominio,
                "Corpus": m.n_corpus,
                "Treino": m.particoes["train"].n_consultas_unicas,
                "Validação": m.particoes["dev"].n_consultas_unicas,
                "Teste": m.particoes["test"].n_consultas_unicas,
                "Relevants/consulta (teste)": _fmt_float(
                    m.particoes["test"].relevants_por_consulta.media, 2
                ),
                "Unidade do corpus": m.unidade_corpus,
            }
        )
    return pd.DataFrame(linhas)


def tabela_comprimentos(report: BenchmarkReport) -> pd.DataFrame:
    linhas: list[dict[str, Any]] = []
    for m in report.datasets.values():
        base = {
            "dataset": m.rotulo,
            "tipo": "corpus",
            "particao": "—",
            "n": m.n_corpus,
            "palavras_media": round(m.palavras_corpus.media, 2),
            "palavras_mediana": round(m.palavras_corpus.percentis["p50"], 2),
            "palavras_p95": round(m.palavras_corpus.percentis["p95"], 2),
            "palavras_max": m.palavras_corpus.maximo,
        }
        if m.tokens_corpus is not None:
            base.update(
                {
                    "tokens_media": round(m.tokens_corpus.media, 2),
                    "tokens_mediana": round(m.tokens_corpus.percentis["p50"], 2),
                    "tokens_p95": round(m.tokens_corpus.percentis["p95"], 2),
                    "tokens_max": m.tokens_corpus.maximo,
                }
            )
        linhas.append(base)

        for split in SPLITS:
            p = m.particoes[split]
            row: dict[str, Any] = {
                "dataset": m.rotulo,
                "tipo": "consulta",
                "particao": split,
                "n": p.n_consultas,
                "palavras_media": round(p.palavras_consulta.media, 2)
                if p.palavras_consulta
                else None,
                "palavras_mediana": round(p.palavras_consulta.percentis["p50"], 2)
                if p.palavras_consulta
                else None,
                "palavras_p95": round(p.palavras_consulta.percentis["p95"], 2)
                if p.palavras_consulta
                else None,
                "palavras_max": p.palavras_consulta.maximo if p.palavras_consulta else None,
            }
            if p.tokens_consulta is not None:
                row.update(
                    {
                        "tokens_media": round(p.tokens_consulta.media, 2),
                        "tokens_mediana": round(p.tokens_consulta.percentis["p50"], 2),
                        "tokens_p95": round(p.tokens_consulta.percentis["p95"], 2),
                        "tokens_max": p.tokens_consulta.maximo,
                    }
                )
            linhas.append(row)

            row_a: dict[str, Any] = {
                "dataset": m.rotulo,
                "tipo": "resposta",
                "particao": split,
                "n": p.n_respostas,
                "palavras_media": round(p.palavras_resposta.media, 2)
                if p.palavras_resposta
                else None,
                "palavras_mediana": round(p.palavras_resposta.percentis["p50"], 2)
                if p.palavras_resposta
                else None,
                "palavras_p95": round(p.palavras_resposta.percentis["p95"], 2)
                if p.palavras_resposta
                else None,
                "palavras_max": p.palavras_resposta.maximo if p.palavras_resposta else None,
            }
            if p.tokens_resposta is not None:
                row_a.update(
                    {
                        "tokens_media": round(p.tokens_resposta.media, 2),
                        "tokens_mediana": round(p.tokens_resposta.percentis["p50"], 2),
                        "tokens_p95": round(p.tokens_resposta.percentis["p95"], 2),
                        "tokens_max": p.tokens_resposta.maximo,
                    }
                )
            linhas.append(row_a)
    return pd.DataFrame(linhas)


def tabela_overlap(report: BenchmarkReport) -> pd.DataFrame:
    linhas: list[dict[str, Any]] = []
    for m in report.datasets.values():
        ov = m.overlap_docs_relevantes
        linha = {
            "dataset": m.rotulo,
            "nivel": "documento_relevante",
            "n_relevants_train": ov.conjuntos.get("train", 0),
            "n_relevants_dev": ov.conjuntos.get("dev", 0),
            "n_relevants_test": ov.conjuntos.get("test", 0),
            "inter_train_dev": ov.intersecoes.get("train∩dev", 0),
            "inter_train_test": ov.intersecoes.get("train∩test", 0),
            "inter_dev_test": ov.intersecoes.get("dev∩test", 0),
            "fracao_test_em_train": round(ov.fracao_test_em_train, 4),
            "fracao_dev_em_train": round(ov.fracao_dev_em_train, 4),
            "overlap_query_train_dev": m.overlap_query_ids.get("train∩dev", 0),
            "overlap_query_train_test": m.overlap_query_ids.get("train∩test", 0),
            "overlap_query_dev_test": m.overlap_query_ids.get("dev∩test", 0),
        }
        linhas.append(linha)
        if m.overlap_docs_pai is not None:
            op = m.overlap_docs_pai
            linhas.append(
                {
                    "dataset": m.rotulo,
                    "nivel": "documento_pai",
                    "n_relevants_train": op.conjuntos.get("train", 0),
                    "n_relevants_dev": op.conjuntos.get("dev", 0),
                    "n_relevants_test": op.conjuntos.get("test", 0),
                    "inter_train_dev": op.intersecoes.get("train∩dev", 0),
                    "inter_train_test": op.intersecoes.get("train∩test", 0),
                    "inter_dev_test": op.intersecoes.get("dev∩test", 0),
                    "fracao_test_em_train": round(op.fracao_test_em_train, 4),
                    "fracao_dev_em_train": round(op.fracao_dev_em_train, 4),
                    "overlap_query_train_dev": None,
                    "overlap_query_train_test": None,
                    "overlap_query_dev_test": None,
                }
            )
    return pd.DataFrame(linhas)


def tabela_relevants(report: BenchmarkReport) -> pd.DataFrame:
    linhas: list[dict[str, Any]] = []
    for m in report.datasets.values():
        for split in SPLITS:
            r = m.particoes[split].relevants_por_consulta
            linhas.append(
                {
                    "dataset": m.rotulo,
                    "particao": split,
                    "n_consultas": r.n,
                    "media": round(r.media, 3),
                    "desvio": round(r.desvio, 3),
                    "minimo": r.minimo,
                    "mediana": round(r.percentis["p50"], 3),
                    "p95": round(r.percentis["p95"], 3),
                    "maximo": r.maximo,
                }
            )
    return pd.DataFrame(linhas)


def tabela_fluxo_bioasq(report: BenchmarkReport) -> pd.DataFrame | None:
    fluxo = report.fluxo_bioasq_resplit
    if fluxo is None:
        return None
    linhas: list[dict[str, Any]] = []
    for src in SPLITS:
        for dst in SPLITS:
            linhas.append(
                {
                    "particao_original": src,
                    "particao_resplit": dst,
                    "n_consultas": fluxo.matriz_transicao[src][dst],
                }
            )
    meta = pd.DataFrame(
        [
            {
                "particao_original": "META",
                "particao_resplit": "n_consultas_originais",
                "n_consultas": fluxo.n_consultas_originais,
            },
            {
                "particao_original": "META",
                "particao_resplit": "n_consultas_resplit",
                "n_consultas": fluxo.n_consultas_resplit,
            },
            {
                "particao_original": "META",
                "particao_resplit": "apenas_original",
                "n_consultas": fluxo.consultas_apenas_original,
            },
            {
                "particao_original": "META",
                "particao_resplit": "apenas_resplit",
                "n_consultas": fluxo.consultas_apenas_resplit,
            },
        ]
    )
    return pd.concat([pd.DataFrame(linhas), meta], ignore_index=True)


def _dataframe_para_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    try:
        corpo = df.to_latex(index=False, escape=True, caption=caption, label=label)
    except TypeError:
        # pandas antigo sem caption/label no to_latex
        corpo = df.to_latex(index=False, escape=True)
        corpo = (
            f"\\begin{{table}}[htbp]\n\\centering\n"
            f"{corpo}\n\\caption{{{caption}}}\\label{{{label}}}\n\\end{{table}}\n"
        )
    return corpo


def exportar_tabelas(
    report: BenchmarkReport,
    output_dir: Path,
    *,
    salvar_json_completo: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    tabelas_dir = output_dir / "tabelas"
    tabelas_dir.mkdir(parents=True, exist_ok=True)

    artefatos: dict[str, Path] = {}

    tabelas = {
        "tabela_resumo_datasets": tabela_resumo(report),
        "tabela_resumo_capitulo": tabela_resumo_latex_friendly(report),
        "tabela_comprimentos": tabela_comprimentos(report),
        "tabela_overlap_documentos": tabela_overlap(report),
        "tabela_relevants_por_consulta": tabela_relevants(report),
    }
    fluxo_df = tabela_fluxo_bioasq(report)
    if fluxo_df is not None:
        tabelas["tabela_bioasq_fluxo_resplit"] = fluxo_df

    captions = {
        "tabela_resumo_datasets": "Estatísticas completas dos conjuntos do benchmark RAG.",
        "tabela_resumo_capitulo": "Visão geral dos conjuntos de dados do benchmark.",
        "tabela_comprimentos": "Estatísticas de comprimento (palavras/tokens).",
        "tabela_overlap_documentos": "Sobreposição de documentos relevantes entre partições.",
        "tabela_relevants_por_consulta": "Distribuição do número de documentos relevantes por consulta.",
        "tabela_bioasq_fluxo_resplit": "Transição das consultas BioASQ original → resplit.",
    }

    for nome, df in tabelas.items():
        csv_path = tabelas_dir / f"{nome}.csv"
        md_path = tabelas_dir / f"{nome}.md"
        tex_path = tabelas_dir / f"{nome}.tex"
        df.to_csv(csv_path, index=False)
        md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
        tex_path.write_text(
            _dataframe_para_latex(
                df,
                caption=captions.get(nome, nome),
                label=f"tab:{nome}",
            ),
            encoding="utf-8",
        )
        artefatos[f"{nome}.csv"] = csv_path
        artefatos[f"{nome}.md"] = md_path
        artefatos[f"{nome}.tex"] = tex_path

    if salvar_json_completo:
        json_path = output_dir / "metricas_benchmark.json"
        json_path.write_text(
            json.dumps(
                report.to_dict(incluir_distribuicoes=False),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artefatos["metricas_benchmark.json"] = json_path

    return artefatos
