"""Geração de figuras em português para o Capítulo 3 (benchmark)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tesis_unicamp.datasets.analysis.metrics import BenchmarkReport
from tesis_unicamp.datasets.analysis.registry import SPLITS

# Estilo sóbrio para tese (evitar paletas genéricas “AI”).
_PALETA_DATASETS = ["#1B4F72", "#148F77", "#B9770E", "#6C3483", "#922B21"]
_PALETA_SPLITS = {"train": "#1B4F72", "dev": "#148F77", "test": "#B9770E"}
_ROTULO_SPLIT = {"train": "Treino", "dev": "Validação", "test": "Teste"}
_LIMITES_TOKEN = (512, 2048)


def _configurar_estilo() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _salvar(fig: plt.Figure, path_sem_ext: Path) -> list[Path]:
    path_sem_ext.parent.mkdir(parents=True, exist_ok=True)
    saidas: list[Path] = []
    for ext in ("pdf", "png"):
        destino = path_sem_ext.with_suffix(f".{ext}")
        fig.savefig(destino)
        saidas.append(destino)
    plt.close(fig)
    return saidas


def _cores_datasets(n: int) -> list[str]:
    if n <= len(_PALETA_DATASETS):
        return _PALETA_DATASETS[:n]
    return sns.color_palette("deep", n).as_hex()


def fig_tamanho_splits(report: BenchmarkReport, path_sem_ext: Path) -> list[Path]:
    registros: list[dict] = []
    for m in report.datasets.values():
        for split in SPLITS:
            registros.append(
                {
                    "Dataset": m.rotulo,
                    "Partição": _ROTULO_SPLIT[split],
                    "Consultas": m.particoes[split].n_consultas_unicas,
                }
            )
    df = pd.DataFrame(registros)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.barplot(
        data=df,
        x="Dataset",
        y="Consultas",
        hue="Partição",
        palette=[_PALETA_SPLITS[s] for s in SPLITS],
        ax=ax,
    )
    ax.set_title("Número de consultas únicas por partição")
    ax.set_xlabel("Conjunto de dados")
    ax.set_ylabel("Consultas únicas")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Partição")
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def fig_tamanho_corpus(report: BenchmarkReport, path_sem_ext: Path) -> list[Path]:
    rotulos = [m.rotulo for m in report.datasets.values()]
    valores = [m.n_corpus for m in report.datasets.values()]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars = ax.bar(rotulos, valores, color=_cores_datasets(len(rotulos)))
    ax.set_title("Tamanho do corpus por conjunto de dados")
    ax.set_xlabel("Conjunto de dados")
    ax.set_ylabel("Documentos / passagens")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, valores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}".replace(",", "."),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def fig_tokens_corpus(
    report: BenchmarkReport,
    path_sem_ext: Path,
    *,
    usar_tokens: bool = True,
    max_exibir: int | None = 4000,
) -> list[Path]:
    registros: list[dict] = []
    tem_tokens = False
    for m in report.datasets.values():
        if usar_tokens and m.dist_tokens_corpus:
            vals = m.dist_tokens_corpus
            eixo = "Tokens"
            tem_tokens = True
        else:
            vals = m.dist_palavras_corpus
            eixo = "Palavras"
        if max_exibir is not None:
            vals = [min(v, max_exibir) for v in vals]
        for v in vals:
            registros.append({"Dataset": m.rotulo, "Comprimento": v})

    if not registros:
        return []

    df = pd.DataFrame(registros)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.violinplot(
        data=df,
        x="Dataset",
        y="Comprimento",
        hue="Dataset",
        palette=_cores_datasets(df["Dataset"].nunique()),
        cut=0,
        inner="quartile",
        legend=False,
        ax=ax,
    )
    titulo = (
        "Distribuição do comprimento do corpus"
        + (" (tokens)" if tem_tokens else " (palavras)")
    )
    ax.set_title(titulo)
    ax.set_xlabel("Conjunto de dados")
    ax.set_ylabel(eixo + (f" (truncado em {max_exibir} p/ visualização)" if max_exibir else ""))
    ax.tick_params(axis="x", rotation=15)
    if tem_tokens:
        for lim in _LIMITES_TOKEN:
            if max_exibir is None or lim <= max_exibir:
                ax.axhline(lim, color="#7B241C", linestyle="--", linewidth=1, alpha=0.8)
                ax.text(
                    0.995,
                    lim,
                    f" {lim}",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#7B241C",
                )
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def fig_tokens_consultas(
    report: BenchmarkReport,
    path_sem_ext: Path,
    *,
    split: str = "test",
    usar_tokens: bool = True,
) -> list[Path]:
    registros: list[dict] = []
    tem_tokens = False
    for m in report.datasets.values():
        if usar_tokens and m.dist_tokens_consulta_por_split.get(split):
            vals = m.dist_tokens_consulta_por_split[split]
            eixo = "Tokens"
            tem_tokens = True
        else:
            vals = m.dist_palavras_consulta_por_split.get(split, [])
            eixo = "Palavras"
        for v in vals:
            registros.append({"Dataset": m.rotulo, "Comprimento": v})

    if not registros:
        return []

    df = pd.DataFrame(registros)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.boxplot(
        data=df,
        x="Dataset",
        y="Comprimento",
        hue="Dataset",
        palette=_cores_datasets(df["Dataset"].nunique()),
        legend=False,
        ax=ax,
        showfliers=False,
    )
    ax.set_title(
        f"Comprimento das consultas ({_ROTULO_SPLIT.get(split, split)})"
        + (" — tokens" if tem_tokens else " — palavras")
    )
    ax.set_xlabel("Conjunto de dados")
    ax.set_ylabel(eixo)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def fig_relevants_por_consulta(
    report: BenchmarkReport,
    path_sem_ext: Path,
    *,
    split: str = "test",
) -> list[Path]:
    registros: list[dict] = []
    for m in report.datasets.values():
        for v in m.dist_relevants_por_split.get(split, []):
            registros.append({"Dataset": m.rotulo, "Relevants": v})
    if not registros:
        return []

    df = pd.DataFrame(registros)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    sns.boxplot(
        data=df,
        x="Dataset",
        y="Relevants",
        hue="Dataset",
        palette=_cores_datasets(df["Dataset"].nunique()),
        legend=False,
        ax=ax,
        showfliers=True,
        fliersize=2,
    )
    ax.set_title(
        f"Documentos relevantes por consulta ({_ROTULO_SPLIT.get(split, split)})"
    )
    ax.set_xlabel("Conjunto de dados")
    ax.set_ylabel("Nº de documentos relevantes")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def fig_overlap_documentos(report: BenchmarkReport, path_sem_ext: Path) -> list[Path]:
    rotulos = []
    matriz = []
    for m in report.datasets.values():
        rotulos.append(m.rotulo)
        ov = m.overlap_docs_relevantes
        matriz.append(
            [
                ov.fracoes_em_relacao_ao_menor.get("train∩dev", 0.0),
                ov.fracao_dev_em_train,
                ov.fracao_test_em_train,
            ]
        )
    data = np.asarray(matriz, dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        xticklabels=[
            "train∩dev / min",
            "dev ⊆ train (fração)",
            "test ⊆ train (fração)",
        ],
        yticklabels=rotulos,
        ax=ax,
        cbar_kws={"label": "Fração de sobreposição"},
    )
    ax.set_title("Sobreposição de documentos relevantes entre partições")
    ax.set_xlabel("Tipo de sobreposição")
    ax.set_ylabel("Conjunto de dados")
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def fig_assimetria_query_doc(report: BenchmarkReport, path_sem_ext: Path) -> list[Path]:
    registros: list[dict] = []
    for m in report.datasets.values():
        q = m.particoes["test"]
        if m.tokens_corpus is not None and q.tokens_consulta is not None:
            registros.append(
                {
                    "Dataset": m.rotulo,
                    "Consulta (média)": q.tokens_consulta.media,
                    "Documento (média)": m.tokens_corpus.media,
                    "unidade": "tokens",
                }
            )
        else:
            registros.append(
                {
                    "Dataset": m.rotulo,
                    "Consulta (média)": q.palavras_consulta.media if q.palavras_consulta else 0,
                    "Documento (média)": m.palavras_corpus.media,
                    "unidade": "palavras",
                }
            )
    df = pd.DataFrame(registros)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(df))
    largura = 0.35
    ax.bar(
        x - largura / 2,
        df["Consulta (média)"],
        width=largura,
        label="Consulta (média)",
        color="#1B4F72",
    )
    ax.bar(
        x + largura / 2,
        df["Documento (média)"],
        width=largura,
        label="Documento (média)",
        color="#B9770E",
    )
    unidade = df["unidade"].iloc[0] if len(df) else "tokens"
    ax.set_xticks(x)
    ax.set_xticklabels(df["Dataset"], rotation=15)
    ax.set_ylabel(f"Comprimento médio ({unidade})")
    ax.set_title("Assimetria de comprimento: consulta vs. documento (teste)")
    ax.legend()
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def fig_bioasq_fluxo_resplit(
    report: BenchmarkReport, path_sem_ext: Path
) -> list[Path]:
    fluxo = report.fluxo_bioasq_resplit
    if fluxo is None:
        return []

    matriz = np.array(
        [[fluxo.matriz_transicao[src][dst] for dst in SPLITS] for src in SPLITS],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sns.heatmap(
        matriz,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=[_ROTULO_SPLIT[s] for s in SPLITS],
        yticklabels=[_ROTULO_SPLIT[s] for s in SPLITS],
        ax=ax,
        cbar_kws={"label": "Nº de consultas"},
    )
    ax.set_xlabel("Partição no BioASQ-Resplit")
    ax.set_ylabel("Partição no BioASQ original")
    ax.set_title("Fluxo de consultas: BioASQ original → resplit")
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def fig_fracao_corpus_relevante(
    report: BenchmarkReport, path_sem_ext: Path
) -> list[Path]:
    rotulos = [m.rotulo for m in report.datasets.values()]
    fracoes = [100.0 * m.fracao_corpus_relevante for m in report.datasets.values()]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars = ax.bar(rotulos, fracoes, color=_cores_datasets(len(rotulos)))
    ax.set_title("Fração do corpus que aparece como relevante em alguma partição")
    ax.set_xlabel("Conjunto de dados")
    ax.set_ylabel("% do corpus")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, fracoes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def gerar_todas_figuras(
    report: BenchmarkReport,
    output_dir: Path,
) -> dict[str, list[Path]]:
    _configurar_estilo()
    figs_dir = Path(output_dir) / "figuras"
    figs_dir.mkdir(parents=True, exist_ok=True)

    usar_tokens = any(
        m.tokens_corpus is not None or m.dist_tokens_corpus for m in report.datasets.values()
    )

    geradores = {
        "fig_tamanho_splits": lambda p: fig_tamanho_splits(report, p),
        "fig_tamanho_corpus": lambda p: fig_tamanho_corpus(report, p),
        "fig_tokens_corpus": lambda p: fig_tokens_corpus(
            report, p, usar_tokens=usar_tokens
        ),
        "fig_tokens_consultas_teste": lambda p: fig_tokens_consultas(
            report, p, split="test", usar_tokens=usar_tokens
        ),
        "fig_relevants_por_consulta_teste": lambda p: fig_relevants_por_consulta(
            report, p, split="test"
        ),
        "fig_overlap_documentos": lambda p: fig_overlap_documentos(report, p),
        "fig_assimetria_query_doc": lambda p: fig_assimetria_query_doc(report, p),
        "fig_fracao_corpus_relevante": lambda p: fig_fracao_corpus_relevante(report, p),
        "fig_bioasq_fluxo_resplit": lambda p: fig_bioasq_fluxo_resplit(report, p),
    }

    saidas: dict[str, list[Path]] = {}
    for nome, fn in geradores.items():
        paths = fn(figs_dir / nome)
        if paths:
            saidas[nome] = paths
    return saidas
