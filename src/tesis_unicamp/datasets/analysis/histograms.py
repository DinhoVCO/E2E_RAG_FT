"""Histogramas por intervalos de comprimento (tokens/palavras)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tesis_unicamp.datasets.analysis.metrics import BenchmarkReport, DatasetMetrics
from tesis_unicamp.datasets.analysis.registry import SPLITS

_PALETA = ["#1B4F72", "#148F77", "#B9770E", "#6C3483"]
_ROTULO_SPLIT = {"train": "Treino", "dev": "Validação", "test": "Teste"}


def _configurar_estilo() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
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


def _valores_por_split(
    por_split: dict[str, list[int]],
    splits: tuple[str, ...] | list[str],
) -> list[int]:
    vals: list[int] = []
    for split in splits:
        vals.extend(por_split.get(split, []))
    return vals


def extrair_comprimentos(
    metrics: DatasetMetrics,
    tipo: str,
    *,
    splits: tuple[str, ...] | list[str],
    usar_tokens: bool,
) -> list[int]:
    """tipo ∈ {corpus, consulta, resposta}."""
    if tipo == "corpus":
        if usar_tokens and metrics.dist_tokens_corpus:
            return list(metrics.dist_tokens_corpus)
        return list(metrics.dist_palavras_corpus)
    if tipo == "consulta":
        if usar_tokens and metrics.dist_tokens_consulta_por_split:
            return _valores_por_split(metrics.dist_tokens_consulta_por_split, splits)
        return _valores_por_split(metrics.dist_palavras_consulta_por_split, splits)
    if tipo == "resposta":
        if usar_tokens and metrics.dist_tokens_resposta_por_split:
            return _valores_por_split(metrics.dist_tokens_resposta_por_split, splits)
        return _valores_por_split(metrics.dist_palavras_resposta_por_split, splits)
    raise ValueError(f"Tipo desconhecido: {tipo!r}")


def construir_bins(
    valores: list[int],
    *,
    bin_width: float,
    xmax: float | None = None,
    incluir_overflow: bool = True,
) -> np.ndarray:
    if not valores:
        return np.asarray([0.0, bin_width], dtype=float)
    arr = np.asarray(valores, dtype=float)
    limite = float(xmax) if xmax is not None else float(np.percentile(arr, 99.5))
    limite = max(limite, bin_width)
    # Bordas [0, w, 2w, ..., até cobrir limite]
    n_bins = int(np.ceil(limite / bin_width))
    edges = np.arange(0, (n_bins + 1) * bin_width, bin_width, dtype=float)
    if incluir_overflow and arr.max() > edges[-1]:
        # Último intervalo aberto à direita: [edges[-1], +inf) representado
        # estendendo um bin extra de mesma largura para visualização.
        edges = np.append(edges, edges[-1] + bin_width)
    return edges


def _rotulos_intervalos(edges: np.ndarray, *, overflow: bool) -> list[str]:
    labels: list[str] = []
    n = len(edges) - 1
    for i in range(n):
        a = int(edges[i])
        b = int(edges[i + 1])
        if overflow and i == n - 1:
            labels.append(f"≥{a}")
        else:
            labels.append(f"[{a}, {b})")
    return labels


def tabela_histogramas(
    report: BenchmarkReport,
    *,
    splits: tuple[str, ...] | list[str],
    usar_tokens: bool,
    bin_width_corpus: float,
    bin_width_consulta: float,
    bin_width_resposta: float,
    xmax_corpus: float | None,
    xmax_consulta: float | None,
    xmax_resposta: float | None,
) -> pd.DataFrame:
    configs = [
        ("corpus", bin_width_corpus, xmax_corpus),
        ("consulta", bin_width_consulta, xmax_consulta),
        ("resposta", bin_width_resposta, xmax_resposta),
    ]
    linhas: list[dict] = []
    for chave, m in report.datasets.items():
        for tipo, width, xmax in configs:
            vals = extrair_comprimentos(m, tipo, splits=splits, usar_tokens=usar_tokens)
            if not vals:
                continue
            edges = construir_bins(
                vals, bin_width=width, xmax=xmax, incluir_overflow=True
            )
            # Clip visual: valores acima do penúltimo edge caem no overflow bin
            clipped = np.minimum(np.asarray(vals, dtype=float), edges[-1] - 1e-9)
            counts, _ = np.histogram(clipped, bins=edges)
            overflow = bool(np.asarray(vals).max() > edges[-2])
            labels = _rotulos_intervalos(edges, overflow=overflow)
            media = float(np.mean(vals))
            mediana = float(np.median(vals))
            for lab, cnt in zip(labels, counts):
                linhas.append(
                    {
                        "dataset": m.rotulo,
                        "chave": chave,
                        "tipo": tipo,
                        "intervalo": lab,
                        "contagem": int(cnt),
                        "fracao": float(cnt / len(vals)) if vals else 0.0,
                        "n_total": len(vals),
                        "media": round(media, 2),
                        "mediana": round(mediana, 2),
                        "unidade": "tokens" if usar_tokens else "palavras",
                    }
                )
    return pd.DataFrame(linhas)


def _desenhar_histograma(
    ax: plt.Axes,
    valores: list[int],
    *,
    titulo: str,
    bin_width: float,
    xmax: float | None,
    cor: str,
    unidade: str,
    mostrar_media: bool = True,
    mostrar_mediana: bool = True,
) -> None:
    if not valores:
        ax.set_title(titulo)
        ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=ax.transAxes)
        return

    arr = np.asarray(valores, dtype=float)
    edges = construir_bins(arr.tolist(), bin_width=bin_width, xmax=xmax)
    clipped = np.minimum(arr, edges[-1] - 1e-9)
    counts, _ = np.histogram(clipped, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges) * 0.92

    ax.bar(centers, counts, width=widths, color=cor, edgecolor="white", linewidth=0.4)

    media = float(arr.mean())
    mediana = float(np.median(arr))
    ymax = max(counts.max(), 1)

    # Linhas de referência: média (principal) + mediana (robusta à cauda).
    if mostrar_media:
        x_media = min(media, edges[-1] - bin_width * 0.05)
        ax.axvline(x_media, color="#922B21", linestyle="--", linewidth=1.6, zorder=5)
        ax.text(
            x_media,
            ymax * 0.98,
            f" média={media:.1f}",
            color="#922B21",
            fontsize=8,
            rotation=90,
            va="top",
            ha="right",
        )
    if mostrar_mediana:
        x_med = min(mediana, edges[-1] - bin_width * 0.05)
        ax.axvline(x_med, color="#1A5276", linestyle=":", linewidth=1.6, zorder=5)
        ax.text(
            x_med,
            ymax * 0.72,
            f" mediana={mediana:.1f}",
            color="#1A5276",
            fontsize=8,
            rotation=90,
            va="top",
            ha="right",
        )

    ax.set_title(titulo)
    ax.set_xlabel(f"Comprimento ({unidade})")
    ax.set_ylabel("Frequência")
    ax.set_xlim(edges[0], edges[-1])


def fig_histogramas_por_tipo(
    report: BenchmarkReport,
    path_sem_ext: Path,
    *,
    tipo: str,
    splits: tuple[str, ...] | list[str],
    usar_tokens: bool,
    bin_width: float,
    xmax: float | None,
) -> list[Path]:
    datasets = list(report.datasets.values())
    n = len(datasets)
    if n == 0:
        return []

    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.6 * nrows),
        squeeze=False,
    )
    unidade = "tokens" if usar_tokens else "palavras"
    tipo_titulo = {
        "corpus": "Corpus",
        "consulta": "Consultas",
        "resposta": "Respostas",
    }[tipo]
    split_txt = (
        "todas as partições"
        if set(splits) == set(SPLITS)
        else ", ".join(_ROTULO_SPLIT.get(s, s) for s in splits)
    )

    for idx, m in enumerate(datasets):
        r, c = divmod(idx, ncols)
        vals = extrair_comprimentos(m, tipo, splits=splits, usar_tokens=usar_tokens)
        _desenhar_histograma(
            axes[r][c],
            vals,
            titulo=m.rotulo,
            bin_width=bin_width,
            xmax=xmax,
            cor=_PALETA[idx % len(_PALETA)],
            unidade=unidade,
        )

    # Esconde eixos vazios
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(
        f"Distribuição por intervalos — {tipo_titulo} ({unidade}; {split_txt})",
        y=1.01,
        fontsize=11,
    )
    fig.tight_layout()
    return _salvar(fig, path_sem_ext)


def fig_histogramas_combinado_dataset(
    report: BenchmarkReport,
    path_sem_ext: Path,
    *,
    splits: tuple[str, ...] | list[str],
    usar_tokens: bool,
    bin_width_corpus: float,
    bin_width_consulta: float,
    bin_width_resposta: float,
    xmax_corpus: float | None,
    xmax_consulta: float | None,
    xmax_resposta: float | None,
) -> list[Path]:
    """Uma figura por dataset: 3 painéis (corpus | consultas | respostas)."""
    unidade = "tokens" if usar_tokens else "palavras"
    saidas: list[Path] = []
    widths = {
        "corpus": bin_width_corpus,
        "consulta": bin_width_consulta,
        "resposta": bin_width_resposta,
    }
    xmaxs = {
        "corpus": xmax_corpus,
        "consulta": xmax_consulta,
        "resposta": xmax_resposta,
    }
    titulos = {
        "corpus": "Corpus",
        "consulta": "Consultas",
        "resposta": "Respostas",
    }

    for idx, m in enumerate(report.datasets.values()):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8))
        for ax, tipo in zip(axes, ("corpus", "consulta", "resposta")):
            vals = extrair_comprimentos(
                m, tipo, splits=splits, usar_tokens=usar_tokens
            )
            _desenhar_histograma(
                ax,
                vals,
                titulo=titulos[tipo],
                bin_width=widths[tipo],
                xmax=xmaxs[tipo],
                cor=_PALETA[idx % len(_PALETA)],
                unidade=unidade,
            )
        fig.suptitle(
            f"{m.rotulo} — distribuição por intervalos ({unidade})",
            fontsize=11,
        )
        fig.tight_layout()
        paths = _salvar(fig, Path(f"{path_sem_ext}_{m.chave}"))
        saidas.extend(paths)
    return saidas


def gerar_histogramas(
    report: BenchmarkReport,
    output_dir: Path,
    *,
    splits: tuple[str, ...] | list[str] = SPLITS,
    usar_tokens: bool = True,
    bin_width_corpus: float = 50.0,
    bin_width_consulta: float = 2.0,
    bin_width_resposta: float = 10.0,
    xmax_corpus: float | None = 1500.0,
    xmax_consulta: float | None = 60.0,
    xmax_resposta: float | None = 300.0,
) -> dict[str, list[Path] | Path]:
    _configurar_estilo()
    figs_dir = Path(output_dir) / "figuras"
    tabs_dir = Path(output_dir) / "tabelas"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tabs_dir.mkdir(parents=True, exist_ok=True)

    artefatos: dict[str, list[Path] | Path] = {}

    df = tabela_histogramas(
        report,
        splits=splits,
        usar_tokens=usar_tokens,
        bin_width_corpus=bin_width_corpus,
        bin_width_consulta=bin_width_consulta,
        bin_width_resposta=bin_width_resposta,
        xmax_corpus=xmax_corpus,
        xmax_consulta=xmax_consulta,
        xmax_resposta=xmax_resposta,
    )
    csv_path = tabs_dir / "tabela_histogramas_intervalos.csv"
    df.to_csv(csv_path, index=False)
    artefatos["tabela_histogramas_intervalos.csv"] = csv_path

    for tipo, width, xmax in (
        ("corpus", bin_width_corpus, xmax_corpus),
        ("consulta", bin_width_consulta, xmax_consulta),
        ("resposta", bin_width_resposta, xmax_resposta),
    ):
        paths = fig_histogramas_por_tipo(
            report,
            figs_dir / f"fig_hist_intervalos_{tipo}",
            tipo=tipo,
            splits=splits,
            usar_tokens=usar_tokens,
            bin_width=width,
            xmax=xmax,
        )
        if paths:
            artefatos[f"fig_hist_intervalos_{tipo}"] = paths

    paths_combo = fig_histogramas_combinado_dataset(
        report,
        figs_dir / "fig_hist_intervalos",
        splits=splits,
        usar_tokens=usar_tokens,
        bin_width_corpus=bin_width_corpus,
        bin_width_consulta=bin_width_consulta,
        bin_width_resposta=bin_width_resposta,
        xmax_corpus=xmax_corpus,
        xmax_consulta=xmax_consulta,
        xmax_resposta=xmax_resposta,
    )
    if paths_combo:
        artefatos["fig_hist_intervalos_por_dataset"] = paths_combo

    return artefatos
