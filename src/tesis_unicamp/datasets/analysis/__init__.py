"""Análise estatística e visualização do benchmark RAG (Capítulo 3)."""

from tesis_unicamp.datasets.analysis.metrics import (
    analisar_benchmark,
    analisar_dataset,
)
from tesis_unicamp.datasets.analysis.plots import gerar_todas_figuras
from tesis_unicamp.datasets.analysis.tables import exportar_tabelas

__all__ = [
    "analisar_benchmark",
    "analisar_dataset",
    "exportar_tabelas",
    "gerar_todas_figuras",
]
