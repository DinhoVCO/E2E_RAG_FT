#!/usr/bin/env python3
"""Extrai métricas, tabelas e figuras do benchmark RAG (Capítulo 3).

Gera, em português:
  - tabelas CSV / Markdown / LaTeX (resumo, comprimentos, overlap, relevants, fluxo BioASQ)
  - figuras PDF/PNG (tamanhos, tokens, overlap, assimetria, fluxo resplit)
  - JSON consolidado com as métricas

Exemplo:
  python scripts/benchmark/analyze_rag_benchmark.py
  python scripts/benchmark/analyze_rag_benchmark.py --datasets qasper bioasq-resplit
  python scripts/benchmark/analyze_rag_benchmark.py --sem-tokens   # só palavras (mais rápido)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesis_unicamp.datasets.analysis.metrics import (
    TOKENIZER_PADRAO,
    analisar_benchmark,
)
from tesis_unicamp.datasets.analysis.plots import gerar_todas_figuras
from tesis_unicamp.datasets.analysis.registry import DATASET_SPECS, SUITE_PRINCIPAL
from tesis_unicamp.datasets.analysis.tables import exportar_tabelas


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_dir() -> Path:
    return _project_root() / "reports" / "benchmark_cap3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisa os datasets do benchmark RAG e exporta tabelas/figuras "
            "em português para o Capítulo 3 da dissertação."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUITE_PRINCIPAL),
        choices=sorted(DATASET_SPECS.keys()),
        help=(
            "Conjuntos a analisar. "
            f"Padrão (suite principal): {', '.join(SUITE_PRINCIPAL)}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Diretório de saída (padrão: reports/benchmark_cap3).",
    )
    parser.add_argument(
        "--tokenizer",
        default=TOKENIZER_PADRAO,
        help=f"Tokenizer Hugging Face para contagem de tokens (padrão: {TOKENIZER_PADRAO}).",
    )
    parser.add_argument(
        "--sem-tokens",
        action="store_true",
        help="Não carrega tokenizer; calcula apenas estatísticas em palavras.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Carrega o tokenizer apenas do cache local (sem rede).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Tamanho do lote para tokenização (padrão: 256).",
    )
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=None,
        help="Limita o número de documentos do corpus (útil para testes rápidos).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limita o número de consultas por partição (útil para testes rápidos).",
    )
    parser.add_argument(
        "--sem-fluxo-bioasq",
        action="store_true",
        help="Não calcula a matriz de transição BioASQ original → resplit.",
    )
    parser.add_argument(
        "--somente-tabelas",
        action="store_true",
        help="Exporta apenas tabelas/JSON (sem figuras).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_nome = None if args.sem_tokens else args.tokenizer
    report = analisar_benchmark(
        datasets=args.datasets,
        tokenizer_nome=tokenizer_nome,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        incluir_fluxo_bioasq=not args.sem_fluxo_bioasq,
    )

    manifesto: dict = {
        "datasets": args.datasets,
        "tokenizer": tokenizer_nome,
        "output_dir": str(output_dir),
        "artefatos": {},
    }

    print("[exportação] Gerando tabelas...")
    artefatos_tab = exportar_tabelas(report, output_dir)
    manifesto["artefatos"]["tabelas"] = {k: str(v) for k, v in artefatos_tab.items()}
    for nome, path in artefatos_tab.items():
        print(f"  - {nome}: {path}")

    if not args.somente_tabelas:
        print("[exportação] Gerando figuras...")
        artefatos_fig = gerar_todas_figuras(report, output_dir)
        manifesto["artefatos"]["figuras"] = {
            k: [str(p) for p in paths] for k, paths in artefatos_fig.items()
        }
        for nome, paths in artefatos_fig.items():
            print(f"  - {nome}: {', '.join(str(p) for p in paths)}")

    manifesto_path = output_dir / "manifesto.json"
    manifesto_path.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nConcluído. Saída em: {output_dir}")
    print(f"Manifesto: {manifesto_path}")


if __name__ == "__main__":
    main()
