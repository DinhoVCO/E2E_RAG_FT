#!/usr/bin/env python3
"""Gera histogramas por intervalos de tokens (corpus, consultas e respostas).

Para cada dataset, plota a distribuição de comprimentos em intervalos (bins),
com linhas de referência para a média e a mediana.

Exemplos:
  python scripts/benchmark/plot_token_histograms.py --local-files-only
  python scripts/benchmark/plot_token_histograms.py --sem-tokens --splits test
  python scripts/benchmark/plot_token_histograms.py \\
      --bin-width-corpus 100 --xmax-corpus 2000 --datasets qasper bioasq-resplit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tesis_unicamp.datasets.analysis.histograms import gerar_histogramas
from tesis_unicamp.datasets.analysis.metrics import (
    TOKENIZER_PADRAO,
    analisar_benchmark,
)
from tesis_unicamp.datasets.analysis.registry import DATASET_SPECS, SPLITS, SUITE_PRINCIPAL


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_dir() -> Path:
    return _project_root() / "reports" / "benchmark_cap3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plota histogramas por intervalos de comprimento (tokens/palavras) "
            "para corpus, consultas e respostas do benchmark RAG."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUITE_PRINCIPAL),
        choices=sorted(DATASET_SPECS.keys()),
        help=f"Datasets a analisar (padrão: {', '.join(SUITE_PRINCIPAL)}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Diretório de saída (padrão: reports/benchmark_cap3).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Partições usadas para consultas/respostas (padrão: train dev test).",
    )
    parser.add_argument(
        "--tokenizer",
        default=TOKENIZER_PADRAO,
        help=f"Tokenizer HF (padrão: {TOKENIZER_PADRAO}).",
    )
    parser.add_argument(
        "--sem-tokens",
        action="store_true",
        help="Usa palavras em vez de tokens (mais rápido).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Carrega o tokenizer apenas do cache local.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-corpus", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--bin-width-corpus",
        type=float,
        default=50.0,
        help="Largura do intervalo (bin) do corpus (padrão: 50).",
    )
    parser.add_argument(
        "--bin-width-consulta",
        type=float,
        default=2.0,
        help="Largura do intervalo das consultas (padrão: 2).",
    )
    parser.add_argument(
        "--bin-width-resposta",
        type=float,
        default=10.0,
        help="Largura do intervalo das respostas (padrão: 10).",
    )
    parser.add_argument(
        "--xmax-corpus",
        type=float,
        default=1500.0,
        help=(
            "Limite visual do eixo X do corpus; valores maiores caem no "
            "intervalo de overflow ≥xmax (padrão: 1500). Use 0 para auto."
        ),
    )
    parser.add_argument(
        "--xmax-consulta",
        type=float,
        default=60.0,
        help="Limite visual do eixo X das consultas (padrão: 60; 0 = auto).",
    )
    parser.add_argument(
        "--xmax-resposta",
        type=float,
        default=300.0,
        help="Limite visual do eixo X das respostas (padrão: 300; 0 = auto).",
    )
    return parser.parse_args()


def _xmax(valor: float) -> float | None:
    return None if valor <= 0 else valor


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_nome = None if args.sem_tokens else args.tokenizer
    print("[análise] Contando comprimentos...")
    report = analisar_benchmark(
        datasets=args.datasets,
        tokenizer_nome=tokenizer_nome,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        incluir_fluxo_bioasq=False,
    )

    print("[exportação] Gerando histogramas por intervalos...")
    artefatos = gerar_histogramas(
        report,
        output_dir,
        splits=tuple(args.splits),
        usar_tokens=not args.sem_tokens,
        bin_width_corpus=args.bin_width_corpus,
        bin_width_consulta=args.bin_width_consulta,
        bin_width_resposta=args.bin_width_resposta,
        xmax_corpus=_xmax(args.xmax_corpus),
        xmax_consulta=_xmax(args.xmax_consulta),
        xmax_resposta=_xmax(args.xmax_resposta),
    )

    manifesto = {
        "datasets": args.datasets,
        "splits": args.splits,
        "tokenizer": tokenizer_nome,
        "bin_width": {
            "corpus": args.bin_width_corpus,
            "consulta": args.bin_width_consulta,
            "resposta": args.bin_width_resposta,
        },
        "xmax": {
            "corpus": _xmax(args.xmax_corpus),
            "consulta": _xmax(args.xmax_consulta),
            "resposta": _xmax(args.xmax_resposta),
        },
        "artefatos": {
            k: ([str(p) for p in v] if isinstance(v, list) else str(v))
            for k, v in artefatos.items()
        },
    }
    manifesto_path = output_dir / "manifesto_histogramas.json"
    manifesto_path.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for nome, paths in artefatos.items():
        if isinstance(paths, list):
            print(f"  - {nome}: {', '.join(str(p) for p in paths)}")
        else:
            print(f"  - {nome}: {paths}")
    print(f"\nConcluído. Manifesto: {manifesto_path}")


if __name__ == "__main__":
    main()
