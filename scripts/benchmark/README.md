# Análise do benchmark RAG (Capítulo 3)

Script e biblioteca para extrair **métricas**, **tabelas** e **figuras** em português
a partir dos datasets padronizados do benchmark (QASPER, BioASQ-Resplit, NarrativeQA, Telco-DPR).

## Uso

A partir da raiz do repositório, com o ambiente virtual ativo:

```bash
source .venv/bin/activate

# Análise completa (tokens com Qwen3-Embedding-4B + tabelas + figuras)
python scripts/benchmark/analyze_rag_benchmark.py --local-files-only

# Mais rápido: só palavras (sem tokenizer)
python scripts/benchmark/analyze_rag_benchmark.py --sem-tokens

# Subconjunto de datasets / teste rápido
python scripts/benchmark/analyze_rag_benchmark.py \
  --datasets qasper telco-dpr \
  --max-corpus 2000 \
  --max-queries 500 \
  --sem-tokens
```

Saída padrão: `reports/benchmark_cap3/`.

## Artefatos gerados

### Tabelas (`reports/benchmark_cap3/tabelas/`)

| Arquivo | Conteúdo |
|---------|----------|
| `tabela_resumo_capitulo.{csv,md,tex}` | Visão geral para colar no capítulo |
| `tabela_resumo_datasets.*` | Estatísticas completas (consultas, qrels, sem positivo, etc.) |
| `tabela_comprimentos.*` | Média / mediana / p95 / máx. de palavras e tokens |
| `tabela_overlap_documentos.*` | Sobreposição de docs relevantes e de query IDs entre splits |
| `tabela_relevants_por_consulta.*` | Distribuição de #relevants por consulta |
| `tabela_bioasq_fluxo_resplit.*` | Matriz original → resplit |

Também: `metricas_benchmark.json` (métricas consolidadas) e `manifesto.json`.

### Figuras (`reports/benchmark_cap3/figuras/`)

| Figura | Descrição |
|--------|-----------|
| `fig_tamanho_splits` | Consultas únicas por partição |
| `fig_tamanho_corpus` | Tamanho do corpus |
| `fig_tokens_corpus` | Violin do comprimento do corpus (+ linhas 512/2048) |
| `fig_tokens_consultas_teste` | Boxplot do comprimento das consultas (teste) |
| `fig_relevants_por_consulta_teste` | Boxplot de relevants/consulta |
| `fig_overlap_documentos` | Heatmap de leakage de documentos relevantes |
| `fig_assimetria_query_doc` | Média consulta vs. documento |
| `fig_fracao_corpus_relevante` | % do corpus que é relevante em alguma partição |
| `fig_bioasq_fluxo_resplit` | Heatmap da transição BioASQ original → resplit |

Cada figura é salva em **PDF** e **PNG**.

## Módulo Python

```python
from tesis_unicamp.datasets.analysis import (
    analisar_benchmark,
    exportar_tabelas,
    gerar_todas_figuras,
)

report = analisar_benchmark(["qasper", "bioasq-resplit"], tokenizer_nome=None)
exportar_tabelas(report, "reports/benchmark_cap3")
gerar_todas_figuras(report, "reports/benchmark_cap3")
```

## Histogramas por intervalos (tokens)

Para figuras de distribuição em intervalos (bins) de **corpus**, **consultas** e **respostas**, com linhas de **média** e **mediana**:

```bash
python scripts/benchmark/plot_token_histograms.py --local-files-only

# Ajustar largura dos intervalos / limite visual do eixo X
python scripts/benchmark/plot_token_histograms.py --local-files-only \
  --bin-width-corpus 100 --xmax-corpus 2000 \
  --bin-width-consulta 2 --xmax-consulta 50 \
  --bin-width-resposta 20 --xmax-resposta 400

# Apenas split de teste, sem tokenizer (palavras)
python scripts/benchmark/plot_token_histograms.py --sem-tokens --splits test
```

Saídas:
- `figuras/fig_hist_intervalos_{corpus,consulta,resposta}.{pdf,png}`
- `figuras/fig_hist_intervalos_{dataset}.{pdf,png}` (3 painéis por dataset)
- `tabelas/tabela_histogramas_intervalos.csv`
- `manifesto_histogramas.json`

A **média** aparece como linha tracejada vermelha e a **mediana** (mais robusta a caudas longas) como linha pontilhada cinza-escuro; os valores ficam numa legenda no canto superior direito.

## Observações

- A contagem de tokens usa o mesmo tokenizer dos experimentos de embedding (`Qwen/Qwen3-Embedding-4B`), com o texto formatado como na indexação/avaliação.
- O fluxo BioASQ carrega também o dataset original `DinoStackAI/bioasq-rag-13b` (além do resplit).
- Para corpora muito longos (ex.: cauda do Telco-DPR), a figura de tokens trunca visualmente em 4000 apenas para legibilidade; as tabelas/JSON mantêm o máximo real.
