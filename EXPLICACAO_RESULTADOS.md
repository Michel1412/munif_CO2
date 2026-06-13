# Explicação dos Resultados Gerados

Este documento descreve todos os artefatos produzidos pelo pipeline de previsão de **CO₂ per capita** por país (2000–2026), com objetivo de estimar o valor para **2027**.

---

## Visão geral do fluxo

```
Planilha Google Sheets
        ↓  step1_prepare.py
data/processed/{país}.csv     (intermediário — não versionado)
        ↓  step2_train.py
models/tree/ e models/svm/      (intermediário — não versionado)
        ↓  step3_report.py + step4_predict_2027.py
reports/                        (resultados finais — versionados)
```

Os arquivos em `reports/` são os **resultados finais em formato de tabela** e devem ser incluídos no repositório.

---

## Pasta `reports/` — Resultados versionados

### `metrics.csv`

Métricas de avaliação de **cada modelo em cada país**, calculadas sobre o conjunto de **teste** (20% final da série temporal, anos ~2022–2026).

| Coluna   | Descrição |
|----------|-----------|
| `country` | Nome do país |
| `model`   | `tree` (Árvore de Decisão) ou `svm` (SVM/SVR) |
| `mae`     | Erro Absoluto Médio — quanto, em média, a previsão erra em toneladas de CO₂ per capita |
| `mse`     | Erro Quadrático Médio |
| `rmse`    | Raiz do MSE — mesma unidade do alvo (toneladas) |
| `r2`      | Coeficiente de determinação — quanto o modelo explica a variância (1 = perfeito; negativo = pior que a média) |
| `mape`    | Erro Percentual Absoluto Médio (%) |

**Exemplo:** Brasil com SVM → R² ≈ 0,90 indica bom ajuste no período de teste.

---

### `metrics.xlsx`

Mesmo conteúdo de `metrics.csv`, organizado em **planilha Excel** com abas adicionais:

| Aba | Conteúdo |
|-----|----------|
| `metrics` | Todas as métricas por país e modelo |
| `best_by_country` | Melhor modelo por país |
| `ranking` | Contagem de vitórias por modelo |

---

### `country_summary.csv`

Tabela **consolidada por país**, com métricas dos dois modelos lado a lado.

| Coluna | Descrição |
|--------|-----------|
| `country` | País |
| `tree_r2`, `tree_mae`, `tree_rmse`, `tree_mape` | Métricas da Árvore de Decisão |
| `svm_r2`, `svm_mae`, `svm_rmse`, `svm_mape` | Métricas do SVM |

Facilita comparar rapidamente qual algoritmo performou melhor em cada nação.

---

### `country_summary.xlsx`

Versão Excel de `country_summary.csv` para entrega ou análise em planilha.

---

### `best_by_country.csv`

Indica o **melhor modelo por país**, escolhido pelo maior **R²** no teste.

| Coluna | Descrição |
|--------|-----------|
| `country` | País |
| `best_model` | `tree` ou `svm` |
| `best_model_label` | Nome legível (`Árvore` ou `SVM`) |
| `best_r2` | R² do modelo vencedor |

**Resultado atual:** SVM venceu nos 50 países analisados.

---

### `ranking.csv`

**Ranking global** — quantas vezes cada modelo foi o melhor entre todos os países.

| Coluna | Descrição |
|--------|-----------|
| `model` | `tree` ou `svm` |
| `model_label` | Nome legível |
| `wins` | Número de países em que o modelo teve maior R² |

---

### `summary.txt`

Resumo **textual automático** gerado pelo pipeline. Exemplo:

```
SVM apresentou melhor desempenho em 50 países.
Árvore apresentou melhor desempenho em 0 países.
SVM foi o modelo superior na maioria dos cenários.
```

---

### `predictions_2027.csv`

**Previsões finais** de CO₂ per capita para o ano de 2027, usando o modelo configurado em `step4_predict_2027.py` (`SELECTED_MODEL = "svm"` por padrão).

| Coluna | Descrição |
|--------|-----------|
| `country` | País |
| `co2_2026` | Valor **observado** em 2026 (toneladas per capita) |
| `co2_predicted_2027` | Valor **previsto** para 2027 |

**Exemplos (modelo SVM):**

| País | CO₂ 2026 | CO₂ previsto 2027 |
|------|----------|-------------------|
| Brazil | 2,11 | 2,07 |
| China | 8,46 | 8,52 |
| United States | 13,97 | 14,03 |

> Para alterar o modelo usado na previsão, mude `SELECTED_MODEL` em `step4_predict_2027.py` ou no notebook `pipeline_co2.ipynb` e execute novamente a Etapa 4.

---

### `predictions_2027.xlsx`

Versão Excel de `predictions_2027.csv`.

---

## Artefatos intermediários (não versionados)

Estes arquivos são **regenerados automaticamente** ao rodar o pipeline e estão listados no `.gitignore`.

### `data/dataset.csv`

Cópia local da planilha Google Sheets após download. Contém todas as colunas originais: `year`, `country`, `co2_per_capita_t`, `population_millions`, etc.

### `data/processed/{país}.csv`

Um arquivo por país (~50), com:

- Colunas originais normalizadas
- **Features temporais:** `lag_1`, `lag_2`, `lag_3`, `rolling_mean_3`, `rolling_mean_5`, `growth_rate`
- Coluna `split`: `train` (80% inicial) ou `test` (20% final)

### `models/tree/` e `models/svm/`

Modelos treinados salvos com **joblib**:

- `{país}.joblib` — modelo serializado
- `{país}.meta.json` — hiperparâmetros, métricas, fingerprint dos dados e colunas de features

Se os dados não mudarem, o step2 **reutiliza** modelos existentes sem retreinar.

### `reports/figures/`

Gráficos PNG gerados pelo step3:

- `r2_by_country.png` — barras comparando R² Árvore vs SVM por país
- `tree_vs_svm_scatter.png` — dispersão R² Árvore × R² SVM

### `logs/pipeline.log`

Registro detalhado de execução (download, treino, erros por país).

---

## Como regenerar os resultados

```bash
python step1_prepare.py   # prepara dados
python step2_train.py     # treina modelos
python step3_report.py    # gera tabelas e gráficos
python step4_predict_2027.py  # previsões 2027
```

Ou execute todas as etapas no notebook **`pipeline_co2.ipynb`**.

---

## Interpretação rápida

| Pergunta | Arquivo |
|----------|---------|
| Qual modelo é melhor no geral? | `ranking.csv` ou `summary.txt` |
| Qual modelo usar para um país? | `best_by_country.csv` |
| Quão bom é o ajuste por país? | `country_summary.csv` |
| Qual a previsão para 2027? | `predictions_2027.csv` |
| Detalhe métrica por métrica | `metrics.csv` |

---

## Modelos utilizados

| Modelo | Algoritmo | Uso |
|--------|-----------|-----|
| **Árvore** | `DecisionTreeRegressor` | Benchmark; grid em `max_depth` e `min_samples_leaf` |
| **SVM** | `SVR` + `StandardScaler` | Modelo principal; grid em `kernel`, `C`, `gamma`, `epsilon` |

A seleção do melhor modelo por país usa **R² no conjunto de teste temporal** (sem embaralhamento dos dados).
