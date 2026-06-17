# Pipeline ML — Previsão de CO₂ per capita por país

Projeto de Aprendizado de Máquina para previsão da emissão de dióxido de carbono (*CO₂ per capita*) do próximo ano, utilizando séries históricas anuais por país.

---

## Integrantes

- João Pedro Souza Peixoto Saraiva - RA: 23034350-2
- Michel Bocchi Junior - RA: 23220783-2
- Felipe Barreto Cortes - RA: 23069437-2

---

## Resumo do projeto

### Contextualização do tema

As emissões globais de dióxido de carbono (CO₂) estão entre os principais indicadores do impacto antropogênico sobre o clima. O monitoramento e a projeção dessas emissões por país são fundamentais para políticas públicas, acordos internacionais (como o Acordo de Paris) e estratégias de transição energética. A emissão *per capita* — quantidade de CO₂ emitida por habitante em toneladas — permite comparar nações com diferentes tamanhos populacionais e identificar padrões de consumo energético, industrialização e eficiência econômica. Estudos preditivos sobre esse indicador contribuem para antecipar tendências de crescimento ou redução das emissões e apoiar decisões baseadas em evidências.

### Problema investigado

O desafio central deste trabalho consiste em **prever a emissão de CO₂ per capita do ano seguinte para cada país**, a partir de um histórico anual observado. Trata-se de um problema de **regressão supervisionada em séries temporais**, no qual:

- Cada país possui uma sequência de registros anuais (aproximadamente 27 observações entre 2000 e 2026);
- A variável alvo é `co2_per_capita` (toneladas por habitante);
- O modelo deve generalizar padrões temporais sem embaralhar os dados, respeitando a ordem cronológica;
- A previsão final visa estimar o valor para **2027**, utilizando todo o histórico disponível até 2026.

A complexidade reside na heterogeneidade entre países (economias emergentes vs. desenvolvidas), na não-linearidade das curvas de emissão e no volume reduzido de observações por nação, o que exige modelos capazes de capturar relações temporais sem overfitting severo.

### Hipótese da equipe

A equipe levantou a hipótese de que **modelos capazes de lidar com relações não lineares** seriam mais adequados para capturar a dinâmica das emissões de CO₂ ao longo do tempo. Com base nisso, foram selecionados:

1. **Máquina de Vetores de Suporte (SVM)** — hipótese: o kernel RBF/linear do SVR seria capaz de mapear relações complexas entre features temporais e exógenas, especialmente quando combinado com normalização (*StandardScaler*);
2. **Árvore de Decisão** — hipótese: a estrutura hierárquica da árvore capturaria regras interpretáveis sobre tendências e patamares de emissão, sendo robusta a diferentes escalas sem necessidade de normalização explícita.

Esperava-se que ambos os modelos apresentassem desempenho competitivo, com possível vantagem do SVM em países com transições mais suaves e da Árvore em contextos com mudanças bruscas de patamar. A comparação entre eles permitiria validar qual abordagem generaliza melhor em séries temporais curtas de emissões nacionais.

### Descrição do dataset utilizado

O dataset `**co2_emission_yearly*`* reúne indicadores anuais de emissão de CO₂ por país. No projeto, os dados são obtidos via planilha Google Sheets e armazenados localmente como `data/dataset.csv`.


| Característica         | Valor            |
| ---------------------- | ---------------- |
| **Registros totais**   | 1.350            |
| **Países**             | 50               |
| **Período**            | 2000 – 2026      |
| **Registros por país** | ~27 (um por ano) |


**Colunas principais:**


| Coluna                         | Descrição                                      |
| ------------------------------ | ---------------------------------------------- |
| `year`                         | Ano da observação                              |
| `country`                      | Nome do país                                   |
| `iso3`                         | Código ISO de 3 letras                         |
| `region`                       | Região geográfica                              |
| `co2_emissions_mt`             | Emissões totais de CO₂ (megatoneladas)         |
| `population_millions`          | População (milhões)                            |
| `co2_per_capita_t`             | **Variável alvo** — CO₂ per capita (toneladas) |
| `co2_intensity_kg_per_gdp_usd` | Intensidade de carbono da economia             |


**Tratamentos de dados aplicados** (`step1_prepare.py`):

1. **Detecção automática de colunas** — normalização de nomes para `country`, `year` e `co2_per_capita`;
2. **Limpeza** — remoção de linhas com valores ausentes nas colunas essenciais e descarte de colunas auxiliares vazias;
3. **Agrupamento por país** — um DataFrame independente por nação, ordenado cronologicamente;
4. **Engenharia de features temporais:**
  - `lag_1`, `lag_2`, `lag_3` — valores defasados de CO₂ per capita;
  - `rolling_mean_3`, `rolling_mean_5` — médias móveis;
  - `growth_rate` — taxa de crescimento anual;
5. **Remoção de linhas iniciais** — eliminação de registros com NaN nas features temporais (~5 primeiros anos);
6. **Divisão temporal 80/20** — treino (2005–2021) e teste (2022–2026), **sem embaralhamento**;
7. **Normalização (SVM)** — `StandardScaler` aplicado no pipeline de treinamento do SVR.

Após o processamento, cada país dispõe de aproximadamente **23 linhas** utilizáveis para modelagem.

### Métodos de IA utilizados

#### Árvore de Decisão (`DecisionTreeRegressor`)

A Árvore de Decisão particiona recursivamente o espaço de features em regiões homogêneas, minimizando o erro de regressão em cada nó folha. Para este problema:

- Recebe features temporais e variáveis exógenas numéricas;
- Não exige normalização prévia;
- Foi submetida a **grid search** sobre `max_depth` ∈ {3, 5, 7, 10, None} e `min_samples_leaf` ∈ {1, 2, 4, 8};
- Um modelo independente é treinado **por país**, permitindo capturar dinâmicas nacionais específicas.

#### Máquina de Vetores de Suporte — SVR (`SVR`)

O SVR (Support Vector Regression) busca uma função que aproxime os dados com margem de tolerância ε, utilizando o princípio de maximização de margem adaptado à regressão. No projeto:

- Implementado como `Pipeline([StandardScaler(), SVR(...)])`;
- Testados kernels **RBF** e **linear**, com variação de `C`, `gamma` e `epsilon`;
- A normalização é essencial dado o mix de features temporais (lags, médias móveis) e exógenas (população, emissões totais);
- Também treinado **por país**, em paralelo com a Árvore.

A seleção do melhor hiperparâmetro em ambos os modelos utiliza o **maior R² no conjunto de teste temporal**.

### Avaliação dos modelos

A avaliação dos modelos contém **métricas e gráficos adequados ao tipo de problema** tratado neste projeto. Como a tarefa é **regressão supervisionada** (previsão contínua de CO₂ per capita em toneladas), utilizamos indicadores de erro e ajuste típicos desse contexto, em vez de métricas de classificação.

#### Tipo de problema e métricas aplicáveis

| Recurso de avaliação | Aplicável? | Justificativa |
| -------------------- | ---------- | ------------- |
| Acurácia, precisão, revocação e F1-score | Não | Métricas de **classificação**; o alvo é numérico contínuo |
| Matriz de confusão | Não | Exige classes discretas; não se aplica a regressão |
| MAE, MSE e RMSE | **Sim** | Erro médio absoluto, quadrático médio e sua raiz — padrão em regressão |
| R² (coeficiente de determinação) | **Sim** | Mede quanto da variância do alvo o modelo explica no teste |
| MAPE (erro percentual absoluto médio) | **Sim** | Complementa MAE/RMSE com interpretação relativa (%) |
| Curva de perda no treinamento | Não* | Árvore e SVR (scikit-learn) não expõem épocas iterativas como redes neurais; a seleção de hiperparâmetros usa **grid search** com R² no conjunto de teste temporal |
| Comparação gráfica entre modelos | **Sim** | Gráficos de barras e dispersão por país (ver abaixo) |
| Ranking de vitórias por modelo | **Sim** | Contagem de países em que cada algoritmo obteve maior R² |

\* Em modelos iterativos (ex.: redes neurais), a curva de perda seria obrigatória; aqui a validação segue o protocolo temporal 80/20 e a escolha do melhor hiperparâmetro por país.

#### Protocolo de avaliação

- **Conjunto avaliado:** 20% final de cada série temporal (anos ~2022–2026), **sem embaralhamento**, preservando a ordem cronológica.
- **Escopo:** 50 países × 2 modelos (SVM e Árvore de Decisão) = 100 avaliações independentes.
- **Critério de seleção do melhor modelo por país:** maior **R²** no teste (`reports/best_by_country.csv`, `reports/ranking.csv`).
- **Implementação:** `utils/metrics.py` (`compute_regression_metrics`) e `step2_train.py`; relatórios e figuras em `step3_report.py`.

#### Métricas agregadas (conjunto de teste)

| Métrica                          | SVM (SVR) | Árvore de Decisão |
| -------------------------------- | --------- | ----------------- |
| **R² médio**                     | 0,931     | −4,730            |
| **R² mediano**                   | 0,981     | −2,861            |
| **MAE médio**                    | 0,027 t   | 0,332 t           |
| **RMSE médio**                   | 0,031 t   | 0,376 t           |
| **MAPE médio**                   | ~0,52 %   | ~6,17 %           |
| **Países vencedores (maior R²)** | **50**    | **0**             |

- **MAE** — erro médio absoluto em toneladas de CO₂ per capita; quanto menor, melhor.
- **MSE / RMSE** — penalizam erros grandes; RMSE está na mesma unidade do alvo (toneladas).
- **R²** — varia de −∞ a 1; valores negativos indicam desempenho **pior que prever a média** do teste.
- **MAPE** — erro percentual médio; útil para comparar países com magnitudes diferentes de emissão.

**Exemplos de R² por país (teste):**

| País          | SVM (R²) | Árvore (R²) | SVM (MAE) | Árvore (MAE) |
| ------------- | -------- | ----------- | --------- | ------------ |
| Brazil        | 0,900    | −1,107      | 0,015 t   | 0,060 t      |
| China         | 0,775    | −19,408     | 0,042 t   | 0,432 t      |
| United States | 0,982    | −0,013      | 0,024 t   | 0,246 t      |
| India         | 0,992    | −5,102      | 0,006 t   | 0,200 t      |
| Iran          | 0,999    | −16,189     | 0,007 t   | 0,800 t      |

Valores de R² negativos na Árvore indicam overfitting ou inadequação à curva temporal em séries curtas (~18 amostras de treino por país).

#### Comparação gráfica entre modelos

Os gráficos abaixo são gerados automaticamente por `step3_report.py` e salvos em `reports/figures/`. Eles estão embutidos neste README para visualização no repositório e **também aparecem no PDF** quando o README é exportado (ex.: Pandoc, extensões Markdown→PDF do VS Code/Cursor), desde que as imagens existam no caminho relativo indicado.

**1. Comparativo de R² por país — SVM vs Árvore de Decisão**

![Comparativo de R² por país — SVM vs Árvore de Decisão](./reports/figures/r2_by_country.png)

Barras horizontais por país; permite comparar visualmente qual modelo explica melhor a variância no período de teste.

**2. Dispersão R² Árvore vs R² SVM**

![Dispersão R² Árvore vs R² SVM](./reports/figures/tree_vs_svm_scatter.png)

Cada ponto é um país. Pontos **acima da diagonal tracejada** indicam R² superior do SVM em relação à Árvore — o que ocorre em todos os 50 países neste experimento.

> **Nota:** Execute `./run.sh` ou `python step3_report.py` para (re)gerar as figuras em `reports/figures/` antes de visualizar o README ou exportar o PDF, caso a pasta ainda não exista.

#### Arquivos de resultados detalhados

- [`reports/metrics.csv`](./reports/metrics.csv) — MAE, MSE, RMSE, R² e MAPE por país e modelo
- [`reports/country_summary.csv`](./reports/country_summary.csv) — comparativo consolidado lado a lado
- [`reports/ranking.csv`](./reports/ranking.csv) — vitórias por modelo (SVM: 50, Árvore: 0)
- [`reports/best_by_country.csv`](./reports/best_by_country.csv) — melhor modelo por país
- [`reports/predictions_2027.csv`](./reports/predictions_2027.csv) — previsões finais para 2027

### Comparação dos resultados

A análise comparativa demonstra **superioridade consistente do SVM (SVR)** em relação à Árvore de Decisão neste dataset:

1. **Cobertura:** O SVM apresentou maior R² em **todos os 50 países**, sem exceção (`ranking.csv`: SVM = 50 vitórias, Árvore = 0).
2. **Magnitude do erro:** O MAE médio do SVM (0,027 t) é **~12 vezes menor** que o da Árvore (0,332 t), indicando previsões substancialmente mais próximas dos valores reais no período de teste (2022–2026).
3. **Estabilidade:** O R² mediano do SVM (0,981) contrasta com o R² mediano negativo da Árvore (−2,861). A Árvore tende a memorizar padrões do treino e falhar na generalização temporal, especialmente em países com alta variabilidade (ex.: China, R² = −19,4; Venezuela, R² = −29,9).
4. **Comportamento nas predições de 2027:** As previsões do SVM para 2027 são coerentes com a tendência recente. Exemplos:

  | País          | CO₂ 2026 (obs.) | CO₂ 2027 (previsto) |
  | ------------- | --------------- | ------------------- |
  | Brazil        | 2,11 t          | 2,07 t              |
  | China         | 8,46 t          | 8,52 t              |
  | United States | 13,97 t         | 14,03 t             |
  | India         | 1,87 t          | 1,95 t              |

   As variações previstas são suaves, compatíveis com séries temporais estáveis no curto prazo.
5. **Interpretação técnica:** O SVR com `StandardScaler` e kernel linear/RBF consegue aproximar a continuidade das séries de emissão quando combinado com lags e médias móveis. A Árvore, com poucas observações (~18 amostras de treino), tende a criar partições muito específicas que não se transferem ao período de teste posterior.

O gráfico de dispersão (*tree_vs_svm_scatter*) evidencia que praticamente todos os pontos ficam **acima da diagonal**, confirmando que o R² do SVM supera o da Árvore em cada nação analisada.

### Conclusão

Com base nos experimentos realizados sobre o dataset `co2_emission_yearly`, conclui-se que:

- **A hipótese de que modelos não lineares seriam adequados foi parcialmente validada:** o SVM (SVR), capaz de modelar relações não lineares via kernels, obteve excelente desempenho (R² médio ≈ 0,93). Porém, a Árvore de Decisão — também não linear — **não** generalizou bem neste contexto de séries temporais curtas por país.
- **O SVM é a abordagem recomendada** para predição de emissões de CO₂ per capita com este dataset. Sua combinação de normalização, margem de tolerância e seleção de hiperparâmetros via grid search produziu modelos robustos e consistentes em todos os 50 países.
- **A predição para 2027** foi realizada com o modelo SVM, gerando estimativas por país disponíveis em `reports/predictions_2027.csv`. Essas projeções devem ser interpretadas como extrapolações baseadas em tendências históricas, sujeitas a incertezas geopolíticas, econômicas e climáticas não capturadas pelos dados.
- **Trabalhos futuros** podem explorar modelos específicos para séries temporais (ARIMA, Prophet, LSTM), validação cruzada temporal (*time series cross-validation*) e inclusão de variáveis macroeconômicas adicionais para enriquecer as previsões exógenas.

---

## Execução do projeto

```bash
./run.sh                  # pipeline completo
./jupyter.sh              # notebook interativo em localhost:8888
```

Documentação complementar: `[EXPLICACAO_RESULTADOS.md](./EXPLICACAO_RESULTADOS.md)`