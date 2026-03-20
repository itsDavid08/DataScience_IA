# Data Science Project - Flight Delay Prediction

## Visão Geral

Este projeto implementa um pipeline completo de análise de dados e machine learning para prever atrasos de voos usando o dataset "Flight Delay and Cancellation Dataset (2019-2023)" do Kaggle.

**Status**: Part 1 (Phases 1-3) - Completa ✅

## Estrutura do Projeto

```
DataScience_IA/
├── DataSet/
│   └── flights_sample_3m.csv          # Dataset raw (3 milhões de linhas)
├── Output Files/                       # Artefatos gerados
├── Project Code/
│   └── PythonCode/
│       ├── main.py                     # Pipeline principal (Part 1)
│       ├── Util/
│       │   ├── DataLoader_IA.py          # Carga e split de dados
│       │   ├── DataVisualization_IA.py   # Visualizações
│       │   └── StatisticalTesting.py  # Testes estatísticos (NOVO)
│       ├── DataPreProcessor/
│       │   └── FlightDataCleaner_IA.py   # Limpeza de dados
│       ├── EDA/
│       │   └── FlightEDA_IA.py           # Análise exploratória + PCA + UMAP
│       └── FeatureEngeneering/
│           └── FlightFeatureEngineer_IA.py # Geração de 20 features (expandido)
├── requirements.txt                    # Dependências Python (NOVO)
└── README.md                          # Este arquivo
```

## Instalação

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

**Principais Bibliotecas**:
- `pandas`, `numpy`: Manipulação de dados
- `scikit-learn`: Machine learning
- `matplotlib`, `seaborn`: Visualização
- `scipy`: Testes estatísticos
- `umap-learn`: Redução não-linear de dimensionalidade
- `tensorflow`, `torch`: Deep learning (Part 2)

### 2. Preparar Dataset

1. Baixar o dataset do Kaggle: [Flight Delay and Cancellation Dataset](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/)
2. Colocar o arquivo `flights_sample_3m.csv` em `DataSet/`

## Part 1: Phases 1-3

### Phase 1: Problem Formulation
Define objetivos de análise:
- **Regressão**: Prever `ARR_DELAY` (minutos exatos)
- **Classificação**: Classificar atrasos em 3 categorias:
  - `On-time`: ARR_DELAY < 15 min
  - `Short delay`: 15 ≤ ARR_DELAY ≤ 30 min
  - `Long delay`: ARR_DELAY > 30 min
- **Clustering**: Identificar padrões operacionais (aeroportos/companhias)

### Phase 2: Data Analysis and Cleansing

#### 2.1 Data Loading (`DataLoader_IA.py`)
```python
from Util.DataLoader_IA import DataLoader_IA

loader = DataLoader_IA("flights_sample_3m.csv", test_size=0.2, random_state=42)
data_train, data_test, target_train, target_test = loader.load_data()
```

**Funcionalidades**:
- Carregamento em chunks (suporte a `nrows` para testes rápidos)
- Split train/test aleatório (80/20)
- Serialização com pickle para checkpoints

#### 2.2 Data Cleansing (`FlightDataCleaner_IA.py`)
```python
from DataPreProcessor.FlightDataCleaner_IA import FlightDataCleaner_IA

cleaner = FlightDataCleaner_IA("flights_sample_3m.csv")
df_clean = cleaner.load_and_clean()
```

**Etapas de Limpeza**:
1. ✅ Remove voos cancelados/desviados
2. ✅ Remove colunas com data leakage (DEP_TIME, ARR_TIME, DELAY_DUE_*, etc.)
3. ✅ Remove nulos em ARR_DELAY (variável alvo)
4. ✅ Trata outliers (IQR) em DISTANCE e CRS_ELAPSED_TIME
5. ✅ Remove colunas redundantes (FL_NUMBER, ORIGIN_CITY, AIRLINE_DOT, etc.)

#### 2.3 Feature Engineering (`FlightFeatureEngineer_IA.py`)

**20 Features Geradas**:

| Categoria | Features | Quantidade |
|-----------|----------|-----------|
| Temporal | MONTH, DAY_OF_WEEK, DAY_OF_YEAR, QUARTER, IS_WEEKEND, IS_HOLIDAY_SEASON, DEP_HOUR, MORNING_FLIGHT, AFTERNOON_FLIGHT, NIGHT_FLIGHT | 10 |
| Rota | ROUTE, ROUTE_FREQUENCY, PLANNED_SPEED_MPM | 3 |
| Caracterização | DISTANCE_CAT, DURATION_CAT, SPEED_CAT | 3 |
| Interações | DISTANCE_x_ELAPSED_TIME, DISTANCE_POW2, ELAPSED_TIME_POW2 | 3 |
| **Targets** | **DELAY_CLASS (On-time, Short delay, Long delay)** | **1** |

```python
from FeatureEngeneering.FlightFeatureEngineer_IA import FlightFeatureEngineer_IA

engineer = FlightFeatureEngineer_IA(df_clean)
df_features = engineer.generate_features()
df_features = engineer.encode_categorical()
df_features = engineer.normalize_features()  # StandardScaler
```

### Phase 3: Model Selection & Exploratory Data Analysis

#### 3.1 EDA Exploratória (`FlightEDA_IA.py`)
```python
from EDA.FlightEDA_IA import FlightEDA_IA

eda = FlightEDA_IA(df_features, target_col='ARR_DELAY')
eda.perform_eda()  # Distribuições, correlações, boxplots
eda.run_pca(n_components=2, explained_variance_threshold=0.8)  # Linear
eda.run_umap_or_tsne(n_components=2, use_umap=True)  # Não-linear
```

**Outputs**:
- `eda_distributions.png`: Histogramas com KDE
- `eda_correlation_matrix.png`: Matriz de correlação
- `eda_boxplots.png`: Boxplots para outliers
- `eda_pca_2d.png`: Projeção PCA
- `eda_umap_2d.png` ou `eda_tsne_2d.png`: Redução não-linear

#### 3.2 Testes Estatísticos (`StatisticalTesting.py`)
```python
from Util.StatisticalTesting import HypothesisTesting

tester = HypothesisTesting(data=df_features, labels=df_features['DELAY_CLASS'])
report = tester.generate_summary_report()
# Executa: Normality (Shapiro), ANOVA, Kruskal-Wallis, Levene, t-tests
```

**Relatórios Gerados**:
- `hypothesis_testing_normality.csv`
- `hypothesis_testing_anova.csv`
- `hypothesis_testing_kruskal_wallis.csv`
- `hypothesis_testing_levene.csv`

### Executar Pipeline Part 1

```bash
cd "Project Code/PythonCode"
python main_IA.py
```

**Output Esperado**:
```
================================================================================
INICIANDO PIPELINE PART 1 - ANÁLISE DE DADOS DE VOOS
================================================================================
...
[Pipeline Phases]
- Phase 1: Problem Formulation
- Phase 2: Data Analysis & Cleansing
- Phase 3: Model Selection & Hypothesis Testing
...
================================================================================
PIPELINE PART 1 CONCLUÍDO COM SUCESSO!
================================================================================

Artefatos gerados em:
- Output Files/ (plots, relatórios)
- pipeline_part1.log (log detalhado)
```

## Checkpoints

O pipeline salva checkpoints em `Output Files/`:

```python
# Checkpoint após limpeza e features
loader.save_checkpoint('checkpoint_cleaned_features.pkl')

# Checkpoint final (pronto para Part 2)
loader.save_checkpoint('checkpoint_part1_complete.pkl')

# Carregar depois
loader = DataLoader_IA.load_checkpoint('checkpoint_part1_complete.pkl')
```

## Responsabilidades de Cada Classe

| Classe | Responsabilidade | Status |
|--------|------------------|--------|
| `DataLoader_IA` | Carga CSV + split train/test | ✅ Refatorizado |
| `FlightDataCleaner_IA` | Remove data leakage + outliers | ✅ Expandido |
| `FlightFeatureEngineer_IA` | Gera 20 features + normaliza | ✅ Expandido |
| `FlightEDA_IA` | EDA + PCA + UMAP | ✅ Expandido |
| `DataVisualization_IA` | Gera gráficos adicionais | ✅ Expandido |
| `HypothesisTesting` | Testes estatísticos | ✅ NOVO |

## Part 2: Ready (Phases 4-6)

A arquitetura Part 1 está preparada para Part 2:

- ✅ Features normalizadas (StandardScaler)
- ✅ Train/test split realizado
- ✅ Checkpoints salvos
- ✅ Variável alvo em 2 formatos (ARR_DELAY para regressão, DELAY_CLASS para classificação)

### Próximas Etapas (Part 2):
1. **Model Building (Phase 4)**
   - kNN from scratch (numpy)
   - Supervised Learning (SKlearn: Regressão Linear, Random Forest, etc.)
   - Ensemble Models (Bagging, Boosting)
   - Deep Learning (TensorFlow/PyTorch)
   - Clustering (K-Means, DBSCAN)

2. **Model Evaluation (Phase 5)**
   - Cross-validation
   - Métricas apropriadas
   - Comparação de modelos

3. **Operationalization (Phase 6)**
   - Deployment
   - Relatório final
   - Apresentação

## Notas Importantes

### Features Seguras (Sem Data Leakage)
✅ Usadas no projeto:
- FL_DATE, CRS_DEP_TIME, CRS_ARR_TIME
- DISTANCE, CRS_ELAPSED_TIME
- AIRLINE_CODE, ORIGIN, DEST

❌ Removidas (Data Leakage):
- DEP_TIME, ARR_TIME (tempos reais)
- DEP_DELAY (diretamente relacionado com ARR_DELAY)
- DELAY_DUE_* (causas do atraso, pós-evento)
- AIR_TIME, TAXI_*, WHEELS_* (pós-voo)

### Split Strategy
- **Aleatório (Part 1)**: 80/20 com `random_state=42`
- **Temporal (Part 2 - recomendado)**: Usar data (FL_DATE) para validação realista

### Normalização
- **StandardScaler** aplicado após feature engineering
- Preserva escalas originais para EDA e testes estatísticos (não requerem normalização)

## Logging

Todo o pipeline é registrado em `pipeline_part1.log`:

```bash
tail -f pipeline_part1.log  # Monitorar em tempo real
```

## Troubleshooting

### Erro: "Dataset não encontrado"
```bash
# Garantir que o arquivo está em:
DataSet/flights_sample_3m.csv
```

### Erro: "umap-learn não instalado"
```bash
pip install umap-learn
# Ou, o código fallback para t-SNE automaticamente
```

### Memória insuficiente
```python
# Carregar amostra no main_IA.py:
data_train, data_test, _ , _ = loader.load_data(nrows=500000)  # 500k linhas
```

## Contribuições

Este projeto segue o padrão de separação de responsabilidades:
- Cada classe tem uma responsabilidade clara
- Fácil estender para Part 2
- Documentação inline completa

## Licença

Projeto acadêmico - Data Science & Machine Learning (2024)

---

**Última Atualização**: 2024-03-18
**Status**: Part 1 (Phases 1-3) Completa ✅

