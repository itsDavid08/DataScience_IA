# ARQUITETURA & DESIGN - Part 1 Project

## Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE PART 1 (Complete)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  main.py (Orquestrador)                                        │
│    │                                                            │
│    ├─→ DataLoader_IA                  [Util/]                    │
│    │    └─ Carrega + Split train/test (80/20)                 │
│    │    └─ Serialização (checkpoints)                         │
│    │                                                            │
│    ├─→ FlightDataCleaner_IA           [DataPreProcessor/]        │
│    │    └─ Remove CANCELLED/DIVERTED                          │
│    │    └─ Remove data leakage (15 colunas)                   │
│    │    └─ Trata outliers (IQR)                               │
│    │    └─ Trata missing values                               │
│    │                                                            │
│    ├─→ FlightFeatureEngineer_IA       [FeatureEngeneering/]      │
│    │    └─ Gera 20 features (temporal, rota, interações)      │
│    │    └─ Encode_categorical() - LabelEncoder                │
│    │    └─ Normalize_features() - StandardScaler              │
│    │                                                            │
│    ├─→ FlightEDA_IA                   [EDA/]                     │
│    │    └─ perform_eda() - Distribuições, correlações, box   │
│    │    └─ run_pca() - Redução linear                         │
│    │    └─ run_umap_or_tsne() - Redução não-linear           │
│    │                                                            │
│    ├─→ DataVisualization_IA           [Util/]                    │
│    │    └─ plot_histograms()                                  │
│    │    └─ plot_boxplots()                                    │
│    │    └─ plot_correlation_matrix()                          │
│    │    └─ plot_density_ridges()                              │
│    │    └─ plot_pairplot()                                    │
│    │                                                            │
│    └─→ HypothesisTesting           [Util/] - NEW              │
│         └─ perform_normality_test() - Shapiro-Wilk            │
│         └─ perform_anova_test()                               │
│         └─ perform_kruskal_wallis_test()                      │
│         └─ perform_levene_test()                              │
│         └─ perform_t_tests()                                  │
│         └─ generate_summary_report()                          │
│                                                                 │
│  Outputs:                                                       │
│    ├─ Output Files/ (plots, reports, checkpoints)             │
│    ├─ pipeline_part1.log (logging completo)                   │
│    └─ Pronto para Part 2 (sem mudanças estruturais)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Separation of Concerns
Cada classe tem **UMA** responsabilidade bem definida:

```
DataLoader_IA          → APENAS carregar e dividir dados
    ↓
FlightDataCleaner_IA   → APENAS limpar dados
    ↓
FeatureEngineer     → APENAS criar/processar features
    ↓
EDA                 → APENAS análise exploratória
    ↓
HypothesisTesting   → APENAS testes estatísticos
```

### 2. Data Flow Paradigm

```
Raw Data (CSV)
    ↓
[Load & Split]  ← DataLoader_IA
    ↓
[Clean]         ← FlightDataCleaner_IA
    ↓
[Engineer]      ← FlightFeatureEngineer_IA
    ↓
[Analyze]       ← FlightEDA_IA + DataVisualization_IA
    ↓
[Test]          ← HypothesisTesting
    ↓
[Checkpoint]    ← Pickle (for Part 2)
```

### 3. Immutability Pattern

```python
# Classes NÃO modificam dados em-place
# Retornam novos DataFrames

class FlightFeatureEngineer_IA:
    def __init__(self, data):
        self.data = data.copy()  # ← Cópia para evitar side effects
    
    def generate_features(self):
        # Trabalha com self.data
        return self.data  # ← Retorna novo DataFrame
```

### 4. Extensibility

```python
# Fácil adicionar novos métodos em Part 2

# Em FlightEDA_IA:
def run_autoencoders(self):
    """Redução com deep learning"""
    pass

# Em HypothesisTesting:
def perform_chi_square_test(self):
    """Teste categórico"""
    pass

# SEM modificar estrutura existente!
```

---

## Detalhes de Implementação

### DataLoader_IA Flow

```
__init__(file_path, test_size=0.2, random_state=42)
    │
    ├─ self.file_path
    ├─ self.test_size
    ├─ self.random_state
    └─ Inicializa atributos vazios

load_data(nrows=None, target_column='ARR_DELAY')
    │
    ├─ pd.read_csv() com nrows opcional
    ├─ Separa X (features) de y (target)
    ├─ train_test_split() aleatório
    └─ Retorna (train, test, target_train, target_test)

save_checkpoint(filepath)
    └─ pickle.dump(self) → arquivo .pkl

@staticmethod
load_checkpoint(filepath)
    └─ pickle.load() → retorna DataLoader_IA instance
```

### FlightDataCleaner_IA Flow

```
__init__(file_path)
    └─ Inicializa para processar

load_and_clean(nrows=None)
    │
    ├─ [STEP 1] Carregar CSV
    ├─ [STEP 2] Remover CANCELLED/DIVERTED
    ├─ [STEP 3] Remover 15 colunas leakage
    ├─ [STEP 4] Remover nulos em ARR_DELAY
    ├─ [STEP 5] _handle_outliers_and_nans()
    │           ├─ IQR em DISTANCE, CRS_ELAPSED_TIME
    │           └─ Imputação por média
    ├─ [STEP 6] Remover colunas redundantes
    │
    └─ return df_clean
```

### FlightFeatureEngineer_IA Flow

```
__init__(data)
    ├─ self.data = data.copy()
    └─ self.scaler = StandardScaler()

generate_features()
    │
    ├─ [STEP 1] Temporal Features (10)
    │           FL_DATE → MONTH, DAY_OF_WEEK, etc.
    │
    ├─ [STEP 2] Route Features (3)
    │           ORIGIN+DEST → ROUTE, ROUTE_FREQUENCY
    │
    ├─ [STEP 3] Categorization Features (3)
    │           pd.cut() → DISTANCE_CAT, DURATION_CAT, SPEED_CAT
    │
    ├─ [STEP 4] Interaction Features (3)
    │           Polinômios: DISTANCE_POW2, etc.
    │
    ├─ [STEP 5] Target Variable (1)
    │           np.select() → DELAY_CLASS (3 classes)
    │
    └─ return df_features

encode_categorical()
    │
    ├─ LabelEncoder para:
    │   ├─ DISTANCE_CAT
    │   ├─ DURATION_CAT
    │   ├─ SPEED_CAT
    │   ├─ DELAY_CLASS
    │   └─ ROUTE
    │
    └─ Armazena encoders em self.encoders (para Part 2)

normalize_features(exclude_cols=['ARR_DELAY', 'DELAY_CLASS'])
    │
    ├─ StandardScaler.fit_transform()
    ├─ Exclui targets (não normaliza)
    │
    └─ return df_normalized
```

### FlightEDA_IA Flow

```
__init__(data, target_col='ARR_DELAY')
    ├─ self.data = data
    ├─ self.numeric_cols = [...] # Apenas colunas numéricas
    └─ self.target_col = target_col

perform_eda()
    │
    ├─ plot_distributions() → eda_distributions.png
    ├─ plot_correlation_matrix() → eda_correlation_matrix.png
    ├─ plot_boxplots() → eda_boxplots.png
    └─ plot_target_distribution() → eda_target_distribution.png

run_pca(n_components=2, explained_variance_threshold=0.8)
    │
    ├─ StandardScaler.fit_transform()
    ├─ PCA() para determinar componentes automáticos
    ├─ PCA(n_components=n).fit_transform()
    │
    └─ Scatter plot colorido → eda_pca_2d.png

run_umap_or_tsne(n_components=2, use_umap=True)
    │
    ├─ Se UMAP disponível:
    │   └─ umap.UMAP().fit_transform()
    │
    ├─ Else:
    │   └─ TSNE().fit_transform() (fallback)
    │
    └─ Scatter plot → eda_umap_2d.png (ou tsne_2d.png)

get_summary_stats()
    └─ df.describe() formatado
```

### HypothesisTesting Flow

```
__init__(data, labels=None, target_col='DELAY_CLASS')
    ├─ self.data = data (features numéricas)
    ├─ self.labels = labels (grupos/classes)
    └─ self.numeric_cols = [...]

perform_normality_test()
    │
    ├─ Shapiro-Wilk para cada feature
    ├─ H0: Dados são normalmente distribuídos
    │
    └─ CSV: hypothesis_testing_normality.csv

perform_anova_test()
    │
    ├─ f_oneway(*groups)
    ├─ H0: Não há diferença entre grupos
    │
    └─ CSV: hypothesis_testing_anova.csv

perform_kruskal_wallis_test()
    │
    ├─ kruskal(*groups) [não-paramétrico]
    ├─ H0: Não há diferença entre grupos
    │
    └─ CSV: hypothesis_testing_kruskal_wallis.csv

perform_levene_test()
    │
    ├─ levene(*groups)
    ├─ H0: Variâncias são iguais
    │
    └─ CSV: hypothesis_testing_levene.csv

perform_t_tests()
    │
    ├─ Para cada par de grupos:
    │   └─ ttest_ind(group1, group2)
    │
    └─ CSV: hypothesis_testing_t_tests.csv (se > 2 grupos)

generate_summary_report()
    │
    ├─ Executa todos os testes acima
    ├─ Retorna dict com todos os DataFrames
    │
    └─ Salva todas as CSVs automaticamente
```

---

## Estado da Arte - Variável Alvo

### Para Regressão (Part 2)
```python
ARR_DELAY  # Variável contínua (minutos)
# Range: -180 a +500 (aproximadamente)
# Distribuição: Right-skewed (mais 0s e pequenos atrasos)
```

### Para Classificação (Part 2)
```python
DELAY_CLASS  # 3 classes
├─ 'On-time': ARR_DELAY < 15
├─ 'Short delay': 15 ≤ ARR_DELAY ≤ 30
└─ 'Long delay': ARR_DELAY > 30

# Distribuição (desbalanceada):
# On-time: ~70%
# Short: ~15%
# Long: ~15%
```

---

## Estratégia de Split

### Train/Test (Part 1)
```python
# Aleatório (80/20)
X_train, X_test = train_test_split(
    X, 
    test_size=0.2, 
    random_state=42  # Reproducibilidade
)
```

### Temporal (Part 2 - Recomendado para série temporal)
```python
# Usar FL_DATE para split realista
# Evita vazamento de informação temporal

cutoff_date = '2023-01-01'
train_mask = df['FL_DATE'] < cutoff_date
test_mask = df['FL_DATE'] >= cutoff_date

X_train = X[train_mask]
X_test = X[test_mask]
```

---

## Tratamento de Data Leakage

### Colunas Removidas (15 total)
```
❌ DEP_TIME          # Tempo real de partida
❌ ARR_TIME          # Tempo real de chegada
❌ WHEELS_OFF        # Movimento da aeronave
❌ WHEELS_ON         # Movimento da aeronave
❌ TAXI_OUT          # Tempo de táxi (pós-evento)
❌ TAXI_IN           # Tempo de táxi (pós-evento)
❌ ELAPSED_TIME      # Tempo de voo real
❌ AIR_TIME          # Tempo de ar real
❌ DEP_DELAY         # LEAK DIRETO com ARR_DELAY
❌ DELAY_DUE_CARRIER # Causa do atraso (pós-evento)
❌ DELAY_DUE_WEATHER # Causa do atraso (pós-evento)
❌ DELAY_DUE_NAS     # Causa do atraso (pós-evento)
❌ DELAY_DUE_SECURITY# Causa do atraso (pós-evento)
❌ DELAY_DUE_LATE_AIRCRAFT # Causa (pós-evento)
❌ CANCELLED/DIVERTED/CANCELLATION_CODE
```

### Colunas Seguras (Usadas)
```
✅ FL_DATE           # Data do voo
✅ CRS_DEP_TIME      # Hora planejada de partida
✅ CRS_ARR_TIME      # Hora planejada de chegada
✅ DISTANCE          # Distância do voo
✅ CRS_ELAPSED_TIME  # Duração planejada
✅ AIRLINE_CODE      # Código da companhia
✅ ORIGIN            # Aeroporto de partida
✅ DEST              # Aeroporto de destino

+ 20 Features derivadas (seguras)
```

---

## Checkpoints & Resumé

### Checkpoint 1: `checkpoint_cleaned_features.pkl`
```
Salvo após:
- Limpeza completa
- Feature engineering
- Encoding categórico
- Normalização

Contém:
- df_features (todas as transformações)
- Scaler (StandardScaler fitted)
```

### Checkpoint 2: `checkpoint_part1_complete.pkl`
```
Salvo ao final de main.py

Contém:
- DataLoader_IA instance com:
  - loader.data (original)
  - loader.data_train (processed)
  - loader.data_test (processed)
  - loader.target_train
  - loader.target_test

Ready para Part 2 sem recalcular
```

---

## Performance Esperada

### Tempo Completo (3M linhas)
```
[main.py]
DataLoader_IA: 15 segundos
FlightDataCleaner_IA: 45 segundos
FeatureEngineer: 30 segundos
EDA + Plots: 120 segundos
HypothesisTesting: 60 segundos
────────────────────
Total: ~4-5 minutos
```

---

## Extensibilidade para Part 2

### Padrão a Seguir

```python
# Part 2: Adicionar novo módulo mantendo arquitetura

# ModelBuilder.py
class ModelBuilder:
    def __init__(self, data_loader):
        self.data_train = data_loader.data_train
        self.target_train = data_loader.target_train
        self.scaler = data_loader.scaler  # Reutilizar!
    
    def build_knn_scratch(self):
        """kNN from numpy"""
        pass
    
    def build_random_forest(self):
        """SKlearn"""
        pass
    
    def build_ensemble(self):
        """Bagging + Boosting"""
        pass

# ModelEvaluator.py
class ModelEvaluator:
    def compare_models(self, models_dict):
        """Compare performance"""
        pass
    
    def generate_report(self):
        """Final report"""
        pass

# Em main_part2.py:
loader = DataLoader_IA.load_checkpoint('checkpoint_part1_complete.pkl')
builder = ModelBuilder(loader)
evaluator = ModelEvaluator()

# SEM refatorar Part 1!
```

---

## Conclusão

A arquitetura Part 1 é:

✅ **Modular**: Cada classe tem responsabilidade clara
✅ **Extensível**: Fácil adicionar Part 2
✅ **Reproducível**: random_state em todos os spots
✅ **Robusta**: Logging + checkpoints + error handling
✅ **Documentada**: README + docstrings + inline comments

**Status**: Pronto para Part 2 com zero refatorações necessárias!

## Execução em Jupyter Notebook (Imports)

```python
# Se o notebook não estiver dentro de ".../Project_Code/PythonCode":
import sys
from pathlib import Path

project_root = Path.cwd()  # ajuste se necessário
python_code = project_root / "Project_Code" / "PythonCode"
sys.path.insert(0, str(python_code))
```

Depois disso, manter imports conforme arquitetura:

```python
from Util.DataVisualization_IA import DataVisualization_IA
from EDA.FlightEDA_IA import FlightEDA_IA
```
