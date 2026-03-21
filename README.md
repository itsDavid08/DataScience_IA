# Data Science Project - Flight Delay Analysis and Prediction

## Overview

This repository implements an end-to-end data pipeline for flight delay analysis using the Kaggle dataset **Flight Delay and Cancellation Dataset (2019-2023)**.

Current status: **Part 1 complete** (data loading, cleaning, feature engineering, EDA, and hypothesis testing).

## Project Structure

```text
DataScience_IA/
|-- DataSet/
|   |-- checkpoint_cleaned_features.pkl
|   |-- checkpoint_part1_complete.pkl
|   |-- cleaned_flight_data.csv
|   `-- flights_sample_3m.csv
|-- Output_Files/
|   |-- eda_boxplots.png
|   |-- eda_correlation_matrix.png
|   |-- eda_distributions.png
|   |-- eda_grouped_boxplots.png
|   |-- eda_grouped_distributions.png
|   |-- eda_pca_2d.png
|   |-- eda_target_distribution.png
|   |-- eda_umap_2d.png
|   |-- hypothesis_testing_anova.csv
|   |-- hypothesis_testing_kruskal_wallis.csv
|   |-- hypothesis_testing_levene.csv
|   |-- hypothesis_testing_normality.csv
|   |-- hypothesis_testing_t_tests.csv
|   |-- pipeline_part1_notebook.log
|   |-- viz_grouped_boxplots_delay_class.png
|   |-- viz_grouped_distributions_delay_class.png
|   `-- viz_heatmap_top_correlations.png
|-- Project_Code/
|   |-- JupyterNotebook/
|   |   `-- Data_Exploration.ipynb
|   `-- PythonCode/
|       |-- main.py
|       |-- DataPreProcessor/
|       |   `-- FlightDataCleaner.py
|       |-- EDA/
|       |   `-- FlightEDA.py
|       |-- FeatureEngeneering/
|       |   `-- FlightFeatureEngineer.py
|       |-- HypothesisTesting/
|       |   `-- HypothesisTester.py
|       `-- Util/
|           |-- DataLoader.py
|           `-- DataVisualization.py
|-- ARCHITECTURE.md
|-- IMPLEMENTATION_SUMMARY.md
|-- config.yml
|-- requirements.txt
`-- README.md
```

> Note: The folder name `FeatureEngeneering` is intentionally kept as-is to match the current repository structure.

## Installation

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Prepare the dataset

1. Download: [Flight Delay and Cancellation Dataset](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/)
2. Place `flights_sample_3m.csv` in `DataSet/`

If the CSV is missing, `DataLoader` can attempt an automatic download using `kagglehub`.

## Main Pipeline (Part 1)

The entry point is `Project_Code/PythonCode/main.py`.

### What the script runs

1. Dataset loading and optional train/test split (`DataLoader`)
2. Data cleaning and target class creation (`FlightDataCleaner`)
3. EDA and dimensionality reduction (`FlightEDA`)
4. Feature engineering, encoding, and normalization (`FlightFeatureEngineer`)
5. Hypothesis testing (`HypothesisTester`)
6. Checkpoint export (`checkpoint_cleaned_features.pkl`, `checkpoint_part1_complete.pkl`)

### Run command

From repository root:

```bash
python Project_Code/PythonCode/main.py
```

Optional example with custom arguments:

```bash
python Project_Code/PythonCode/main.py --nrows 500000 --test-size 0.2 --random-state 42
```

## Key Components

### `DataLoader`
- Loads CSV data.
- Supports `nrows` for smaller test runs.
- Performs train/test split.
- Saves and loads checkpoints.

### `FlightDataCleaner`
- Removes canceled/diverted flights.
- Drops null target rows (`ARR_DELAY`).
- Normalizes negative delays to zero.
- Balances delayed vs. non-delayed records.
- Handles outliers and numeric missing values.
- Adds `DELAY_CLASS` (`On-time`, `Short delay`, `Long delay`).

### `FlightFeatureEngineer`
Creates engineered features in four groups:
- Temporal (month, weekday, hour, period flags)
- Route-level (`ROUTE`, `ROUTE_FREQUENCY`, `PLANNED_SPEED_MPM`)
- Categorized bins (`DISTANCE_CAT`, `DURATION_CAT`, `SPEED_CAT`)
- Interaction and polynomial features

Then it label-encodes categorical fields and applies `StandardScaler` to numeric features.

### `FlightEDA`
- Descriptive and quality diagnostics.
- Correlation and outlier analysis.
- Visual outputs (distribution, boxplot, correlation heatmap).
- PCA and UMAP (or t-SNE fallback).

### `HypothesisTester`
Generates statistical reports for:
- Normality (Shapiro)
- ANOVA
- Kruskal-Wallis
- Levene
- Pairwise t-tests

## Outputs

Main outputs are saved to `Output_Files/`:
- EDA charts (`eda_*.png`, `viz_*.png`)
- Statistical CSV reports
- Pipeline log: `pipeline_part1_notebook.log`

## Checkpoints

The pipeline writes serialized checkpoints to `DataSet/`:
- `checkpoint_cleaned_features.pkl`
- `checkpoint_part1_complete.pkl`

Load a checkpoint:

```python
from Project_Code.PythonCode.Util.DataLoader import DataLoader
loader = DataLoader.load_checkpoint("DataSet/checkpoint_part1_complete.pkl")
```

## Notes

- The codebase currently targets Part 1 workflow.
- Existing outputs may include legacy filenames from previous runs.
- For Part 2, this project is ready to extend with model training and evaluation modules.

## Troubleshooting

### Dataset not found
Ensure this file exists:

```text
DataSet/flights_sample_3m.csv
```

### UMAP not installed
Install optional dependency:

```bash
pip install umap-learn
```

If UMAP is unavailable, the EDA module falls back to t-SNE.

## License

Academic project for Data Science and Machine Learning coursework.
