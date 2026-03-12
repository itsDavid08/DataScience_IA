from pathlib import Path

from DataPreProcessor.FlightDataCleaner import FlightDataCleaner
from EDA.FlightEDA import FlightEDA
from FeatureEngeneering.FlightFeatureEngineer import FlightFeatureEngineer


def main():
    dataset_path = Path(__file__).resolve().parents[2] / 'DataSet' / 'flights_sample_3m.csv'
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')

    # 1. Limpar
    cleaner = FlightDataCleaner(str(dataset_path))
    df_clean = cleaner.load_and_clean()

    # 2. Criar Features
    engineer = FlightFeatureEngineer(df_clean)
    df_features = engineer.generate_features()

    # 3. Explorar
    eda = FlightEDA(df_features)
    eda.plot_correlation_matrix()
    eda.run_pca()


if __name__ == '__main__':
    main()
