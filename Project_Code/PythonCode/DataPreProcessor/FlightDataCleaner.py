import pandas as pd
import numpy as np
from typing import Union


class FlightDataCleaner:
    """Single cleaner class supporting both DataFrame and file-path entrypoints."""

    def __init__(self, df: pd.DataFrame = None, file_path: str = None):
        # Backward compatibility: allow FlightDataCleaner_IA("path/to/file.csv").
        if isinstance(df, str) and file_path is None:
            file_path = df
            df = None

        self.file_path = file_path
        self.original_shape = None

        if df is not None and not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        if df is not None:
            self.df = df.copy()
        else:
            self.df = pd.DataFrame()

        # Keep `data` for backward compatibility with existing code.
        self.data = self.df

    def fill_missing(
        self,
        strategy: str = "mean",
        value: Union[int, float, str, None] = None,
    ) -> None:
        """Fill missing values using mean, median, mode, or constant value."""
        valid_strategies = ["mean", "median", "mode", "constant"]
        if strategy not in valid_strategies:
            raise ValueError("Strategy must be one of 'mean', 'median', 'mode', or 'constant'")

        if strategy == "mean":
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == "median":
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == "mode":
            for col in self.df.columns:
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    self.df[col] = self.df[col].fillna(mode_val.iloc[0])
        else:
            if value is None:
                raise ValueError("Must provide a value for constant filling")
            self.df = self.df.fillna(value)

    def handle_missing_values(
        self,
        method: str = "drop",
        fill_value: Union[int, float, str, None] = None,
    ) -> None:
        """Handle missing values by dropping rows or filling with a constant."""
        if method not in ["drop", "fill"]:
            raise ValueError("Method must be 'drop' or 'fill'")

        if method == "drop":
            self.df = self.df.dropna()
        else:
            if fill_value is None:
                raise ValueError("fill_value must be provided when method='fill'")
            self.df = self.df.fillna(fill_value)

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    def remove_cancelled_diverted(self) -> pd.DataFrame:
        self.df = self.df[(self.df["CANCELLED"] == 0) & (self.df["DIVERTED"] == 0)]
        self.data = self.df
        return self.df

    def remove_data_leak_cols(self) -> None:
        leakage_cols = [
            "DEP_DELAY",
            "DELAY_DUE_CARRIER",
            "DELAY_DUE_WEATHER",
            "DELAY_DUE_NAS",
            "DELAY_DUE_SECURITY",
            "DELAY_DUE_LATE_AIRCRAFT",
            "ARR_TIME",
            "DEP_TIME",
            "WHEELS_OFF",
            "WHEELS_ON",
            "TAXI_OUT",
            "TAXI_IN",
            "ELAPSED_TIME",
            "AIR_TIME",
            "CANCELLED",
            "CANCELLATION_CODE",
            "DIVERTED",
        ]
        self.df = self.df.drop(columns=leakage_cols, errors="ignore")
        self.data = self.df

    def save(self, filename: str) -> None:
        """Save current dataframe as CSV or XLSX."""
        if filename.endswith(".csv"):
            self.df.to_csv(filename, index=False)
        elif filename.endswith(".xlsx"):
            self.df.to_excel(filename, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

    def _handle_outliers_and_nans(self) -> None:
        """Treat outliers via IQR and impute resulting numeric NaNs with mean."""
        cols_for_outliers = ["DISTANCE", "CRS_ELAPSED_TIME"]

        for col in cols_for_outliers:
            if col in self.df.columns:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()

                self.df[col] = np.where(
                    (self.df[col] < lower_bound) | (self.df[col] > upper_bound),
                    np.nan,
                    self.df[col],
                )

                self.df[col] = self.df[col].fillna(self.df[col].mean())
                print(f"   ✓ {col}: {outliers_count} outliers tratados")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        missing_count = self.df[numeric_cols].isnull().sum().sum()

        if missing_count > 0:
            for col in numeric_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            print(f"   ✓ {missing_count} missing values imputados (média)")
        else:
            print("   ✓ Nenhum missing value encontrado")
        self.data = self.df

    def load_and_clean(self, nrows=None):
        """Load CSV and execute the standardized flight cleaning pipeline."""
        print("=" * 60)
        print("INICIANDO PROCESSO DE LIMPEZA DE DADOS")
        print("=" * 60)

        if self.df.empty:
            if not self.file_path:
                raise ValueError("Provide file_path or initialize with a non-empty DataFrame")
            print("\n1. A carregar os dados...")
            self.df = pd.read_csv(self.file_path, nrows=nrows)
        else:
            print("\n1. A usar DataFrame fornecido no construtor...")

        self.data = self.df
        self.original_shape = self.df.shape
        print(f"   ✓ Dataset carregado: {self.original_shape[0]} linhas × {self.original_shape[1]} colunas")

        print("\n2. A remover voos cancelados e desviados...")
        initial_rows = len(self.df)
        self.remove_cancelled_diverted()
        self.data = self.df
        removed = initial_rows - len(self.df)
        print(f"   ✓ Removidos {removed} voos (cancelados/desviados)")

        print("\n3. A remover colunas com data leakage...")
        cols_before = self.df.shape[1]
        self.remove_data_leak_cols()
        self.data = self.df
        print(f"   ✓ Removidas {cols_before - self.df.shape[1]} colunas com data leakage")

        print("\n4. A remover nulos na variável alvo...")
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=["ARR_DELAY"])
        self.data = self.df
        removed = initial_rows - len(self.df)
        print(f"   ✓ Removidas {removed} linhas com ARR_DELAY nulo")

        print("\n5. A tratar outliers e missing values...")
        self._handle_outliers_and_nans()
        self.data = self.df

        print("\n6. A remover colunas redundantes...")
        redundant_cols = ["FL_NUMBER", "ORIGIN_CITY", "DEST_CITY", "AIRLINE_DOT", "DOT_CODE"]
        cols_before = self.df.shape[1]
        self.df = self.df.drop(columns=redundant_cols, errors="ignore")
        self.data = self.df
        print(f"   ✓ Removidas {cols_before - self.df.shape[1]} colunas redundantes")

        print("\n" + "=" * 60)
        print("LIMPEZA CONCLUÍDA!")
        print(f"Dimensão final: {self.df.shape[0]} linhas × {self.df.shape[1]} colunas")
        print(
            f"Redução: {self.original_shape[0] - self.df.shape[0]} linhas removidas "
            f"({100 * (self.original_shape[0] - self.df.shape[0]) / self.original_shape[0]:.1f}%)"
        )
        print("=" * 60 + "\n")

        return self.df
