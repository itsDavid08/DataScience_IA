import pandas as pd
import numpy as np
from typing import Union


class FlightDataCleaner:
    """Clean flight datasets from DataFrame or CSV entrypoints.

    This class centralizes the cleaning routine used in the pipeline, including
    filtering canceled/diverted flights, removing leakage columns, handling
    missing values, and basic outlier treatment.

    Attributes:
        file_path: Optional CSV path used when data is not provided in memory.
        original_shape: Dataset shape captured before pipeline cleaning.
        df: Working pandas DataFrame updated by each cleaning step.
        data: Backward-compatible alias to the current dataframe reference.
    """

    def __init__(self, df: pd.DataFrame = None, file_path: str = None):
        """Initialize the cleaner with either in-memory data or a CSV path.

        Args:
            df: Optional pandas DataFrame to clean. For backward compatibility,
                a string value is interpreted as `file_path` when `file_path`
                is not provided.
            file_path: Optional CSV path to load during `load_and_clean`.

        Raises:
            ValueError: If `df` is provided and is not a pandas DataFrame.
        """
        # Backward compatibility: allow FlightDataCleaner("path/to/file.csv").
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

        # Keep `data` as a compatibility alias used by older pipeline code.
        self.data = self.df

    def fill_missing(
        self,
        strategy: str = "mean",
        value: Union[int, float, str, None] = None,
    ) -> None:
        """Fill missing values using a selected imputation strategy.

        Inputs:
            Uses `self.df` as the active dataframe.

        Args:
            strategy: Imputation mode. Supported values are `mean`, `median`,
                `mode`, and `constant`.
            value: Constant replacement value required when
                `strategy == "constant"`.

        Returns:
            None. The operation updates `self.df` in place or by reassignment.

        Raises:
            ValueError: If `strategy` is not supported.
            ValueError: If `strategy` is `constant` and `value` is None.
        """
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
        """Handle missing values by dropping rows or filling all nulls.

        Inputs:
            Uses `self.df` as the active dataframe.

        Args:
            method: Missing-value action. Use `drop` to remove rows with nulls,
                or `fill` to replace nulls with `fill_value`.
            fill_value: Replacement value used only when `method == "fill"`.

        Returns:
            None. The method reassigns `self.df` with the transformed data.

        Raises:
            ValueError: If `method` is not `drop` or `fill`.
            ValueError: If `method` is `fill` and `fill_value` is None.
        """
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
        """Return a dataframe copy with duplicate rows removed.

        Args:
            df: Input dataframe to deduplicate.

        Returns:
            pd.DataFrame: Dataframe without duplicated rows.
        """
        return df.drop_duplicates()

    def remove_cancelled_diverted(self) -> pd.DataFrame:
        """Remove rows flagged as canceled or diverted flights.

        Inputs:
            Expects `CANCELLED` and `DIVERTED` columns in `self.df`.

        Returns:
            pd.DataFrame: Filtered dataframe keeping only completed flights.

        Side effects:
            Updates `self.df` and syncs `self.data`.
        """
        self.df = self.df[(self.df["CANCELLED"] == 0) & (self.df["DIVERTED"] == 0)]
        self.data = self.df
        return self.df

    def remove_data_leak_cols(self) -> None:
        """Drop post-outcome and leakage-prone columns from the dataset.

        Inputs:
            Uses `self.df` and removes predefined leakage columns when present.

        Returns:
            None. The method reassigns `self.df` and updates `self.data`.
        """
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
        """Save the current dataframe to disk as CSV or XLSX.

        Args:
            filename: Output file path ending in `.csv` or `.xlsx`.

        Returns:
            None.

        Raises:
            ValueError: If `filename` extension is not supported.
        """
        if filename.endswith(".csv"):
            self.df.to_csv(filename, index=False)
        elif filename.endswith(".xlsx"):
            self.df.to_excel(filename, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

    def _handle_outliers_and_nans(self) -> None:
        """Treat outliers with IQR and impute numeric missing values with mean.

        Inputs:
            Uses `self.df`. Outlier detection is applied to `DISTANCE` and
            `CRS_ELAPSED_TIME` when those columns exist.

        Processing:
            - Computes IQR bounds per target column.
            - Replaces outlier values with NaN.
            - Imputes resulting NaNs using the column mean.
            - Fills remaining numeric NaNs using per-column means.

        Returns:
            None. Updates `self.df` and `self.data`.
        """
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

    def normalize_arr_delay(self) -> None:
        """Clamp negative `ARR_DELAY` values to zero.

        Inputs:
            Uses `self.df` and expects the `ARR_DELAY` column.

        Returns:
            None. Updates `self.df` and `self.data`.

        Raises:
            ValueError: If `ARR_DELAY` is not present in the dataframe.
        """
        if "ARR_DELAY" not in self.df.columns:
            raise ValueError("Column 'ARR_DELAY' is required to normalize delays")

        negative_count = int((self.df["ARR_DELAY"] < 0).sum())
        if negative_count > 0:
            self.df["ARR_DELAY"] = self.df["ARR_DELAY"].clip(lower=0)
        self.data = self.df
        print(f"   ✓ {negative_count} valores negativos de ARR_DELAY convertidos para 0")

    def balance_delay_dataset(self, random_state: int = 42) -> None:
        """Balance delayed and non-delayed records for ARR_DELAY.

        The resulting dataset includes:
            - All rows where `ARR_DELAY > 0`.
            - The same number of rows sampled from `ARR_DELAY == 0`.

        Inputs:
            Uses normalized `self.df` where negative delays were already mapped
            to zero.

        Args:
            random_state: Seed used for deterministic sampling/shuffling.

        Returns:
            None. Reassigns `self.df` and updates `self.data`.
        """
        positive_df = self.df[self.df["ARR_DELAY"] > 0]
        zero_df = self.df[self.df["ARR_DELAY"] == 0]

        positive_count = len(positive_df)
        zero_count = len(zero_df)

        if positive_count == 0 or zero_count == 0:
            print(
                "   ! Balanceamento ignorado: não há amostras suficientes "
                "em uma das classes (ARR_DELAY > 0 ou ARR_DELAY == 0)"
            )
            self.data = self.df
            return

        sample_with_replacement = zero_count < positive_count
        sampled_zero_df = zero_df.sample(
            n=positive_count,
            replace=sample_with_replacement,
            random_state=random_state,
        )

        self.df = pd.concat([positive_df, sampled_zero_df], axis=0)
        self.df = self.df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        self.data = self.df

        replacement_note = " (com reposição)" if sample_with_replacement else ""
        print(
            f"   ✓ Dataset balanceado: {positive_count} atrasos positivos + "
            f"{positive_count} zeros{replacement_note}"
        )

    def load_and_clean(self, nrows=None, random_state: int = 42):
        """Run the end-to-end cleaning pipeline and return cleaned data.

        Inputs:
            - Uses in-memory `self.df` when provided at initialization.
            - Otherwise loads data from `self.file_path`.

        Args:
            nrows: Optional number of rows to read from CSV.
            random_state: Seed used for deterministic balancing sampling.

        Pipeline steps:
            1. Load dataset (from memory or file).
            2. Remove canceled/diverted flights.
            3. Drop leakage columns.
            4. Drop rows with null `ARR_DELAY`.
            5. Convert negative `ARR_DELAY` values to `0`.
            6. Balance dataset: all positive delays + equal zero delays.
            7. Treat outliers and impute numeric nulls.
            8. Drop redundant identifier/location columns.

        Returns:
            pd.DataFrame: Fully cleaned dataframe.

        Raises:
            ValueError: If no in-memory dataframe is set and `file_path` is missing.
        """
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

        print("\n5. A converter atrasos negativos para zero...")
        self.normalize_arr_delay()

        print("\n6. A balancear dataset (atrasos positivos vs zeros)...")
        self.balance_delay_dataset(random_state=random_state)

        print("\n7. A tratar outliers e missing values...")
        self._handle_outliers_and_nans()
        self.data = self.df

        print("\n8. A remover colunas redundantes...")
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
