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

        self.data = self.df

    def fill_missing(
        self,
        strategy: str = "mean",
        value: Union[int, float, str, None] = None,
    ) -> None:
        """Fill missing values using a selected imputation strategy.

        Args:
            strategy: Imputation mode — 'mean', 'median', 'mode', or 'constant'.
            value: Constant replacement value required when strategy == 'constant'.

        Raises:
            ValueError: If strategy is not supported.
            ValueError: If strategy is 'constant' and value is None.
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

        Args:
            method: 'drop' to remove rows with nulls, or 'fill' to replace nulls.
            fill_value: Replacement value used only when method == 'fill'.

        Raises:
            ValueError: If method is not 'drop' or 'fill'.
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
        """Return a dataframe copy with duplicate rows removed."""
        return df.drop_duplicates()

    def remove_cancelled_diverted(self) -> pd.DataFrame:
        """Remove rows flagged as canceled or diverted flights."""
        self.df = self.df[(self.df["CANCELLED"] == 0) & (self.df["DIVERTED"] == 0)]
        self.data = self.df
        return self.df

    def remove_data_leak_cols(self) -> pd.DataFrame:
        """Drop post-outcome and leakage-prone columns from the dataset.

        These columns are forbidden as predictors because they either encode the
        target variable, are derived from it, or contain post-event information
        (i.e., information only available AFTER the flight lands). Using them
        would cause data leakage and invalidate any predictive model.

        Columns removed and rationale:
            - DEP_DELAY      : directly correlated with ARR_DELAY (post-departure info).
            - DELAY_DUE_*    : cause-of-delay fields only populated when delay >= 15 min
                               (derived from the target, known only post-event).
            - ARR_TIME       : actual arrival time — only known after landing.
            - DEP_TIME       : actual departure time — known after departure.
            - WHEELS_OFF/ON  : recorded during the flight.
            - TAXI_OUT/IN    : recorded during the flight.
            - ELAPSED_TIME   : actual elapsed time — known after landing.
            - AIR_TIME       : known only after landing.
            - CANCELLED      : already filtered out in step 2.
            - DIVERTED       : already filtered out in step 2.

        Returns:
            pd.DataFrame: Dataset without leakage columns.
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
        removed = [c for c in leakage_cols if c in self.df.columns]
        self.df = self.df.drop(columns=leakage_cols, errors="ignore")
        self.data = self.df
        return self.df

    def remove_cancel_diverted(self):
        cols = ["CANCELLED", "CANCELLATION_CODE", "DIVERTED"]
        self.df = self.df.drop(columns=cols, errors="ignore")
        self.data = self.df

    def _handle_outliers_and_nans(self) -> None:
        """Treat outliers with IQR and impute numeric missing values with mean."""
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
        """Clamp negative ARR_DELAY values to zero.

        Negative values indicate early arrivals. For the purpose of delay
        prediction, we treat early arrivals as zero delay (on-time).

        Raises:
            ValueError: If ARR_DELAY is not present in the dataframe.
        """
        if "ARR_DELAY" not in self.df.columns:
            raise ValueError("Column 'ARR_DELAY' is required to normalize delays")

        negative_count = int((self.df["ARR_DELAY"] < 0).sum())
        if negative_count > 0:
            self.df["ARR_DELAY"] = self.df["ARR_DELAY"].clip(lower=0)
        self.data = self.df
        print(f"   ✓ {negative_count} valores negativos de ARR_DELAY convertidos para 0")

    def balance_delay_dataset(self, method: str = "smote", random_state: int = 42) -> None:
        """Balance the dataset for ARR_DELAY using SMOTE or undersampling.

        Three strategies are supported:
            - 'smote'       : Synthetic Minority Over-sampling Technique (preferred).
                              Generates synthetic samples for the minority class,
                              preserving information from the majority class.
            - 'oversample'  : Random oversampling (duplicates minority samples).
            - 'undersample' : Random undersampling (reduces majority class).

        SMOTE is the preferred method as it avoids information loss (undersampling)
        and overfitting from simple duplication (random oversampling).

        Requires imblearn: pip install imbalanced-learn

        Args:
            method: Balancing strategy — 'smote', 'oversample', or 'undersample'.
            random_state: Seed for reproducibility.
        """
        # Build binary target for balancing: delayed (>0) vs on-time (==0)
        y_binary = (self.df["ARR_DELAY"] > 0).astype(int)
        class_counts = y_binary.value_counts()
        print(f"   Distribuição antes do balanceamento: {dict(class_counts)}")

        if class_counts.min() == 0:
            print("   ! Balanceamento ignorado: uma das classes está vazia.")
            return

        # Attempt imblearn-based balancing; fall back to manual undersampling.
        try:
            from imblearn.over_sampling import SMOTE, RandomOverSampler
            from imblearn.under_sampling import RandomUnderSampler

            # Use only numeric columns for SMOTE (it cannot handle categoricals).
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = self.df[numeric_cols]

            if method == "smote":
                sampler = SMOTE(random_state=random_state)
                strategy_label = "SMOTE (oversampling sintético)"
            elif method == "oversample":
                sampler = RandomOverSampler(random_state=random_state)
                strategy_label = "RandomOverSampler"
            else:
                sampler = RandomUnderSampler(random_state=random_state)
                strategy_label = "RandomUnderSampler"

            X_res, y_res = sampler.fit_resample(X_numeric, y_binary)

            # Rebuild DataFrame from resampled indices (for non-numeric cols, use idx).
            if hasattr(sampler, "sample_indices_"):
                idx = sampler.sample_indices_
                self.df = self.df.iloc[idx].reset_index(drop=True)
            else:
                # SMOTE creates synthetic rows — rebuild numeric cols only.
                self.df = pd.DataFrame(X_res, columns=numeric_cols)

            print(f"   ✓ Dataset balanceado com {strategy_label}")
            print(f"   Distribuição após balanceamento: {dict(pd.Series(y_res).value_counts())}")

        except ImportError:
            print("   ! imblearn não instalado. A usar undersampling manual como fallback.")
            print("     Instala com: pip install imbalanced-learn")
            self._manual_undersample(random_state)

        self.data = self.df

    def _manual_undersample(self, random_state: int = 42) -> None:
        """Fallback undersampling: keep all delayed rows, sample equal non-delayed."""
        positive_df = self.df[self.df["ARR_DELAY"] > 0]
        zero_df = self.df[self.df["ARR_DELAY"] == 0]

        positive_count = len(positive_df)
        zero_count = len(zero_df)

        if positive_count == 0 or zero_count == 0:
            print("   ! Balanceamento ignorado: classes insuficientes.")
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
        print(f"   ✓ Dataset balanceado (manual): {positive_count} delayed + {positive_count} on-time")

    def classify_target(self):
        """Add DELAY_CLASS column with three classes as per project specification:
            - 'On-time'     : ARR_DELAY < 15 minutes
            - 'Short delay' : 15 <= ARR_DELAY <= 30 minutes
            - 'Long delay'  : ARR_DELAY > 30 minutes
        """
        conditions = [
            (self.df['ARR_DELAY'] < 15),
            (self.df['ARR_DELAY'] >= 15) & (self.df['ARR_DELAY'] <= 30),
            (self.df['ARR_DELAY'] > 30)
        ]
        choices = ['On-time', 'Short delay', 'Long delay']
        self.df['DELAY_CLASS'] = np.select(conditions, choices, default='Unknown')
        return self.df

    def load_and_clean(self, nrows=None, random_state: int = 42, balance_method: str = "smote"):
        """Run the end-to-end cleaning pipeline and return cleaned data.

        Pipeline steps:
            1. Load dataset (from memory or file).
            2. Remove canceled/diverted flights.
            3. Drop leakage columns (data leakage prevention).
            4. Drop rows with null ARR_DELAY.
            5. Convert negative ARR_DELAY values to 0.
            6. Balance dataset using SMOTE or fallback undersampling.
            7. Treat outliers and impute numeric nulls.
            8. Drop redundant identifier/location columns.

        Args:
            nrows: Optional number of rows to read from CSV.
            random_state: Seed for reproducibility.
            balance_method: Balancing strategy — 'smote', 'oversample', or 'undersample'.

        Returns:
            pd.DataFrame: Fully cleaned dataframe.
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

        # FIX: Data leakage removal is now ENABLED (was previously commented out).
        # Columns like DEP_DELAY, ARR_TIME, TAXI_OUT, etc. are removed here to
        # prevent the model from learning from information unavailable at flight time.
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

        print(f"\n6. A balancear dataset (método: {balance_method})...")
        self.balance_delay_dataset(method=balance_method, random_state=random_state)

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