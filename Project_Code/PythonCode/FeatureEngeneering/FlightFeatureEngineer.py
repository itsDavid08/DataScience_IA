import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


class FlightFeatureEngineer:
    """Create and process features from flight data.

    Generates 20+ new features and supports two encoding strategies:
    - Label Encoding for ordinal/binary features.
    - One-Hot Encoding for nominal categorical features (AIRLINE, ORIGIN, DEST).

    Also supports two scaling strategies:
    - StandardScaler (z-score): recommended for models sensitive to scale
      (linear regression, SVM, neural networks).
    - MinMaxScaler (0-1 range): useful when bounded input range is required
      (kNN, neural networks with sigmoid activation).
    """

    def __init__(self, data):
        self.data = data.copy()
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        self.encoders = {}
        self._ohe_feature_names = []

    def generate_features(self):
        """Generate 20+ new features from existing columns.

        Feature groups created:
            - Temporal (month, day of week, hour, quarter, flags)
            - Route (route string, frequency, planned speed)
            - Categorization bins (distance, duration, speed)
            - Interaction and polynomial features

        Returns:
            pd.DataFrame: Dataset with new features appended.
        """
        print("=" * 60)
        print("INICIANDO FEATURE ENGINEERING")
        print("=" * 60)

        print("\n1. Processando features temporais...")
        self.data['FL_DATE'] = pd.to_datetime(self.data['FL_DATE'])
        self.data['MONTH'] = self.data['FL_DATE'].dt.month
        self.data['DAY_OF_WEEK'] = self.data['FL_DATE'].dt.dayofweek
        self.data['DAY_OF_YEAR'] = self.data['FL_DATE'].dt.dayofyear
        self.data['QUARTER'] = self.data['FL_DATE'].dt.quarter
        self.data['IS_WEEKEND'] = (self.data['DAY_OF_WEEK'] >= 5).astype(int)
        self.data['IS_HOLIDAY_SEASON'] = self.data['MONTH'].apply(lambda x: 1 if x in [7, 8, 12] else 0)
        self.data['DEP_HOUR'] = (self.data['CRS_DEP_TIME'] // 100).astype(int)
        self.data['MORNING_FLIGHT'] = self.data['DEP_HOUR'].apply(lambda x: 1 if 6 <= x <= 11 else 0)
        self.data['AFTERNOON_FLIGHT'] = self.data['DEP_HOUR'].apply(lambda x: 1 if 12 <= x <= 17 else 0)
        self.data['NIGHT_FLIGHT'] = self.data['DEP_HOUR'].apply(lambda x: 1 if x >= 18 or x < 6 else 0)

        conditions = [
            (self.data['MORNING_FLIGHT'] == 1),
            (self.data['AFTERNOON_FLIGHT'] == 1),
            (self.data['NIGHT_FLIGHT'] == 1),
        ]
        choices = ['Morning', 'Afternoon', 'Night']
        self.data['TIME_PERIOD'] = np.select(conditions, choices, default='Other')
        print("   ✓ 10 features temporais criadas")

        print("\n2. Processando features de rota...")
        self.data['ROUTE'] = self.data['ORIGIN'] + "_" + self.data['DEST']
        self.data['ROUTE_FREQUENCY'] = self.data['ROUTE'].map(self.data['ROUTE'].value_counts())
        self.data['PLANNED_SPEED_MPM'] = self.data['DISTANCE'] / self.data['CRS_ELAPSED_TIME']
        print("   ✓ 3 features de rota criadas")

        print("\n3. Processando features de categorização...")
        self.data['DISTANCE_CAT'] = pd.cut(
            self.data['DISTANCE'],
            bins=[0, 500, 1500, 10000],
            labels=['Short', 'Medium', 'Long'],
        )
        self.data['DURATION_CAT'] = pd.cut(
            self.data['CRS_ELAPSED_TIME'],
            bins=[0, 120, 360, 1200],
            labels=['Short', 'Medium', 'Long'],
        )
        self.data['SPEED_CAT'] = pd.cut(
            self.data['PLANNED_SPEED_MPM'],
            bins=[0, 5, 8, 15],
            labels=['Slow', 'Normal', 'Fast'],
        )
        print("   ✓ 3 features de categorização criadas")

        print("\n4. Processando features de interação...")
        self.data['DISTANCE_x_ELAPSED_TIME'] = self.data['DISTANCE'] * self.data['CRS_ELAPSED_TIME']
        self.data['DISTANCE_POW2'] = self.data['DISTANCE'] ** 2
        self.data['ELAPSED_TIME_POW2'] = self.data['CRS_ELAPSED_TIME'] ** 2
        print("   ✓ 3 features de interação criadas")

        self.data = self.data.drop(columns=['FL_DATE'], errors='ignore')

        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING CONCLUÍDO!")
        print(f"Total de features novas: 20+")
        print(f"Dimensão do dataset: {self.data.shape[0]} linhas × {self.data.shape[1]} colunas")
        print("=" * 60 + "\n")

        return self.data

    def encode_categorical(self, ohe_cols=None, label_cols=None):
        """Encode categorical variables using the appropriate strategy.

        Encoding strategy rationale:
            - One-Hot Encoding (OHE): used for NOMINAL variables — categories with
              no natural order (AIRLINE, ORIGIN, DEST, TIME_PERIOD). Label encoding
              these would imply a false ordinal relationship (e.g., Delta=1 < United=2).
            - Label Encoding: used for ORDINAL variables — categories with a natural
              order (DISTANCE_CAT: Short < Medium < Long; DURATION_CAT; SPEED_CAT)
              and for high-cardinality variables like ROUTE where OHE would create
              thousands of columns.
            - DELAY_CLASS is label-encoded as it is the target variable.

        Args:
            ohe_cols: Columns to one-hot encode. Defaults to nominal columns.
            label_cols: Columns to label encode. Defaults to ordinal/high-card columns.

        Returns:
            pd.DataFrame: Dataset with encoded categorical variables.
        """
        print("Codificando variáveis categóricas...")

        # Default nominal columns → One-Hot Encoding
        if ohe_cols is None:
            ohe_cols = []
            for candidate in ['AIRLINE', 'AIRLINE_CODE', 'TIME_PERIOD']:
                if candidate in self.data.columns:
                    ohe_cols.append(candidate)

        # Default ordinal / high-cardinality columns → Label Encoding
        if label_cols is None:
            label_cols = []
            for candidate in ['DISTANCE_CAT', 'DURATION_CAT', 'SPEED_CAT', 'DELAY_CLASS', 'ROUTE']:
                if candidate in self.data.columns:
                    label_cols.append(candidate)

        # --- One-Hot Encoding ---
        if ohe_cols:
            print(f"\n   [One-Hot Encoding] colunas nominais: {ohe_cols}")
            print("   Justificação: variáveis nominais sem ordem natural — OHE evita")
            print("   que o modelo assuma relações ordinais falsas entre categorias.")
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.int8)
            ohe_array = ohe.fit_transform(self.data[ohe_cols].astype(str))
            ohe_feature_names = ohe.get_feature_names_out(ohe_cols)
            self._ohe_feature_names = list(ohe_feature_names)
            ohe_df = pd.DataFrame(ohe_array, columns=ohe_feature_names, index=self.data.index)

            self.data = self.data.drop(columns=ohe_cols)
            self.data = pd.concat([self.data, ohe_df], axis=1)
            self.encoders['ohe'] = ohe
            print(f"   ✓ {len(ohe_feature_names)} colunas OHE criadas a partir de {len(ohe_cols)} variáveis")

        # --- Label Encoding ---
        if label_cols:
            print(f"\n   [Label Encoding] colunas ordinais/alta cardinalidade: {label_cols}")
            print("   Justificação: variáveis com ordem natural (Short<Medium<Long) ou")
            print("   alta cardinalidade (ROUTE) onde OHE criaria demasiadas colunas.")
            for col in label_cols:
                le = LabelEncoder()
                series = self.data[col]
                non_null_mask = series.notna()
                if non_null_mask.any():
                    encoded_col = pd.Series(pd.NA, index=series.index, dtype="Int64")
                    encoded_col.loc[non_null_mask] = le.fit_transform(
                        series.loc[non_null_mask].astype(str)
                    )
                    self.data[col] = encoded_col
                    self.encoders[col] = le
                    print(f"   ✓ {col} codificada ({len(le.classes_)} classes)")

        return self.data

    def normalize_features(self, method: str = "standard", exclude_cols=None):
        """Normalize numeric features using the chosen scaling strategy.

        Scaling strategy rationale:
            - StandardScaler (z-score, mean=0, std=1): best for algorithms that
              assume normally distributed inputs or compute distances using absolute
              magnitudes (Linear Regression, SVM, Logistic Regression, Neural Networks,
              PCA). Does not bound the output range.
            - MinMaxScaler (0–1 range): best for algorithms sensitive to bounded
              input range (kNN, Neural Networks with sigmoid/tanh activation). More
              sensitive to outliers than StandardScaler.

        Args:
            method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler,
                    or 'both' to apply StandardScaler and return both versions.
            exclude_cols: Columns to exclude from scaling (defaults to targets).

        Returns:
            pd.DataFrame: Dataset with scaled numeric features (using chosen method).
        """
        if exclude_cols is None:
            exclude_cols = ['ARR_DELAY', 'DELAY_CLASS']

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]

        if method == "standard":
            print(f"\nNormalizando com StandardScaler (z-score) — {len(cols_to_scale)} features...")
            print("   Justificação: z-score é robusto para modelos lineares e PCA,")
            print("   não impõe limite ao intervalo de saída.")
            self.data[cols_to_scale] = self.scaler_standard.fit_transform(self.data[cols_to_scale])
            print(f"   ✓ {len(cols_to_scale)} features normalizadas (StandardScaler)")

        elif method == "minmax":
            print(f"\nNormalizando com MinMaxScaler (0–1) — {len(cols_to_scale)} features...")
            print("   Justificação: intervalo [0,1] necessário para kNN e redes neuronais")
            print("   com funções de activação limitadas (sigmoid, tanh).")
            self.data[cols_to_scale] = self.scaler_minmax.fit_transform(self.data[cols_to_scale])
            print(f"   ✓ {len(cols_to_scale)} features normalizadas (MinMaxScaler)")

        elif method == "both":
            print(f"\nA gerar versões com StandardScaler e MinMaxScaler ({len(cols_to_scale)} features)...")
            df_standard = self.data.copy()
            df_standard[cols_to_scale] = self.scaler_standard.fit_transform(df_standard[cols_to_scale])

            df_minmax = self.data.copy()
            df_minmax[cols_to_scale] = self.scaler_minmax.fit_transform(df_minmax[cols_to_scale])

            # Apply StandardScaler to self.data as default for pipeline continuity
            self.data[cols_to_scale] = df_standard[cols_to_scale]
            print("   ✓ Ambas as versões criadas. StandardScaler aplicado ao dataset principal.")
            print("   ✓ Use scaler_standard e scaler_minmax para transformar novos dados.")
            return self.data

        else:
            raise ValueError("method must be 'standard', 'minmax', or 'both'")

        return self.data

    def get_feature_summary(self):
        """Return a summary of the engineered features."""
        print("\n" + "=" * 60)
        print("RESUMO DAS FEATURES")
        print("=" * 60)
        print(f"Total de features: {self.data.shape[1]}")
        print(f"Total de amostras: {self.data.shape[0]}")
        print(f"\nTipos de dados:")
        print(self.data.dtypes)
        print(f"\nMissing values:")
        print(self.data.isnull().sum())
        print("=" * 60 + "\n")
        return self.data.info()