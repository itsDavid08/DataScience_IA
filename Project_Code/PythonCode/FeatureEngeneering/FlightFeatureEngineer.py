import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FlightFeatureEngineer:
    """
    Classe para criar e processar features dos dados de voos.
    Gera 12+ features novas e normaliza dados para Part 2 (modelação).
    """

    def __init__(self, data):
        self.data = data.copy()
        self.scaler = StandardScaler()
        self.encoders = {}

    def generate_features(self):
        """
        Gera 12+ novas features a partir dos dados existentes.
        Inclui: temporal, rota, características, interações.

        Returns:
            pd.DataFrame: Dataset com novas features.
        """
        print("=" * 60)
        print("INICIANDO FEATURE ENGINEERING")
        print("=" * 60)

        # Converter FL_DATE para datetime
        print("\n1. Processando features temporais...")
        self.data['FL_DATE'] = pd.to_datetime(self.data['FL_DATE'])

        # Feature 1-4: Temporal features
        self.data['MONTH'] = self.data['FL_DATE'].dt.month
        self.data['DAY_OF_WEEK'] = self.data['FL_DATE'].dt.dayofweek
        self.data['DAY_OF_YEAR'] = self.data['FL_DATE'].dt.dayofyear
        self.data['QUARTER'] = self.data['FL_DATE'].dt.quarter

        # Feature 5-7: Indicadores temporais
        self.data['IS_WEEKEND'] = (self.data['DAY_OF_WEEK'] >= 5).astype(int)
        self.data['IS_HOLIDAY_SEASON'] = self.data['MONTH'].apply(lambda x: 1 if x in [7, 8, 12] else 0)

        # Feature 8: Hora de partida
        self.data['DEP_HOUR'] = (self.data['CRS_DEP_TIME'] // 100).astype(int)
        self.data['MORNING_FLIGHT'] = self.data['DEP_HOUR'].apply(lambda x: 1 if 6 <= x <= 11 else 0)
        self.data['AFTERNOON_FLIGHT'] = self.data['DEP_HOUR'].apply(lambda x: 1 if 12 <= x <= 17 else 0)
        self.data['NIGHT_FLIGHT'] = self.data['DEP_HOUR'].apply(lambda x: 1 if x >= 18 or x < 6 else 0)

        print(f"   ✓ {10} features temporais criadas")

        # Feature 11-13: Rota e velocidade
        print("\n2. Processando features de rota...")
        self.data['ROUTE'] = self.data['ORIGIN'] + "_" + self.data['DEST']
        self.data['ROUTE_FREQUENCY'] = self.data['ROUTE'].map(self.data['ROUTE'].value_counts())
        self.data['PLANNED_SPEED_MPM'] = self.data['DISTANCE'] / self.data['CRS_ELAPSED_TIME']

        print(f"   ✓ 3 features de rota criadas")

        # Feature 14-16: Categorização de características
        print("\n3. Processando features de categorização...")
        self.data['DISTANCE_CAT'] = pd.cut(self.data['DISTANCE'],
                                           bins=[0, 500, 1500, 10000],
                                           labels=['Short', 'Medium', 'Long'])

        self.data['DURATION_CAT'] = pd.cut(self.data['CRS_ELAPSED_TIME'],
                                          bins=[0, 120, 360, 1200],
                                          labels=['Short', 'Medium', 'Long'])

        self.data['SPEED_CAT'] = pd.cut(self.data['PLANNED_SPEED_MPM'],
                                       bins=[0, 5, 8, 15],
                                       labels=['Slow', 'Normal', 'Fast'])

        print(f"   ✓ 3 features de categorização criadas")

        # Feature 17-19: Interações e polinômios baixo grau
        print("\n4. Processando features de interação...")
        self.data['DISTANCE_x_ELAPSED_TIME'] = self.data['DISTANCE'] * self.data['CRS_ELAPSED_TIME']
        self.data['DISTANCE_POW2'] = self.data['DISTANCE'] ** 2
        self.data['ELAPSED_TIME_POW2'] = self.data['CRS_ELAPSED_TIME'] ** 2

        print(f"   ✓ 3 features de interação criadas")

        # Feature 20: Variável de classificação (OBRIGATÓRIA no guião)
        print("\n5. Criando variável de classificação (DELAY_CLASS)...")
        conditions = [
            (self.data['ARR_DELAY'] < 15),
            (self.data['ARR_DELAY'] >= 15) & (self.data['ARR_DELAY'] <= 30),
            (self.data['ARR_DELAY'] > 30)
        ]
        choices = ['On-time', 'Short delay', 'Long delay']
        self.data['DELAY_CLASS'] = np.select(conditions, choices, default='Unknown')

        print(f"   ✓ Variável DELAY_CLASS criada (3 classes)")

        # Remover FL_DATE (não é feature, é apenas usada para extrair features temporais)
        self.data = self.data.drop(columns=['FL_DATE'], errors='ignore')

        print("\n" + "=" * 60)
        print(f"FEATURE ENGINEERING CONCLUÍDO!")
        print(f"Total de features novas: 20")
        print(f"Dimensão do dataset: {self.data.shape[0]} linhas × {self.data.shape[1]} colunas")
        print("=" * 60 + "\n")

        return self.data

    def encode_categorical(self):
        """
        Codifica variáveis categóricas (DISTANCE_CAT, DURATION_CAT, SPEED_CAT, DELAY_CLASS).
        """
        print("Codificando variáveis categóricas...")

        categorical_cols = ['DISTANCE_CAT', 'DURATION_CAT', 'SPEED_CAT', 'DELAY_CLASS', 'ROUTE']

        for col in categorical_cols:
            if col in self.data.columns:
                le = LabelEncoder()
                series = self.data[col]
                non_null_mask = series.notna()

                if non_null_mask.any():
                    encoded_col = pd.Series(pd.NA, index=series.index, dtype="Int64")
                    encoded_col.loc[non_null_mask] = le.fit_transform(series.loc[non_null_mask].astype(str))
                    self.data[col] = encoded_col

                    self.encoders[col] = le
                    print(f"   ✓ {col} codificada")

        return self.data

    def normalize_features(self, exclude_cols=None):
        """
        Normaliza features numéricas usando StandardScaler (para Part 2).
        Exclui a variável alvo (ARR_DELAY) e DELAY_CLASS (são targets).

        Args:
            exclude_cols (list): Colunas a não normalizar.

        Returns:
            pd.DataFrame: Dataset normalizado.
        """
        if exclude_cols is None:
            exclude_cols = ['ARR_DELAY', 'DELAY_CLASS']

        print("\nNormalizando features numéricas (StandardScaler)...")

        # Selecionar colunas numéricas que não estão em exclude_cols
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        # Fit e transform
        self.data[cols_to_normalize] = self.scaler.fit_transform(self.data[cols_to_normalize])

        print(f"   ✓ {len(cols_to_normalize)} features normalizadas")

        return self.data

    def get_feature_summary(self):
        """
        Retorna um resumo das features criadas.
        """
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
