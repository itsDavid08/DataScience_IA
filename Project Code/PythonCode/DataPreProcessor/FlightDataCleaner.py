import pandas as pd
import numpy as np


class FlightDataCleaner:
    """
    Classe responsável pelo carregamento e limpeza inicial dos dados dos voos.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_and_clean(self):
        print("A carregar os dados...")
        self.data = pd.read_csv(self.file_path)

        # 1. Remover voos cancelados e desviados [cite: 197]
        self.data = self.data[(self.data['CANCELLED'] == 0) & (self.data['DIVERTED'] == 0)]

        # 2. Remover Data Leakage (colunas proibidas) [cite: 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197]
        leakage_cols = [
            'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
            'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT', 'ARR_TIME', 'DEP_TIME',
            'WHEELS_OFF', 'WHEELS_ON', 'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME',
            'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED'
        ]
        self.data = self.data.drop(columns=leakage_cols, errors='ignore')

        # 3. Remover Nulos na Variável Alvo [cite: 184]
        self.data = self.data.dropna(subset=['ARR_DELAY'])

        self._handle_outliers_and_nans()
        print("Limpeza concluída! Dimensão atual:", self.data.shape)
        return self.data

    def _handle_outliers_and_nans(self):
        """Método privado para tratar outliers usando IQR nas features contínuas."""
        cols_for_outliers = ['DISTANCE', 'CRS_ELAPSED_TIME']
        for col in cols_for_outliers:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Substituir outliers por NaN
                self.data[col] = np.where(
                    (self.data[col] < lower_bound) | (self.data[col] > upper_bound),
                    np.nan,
                    self.data[col]
                )
                # Imputar NaNs com a média
                self.data[col] = self.data[col].fillna(self.data[col].mean())