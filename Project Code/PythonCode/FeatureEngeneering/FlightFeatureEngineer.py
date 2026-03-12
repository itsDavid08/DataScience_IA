import pandas as pd
import numpy as np

class FlightFeatureEngineer:
    """
    Classe para criar novas features a partir dos dados existentes.
    """

    def __init__(self, data):
        self.data = data.copy()

    def generate_features(self):
        print("A gerar 10+ novas features...")

        # Converter FL_DATE para datetime para extrair componentes
        self.data['FL_DATE'] = pd.to_datetime(self.data['FL_DATE'])

        # --- Feature 1 a 4: Temporais ---
        self.data['MONTH'] = self.data['FL_DATE'].dt.month
        self.data['DAY_OF_WEEK'] = self.data['FL_DATE'].dt.dayofweek
        self.data['IS_WEEKEND'] = self.data['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 5 else 0)

        # Extrair a hora de partida planeada (CRS_DEP_TIME está em formato HHMM ou HMM)
        self.data['DEP_HOUR'] = self.data['CRS_DEP_TIME'] // 100

        # --- Feature 5 a 6: Rotas e Velocidade ---
        self.data['ROUTE'] = self.data['ORIGIN'] + "_" + self.data['DEST']
        # Velocidade planeada (Milhas por minuto)
        self.data['PLANNED_SPEED_MPM'] = self.data['DISTANCE'] / self.data['CRS_ELAPSED_TIME']

        # --- Feature 7 a 9: Categorias e Bins ---
        # Categorizar a distância (Curto, Médio, Longo Curso)
        self.data['DISTANCE_CAT'] = pd.cut(self.data['DISTANCE'], bins=[0, 500, 1500, 10000],
                                           labels=['Short', 'Medium', 'Long'])
        self.data['IS_HOLIDAY_SEASON'] = self.data['MONTH'].apply(lambda x: 1 if x in [7, 8, 12] else 0)
        self.data['MORNING_FLIGHT'] = self.data['DEP_HOUR'].apply(lambda x: 1 if 6 <= x <= 11 else 0)

        # --- Feature 10: Criação da Variável de Classificação (Obrigatório no Guião) ---
        # Classes: On-time (< 15), Short delay (15-30), Long delay (> 30)
        conditions = [
            (self.data['ARR_DELAY'] < 15),
            (self.data['ARR_DELAY'] >= 15) & (self.data['ARR_DELAY'] <= 30),
            (self.data['ARR_DELAY'] > 30)
        ]
        choices = ['On-time', 'Short delay', 'Long delay']
        self.data['DELAY_CLASS'] = np.select(conditions, choices, default='Unknown')

        print("Feature Engineering concluída!")
        return self.data