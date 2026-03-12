import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler


class FlightEDA:
    """
    Classe para Análise Exploratória e Redução de Dimensionalidade.
    """

    def __init__(self, data):
        self.data = data

    def plot_correlation_matrix(self):
        plt.figure(figsize=(10, 8))
        # Selecionar apenas colunas numéricas
        numeric_cols = self.data.select_dtypes(include=[np.number])
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matriz de Correlação das Variáveis Numéricas')
        plt.show()

    def run_pca(self, n_components=2):
        print("A executar PCA...")
        # Filtrar features numéricas e remover Nulos caso existam
        numeric_data = self.data.select_dtypes(include=[np.number]).dropna()

        # É obrigatório normalizar antes do PCA
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(numeric_data)

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_scaled)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title('Projeção PCA (2 Componentes)')
        plt.xlabel(f'Componente 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
        plt.ylabel(f'Componente 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
        plt.grid(True)
        plt.show()