import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataVisualization:
    """
    Classe para Análise Exploratória de Dados (EDA) e geração de gráficos.
    """

    def __init__(self, data):
        self.data = data

    def plot_histograms(self, columns):
        """
        Gera histogramas para entender a distribuição das variáveis numéricas.
        Útil para ver a distribuição dos atrasos (ARR_DELAY).
        """
        print(f"A gerar histogramas para as colunas: {columns}...")
        num_cols = len(columns)
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 4))

        # Garantir que axes é iterável mesmo com apenas 1 coluna
        if num_cols == 1:
            axes = [axes]

        for i, col in enumerate(columns):
            if col in self.data.columns:
                sns.histplot(self.data[col], bins=30, kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f'Distribuição de {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequência')

        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, columns):
        """
        Gera boxplots. Muito útil para visualizar outliers antes e depois da limpeza.
        """
        print(f"A gerar boxplots para as colunas: {columns}...")
        num_cols = len(columns)
        fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 4))

        if num_cols == 1:
            axes = [axes]

        for i, col in enumerate(columns):
            if col in self.data.columns:
                sns.boxplot(y=self.data[col], ax=axes[i], color='lightgreen')
                axes[i].set_title(f'Boxplot de {col}')

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self):
        """
        Gera uma matriz de correlação das variáveis numéricas contínuas.
        """
        print("A gerar matriz de correlação...")
        plt.figure(figsize=(10, 8))

        # Selecionar apenas colunas numéricas para evitar erros de correlação com strings
        numeric_cols = self.data.select_dtypes(include=[np.number])

        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title('Matriz de Correlação das Variáveis Numéricas')
        plt.show()