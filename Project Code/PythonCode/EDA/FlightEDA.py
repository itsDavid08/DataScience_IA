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

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return len(self.df)

    def __getitem__(self, column: str) -> pd.Series:
        """
        Returns a column from the dataset.

        Parameters:
            column (str): The column name to retrieve.

        Raises:
            KeyError: If the column does not exist in the dataset.

        Returns:
            pd.Series: The data for the specified column.
        """
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in dataset")
        return self.df[column]

    def __setitem__(self, column: str, value) -> None:
        """
        Sets a column in the dataset.

        Parameters:
            column (str): The column name.
            value: The data to assign to the column.
        """
        self.df[column] = value

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        return f"DatasetAnalyzer({len(self.df)} rows, {len(self.df.columns)} columns)"

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Returns the first n rows of the dataset.

        Parameters:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            pd.DataFrame: The first n rows of the dataset.
        """
        return self.df.head(n)


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

        def assess_outliers(self, method: str = 'zscore', threshold: float = 3.0) -> Dict[str, int]:
            """
            Detects outliers in numerical columns using either the Z-score or IQR method.

            Parameters:
                method (str): The method to use for outlier detection ('zscore' or 'iqr').
                threshold (float): The threshold to define an outlier.

            Returns:
                dict: A dictionary with numerical column names as keys and outlier counts as values.

            Raises:
                ValueError: If the method is not 'zscore' or 'iqr'.
            """
            outliers = {}
            numeric_cols = self.df.select_dtypes(include=[np.number])
            for col in numeric_cols:
                if method == 'zscore':
                    std = self.df[col].std()
                    # Avoid division by zero in case of constant columns.
                    if std == 0:
                        outliers[col] = 0
                        continue
                    z_scores = np.abs((self.df[col] - self.df[col].mean()) / std)
                    outliers[col] = (z_scores > threshold).sum()
                elif method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers[col] = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                else:
                    raise ValueError("Method must be 'zscore' or 'iqr'")
            return outliers

        def get_dtype_info(self) -> pd.Series:
            """
            Returns the data types for all columns in the dataset.

            Returns:
                pd.Series: A series with column names as index and data types as values.
            """
            return self.df.dtypes

        def get_summary(self) -> pd.DataFrame:
            """
            Returns summary statistics for the dataset.

            Returns:
                pd.DataFrame: A table of summary statistics including count, mean, std, min, and percentiles.
            """
            return self.df.describe(include='all')
