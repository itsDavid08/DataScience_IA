import pandas as pd


class DataLoader:
    """
    Classe responsável apenas pelo carregamento e leitura inicial dos dados.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self, nrows=None):
        """
        Carrega o dataset. O parâmetro nrows é útil se quiseres testar o código
        rapidamente com apenas uma amostra (ex: nrows=10000) sem carregar os 3 milhões.
        """
        print(f"A carregar o dataset a partir de: {self.file_path}...")
        self.data = pd.read_csv(self.file_path, nrows=nrows)

        print(f"Dataset carregado com sucesso!")
        print(f"Número de linhas: {self.data.shape[0]}")
        print(f"Número de colunas: {self.data.shape[1]}")

        return self.data

    def get_basic_info(self):
        """Devolve informações básicas sobre os tipos de dados e valores não nulos."""
        if self.data is not None:
            print("\n--- Informação Básica do Dataset ---")
            self.data.info()
        else:
            print("Os dados ainda não foram carregados.")