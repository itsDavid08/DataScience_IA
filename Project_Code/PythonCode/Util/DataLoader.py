import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Classe responsável pelo carregamento inicial e split train/test dos dados dos voos.
    Responsabilidade única: Carregar CSV e dividir em conjuntos de treino/teste.
    Sem lógica de limpeza (delegada para FlightDataCleaner_IA).
    """

    def __init__(self, file_path, test_size=0.2, random_state=42):
        """
        Inicializa o DataLoader_IA com caminho do arquivo, tamanho do teste e seed aleatória.

        Args:
            file_path (str): Caminho para o arquivo CSV.
            test_size (float): Proporção dos dados para teste (padrão: 0.2).
            random_state (int): Seed para reprodutibilidade (padrão: 42).
        """
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.data_train = None
        self.data_test = None
        self.target_train = None
        self.target_test = None

        # Kaggle dataset source used when local CSV is missing.
        self._kaggle_dataset = "patrickzel/flight-delay-and-cancellation-dataset-2019-2023"

    def _resolve_dataset_path(self):
        """Resolve dataset path relative to project root when needed."""
        dataset_path = Path(self.file_path)
        if dataset_path.is_absolute():
            return dataset_path

        project_root = Path(__file__).resolve().parents[3]
        return project_root / dataset_path

    def _download_dataset_if_missing(self, dataset_path):
        """Download and persist the dataset if the target CSV does not exist locally."""
        if dataset_path.exists():
            return

        print(f"Dataset not found at: {dataset_path}")
        print("Attempting to download dataset from KaggleHub...")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'kagglehub'. Install it with: pip install kagglehub[pandas-datasets]"
            ) from exc

        try:
            # Empty file_path lets KaggleHub resolve the default dataset file.
            downloaded_df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                self._kaggle_dataset,
                "",
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to download dataset from KaggleHub. "
                "Make sure Kaggle access is configured and internet is available."
            ) from exc

        if downloaded_df is None or downloaded_df.empty:
            raise RuntimeError("KaggleHub returned an empty dataset; CSV was not created.")

        downloaded_df.to_csv(dataset_path, index=False)
        print(f"Dataset downloaded and saved to: {dataset_path}")

    def load_data(self, nrows=None, target_column='ARR_DELAY'):
        """
        Carrega o dataset e realiza split train/test aleatório.

        Args:
            nrows (int): Número máximo de linhas a carregar (opcional, para testes).
            target_column (str): Nome da coluna alvo (padrão: 'ARR_DELAY').

        Returns:
            tuple: (data_train, data_test, target_train, target_test)
        """
        dataset_path = self._resolve_dataset_path()
        self._download_dataset_if_missing(dataset_path)

        # Keep canonical path after resolution/download.
        self.file_path = str(dataset_path)

        print(f"A carregar o dataset a partir de: {self.file_path}...")
        self.data = pd.read_csv(dataset_path, nrows=nrows)

        print(f"Dataset carregado com sucesso!")
        print(f"Dimensão original: {self.data.shape[0]} linhas × {self.data.shape[1]} colunas")

        # Separar features e alvo
        X = self.data.drop(columns=[target_column], errors='ignore')
        y = self.data[target_column] if target_column in self.data.columns else None

        # Split train/test aleatório (80/20)
        if y is not None:
            self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            print(f"Split realizado: {self.data_train.shape[0]} treino × {self.data_test.shape[0]} teste")
        else:
            print(f"Aviso: Coluna alvo '{target_column}' não encontrada.")
            self.data_train, self.data_test = train_test_split(
                X, test_size=self.test_size, random_state=self.random_state
            )
            self.target_train = None
            self.target_test = None

        return self.data_train, self.data_test, self.target_train, self.target_test

    def get_basic_info(self):
        """Devolve informações básicas sobre os tipos de dados e valores não nulos."""
        if self.data is not None:
            print("\n--- Informação Básica do Dataset ---")
            self.data.info()
            print("\n--- Estatísticas Descritivas ---")
            print(self.data.describe())
        else:
            print("Os dados ainda não foram carregados.")

    @staticmethod
    def _resolve_checkpoint_path(filepath):
        """Resolve checkpoint path relative to project root when needed."""
        checkpoint_path = Path(filepath)
        if checkpoint_path.is_absolute():
            return checkpoint_path

        project_root = Path(__file__).resolve().parents[3]
        return project_root / checkpoint_path

    def save_checkpoint(self, filepath):
        """Serializa o DataLoader_IA para checkpoint (salva estado atual)."""
        checkpoint_path = self._resolve_checkpoint_path(filepath)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Checkpoint salvo: {checkpoint_path}")

    @staticmethod
    def load_checkpoint(filepath):
        """Carrega um DataLoader_IA previamente serializado."""
        checkpoint_path = DataLoader._resolve_checkpoint_path(filepath)

        with open(checkpoint_path, 'rb') as f:
            loader = pickle.load(f)
        print(f"Checkpoint carregado: {checkpoint_path}")
        return loader
