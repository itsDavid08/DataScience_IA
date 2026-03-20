# %% Analyze all steps of PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PCAAnalysis:
    """
    Perform Principal Component Analysis (PCA) on a dataset using a developed method and a standard library.

    Attributes:
        X (numpy.ndarray): Input data array.
        y (numpy.ndarray): Target labels.
        num_components (int): Number of principal components.
        X_standardized (numpy.ndarray): Standardized input data.
        cov_matrix (numpy.ndarray): Covariance matrix of standardized data.
        eigenvalues (numpy.ndarray): Eigenvalues of covariance matrix.
        eigenvectors (numpy.ndarray): Eigenvectors of covariance matrix.
        pca_projection (numpy.ndarray): Projection of data onto principal components (manual PCA).
        sklearn_pca_projection (numpy.ndarray): Projection of data onto principal components (library PCA).
        sklearn_pca (sklearn.decomposition.PCA): Trained sklearn PCA object.
    """

    def __init__(self, data, targets, num_components):
        """
        Initialize PCAAnalysis with input data, targets, and number of principal components.

        Args:
            data (numpy.ndarray): Input data.
            targets (numpy.ndarray): Target labels.
            num_components (int): Number of principal components.

        Raises:
            ValueError: If the number of components exceeds the number of features.
        """
        if num_components > data.shape[1]:
            raise ValueError("Number of components cannot exceed the number of features.")

        # Dataset
        self.X = data
        self.y = targets

        # Configuration
        self.num_components = num_components

        # Plots
        self.fig, self.axes = None, None

        # Implemented PCA
        self.X_standardized = self._standardize_data()
        self.cov_matrix = self._compute_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self._compute_eigenvalues_eigenvectors()
        self.eigenvalues, self.eigenvectors = self._sort_eigenvectors()
        self.pca_projection = self._project_data()

        # Library PCA
        self.sklearn_pca_projection, self.sklearn_pca = self._apply_sklearn_pca()

    # Single underscore _: Indicates that the attribute is protected, but it's still accessible from outside the class.
    # It's a convention, not enforced by the language itself.
    # Double underscores __: Indicates that the attribute is private, and name mangling is applied. Accessing these
    # attributes from outside the class is more difficult and discouraged (example, for a method __foo the interpreter
    # replaces this name with _classname__foo).
    def _standardize_data(self):
        """
        Step 1: Standardize the dataset.
        """
        return StandardScaler().fit_transform(self.X)

    def _compute_covariance_matrix(self):
        """
        Step 2: Compute the covariance matrix.
        """
        return np.cov(self.X_standardized.T)

    def _compute_eigenvalues_eigenvectors(self):
        """
        Step 3: Compute the eigenvectors and eigenvalues.
        """
        return np.linalg.eig(self.cov_matrix)

    def _sort_eigenvectors(self):
        """
        Step 4: Sort eigenvectors based on eigenvalues.
        """
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        return self.eigenvalues[sorted_indices], self.eigenvectors[:, sorted_indices]

    def _project_data(self):
        """
        Step 5: Select the number of principal components and project the data onto them.
        """
        return self.X_standardized.dot(self.eigenvectors[:, :self.num_components])

    def _apply_sklearn_pca(self):
        """
        Apply PCA using sklearn (for comparison).
        """
        pca = PCA(n_components=self.num_components)
        return pca.fit_transform(self.X_standardized), pca

    def print_pca_projection(self):
        """
        Show the first lines of the developed and the library PCA projection.
        """
        print("PCA Projection (Manual):\n", self.pca_projection[:5])
        print("\nPCA Projection (Sklearn):\n", self.sklearn_pca_projection[:5])

    def display_feature_contributions(self):
        """
        Display feature contributions to principal components.
        """
        print("Feature Contributions to Principal Components:")
        for i, eigenvector in enumerate(self.eigenvectors.T):
            print(f"Principal Component {i + 1}:")
            for j, feature_contribution in enumerate(eigenvector):
                print(f"   Feature {j + 1}: {feature_contribution:.4f}")

    def plot_pca_projections(self):
        """
        Plot PCA projections for principal and for the two principal components in a 4 by 4 grid.
        """

        # For the developed PCA
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.axes[0, 0].scatter(self.pca_projection[:, 0], self.pca_projection[:, 0], c=self.y, cmap='viridis',
                                alpha=0.8)
        self.axes[0, 0].set_title('PCA Projection of the First Principal Component (Manual)')
        self.axes[0, 0].set_xlabel('Principal Component 1')
        self.axes[0, 0].set_ylabel('Principal Component 1')
        self.axes[0, 0].grid(True)

        self.axes[0, 1].scatter(self.pca_projection[:, 0], self.pca_projection[:, 1], c=self.y, cmap='viridis',
                                alpha=0.8)
        self.axes[0, 1].set_title('PCA Projection of the First Two Principal Components (Manual)')
        self.axes[0, 1].set_xlabel('Principal Component 1')
        self.axes[0, 1].set_ylabel('Principal Component 2')
        self.axes[0, 1].grid(True)

        # For th library PCA
        self.axes[1, 0].scatter(self.sklearn_pca_projection[:, 0], self.sklearn_pca_projection[:, 0], c=self.y,
                                cmap='viridis', alpha=0.8)
        self.axes[1, 0].set_title('PCA Projection of the First Principal Component (Sklearn)')
        self.axes[1, 0].set_xlabel('Principal Component 1')
        self.axes[1, 0].set_ylabel('Principal Component 1')
        self.axes[1, 0].grid(True)

        scatter = self.axes[1, 1].scatter(self.sklearn_pca_projection[:, 0], self.sklearn_pca_projection[:, 1],
                                          c=self.y, cmap='viridis', alpha=0.8)
        self.axes[1, 1].set_title('PCA Projection of the First Two Principal Components (Sklearn)')
        self.axes[1, 1].set_xlabel('Principal Component 1')
        self.axes[1, 1].set_ylabel('Principal Component 2')
        self.axes[1, 1].grid(True)

        # The * before scatter.legend_elements() is the unpacking operator in Python, when used before an iterable
        # (such as a list or a tuple), it unpacks the elements of the iterable into positional arguments of a function
        # or method call. In this specific context, scatter.legend_elements() returns a tuple containing two elements:
        # handles and labels. The handles represent the plotted elements (in this case, the points in the scatter plot),
        # and the labels represent the corresponding labels for those elements (in this case, the class labels). By
        # using * before scatter.legend_elements(), we are unpacking the tuple returned by scatter.legend_elements()
        # into separate arguments, which are then passed as positional arguments to the legend() method of the
        # matplotlib.axes.Axes object.
        self.axes[1, 1].add_artist(
            self.axes[1, 1].legend(*scatter.legend_elements(), title="Classes", loc="lower right"))
        plt.tight_layout()
        plt.show()

    def calculate_explained_variance_ratio(self):
        """
        Calculate explained variance ratio for both developed and library PCA.
        """
        explained_variance_ratio = self.eigenvalues[:self.num_components] / np.sum(self.eigenvalues)
        print(f"Explained Variance of the developed PCA using {self.num_components} component(s): ",
              np.sum(explained_variance_ratio))
        print(f"Explained Variance of the library PCA using {self.num_components} component(s): ",
              np.sum(self.sklearn_pca.explained_variance_ratio_))

    def plot_explained_variance_ratio(self):
        """
        Plot the explained variance ratio of the developed PCA.
        """
        explained_variance_ratio = self.eigenvalues[:self.num_components] / np.sum(self.eigenvalues)
        plt.figure(figsize=(8, 6))
        bars = plt.bar(range(1, self.num_components + 1), explained_variance_ratio, alpha=0.5, align='center')
        for bar, value in zip(bars, explained_variance_ratio):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{value:.2f}', ha='center',
                     va='bottom')
        plt.ylabel('Explained Variance Ratio')
        plt.xlabel('Principal Components')
        plt.title('Explained Variance Ratio per Principal Component')
        plt.grid(True)
        plt.show()

# Main entry point of the script.
# if __name__ == "__main__" is used to check whether the script is being run directly by the Python interpreter or if it
# is being imported as a module into another script.
# When a Python script is executed, Python sets the special variable __name__ to "__main__" if the script is being run
# directly. However, if the script is imported as a module into another script, the value of __name__ is set to the name
# of the module.
if __name__ == "__main__":
    iris = load_iris()
    pca_analysis = PCAAnalysis(iris.data, iris.target, 2)
    pca_analysis.display_feature_contributions()
    pca_analysis.calculate_explained_variance_ratio()
    pca_analysis.plot_explained_variance_ratio()
    # Interpretation:
    # Principal Component 1 is primarily influenced by positive contributions (increase in the value of the feature tends to
    # correspond to an increasse in the value of the corresponding principal component) from Feature 1, Feature 3, and
    # Feature 4, while being negatively influenced (increase in the value of a particular feature tends to correspond to a
    # decrease in the value of the corresponding principal component) by Feature 2; Principal Component 2 is negatively
    # influenced by all features, with the strongest negative influence coming from Feature 2.
    pca_analysis.print_pca_projection()
    pca_analysis.plot_pca_projections()

# %% Examine dimensionality reduction algorithms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap


class DimensionalityReduction:
    def __init__(self, data, targets):
        """
        Initialize the DimensionalityReduction object with the dataset.

        Parameters:
        - data: The dataset to perform dimensionality reduction on.
        - targets: The targets of the samples.
        """
        self.data = StandardScaler().fit_transform(data)
        self.targets = targets

    def compute_pca(self, n_components=2):
        """
        Compute Principal Component Analysis (PCA) on the dataset.

        Parameters:
        - n_components: The number of components to keep.

        Returns:
        - pca_projection: The projected data using PCA.
        """
        return PCA(n_components=n_components).fit_transform(self.data)

    def compute_lda(self, n_components=2):
        """
        Perform Linear Discriminant Analysis (LDA) on the input data.

        Parameters:
        - n_components: The number of components to keep

        Returns:
            array-like: The reduced-dimensional representation of the data using LDA.
        """
        return LinearDiscriminantAnalysis(n_components=n_components).fit_transform(self.data, self.targets)
    def compute_tsne(self, n_components=2, perplexity=3):
        """
        Compute t-Distributed Stochastic Neighbor Embedding (t-SNE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - perplexity: The perplexity parameter for t-SNE.

        Returns:
        - tsne_projection: The projected data using t-SNE.
        """
        return TSNE(n_components=n_components, perplexity=perplexity).fit_transform(self.data)

    def compute_umap(self, n_components=2, n_neighbors=8, min_dist=0.5, metric='euclidean'):
        """
        Compute Uniform Manifold Approximation and Projection (UMAP) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.
        - min_dist: The minimum distance between embedded points.
        - metric: The distance metric to use.

        Returns:
        - umap_projection: The projected data using UMAP.
        """
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric).fit_transform(self.data)

    def compute_lle(self, n_components=2, n_neighbors=20):
        """
        Compute Locally Linear Embedding (LLE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.

        Returns:
        - lle_projection: The projected data using LLE.
        """
        return LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components).fit_transform(self.data)

    def plot_projection(self, projection, title):
        """
        Plot the 2D projection of the dataset.

        Parameters:
        - projection: The projected data.
        - title: The title of the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(projection[:, 0], projection[:, 1], c=self.targets, alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()

    # Initialize DimensionalityReduction object with the dataset
    dr = DimensionalityReduction(iris.data, iris.target)

    # Compute and plot PCA projection
    dr.plot_projection(dr.compute_pca(), 'PCA Projection')
    # Compute and plot LDA projection
    dr.plot_projection(dr.compute_lda(), 'LDA Projection')
    # Compute and plot t-SNE projection
    dr.plot_projection(dr.compute_tsne(), 't-SNE Projection')
    # Compute and plot UMAP projection
    dr.plot_projection(dr.compute_umap(), 'UMAP Projection')
    # Compute and plot LLE projection
    dr.plot_projection(dr.compute_lle(), 'LLE Projection')

#%% Feature creation and selection

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from scipy.stats import entropy
from scipy.fftpack import fft
from mrmr import mrmr_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

class FeatureExtractor:
    """
    A class to extract various types of features from a dataset.
    """

    def __init__(self, data, feature_names, labels):
        """
        Initializes the FeatureExtractor object.

        Parameters:
        - data (numpy.ndarray): The input dataset.
        """
        self.data = data
        self.feature_names = feature_names
        self.labels = labels
        self.all_features = [];

    def _statistical_features(self):
        """
        Computes statistical features per sample (row) of the dataset.

        Returns:
        - numpy.ndarray: An array containing statistical features per sample.
        """
        # Compute statistical features
        mean = np.mean(self.data, axis=1)
        std_dev = np.std(self.data, axis=1)
        median = np.median(self.data, axis=1)
        min_val = np.min(self.data, axis=1)
        max_val = np.max(self.data, axis=1)
        # Append feature names
        self.feature_names.extend(['Mean'])
        self.feature_names.extend(['Std_Dev'])
        self.feature_names.extend(['Median'])
        self.feature_names.extend(['Min'])
        self.feature_names.extend(['Max'])
        return np.column_stack((mean, std_dev, median, min_val, max_val))

    def _pairwise_differences(self):
        """
        Computes the pairwise absolute differences between each pair of features per line.

        Returns:
        -------
        pd.DataFrame:
            A DataFrame containing pairwise absolute differences between each pair of features.
            The DataFrame has (n-1) * n / 2 columns, where n is the number of features in the dataset.
            Each row represents the absolute differences between pairs of features for a single data point.
        """
        # Calculate the number of features
        num_features = self.data.shape[1]

        # Initialize an empty DataFrame to store the pairwise differences
        pairwise_diff_df = pd.DataFrame()

        # Compute pairwise absolute differences for each pair of features
        for i in range(num_features - 1):
            for j in range(i + 1, num_features):
                feature_name = f'pairwise_diff_{i + 1}_vs_{j + 1}'
                pairwise_diff_df[feature_name] = np.abs(self.data[:, i] - self.data[:, j])
                self.feature_names.extend([feature_name])

        return pairwise_diff_df

    def _frequency_domain_features(self):
        """
        Computes frequency domain features per sample (row) of the dataset using FFT.

        Returns:
        - numpy.ndarray: An array containing frequency domain features per sample.
        """
        # Compute frequency domain features using FFT
        fft_result = fft(self.data)
        # Append feature names
        self.feature_names.extend(['FFT'])
        return np.abs(fft_result).mean(axis=1).reshape(-1, 1)

    def _entropy_features(self):
        """
        Computes entropy-based features per sample (row) of the dataset.

        Returns:
        - numpy.ndarray: An array containing entropy-based features per sample.
        """
        # Compute entropy-based features
        entropy_vals = [entropy(self.data[i]) for i in range(len(self.data))]
        # Append feature names
        self.feature_names.extend(['Entropy'])
        return np.array(entropy_vals).reshape(-1, 1)

    def _area_based_features(self):
        """
        Calculate area-based features including petal area, sepal area, and petal to sepal area ratio.

        Returns:
        - numpy.ndarray: An array containing the calculated features.
        """
        petal_length = self.data[:, 2]
        petal_width = self.data[:, 3]
        sepal_length = self.data[:, 0]
        sepal_width = self.data[:, 1]
        # Calculate petal area (assuming petal shape is close to an ellipse)
        petal_area = np.pi * (petal_length / 2) * (petal_width / 2)
        # Calculate sepal area (assuming sepal shape is close to a rectangle)
        sepal_area = sepal_length * sepal_width
        # Calculate petal to sepal area ratio
        ratio_petal_sepal_area = petal_area / sepal_area
        # Add the features names
        self.feature_names.extend(['Petal_area'])
        self.feature_names.extend(['Sepal_area'])
        self.feature_names.extend(['Ratio_petal_sepal_area'])
        # Stack the calculated features horizontally
        return np.column_stack((petal_area, sepal_area, ratio_petal_sepal_area))

    def extract_features(self):
        """
        Extracts various types of features from the dataset.

        Returns:
        - pandas.DataFrame: A dataframe containing all extracted features.
        """
        # Extract and combine all features with the original features passed in data
        self.all_features = np.hstack((self.data, self._statistical_features(), self._pairwise_differences(),
                                  self._frequency_domain_features(), self._entropy_features(), self._area_based_features()))
        # Create pandas dataframe
        return pd.DataFrame(data=self.all_features, columns=self.feature_names)

    def plot_all_features(self):
        """
        Plot histograms for all extracted features.
        Each histogram is plotted separately for each feature, with optional coloring based on provided labels.
        """
        num_features = self.all_features.shape[1]
        num_rows = int(np.ceil(np.sqrt(num_features)))
        num_cols = int(np.ceil(num_features / num_rows))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        fig.suptitle('All Features', fontsize=20)

        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < num_features:
                    ax = axes[i, j]
                    ax.set_title(f'Feature {self.feature_names[idx]}', fontsize=12)
                    ax.set_xlabel('Value', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    ax.grid(True)
                    if self.labels is not None:
                        # Add a plot per feature
                        unique_labels = np.unique(self.labels)
                        for label in unique_labels:
                            ax.hist(self.all_features[self.labels == label, idx], bins=20, alpha=0.7, label=label)
                        ax.legend()

        plt.tight_layout()
        plt.show()


class FeatureSelector:
    def __init__(self, data, labels):
        """
        Initialize the FeatureSelector instance.

        Parameters:
        - data (numpy.ndarray): The input data array with shape (n_samples, n_features).
        - labels (numpy.ndarray): The labels array with shape (n_samples,).
        """
        self.data = data
        self.labels = labels

    def select_features_mrmr(self, k=5):
        """
        Select features using mRMR (minimum Redundancy Maximum Relevance).

        Parameters:
        - k (int): The number of features to select. Default is 5.

        Returns:
        - List: The selected features as a list.
        """
        # Return the selected features
        return mrmr_classif(X=self.data, y=self.labels, K=k)

    def select_features_sequential(self, k=5):
        """
        Select features using sequential feature selection with LDA as the classifier.

        Parameters:
        - k (int): The number of features to select. Default is 5.

        Returns:
        - numpy.ndarray: The selected features array with shape (n_samples, k).
        """
        # Sequential forward feature selection
        sfs = SequentialFeatureSelector(LinearDiscriminantAnalysis(), n_features_to_select=k, direction='forward').fit(self.data, self.labels)
        # Return the selected features
        return self.data.loc[:, sfs.get_support()].columns


if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    # Create an instance of the FeatureExtractor class
    extractor = FeatureExtractor(iris.data, iris.feature_names, iris.target)
    # Extract features
    feature_df = extractor.extract_features()
    # Display the dataframe
    print(feature_df)
    # Plot the features per class
    extractor.plot_all_features()

    # Create an instance of the FeatureSelector class
    feature_selector = FeatureSelector(feature_df, iris.target)
    # Use the select_features_mrmr method
    selected_features_mrmr = feature_selector.select_features_mrmr()
    print("Selected features (mRMR):", selected_features_mrmr)
    # Use the select_features_sequential method
    selected_features_seq = feature_selector.select_features_sequential()
    print("Selected features (sequential):", selected_features_seq)
