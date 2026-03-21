"""Exploratory data analysis utilities for the flight delay project.

This module centralizes analytical EDA routines and delegates chart rendering
responsibilities to ``DataVisualization``.
"""

from pathlib import Path
import sys

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
from Project_Code.PythonCode.Util.DataVisualization import DataVisualization


# Try to import UMAP and gracefully fall back to t-SNE if it is unavailable.
try:
    import umap as _umap_check
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Aviso: umap-learn não está instalado. Usar t-SNE como fallback.")


class FlightEDA:
    """Run analytical EDA and dimensionality reduction for flight data.

    The class computes descriptive reports and quality checks, while all plot
    generation is delegated to ``DataVisualization_IA``.
    """

    def __init__(self, data, target_col='ARR_DELAY', output_dir='Output_Files', group_col='DELAY_CLASS'):
        """Initialize EDA state, feature metadata, and plotting helper.

        Args:
            data: Input dataframe for EDA.
            target_col: Name of the regression target column.
            output_dir: Folder where generated artifacts are saved.
            group_col: Preferred grouping column for grouped analysis.
        """
        self.data = data.copy()
        self.target_col = target_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in self.numeric_cols:
            self.numeric_cols.remove(target_col)

        self.group_col = group_col if group_col in self.data.columns else None

        if self.group_col is None:
            print("Aviso: group_col não encontrado. EDA agrupada será ignorada.")

        self.viz = DataVisualization(
            data=self.data,
            numeric_cols=self.numeric_cols,
            target_col=self.target_col,
            output_dir=self.output_dir,
        )

    def describe_variables(self):
        """Compute and print descriptive statistics globally and by group.

        Returns:
            dict: ``{'overall': DataFrame, 'by_group': DataFrame | None}``.
        """
        print("\n" + "=" * 60)
        print("ESTATISTICAS DESCRITIVAS")
        print("=" * 60)

        overall_stats = self.data[self.numeric_cols + [self.target_col]].describe(include='all')
        print("Resumo geral:")
        print(overall_stats)

        grouped_stats = None
        if self.group_col is not None:
            grouped_stats = self.data.groupby(self.group_col)[self.numeric_cols + [self.target_col]].describe()
            print(f"\nResumo por grupo ({self.group_col}):")
            print(grouped_stats)

        print("=" * 60 + "\n")
        return {'overall': overall_stats, 'by_group': grouped_stats}

    def determine_range(self):
        """Compute feature ranges (max-min) globally and by group.

        Returns:
            dict: ``{'overall': Series, 'by_group': DataFrame | None}``.
        """
        print("\n" + "=" * 60)
        print("AMPLITUDE DAS VARIAVEIS")
        print("=" * 60)

        range_overall = self.data[self.numeric_cols + [self.target_col]].max() - self.data[self.numeric_cols + [self.target_col]].min()
        print("Amplitude geral:")
        print(range_overall)

        range_by_group = None
        if self.group_col is not None:
            grouped_max = self.data.groupby(self.group_col)[self.numeric_cols + [self.target_col]].max()
            grouped_min = self.data.groupby(self.group_col)[self.numeric_cols + [self.target_col]].min()
            range_by_group = grouped_max - grouped_min
            print(f"\nAmplitude por grupo ({self.group_col}):")
            print(range_by_group)

        print("=" * 60 + "\n")
        return {'overall': range_overall, 'by_group': range_by_group}

    def assess_correlation(self):
        """Compute Pearson correlations globally and per group.

        Returns:
            dict: ``{'overall': DataFrame, 'by_group': dict | None}``.
        """
        print("\n" + "=" * 60)
        print("CORRELACAO ENTRE VARIAVEIS")
        print("=" * 60)

        corr_overall = self.data[self.numeric_cols + [self.target_col]].corr()
        print("Correlacao geral:")
        print(corr_overall)

        corr_by_group = None
        if self.group_col is not None:
            corr_by_group = {
                group_name: group_df[self.numeric_cols + [self.target_col]].corr()
                for group_name, group_df in self.data.groupby(self.group_col)
                if len(group_df) > 2
            }
            print(f"\nCorrelacao por grupo ({self.group_col}):")
            for group_name, group_corr in corr_by_group.items():
                print(f"\n[{group_name}]")
                print(group_corr)

        print("=" * 60 + "\n")
        return {'overall': corr_overall, 'by_group': corr_by_group}

    def assess_quality(self):
        """Assess dataset quality using missing, duplicates, and IQR outliers.

        Returns:
            dict: Missing value counts, duplicate count, and outlier counts.
        """
        print("\n" + "=" * 60)
        print("QUALIDADE DO DATASET")
        print("=" * 60)

        subset_cols = self.numeric_cols + [self.target_col]
        numeric_df = self.data[subset_cols].copy()

        missing = numeric_df.isnull().sum().sort_values(ascending=False)
        duplicates = int(self.data.duplicated().sum())

        outlier_counts = {}
        for feature in subset_cols:
            q1 = numeric_df[feature].quantile(0.25)
            q3 = numeric_df[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = numeric_df[(numeric_df[feature] < lower_bound) | (numeric_df[feature] > upper_bound)]
            outlier_counts[feature] = int(len(outliers))

        outlier_series = pd.Series(outlier_counts).sort_values(ascending=False)

        print("Missing values por coluna:")
        print(missing)
        print(f"\nDuplicados no dataframe: {duplicates}")
        print("\nOutliers por coluna (IQR):")
        print(outlier_series)
        print("=" * 60 + "\n")

        return {
            'missing_values': missing,
            'duplicate_count': duplicates,
            'outlier_count': outlier_series,
        }

    def run_analytical_eda(self):
        """Run only analytical EDA checks and return a consolidated report.

        Returns:
            dict: Consolidated analytical outputs from EDA checks.
        """
        print("=" * 60)
        print("INICIANDO EDA ANALITICA")
        print("=" * 60)

        report = {
            'descriptive_stats': self.describe_variables(),
            'ranges': self.determine_range(),
            'correlations': self.assess_correlation(),
            'quality': self.assess_quality(),
        }

        print("=" * 60)
        print("EDA ANALITICA CONCLUIDA!")
        print("=" * 60 + "\n")
        return report

    def run_core_visual_eda(self):
        """Run only core visual EDA charts (non-grouped)."""
        print("=" * 60)
        print("INICIANDO EDA VISUAL (GRAFICOS BASE)")
        print("=" * 60)

        self.viz.plot_distributions()
        self.viz.plot_correlation_matrix(filename='eda_correlation_matrix.png')
        self.viz.plot_boxplots(columns=self.numeric_cols, filename='eda_boxplots.png')
        self.viz.plot_target_distribution(filename='eda_target_distribution.png')

        print("=" * 60)
        print("EDA VISUAL (GRAFICOS BASE) CONCLUIDA!")
        print("=" * 60 + "\n")

    def run_grouped_visual_eda(self, focus_cols=None):
        """Run grouped visual EDA charts when grouping is available.

        Args:
            focus_cols: Optional subset of columns for grouped charts.
        """
        if self.group_col is None:
            print("Coluna de grupo nao disponivel. Graficos agrupados ignorados.")
            return

        if focus_cols is None:
            # Keep grouped charts focused to avoid overcrowded figures.
            focus_cols = self.numeric_cols[:6]

        print("=" * 60)
        print(f"INICIANDO EDA VISUAL AGRUPADA ({self.group_col})")
        print("=" * 60)

        self.viz.plot_grouped_feature_distributions(
            columns=focus_cols,
            group_col=self.group_col,
            filename='eda_grouped_distributions.png'
        )
        self.viz.plot_grouped_boxplots(
            columns=focus_cols,
            group_col=self.group_col,
            filename='eda_grouped_boxplots.png'
        )

        print("=" * 60)
        print("EDA VISUAL AGRUPADA CONCLUIDA!")
        print("=" * 60 + "\n")

    def run_eda_steps(self, run_analytics=True, run_core_visuals=True, run_grouped_visuals=True, grouped_focus_cols=None):
        """Run EDA in configurable stages for notebook-friendly execution.

        Args:
            run_analytics: Whether to run analytical EDA checks.
            run_core_visuals: Whether to run core non-grouped charts.
            run_grouped_visuals: Whether to run grouped charts.
            grouped_focus_cols: Optional columns for grouped charts.

        Returns:
            dict: Analytical report when requested, else empty dict.
        """
        report = {}

        if run_analytics:
            report = self.run_analytical_eda()

        if run_core_visuals:
            self.run_core_visual_eda()

        if run_grouped_visuals:
            self.run_grouped_visual_eda(focus_cols=grouped_focus_cols)

        return report

    def perform_eda(self):
        """Run the full EDA workflow (analytical + visual) and return a report.

        Returns:
            dict: Consolidated analytical outputs from all EDA checks.
        """
        print("=" * 60)
        print("INICIANDO ANALISE EXPLORATORIA (EDA)")
        print("=" * 60)

        report = self.run_eda_steps(
            run_analytics=True,
            run_core_visuals=True,
            run_grouped_visuals=True,
        )

        print("=" * 60)
        print("EDA CONCLUIDA!")
        print("=" * 60 + "\n")
        return report

    def run_pca(self, n_components=2, explained_variance_threshold=0.8):
        """Run PCA and return projected components.

        Args:
            n_components: Number of principal components to return.
            explained_variance_threshold: Threshold used to report auto component count.

        Returns:
            np.ndarray: PCA transformed matrix.
        """

        print("\n" + "=" * 60)
        print("EXECUTANDO PCA")
        print("=" * 60)

        if n_components < 2:
            raise ValueError("n_components deve ser >= 2 para visualização 2D.")

        # --- Features only ---
        df = self.data[self.numeric_cols].dropna()
        X = df

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Fit PCA ---
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_scaled)

        # --- Labels for visualization ---
        if "DELAY_CLASS" in self.data.columns:
            labels = self.data.loc[df.index, "DELAY_CLASS"].astype("category").cat.codes
        else:
            labels = self.data.loc[df.index, self.target_col]

        # --- Plot ---
        self.viz.plot_reduction_scatter(
            components=pca_result,
            labels=labels,
            method_name="PCA",
            x_label="Dimensão 1",
            y_label="Dimensão 2",
            filename="eda_pca_2d.png",
        )

        print("\n✓ PCA executado e visualizado")
        print("=" * 60 + "\n")

        return pca_result

    def run_umap_or_tsne(self, n_components=2, use_umap=True):
        """Run UMAP (preferred) or t-SNE (fallback) for nonlinear projection.

        Args:
            n_components: Number of dimensions for embedding.
            use_umap: Whether UMAP should be attempted first.

        Returns:
            np.ndarray: Nonlinear embedding matrix.
        """
        print("\n" + "=" * 60)
        print("EXECUTANDO UMAP / t-SNE")
        print("=" * 60)

        # --- Features only ---
        df = self.data[self.numeric_cols].dropna()
        X = df

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Reduce ---
        if use_umap and HAS_UMAP:
            import umap as umap_module

            print("\nAplicando UMAP...")
            reducer = umap_module.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=15
            )
            result = reducer.fit_transform(X_scaled)
            method_name = "UMAP"
        else:
            print("\nAplicando t-SNE...")
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=30
            )
            result = reducer.fit_transform(X_scaled)
            method_name = "t-SNE"

        # --- Labels ---
        if "DELAY_CLASS" in self.data.columns:
            labels = self.data.loc[df.index, "DELAY_CLASS"].astype("category").cat.codes
        else:
            labels = self.data.loc[df.index, self.target_col]

        # --- Plot ---
        self.viz.plot_reduction_scatter(
            components=result,
            labels=labels,
            method_name=method_name,
            x_label="Dimensão 1",
            y_label="Dimensão 2",
            filename=f"eda_{method_name.lower()}_2d.png",
        )

        print(f"\n✓ {method_name} executado e visualizado")
        print("=" * 60 + "\n")

        return result

    def get_summary_stats(self):
        """Return the global descriptive statistics table."""
        return self.describe_variables()['overall']
