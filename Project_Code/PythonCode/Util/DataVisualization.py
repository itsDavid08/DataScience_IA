"""Visualization utilities for exploratory analysis of flight datasets."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataVisualization:
    """Generate EDA visualizations for flight datasets and save them as PNG files.

    Attributes:
        self.data: Source pandas DataFrame used in every chart.
        self.target_col: Target column name (default: 'ARR_DELAY').
        self.output_dir: Output directory where figures are saved.
        self.numeric_cols: Numeric columns used by plotting methods (auto-detected when omitted).
    """

    def __init__(self, data, numeric_cols=None, target_col='ARR_DELAY', output_dir='.'):
        """Initialize plotting context and resolve numeric columns.

        Args:
            data: Source dataframe for all charts.
            numeric_cols: Optional numeric column list. If None, numeric columns are auto-detected.
            target_col: Target column used by target-oriented plots.
            output_dir: Folder where PNG files are saved (created if it does not exist).
        """
        self.data = data
        self.target_col = target_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        detected_numeric = data.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = numeric_cols if numeric_cols is not None else detected_numeric

        self._default_dpi = 100
        self._default_grid_cols = 3

    def _resolve_columns(self, columns=None, default_limit=None, exclude=None):
        """Resolve plot columns from defaults/user input and keep only valid dataframe columns."""
        if columns is None:
            selected = self.numeric_cols[:default_limit] if default_limit is not None else self.numeric_cols
        elif isinstance(columns, int) and not isinstance(columns, bool):
            if columns <= 0:
                return []
            selected = self.numeric_cols[:columns]
        else:
            selected = list(columns)

        excluded = set(exclude or [])
        return [col for col in selected if col in self.data.columns and col not in excluded]

    def _create_grid_axes(self, num_plots, ncols=None, base_figsize=(15, 4)):
        """Create a subplot grid and return flattened axes."""
        if num_plots <= 0:
            return None, np.array([])

        ncols = ncols or self._default_grid_cols
        nrows = (num_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(base_figsize[0], base_figsize[1] * nrows))
        return fig, np.array(axes).reshape(-1)

    def _remove_unused_axes(self, fig, axes, used_axes_count):
        """Remove axes that are not used by the current chart."""
        for idx in range(used_axes_count, len(axes)):
            fig.delaxes(axes[idx])

    def _save_show(self, filename, success_message=None):
        """Apply layout, save figure, show figure, and optionally print a success message."""
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=self._default_dpi, bbox_inches='tight')
        plt.show()
        if success_message:
            print(success_message)

    def plot_histograms(self, columns=None, bins=30, filename='viz_histograms.png'):
        """Plot histograms with KDE overlays for selected numeric columns.

        Args:
            columns: Columns to plot. Uses up to 10 numeric columns when omitted.
            bins: Histogram bin count (default: 30).
            filename: Output image file name (default: 'viz_histograms.png').

        Visual details:
            - Layout: 3-column grid with dynamic row count.
            - Color: 'skyblue'.
            - Result: frequency distribution for each selected feature.
        """
        columns = self._resolve_columns(columns=columns, default_limit=10)
        if not columns:
            print("No valid columns found for plot_histograms.")
            return

        print(f"Generating histograms for {len(columns)} columns...")
        fig, axes = self._create_grid_axes(num_plots=len(columns))

        for i, col in enumerate(columns):
            sns.histplot(self.data[col], bins=bins, kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        self._remove_unused_axes(fig, axes, len(columns))
        self._save_show(filename, f"[OK] Histograms saved: {filename}")

    def plot_boxplots(self, columns=None, filename='viz_boxplots.png'):
        """Plot boxplots to inspect spread and outliers of numeric columns.

        Args:
            columns: Columns to plot. Uses up to 10 numeric columns when omitted.
            filename: Output image file name (default: 'viz_boxplots.png').

        Visual details:
            - Layout: 3-column grid with dynamic row count.
            - Color: 'lightgreen'.
            - Result: median, quartiles, whiskers, and outlier points.
        """
        columns = self._resolve_columns(columns=columns, default_limit=10)
        if not columns:
            print("No valid columns found for plot_boxplots.")
            return

        print(f"Generating boxplots for {len(columns)} columns...")
        fig, axes = self._create_grid_axes(num_plots=len(columns))

        for i, col in enumerate(columns):
            sns.boxplot(y=self.data[col], ax=axes[i], color='lightgreen')
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_ylabel(col)

        self._remove_unused_axes(fig, axes, len(columns))
        self._save_show(filename, f"[OK] Boxplots saved: {filename}")

    def plot_correlation_matrix(self, figsize=(12, 10), filename='viz_correlation_matrix.png'):
        """Plot a full correlation heatmap for numeric variables.

        Args:
            figsize: Figure size tuple (default: (12, 10)).
            filename: Output image file name (default: 'viz_correlation_matrix.png').

        Visual details:
            - Colormap: 'coolwarm' (blue to red).
            - Annotations: disabled (`annot=False`).
            - Result: pairwise correlation strength across numeric features.
        """
        print("Generating correlation matrix...")
        plt.figure(figsize=figsize)

        numeric_data = self.data[self.numeric_cols]
        if numeric_data.empty:
            print("No valid numeric columns for correlation matrix.")
            return

        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f',
                    square=True, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix of Numeric Variables')
        self._save_show(filename, f"[OK] Correlation matrix saved: {filename}")

    def plot_pairplot(self, columns=None, sample_size=None, filename='viz_pairplot.png'):
        """Plot pairwise relationships among selected numeric columns.

        Args:
            columns: Columns to include. Uses first five numeric columns when omitted.
            sample_size: Optional row sample size to speed up plotting on large datasets.
            filename: Output image file name (default: 'viz_pairplot.png').

        Visual details:
            - Diagonal: KDE density curves.
            - Off-diagonal: scatter plots.
            - Result: quick pairwise relationship inspection.
        """
        columns = self._resolve_columns(columns=columns, default_limit=5)
        if not columns:
            print("No valid columns found for plot_pairplot.")
            return

        print(f"Generating pairplot for {len(columns)} columns...")

        data_subset = self.data[columns].copy()

        if sample_size is not None and len(data_subset) > sample_size:
            data_subset = data_subset.sample(n=sample_size, random_state=42)
            print(f"  (Using sample of {sample_size} rows)")

        pair_grid = sns.pairplot(data_subset, diag_kind='kde')
        pair_grid.figure.suptitle("Feature Pairplot", y=1.001)
        pair_grid.figure.tight_layout()
        pair_grid.figure.savefig(self.output_dir / filename, dpi=self._default_dpi, bbox_inches='tight')
        plt.show()
        print(f"[OK] Pairplot saved: {filename}")

    def plot_density_ridges(self, columns=None, filename='viz_density_ridges.png'):
        """Plot stacked density curves (ridge style) for numeric columns.

        Args:
            columns: Columns to include. Uses first eight numeric columns when omitted.
            filename: Output image file name (default: 'viz_density_ridges.png').

        Visual details:
            - Layout: single column with one subplot per feature.
            - Colors: gradient from 'Blues'.
            - Styling: minimal spines for a cleaner ridge look.
        """
        columns = self._resolve_columns(columns=columns, default_limit=8)
        if not columns:
            print("No valid columns found for plot_density_ridges.")
            return

        print(f"Generating density ridges for {len(columns)} columns...")

        num_plots = len(columns)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.5 * num_plots))

        if num_plots == 1:
            axes = [axes]

        # Build a smooth color gradient across subplots.
        cmap = plt.get_cmap('Blues')
        colors = [cmap(1 - i / (num_plots + 1)) for i in range(1, num_plots + 1)]

        for i, (col, color) in enumerate(zip(columns, colors)):
            sns.kdeplot(data=self.data[col], ax=axes[i], color=color, fill=True, linewidth=2)
            axes[i].set_ylabel(col, rotation=0, labelpad=40)
            axes[i].yaxis.set_label_coords(-0.1, 0.5)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)

        axes[-1].set_xlabel("Value")
        self._save_show(filename, f"[OK] Density ridges saved: {filename}")

    def plot_scatter_with_regression(self, x_col, y_col, filename=None):
        """Plot scatter points with a linear regression fit.

        Args:
            x_col: Feature used on the x-axis.
            y_col: Feature used on the y-axis.
            filename: Optional output file name. Auto-generates when omitted.

        Visual details:
            - Scatter alpha: 0.5.
            - Regression: linear fit with seaborn regplot defaults.
            - Result: visual check of linear relationship between two features.
        """
        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"Columns {x_col} or {y_col} not found.")
            return

        print(f"Generating scatter plot: {x_col} vs {y_col}...")

        plt.figure(figsize=(10, 6))
        sns.regplot(x=x_col, y=y_col, data=self.data, scatter_kws={'alpha':0.5})
        plt.title(f'Relationship between {x_col} and {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        if filename is None:
            filename = f'viz_scatter_{x_col}_{y_col}.png'
        self._save_show(filename, f"[OK] Scatter plot saved: {filename}")

    def plot_heatmap_top_correlations(self, top_n=20, filename='viz_heatmap_top_correlations.png'):
        """Plot a heatmap for features involved in top absolute correlations.

        Args:
            top_n: Number of strongest correlation pairs to select (default: 20).
            filename: Output image file name (default: 'viz_heatmap_top_correlations.png').

        Visual details:
            - Colormap: 'coolwarm'.
            - Annotations: enabled with 2 decimal places (`fmt='.2f'`).
            - Result: compact heatmap focused on most correlated features.
        """
        print(f"Generating top {top_n} correlations heatmap...")

        numeric_data = self.data[self.numeric_cols]
        if numeric_data.empty:
            print("No valid numeric columns for correlations heatmap.")
            return

        corr_matrix = numeric_data.corr().abs()

        # Keep only upper-triangle pairs to avoid duplicate correlations.
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        sorted_pairs = upper_triangle.stack().sort_values(ascending=False)

        top_pairs = sorted_pairs.head(top_n)
        if top_pairs.empty:
            print("Not enough pairs to generate correlations heatmap.")
            return

        top_features = list(set([pair[0] for pair in top_pairs.index] +
                                [pair[1] for pair in top_pairs.index]))

        # Build a compact heatmap from top-related features.
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data[top_features].corr(), annot=True, cmap='coolwarm',
                   square=True, fmt='.2f', cbar_kws={"shrink": 0.8})
        plt.title(f'Top {len(top_features)} Features with Highest Correlations')
        self._save_show(filename, f"[OK] Heatmap saved: {filename}")

    def plot_distributions(self):
        """Convenience wrapper that plots distributions for all configured numeric columns.

        Uses:
            - plot_histograms(columns=self.numeric_cols, bins=30)
            - Output file: 'eda_distributions.png'
        """
        self.plot_histograms(columns=self.numeric_cols, bins=30, filename='eda_distributions.png')

    def plot_target_distribution(self, filename='eda_target_distribution.png'):
        """Plot target distribution with histogram and boxplot side by side.

        Args:
            filename: Output image file name (default: 'eda_target_distribution.png').

        Visual details:
            - Left subplot: histogram + KDE, color 'coral', bins=50.
            - Right subplot: boxplot, color 'lightblue'.
            - Result: target shape, spread, and outlier overview.
        """
        if self.target_col not in self.data.columns:
            print(f"Target column {self.target_col} not found.")
            return

        print("\n4. Generating target variable distribution...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.histplot(self.data[self.target_col], bins=50, kde=True, ax=axes[0], color='coral')
        axes[0].set_title(f'Distribution of {self.target_col}')
        axes[0].set_xlabel(self.target_col)
        axes[0].set_ylabel('Frequency')

        sns.boxplot(y=self.data[self.target_col], ax=axes[1], color='lightblue')
        axes[1].set_title(f'Boxplot of {self.target_col}')
        axes[1].set_ylabel(self.target_col)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=100, bbox_inches='tight')
        plt.show()
        print(f"[OK] Target distribution saved: {filename}")

    def plot_reduction_scatter(self, components, labels, method_name, x_label, y_label, filename):
        """Plot a standardized 2D scatter for PCA/UMAP/t-SNE style projections.

        Args:
            components: 2D projected matrix with shape (n_samples, 2).
            labels: Values used for point coloring.
            method_name: Projection method name used in the title.
            x_label: X-axis label.
            y_label: Y-axis label.
            filename: Output image file name.

        Visual details:
            - Colormap: 'viridis'.
            - Point alpha: 0.6.
            - Grid: enabled with alpha=0.3.
            - Colorbar: labeled with `self.target_col`.
        """
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(components[:, 0], components[:, 1], alpha=0.6, c=labels, cmap='viridis')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'2D {method_name} Projection')
        plt.colorbar(scatter, label=self.target_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=100, bbox_inches='tight')
        plt.show()

    def plot_grouped_feature_distributions(self, columns=None, group_col='DELAY_CLASS', bins=30,
                                           filename='viz_grouped_distributions.png'):
        """Plot grouped histograms to compare feature distributions across classes.

        Args:
            columns: Numeric columns to compare. Uses first six numeric columns when omitted.
            group_col: Grouping column mapped to hue (default: 'DELAY_CLASS').
            bins: Histogram bin count (default: 30).
            filename: Output image file name (default: 'viz_grouped_distributions.png').

        Visual details:
            - Layout: 3-column grid with dynamic row count.
            - Group overlays: hue by `group_col` with alpha=0.4.
            - Result: class-level distribution comparison per feature.
        """
        if group_col not in self.data.columns:
            print(f"Group column {group_col} not found.")
            return

        columns = self._resolve_columns(columns=columns, default_limit=6, exclude={group_col})
        if not columns:
            print("No valid columns found for plot_grouped_feature_distributions.")
            return

        print(f"Generating grouped histograms for {len(columns)} columns...")
        fig, axes = self._create_grid_axes(num_plots=len(columns))

        for i, col in enumerate(columns):
            sns.histplot(
                data=self.data,
                x=col,
                hue=group_col,
                kde=True,
                bins=bins,
                alpha=0.4,
                ax=axes[i]
            )
            axes[i].set_title(f'Distribution of {col} by {group_col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        self._remove_unused_axes(fig, axes, len(columns))
        self._save_show(filename, f"[OK] Grouped histograms saved: {filename}")

    def plot_grouped_boxplots(self, columns=None, group_col='DELAY_CLASS', filename='viz_grouped_boxplots.png'):
        """Plot grouped boxplots with strip overlays for class comparisons.

        Args:
            columns: Numeric columns to compare. Uses first six numeric columns when omitted.
            group_col: Grouping column used on x-axis (default: 'DELAY_CLASS').
            filename: Output image file name (default: 'viz_grouped_boxplots.png').

        Visual details:
            - Layout: 3-column grid with dynamic row count.
            - Box color: 'lightgreen'.
            - Strip points: 'black', alpha=0.15, size=2.
            - Result: distribution, spread, and outlier view by group.
        """
        if group_col not in self.data.columns:
            print(f"Group column {group_col} not found.")
            return

        columns = self._resolve_columns(columns=columns, default_limit=6, exclude={group_col})
        if not columns:
            print("No valid columns found for plot_grouped_boxplots.")
            return

        print(f"Generating grouped boxplots for {len(columns)} columns...")
        fig, axes = self._create_grid_axes(num_plots=len(columns))

        for i, col in enumerate(columns):
            sns.boxplot(data=self.data, x=group_col, y=col, ax=axes[i], color='lightgreen')
            sns.stripplot(data=self.data, x=group_col, y=col, ax=axes[i], color='black', alpha=0.15, size=2)
            axes[i].set_title(f'Boxplot of {col} by {group_col}')
            axes[i].set_xlabel(group_col)
            axes[i].set_ylabel(col)

        self._remove_unused_axes(fig, axes, len(columns))
        self._save_show(filename, f"[OK] Grouped boxplots saved: {filename}")
