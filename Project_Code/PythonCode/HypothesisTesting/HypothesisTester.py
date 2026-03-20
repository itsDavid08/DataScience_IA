#%% Hypothesis Testing
import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    f_oneway,
    friedmanchisquare,
    kruskal,
    levene,
    mannwhitneyu,
    probplot,
    shapiro,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)


class HypothesisTester:
    """
    Unified statistical testing utility.

    This class combines:
    - Manual/stateless-style methods (explicit groups per call), and
    - Dataset-aware bulk methods (iterate numeric features by class labels).
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        labels: Optional[pd.Series] = None,
        target_col: str = "DELAY_CLASS",
        alpha: float = 0.05,
        verbose: bool = True,
    ):
        self.data = data
        self.labels = labels
        self.target_col = target_col
        self.alpha = alpha
        self.verbose = verbose
        self.numeric_cols: List[str] = []
        self._refresh_schema()

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def set_data(
        self,
        data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        target_col: Optional[str] = None,
    ) -> None:
        """Set or replace dataset context for bulk tests."""
        self.data = data
        self.labels = labels
        if target_col is not None:
            self.target_col = target_col
        self._refresh_schema()

    def _refresh_schema(self) -> None:
        if self.data is None:
            self.numeric_cols = []
            return

        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if self.labels is None and self.target_col in self.data.columns:
            self.labels = self.data[self.target_col]

        if self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)

    def _require_dataset_mode(self) -> None:
        if self.data is None or self.labels is None:
            raise ValueError(
                "Dataset mode requires 'data' and 'labels' (or target_col present in data)."
            )

    def _feature_list(self, columns: Optional[Sequence[str]] = None) -> List[str]:
        return list(columns) if columns is not None else list(self.numeric_cols)

    def _groups_for_feature(self, feature: str) -> List[np.ndarray]:
        labels_series = pd.Series(self.labels).dropna()
        unique_labels = labels_series.unique()

        groups = [
            self.data.loc[self.labels == label, feature].dropna().values
            for label in unique_labels
        ]
        return [g for g in groups if len(g) > 0]

    # ---------------------------
    # Legacy/manual methods
    # ---------------------------
    def unpaired_t_test(self, group1, group2, equal_var: bool = False) -> Tuple[float, float]:
        """Perform an unpaired t-test for two independent groups."""
        statistic, p_value = ttest_ind(group1, group2, equal_var=equal_var, nan_policy="omit")
        return float(statistic), float(p_value)

    def unpaired_anova(self, *groups) -> Tuple[float, float]:
        """Perform one-way ANOVA for two or more independent groups."""
        statistic, p_value = f_oneway(*groups)
        return float(statistic), float(p_value)

    def paired_t_test(self, group1, group2) -> Tuple[float, float]:
        """Perform a paired t-test for two related samples."""
        statistic, p_value = ttest_rel(group1, group2, nan_policy="omit")
        return float(statistic), float(p_value)

    def paired_anova(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Perform repeated-measures style ANOVA with long-format data.
        Expected columns: 'value', 'condition'.
        """
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        model = ols("value ~ C(condition)", data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return float(anova_table["F"].iloc[0]), float(anova_table["PR(>F)"].iloc[0])

    def mann_whitney_u_test(
        self, group1, group2, alternative: str = "two-sided"
    ) -> Tuple[float, float]:
        """Perform Mann-Whitney U test for two independent samples."""
        statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)
        return float(statistic), float(p_value)

    def wilcoxon_ranksum_test(
        self, group1, group2, alternative: str = "two-sided"
    ) -> Tuple[float, float]:
        """Backward-compatible alias for Mann-Whitney U test."""
        return self.mann_whitney_u_test(group1, group2, alternative=alternative)

    def wilcoxon_signedrank_test(
        self, group1, group2, alternative: str = "two-sided"
    ) -> Tuple[float, float]:
        """Perform Wilcoxon signed-rank test for paired samples."""
        result = wilcoxon(group1, group2, alternative=alternative)
        return float(result.statistic), float(result.pvalue)

    def kruskal_wallis_test(self, *groups) -> Tuple[float, float]:
        """Perform Kruskal-Wallis test for independent groups."""
        statistic, p_value = kruskal(*groups)
        return float(statistic), float(p_value)

    def friedman_test(self, *groups) -> Tuple[float, float]:
        """Perform Friedman test for related groups."""
        statistic, p_value = friedmanchisquare(*groups)
        return float(statistic), float(p_value)

    def qq_plots(
        self,
        variable_names: Sequence[str],
        *data_samples,
        distribution: str = "norm",
    ) -> None:
        """Generate Q-Q plots for multiple samples."""
        num_samples = len(data_samples)
        if num_samples == 0:
            return

        num_rows = (num_samples + 1) // 2
        num_cols = 2 if num_samples > 1 else 1

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
        axes = np.atleast_1d(axes).ravel()

        for i, data in enumerate(data_samples):
            ax = axes[i]
            probplot(data, dist=distribution, plot=ax)
            ax.set_title(f"Q-Q Plot ({distribution})")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel(variable_names[i])

        for j in range(num_samples, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def test_normality(
        self, variable_names: Sequence[str], *data_samples
    ) -> Dict[str, Tuple[float, float]]:
        """Run Shapiro-Wilk normality test on provided samples."""
        results: Dict[str, Tuple[float, float]] = {}

        for name, data in zip(variable_names, data_samples):
            statistic, p_value = shapiro(data)
            results[name] = (float(statistic), float(p_value))
            if self.verbose:
                print(
                    f"{name}: Shapiro-Wilk statistic={statistic:.6f}, "
                    f"p-value={p_value:.6f}"
                )

        return results

    # ---------------------------
    # Dataset-aware bulk methods
    # ---------------------------
    def perform_normality_test(
        self,
        sample_size: int = 5000,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Run Shapiro-Wilk test per numeric feature."""
        self._require_dataset_mode()
        results = []

        for col in self._feature_list(columns):
            data_col = self.data[col].dropna()
            if len(data_col) == 0:
                continue

            if len(data_col) > sample_size:
                data_col = data_col.sample(n=sample_size, random_state=42)

            statistic, p_value = shapiro(data_col)
            results.append(
                {
                    "Feature": col,
                    "Statistic": float(statistic),
                    "P-Value": float(p_value),
                    "Normal (α=0.05)": bool(p_value > self.alpha),
                }
            )

        return pd.DataFrame(results)

    def perform_anova_test(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Run one-way ANOVA per numeric feature across label groups."""
        self._require_dataset_mode()
        results = []

        for col in self._feature_list(columns):
            groups = self._groups_for_feature(col)
            if len(groups) < 2:
                continue

            statistic, p_value = f_oneway(*groups)
            results.append(
                {
                    "Feature": col,
                    "F-Statistic": float(statistic),
                    "P-Value": float(p_value),
                    "Significant (α=0.05)": bool(p_value < self.alpha),
                }
            )

        return pd.DataFrame(results)

    def perform_kruskal_wallis_test(
        self, columns: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Run Kruskal-Wallis test per numeric feature across label groups."""
        self._require_dataset_mode()
        results = []

        for col in self._feature_list(columns):
            groups = self._groups_for_feature(col)
            if len(groups) < 2:
                continue

            statistic, p_value = kruskal(*groups)
            results.append(
                {
                    "Feature": col,
                    "H-Statistic": float(statistic),
                    "P-Value": float(p_value),
                    "Significant (α=0.05)": bool(p_value < self.alpha),
                }
            )

        return pd.DataFrame(results)

    def perform_t_tests(
        self,
        columns: Optional[Sequence[str]] = None,
        equal_var: bool = False,
    ) -> pd.DataFrame:
        """Run pairwise independent t-tests between all label pairs per feature."""
        self._require_dataset_mode()
        labels_series = pd.Series(self.labels).dropna()
        unique_labels = labels_series.unique()

        results = []
        for col in self._feature_list(columns):
            for label1, label2 in itertools.combinations(unique_labels, 2):
                data1 = self.data.loc[self.labels == label1, col].dropna().values
                data2 = self.data.loc[self.labels == label2, col].dropna().values

                if len(data1) == 0 or len(data2) == 0:
                    continue

                statistic, p_value = ttest_ind(
                    data1, data2, equal_var=equal_var, nan_policy="omit"
                )
                results.append(
                    {
                        "Feature": col,
                        "Group1": str(label1),
                        "Group2": str(label2),
                        "T-Statistic": float(statistic),
                        "P-Value": float(p_value),
                        "Significant (α=0.05)": bool(p_value < self.alpha),
                    }
                )

        return pd.DataFrame(results)

    def perform_levene_test(
        self, columns: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Run Levene test per numeric feature across label groups."""
        self._require_dataset_mode()
        results = []

        for col in self._feature_list(columns):
            groups = self._groups_for_feature(col)
            if len(groups) < 2:
                continue

            statistic, p_value = levene(*groups)
            results.append(
                {
                    "Feature": col,
                    "Statistic": float(statistic),
                    "P-Value": float(p_value),
                    "Different Variances (α=0.05)": bool(p_value < self.alpha),
                }
            )

        return pd.DataFrame(results)

    def generate_summary_report(
        self,
        include_pairwise_ttests: bool = True,
        sample_size: int = 5000,
        columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Generate a dictionary with all bulk test outputs."""
        self._require_dataset_mode()

        report = {
            "normality": self.perform_normality_test(sample_size=sample_size, columns=columns),
            "anova": self.perform_anova_test(columns=columns),
            "kruskal_wallis": self.perform_kruskal_wallis_test(columns=columns),
            "levene": self.perform_levene_test(columns=columns),
        }

        if include_pairwise_ttests and len(pd.Series(self.labels).dropna().unique()) > 1:
            report["t_tests"] = self.perform_t_tests(columns=columns)

        return report

