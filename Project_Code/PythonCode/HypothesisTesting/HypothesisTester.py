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
    """Unified statistical testing utility.

    Each bulk test method follows the formal hypothesis testing procedure:
        1. State the null hypothesis (H0) and alternative hypothesis (H1).
        2. Choose the appropriate statistical test.
        3. Execute the test and collect the test statistic and p-value.
        4. Interpret: reject H0 if p-value < alpha; fail to reject otherwise.

    This class combines:
        - Manual/stateless methods (explicit groups per call).
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
    def set_data(self, data, labels=None, target_col=None):
        self.data = data
        self.labels = labels
        if target_col is not None:
            self.target_col = target_col
        self._refresh_schema()

    def _refresh_schema(self):
        if self.data is None:
            self.numeric_cols = []
            return

        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        if self.labels is None and self.target_col in self.data.columns:
            self.labels = self.data[self.target_col]

        if self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)

    def _require_dataset_mode(self):
        if self.data is None or self.labels is None:
            raise ValueError(
                "Dataset mode requires 'data' and 'labels' (or target_col present in data)."
            )

    def _feature_list(self, columns=None):
        return list(columns) if columns is not None else list(self.numeric_cols)

    def _groups_for_feature(self, feature):
        labels_series = pd.Series(self.labels).dropna()
        unique_labels = labels_series.unique()
        groups = [
            self.data.loc[self.labels == label, feature].dropna().values
            for label in unique_labels
        ]
        return [g for g in groups if len(g) > 0]

    def _print_hypothesis(self, h0: str, h1: str, test_name: str) -> None:
        """Print the formal hypothesis formulation before executing a test."""
        if self.verbose:
            print(f"\n{'─' * 55}")
            print(f"  Test: {test_name}")
            print(f"  H0 (null hypothesis)       : {h0}")
            print(f"  H1 (alternative hypothesis): {h1}")
            print(f"  Significance level (α)     : {self.alpha}")
            print(f"{'─' * 55}")

    def _print_decision(self, p_value: float, feature: str = "") -> None:
        """Print the statistical decision based on p-value vs alpha."""
        if self.verbose:
            label = f" [{feature}]" if feature else ""
            if p_value < self.alpha:
                print(f"  → REJECT H0{label}: p={p_value:.6f} < α={self.alpha} — significant result.")
            else:
                print(f"  → FAIL TO REJECT H0{label}: p={p_value:.6f} ≥ α={self.alpha} — not significant.")

    # ---------------------------
    # Manual/stateless methods
    # ---------------------------
    def unpaired_t_test(self, group1, group2, equal_var: bool = False) -> Tuple[float, float]:
        """Perform an unpaired t-test for two independent groups.

        H0: The means of group1 and group2 are equal (μ1 = μ2).
        H1: The means of group1 and group2 are different (μ1 ≠ μ2).
        """
        statistic, p_value = ttest_ind(group1, group2, equal_var=equal_var, nan_policy="omit")
        return float(statistic), float(p_value)

    def unpaired_anova(self, *groups) -> Tuple[float, float]:
        """Perform one-way ANOVA for two or more independent groups.

        H0: All group means are equal (μ1 = μ2 = ... = μk).
        H1: At least one group mean differs from the others.
        """
        statistic, p_value = f_oneway(*groups)
        return float(statistic), float(p_value)

    def paired_t_test(self, group1, group2) -> Tuple[float, float]:
        """Perform a paired t-test for two related samples.

        H0: The mean difference between paired observations is zero (μd = 0).
        H1: The mean difference is not zero (μd ≠ 0).
        """
        statistic, p_value = ttest_rel(group1, group2, nan_policy="omit")
        return float(statistic), float(p_value)

    def mann_whitney_u_test(self, group1, group2, alternative: str = "two-sided") -> Tuple[float, float]:
        """Perform Mann-Whitney U test for two independent samples.

        H0: The distributions of group1 and group2 are equal.
        H1: The distributions differ (two-sided) or one stochastically dominates the other.
        Non-parametric alternative to the independent t-test when normality is not met.
        """
        statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)
        return float(statistic), float(p_value)

    def wilcoxon_ranksum_test(self, group1, group2, alternative: str = "two-sided") -> Tuple[float, float]:
        """Alias for Mann-Whitney U test (rank-sum formulation)."""
        return self.mann_whitney_u_test(group1, group2, alternative=alternative)

    def wilcoxon_signedrank_test(self, group1, group2, alternative: str = "two-sided") -> Tuple[float, float]:
        """Perform Wilcoxon signed-rank test for paired samples.

        H0: The median difference between paired observations is zero.
        H1: The median difference is not zero.
        Non-parametric alternative to the paired t-test.
        """
        result = wilcoxon(group1, group2, alternative=alternative)
        return float(result.statistic), float(result.pvalue)

    def kruskal_wallis_test(self, *groups) -> Tuple[float, float]:
        """Perform Kruskal-Wallis test for independent groups.

        H0: All groups come from the same distribution (medians are equal).
        H1: At least one group comes from a different distribution.
        Non-parametric alternative to one-way ANOVA.
        """
        statistic, p_value = kruskal(*groups)
        return float(statistic), float(p_value)

    def friedman_test(self, *groups) -> Tuple[float, float]:
        """Perform Friedman test for related groups.

        H0: The distributions of all related groups are equal.
        H1: At least one group distribution differs.
        Non-parametric alternative to repeated-measures ANOVA.
        """
        statistic, p_value = friedmanchisquare(*groups)
        return float(statistic), float(p_value)

    def qq_plots(self, variable_names, *data_samples, distribution: str = "norm") -> None:
        """Generate Q-Q plots to visually assess normality."""
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

    def test_normality(self, variable_names, *data_samples) -> Dict[str, Tuple[float, float]]:
        """Run Shapiro-Wilk normality test on provided samples.

        H0: The data follows a normal distribution.
        H1: The data does not follow a normal distribution.
        """
        results: Dict[str, Tuple[float, float]] = {}
        for name, data in zip(variable_names, data_samples):
            statistic, p_value = shapiro(data)
            results[name] = (float(statistic), float(p_value))
            if self.verbose:
                print(
                    f"{name}: Shapiro-Wilk W={statistic:.6f}, p={p_value:.6f} "
                    f"→ {'Normal' if p_value > self.alpha else 'NOT normal'}"
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
        """Run Shapiro-Wilk normality test per numeric feature.

        Formal hypothesis:
            H0: The feature follows a normal distribution.
            H1: The feature does not follow a normal distribution.

        Choice justification: Shapiro-Wilk is one of the most powerful tests for
        normality, especially for small to medium samples. Results inform whether
        parametric (t-test, ANOVA) or non-parametric tests (Kruskal-Wallis,
        Mann-Whitney) are appropriate for subsequent analyses.

        Args:
            sample_size: Max rows to sample (Shapiro-Wilk is limited to ~5000).
            columns: Optional subset of features to test.

        Returns:
            pd.DataFrame: Feature-level normality test results.
        """
        self._require_dataset_mode()

        self._print_hypothesis(
            h0="The feature follows a normal distribution.",
            h1="The feature does not follow a normal distribution.",
            test_name="Shapiro-Wilk Normality Test",
        )

        results = []
        for col in self._feature_list(columns):
            data_col = self.data[col].dropna()
            if len(data_col) == 0:
                continue
            if len(data_col) > sample_size:
                data_col = data_col.sample(n=sample_size, random_state=42)

            statistic, p_value = shapiro(data_col)
            is_normal = bool(p_value > self.alpha)
            results.append({
                "Feature": col,
                "Statistic (W)": float(statistic),
                "P-Value": float(p_value),
                f"Normal (α={self.alpha})": is_normal,
                "Decision": f"Fail to reject H0 (normal)" if is_normal else "Reject H0 (not normal)",
            })
            self._print_decision(p_value, feature=col)

        return pd.DataFrame(results)

    def perform_anova_test(self, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Run one-way ANOVA per numeric feature across label groups.

        Formal hypothesis:
            H0: The means of the feature are equal across all delay classes
                (μ_OnTime = μ_ShortDelay = μ_LongDelay).
            H1: At least one delay class has a significantly different mean.

        Choice justification: ANOVA tests whether group membership (delay class)
        explains a significant portion of the variance in each feature. A significant
        result suggests the feature is discriminative between delay classes.
        Assumes normality within groups and homogeneity of variance (verified with
        Levene's test). If normality is violated, use Kruskal-Wallis instead.

        Args:
            columns: Optional subset of features to test.

        Returns:
            pd.DataFrame: Feature-level ANOVA results.
        """
        self._require_dataset_mode()

        unique_labels = pd.Series(self.labels).dropna().unique()
        label_str = " = ".join([f"μ_{l}" for l in unique_labels])
        self._print_hypothesis(
            h0=f"Feature means are equal across all delay classes ({label_str}).",
            h1="At least one delay class has a significantly different mean.",
            test_name="One-Way ANOVA (F-test)",
        )

        results = []
        for col in self._feature_list(columns):
            groups = self._groups_for_feature(col)
            if len(groups) < 2:
                continue

            statistic, p_value = f_oneway(*groups)
            is_significant = bool(p_value < self.alpha)
            results.append({
                "Feature": col,
                "F-Statistic": float(statistic),
                "P-Value": float(p_value),
                f"Significant (α={self.alpha})": is_significant,
                "Decision": "Reject H0 (means differ)" if is_significant else "Fail to reject H0",
            })
            self._print_decision(p_value, feature=col)

        return pd.DataFrame(results)

    def perform_kruskal_wallis_test(
        self, columns: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Run Kruskal-Wallis test per numeric feature across label groups.

        Formal hypothesis:
            H0: All delay class groups come from the same distribution
                (medians are equal across On-time, Short delay, Long delay).
            H1: At least one group comes from a stochastically different distribution.

        Choice justification: Non-parametric alternative to one-way ANOVA.
        Appropriate when normality cannot be assumed (confirmed by Shapiro-Wilk).
        Does not require equal variances. Tests whether at least one group median
        is significantly different from the others.

        Args:
            columns: Optional subset of features to test.

        Returns:
            pd.DataFrame: Feature-level Kruskal-Wallis results.
        """
        self._require_dataset_mode()

        self._print_hypothesis(
            h0="All delay class groups come from the same distribution (medians are equal).",
            h1="At least one group comes from a stochastically different distribution.",
            test_name="Kruskal-Wallis H-Test (non-parametric)",
        )

        results = []
        for col in self._feature_list(columns):
            groups = self._groups_for_feature(col)
            if len(groups) < 2:
                continue

            statistic, p_value = kruskal(*groups)
            is_significant = bool(p_value < self.alpha)
            results.append({
                "Feature": col,
                "H-Statistic": float(statistic),
                "P-Value": float(p_value),
                f"Significant (α={self.alpha})": is_significant,
                "Decision": "Reject H0 (distributions differ)" if is_significant else "Fail to reject H0",
            })
            self._print_decision(p_value, feature=col)

        return pd.DataFrame(results)

    def perform_t_tests(
        self,
        columns: Optional[Sequence[str]] = None,
        equal_var: bool = False,
    ) -> pd.DataFrame:
        """Run pairwise independent t-tests between all label pairs per feature.

        Formal hypothesis (per pair):
            H0: The mean of the feature is equal between group A and group B (μA = μB).
            H1: The mean of the feature differs between group A and group B (μA ≠ μB).

        Choice justification: Welch's t-test (equal_var=False) does not assume equal
        variances between groups, making it more robust than Student's t-test when
        groups have different spreads (confirmed by Levene's test). Pairwise tests
        identify which specific class pairs drive the ANOVA/Kruskal-Wallis significance.

        Args:
            columns: Optional subset of features to test.
            equal_var: If False, uses Welch's t-test (recommended).

        Returns:
            pd.DataFrame: Pairwise t-test results per feature.
        """
        self._require_dataset_mode()
        labels_series = pd.Series(self.labels).dropna()
        unique_labels = labels_series.unique()

        self._print_hypothesis(
            h0="The mean of the feature is equal between the two delay class groups (μA = μB).",
            h1="The mean of the feature differs between the two delay class groups (μA ≠ μB).",
            test_name=f"Independent Welch's t-test (equal_var={equal_var}) — pairwise",
        )

        results = []
        for col in self._feature_list(columns):
            for label1, label2 in itertools.combinations(unique_labels, 2):
                data1 = self.data.loc[self.labels == label1, col].dropna().values
                data2 = self.data.loc[self.labels == label2, col].dropna().values

                if len(data1) == 0 or len(data2) == 0:
                    continue

                statistic, p_value = ttest_ind(data1, data2, equal_var=equal_var, nan_policy="omit")
                is_significant = bool(p_value < self.alpha)
                results.append({
                    "Feature": col,
                    "Group A": str(label1),
                    "Group B": str(label2),
                    "T-Statistic": float(statistic),
                    "P-Value": float(p_value),
                    f"Significant (α={self.alpha})": is_significant,
                    "Decision": (
                        f"Reject H0 ({label1} ≠ {label2})"
                        if is_significant
                        else f"Fail to reject H0 ({label1} = {label2})"
                    ),
                })

        return pd.DataFrame(results)

    def perform_levene_test(
        self, columns: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Run Levene's test for equality of variances per feature.

        Formal hypothesis:
            H0: The variances of the feature are equal across all delay class groups
                (σ²_OnTime = σ²_ShortDelay = σ²_LongDelay).
            H1: At least one group has a significantly different variance.

        Choice justification: Levene's test is a prerequisite check for ANOVA and
        the independent t-test, which assume homogeneity of variances. A significant
        result (reject H0) indicates heterogeneous variances, recommending the use
        of Welch's t-test instead of Student's t-test.

        Args:
            columns: Optional subset of features to test.

        Returns:
            pd.DataFrame: Feature-level Levene test results.
        """
        self._require_dataset_mode()

        self._print_hypothesis(
            h0="Variances are equal across all delay class groups (homoscedasticity).",
            h1="At least one group has a significantly different variance (heteroscedasticity).",
            test_name="Levene's Test for Equality of Variances",
        )

        results = []
        for col in self._feature_list(columns):
            groups = self._groups_for_feature(col)
            if len(groups) < 2:
                continue

            statistic, p_value = levene(*groups)
            is_significant = bool(p_value < self.alpha)
            results.append({
                "Feature": col,
                "Statistic": float(statistic),
                "P-Value": float(p_value),
                f"Unequal Variances (α={self.alpha})": is_significant,
                "Decision": "Reject H0 (variances differ)" if is_significant else "Fail to reject H0",
            })
            self._print_decision(p_value, feature=col)

        return pd.DataFrame(results)

    def test_airline_delays(self, arr_delay_col: str = "ARR_DELAY") -> pd.DataFrame:
        """Test hypothesis: certain airlines are systematically more delayed.

        Formal hypothesis:
            H0: The distribution of arrival delays is equal across all airlines
                — no airline is systematically more delayed than others.
            H1: At least one airline has a statistically different delay distribution
                — certain airlines are systematically more delayed.

        Test chosen: Kruskal-Wallis H-test.
        Justification: ARR_DELAY is highly skewed (confirmed by Shapiro-Wilk),
        violating ANOVA's normality assumption. Kruskal-Wallis is the appropriate
        non-parametric alternative for comparing distributions across multiple
        independent groups (airlines) without assuming normality.

        Returns:
            pd.DataFrame: Test result with statistic, p-value, and decision.
        """
        self._print_hypothesis(
            h0="The distribution of ARR_DELAY is equal across all airlines (no systematic differences).",
            h1="At least one airline has a significantly different delay distribution.",
            test_name="Kruskal-Wallis — Airline Systematic Delays",
        )

        if self.target_col not in self.data.columns or arr_delay_col not in self.data.columns:
            raise ValueError(f"Columns '{self.target_col}' and '{arr_delay_col}' required.")

        airline_groups = [
            group[arr_delay_col].dropna().values
            for _, group in self.data.groupby(self.target_col)
            if len(group) > 0
        ]

        statistic, p_value = kruskal(*airline_groups)
        is_significant = bool(p_value < self.alpha)
        self._print_decision(p_value)

        return pd.DataFrame([{
            "Test": "Kruskal-Wallis",
            "Hypothesis": "Airline systematic delays",
            "H-Statistic": float(statistic),
            "P-Value": float(p_value),
            f"Significant (α={self.alpha})": is_significant,
            "Decision": (
                "Reject H0 — airlines differ systematically in delays."
                if is_significant
                else "Fail to reject H0 — no significant difference between airlines."
            ),
        }])

    def test_time_of_day_impact(self, dep_delay_col: str = "DEP_DELAY") -> pd.DataFrame:
        """Test hypothesis: time of day has a significant impact on flight delays.

        Formal hypothesis:
            H0: The distribution of departure delays is equal across all time-of-day
                groups (Morning, Afternoon, Night) — time of day does not affect delays.
            H1: At least one time-of-day group has a significantly different delay
                distribution — time of day significantly impacts flight delays.

        Test chosen: Kruskal-Wallis H-test.
        Justification: Same rationale as airline test — delay distributions are
        heavily right-skewed, making non-parametric tests more appropriate.

        Returns:
            pd.DataFrame: Test result with statistic, p-value, and decision.
        """
        self._print_hypothesis(
            h0="Delay distributions are equal across Morning, Afternoon, and Night flights.",
            h1="At least one time-of-day group has a significantly different delay distribution.",
            test_name="Kruskal-Wallis — Time of Day Impact on Delays",
        )

        if self.target_col not in self.data.columns or dep_delay_col not in self.data.columns:
            print(f"   Warning: column(s) not found — skipping time-of-day test.")
            return pd.DataFrame()

        groups = [
            group[dep_delay_col].dropna().values
            for _, group in self.data.groupby(self.target_col)
            if len(group) > 0
        ]

        if len(groups) < 2:
            return pd.DataFrame()

        statistic, p_value = kruskal(*groups)
        is_significant = bool(p_value < self.alpha)
        self._print_decision(p_value)

        return pd.DataFrame([{
            "Test": "Kruskal-Wallis",
            "Hypothesis": "Time of day impact on delays",
            "H-Statistic": float(statistic),
            "P-Value": float(p_value),
            f"Significant (α={self.alpha})": is_significant,
            "Decision": (
                "Reject H0 — time of day significantly affects delays."
                if is_significant
                else "Fail to reject H0 — time of day does not significantly affect delays."
            ),
        }])

    def generate_summary_report(
        self,
        include_pairwise_ttests: bool = True,
        sample_size: int = 5000,
        columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Generate a dictionary with all bulk test outputs.

        Runs the full hypothesis testing battery:
            1. Shapiro-Wilk normality test (prerequisite for parametric tests).
            2. Levene's test (prerequisite for ANOVA / t-test variance assumption).
            3. One-way ANOVA (parametric group comparison).
            4. Kruskal-Wallis (non-parametric group comparison — recommended when
               normality is violated, as is typically the case for delay data).
            5. Pairwise Welch's t-tests (identify which specific class pairs differ).

        Returns:
            Dict[str, pd.DataFrame]: Named test results ready for CSV export.
        """
        self._require_dataset_mode()

        print("\n" + "=" * 60)
        print("HYPOTHESIS TESTING — FULL BATTERY")
        print("=" * 60)

        report = {
            "normality":      self.perform_normality_test(sample_size=sample_size, columns=columns),
            "levene":         self.perform_levene_test(columns=columns),
            "anova":          self.perform_anova_test(columns=columns),
            "kruskal_wallis": self.perform_kruskal_wallis_test(columns=columns),
        }

        if include_pairwise_ttests and len(pd.Series(self.labels).dropna().unique()) > 1:
            report["t_tests"] = self.perform_t_tests(columns=columns)

        print("\n" + "=" * 60)
        print("HYPOTHESIS TESTING COMPLETE")
        print("=" * 60 + "\n")

        return report