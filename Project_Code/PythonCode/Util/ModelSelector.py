"""Model Selection — Phase 3 of the Data Analytics Lifecycle.

This module implements Phase 3: examining candidate models suitable for the
flight delay prediction problem, and assessing the most appropriate model
validation strategy.

Tasks covered:
    1. Candidate model identification and justification.
    2. Baseline evaluation with train/test split.
    3. Cross-validation strategy assessment (k-fold, stratified, bootstrap).
    4. Comparison summary table.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    ShuffleSplit,
    train_test_split,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class ModelSelector:
    """Examine, compare, and select candidate models for flight delay classification.

    This class covers Phase 3 of the data analytics lifecycle:
        - Identify candidate models suitable for the problem.
        - Justify model choices based on dataset characteristics.
        - Compare validation strategies (train/test split, k-fold CV, bootstrap).
        - Produce a comparison table to guide Phase 4 model building.

    The flight delay problem is formulated as a 3-class classification task:
        - On-time    (ARR_DELAY < 15 min)
        - Short delay (15 <= ARR_DELAY <= 30 min)
        - Long delay  (ARR_DELAY > 30 min)
    """

    # Candidate models with brief justification for inclusion.
    # These are the models examined in Phase 3; a subset will be built in Phase 4.
    CANDIDATE_MODELS = {
        "Baseline (majority class)": {
            "model": DummyClassifier(strategy="most_frequent"),
            "justification": (
                "Reference baseline. Any useful model must outperform random/majority guessing. "
                "Provides a lower bound on acceptable performance."
            ),
            "suitable": True,
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs"),
            "justification": (
                "Linear probabilistic classifier. Interpretable coefficients allow understanding "
                "which features drive delay predictions. Fast training. Suitable as a strong "
                "linear baseline. Requires feature scaling (already applied). "
                "Limitation: assumes linear decision boundaries."
            ),
            "suitable": True,
        },
        "k-Nearest Neighbors (kNN)": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "justification": (
                "Non-parametric, instance-based learner. No training phase — predictions are "
                "based on similarity to neighbors. Chosen because it will be implemented from "
                "scratch using NumPy in Phase 4 (project requirement). Works well with scaled "
                "features. Limitation: slow prediction on large datasets; sensitive to irrelevant "
                "features."
            ),
            "suitable": True,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=10, random_state=42),
            "justification": (
                "Rule-based classifier. Highly interpretable (visualizable tree). Handles both "
                "numeric and categorical features. Useful for understanding decision boundaries "
                "in delay data. Limitation: prone to overfitting without pruning."
            ),
            "suitable": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "justification": (
                "Ensemble of decision trees using bagging. Reduces overfitting by averaging "
                "predictions from many diverse trees. Provides feature importance scores. "
                "Strong performer on tabular data. Suitable as the bagging ensemble in Phase 4."
            ),
            "suitable": True,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "justification": (
                "Sequential ensemble where each tree corrects errors from the previous one. "
                "Typically achieves the best performance on tabular datasets. Suitable as "
                "the boosting ensemble in Phase 4. Limitation: longer training time."
            ),
            "suitable": True,
        },
        "Linear SVM": {
            "model": LinearSVC(max_iter=2000, random_state=42),
            "justification": (
                "Finds the maximum-margin hyperplane between classes. Effective in "
                "high-dimensional feature spaces. Requires feature scaling. "
                "Limitation: does not natively produce probability estimates."
            ),
            "suitable": True,
        },
    }

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "DELAY_CLASS",
        output_dir: str = "Output_Files",
        random_state: int = 42,
        cv_folds: int = 5,
        bootstrap_n: int = 100,
    ):
        """Initialize ModelSelector.

        Args:
            data: Preprocessed and encoded dataset (output of FlightFeatureEngineer).
            target_col: Name of the classification target column.
            output_dir: Directory to save output plots and CSVs.
            random_state: Seed for reproducibility.
            cv_folds: Number of folds for k-fold cross-validation.
            bootstrap_n: Number of bootstrap iterations.
        """
        self.data = data.copy()
        self.target_col = target_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.bootstrap_n = bootstrap_n

        self.X, self.y = self._prepare_features()
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y.astype(str))

        self._results: List[Dict] = []

    def _prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix and target vector from dataset."""
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data.")

        # Exclude the target, and also ARR_DELAY (the regression target).
        exclude = [self.target_col, "ARR_DELAY"]
        feature_cols = [
            c for c in self.data.columns
            if c not in exclude and self.data[c].dtype != object
        ]
        X = self.data[feature_cols].fillna(0)
        y = self.data[self.target_col]
        return X, y

    # ---------------------------
    # Phase 3.1 — Model catalogue
    # ---------------------------
    def print_candidate_models(self) -> None:
        """Print the catalogue of candidate models with justifications."""
        print("\n" + "=" * 70)
        print("PHASE 3 — CANDIDATE MODEL EXAMINATION")
        print("=" * 70)
        print(
            "Problem type : Multi-class classification (3 classes)\n"
            "Target       : DELAY_CLASS — On-time | Short delay | Long delay\n"
            "Dataset size : Large (100k+ rows) — favour scalable models\n"
            "Features     : Mix of numeric and encoded categorical\n"
        )
        for name, info in self.CANDIDATE_MODELS.items():
            status = "✓ Suitable" if info["suitable"] else "✗ Not recommended"
            print(f"  [{status}] {name}")
            for line in info["justification"].split(". "):
                print(f"    {line.strip()}.")
            print()
        print("=" * 70 + "\n")

    # ---------------------------
    # Phase 3.2 — Validation strategies
    # ---------------------------
    def assess_validation_strategies(self) -> pd.DataFrame:
        """Compare validation strategies: train/test split, k-fold CV, bootstrap.

        Validation strategy rationale:
            - Train/Test Split (80/20): fast, simple baseline evaluation. Suitable for
              large datasets where a held-out set is statistically representative.
              Limitation: high variance for small datasets; result depends on split.

            - Stratified K-Fold CV (k=5): partitions data into k folds, trains on k-1
              and tests on 1, repeating k times. Stratification preserves class balance
              in each fold. Provides lower-variance performance estimate than a single
              split. Recommended for this problem given the 3-class imbalance.

            - Bootstrap (n=100): samples with replacement; unselected rows (~36.8%) form
              the test set. Provides confidence intervals on performance. Computationally
              expensive but more informative for small datasets.

        Returns:
            pd.DataFrame: Comparison of validation strategy results per model.
        """
        print("\n" + "=" * 70)
        print("PHASE 3 — VALIDATION STRATEGY ASSESSMENT")
        print("=" * 70)

        # Use a fast subset of models for strategy comparison.
        strategy_models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=self.random_state),
            "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=self.random_state),
            "Random Forest":       RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1),
        }

        results = []
        X_arr = self.X.values
        y_arr = self.y_encoded

        for model_name, model in strategy_models.items():
            print(f"\n  Evaluating: {model_name}")

            # --- Strategy 1: Train/Test Split (80/20) ---
            X_train, X_test, y_train, y_test = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=self.random_state, stratify=y_arr
            )
            model.fit(X_train, y_train)
            split_acc = accuracy_score(y_test, model.predict(X_test))
            split_f1 = f1_score(y_test, model.predict(X_test), average="weighted")

            # --- Strategy 2: Stratified K-Fold CV ---
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_arr, y_arr, cv=skf, scoring="accuracy", n_jobs=-1)

            # --- Strategy 3: Bootstrap (limited iterations for speed) ---
            bootstrap_iters = min(self.bootstrap_n, 20)
            bs = ShuffleSplit(n_splits=bootstrap_iters, test_size=0.368, random_state=self.random_state)
            bs_scores = cross_val_score(model, X_arr, y_arr, cv=bs, scoring="accuracy", n_jobs=-1)

            results.append({
                "Model": model_name,
                "Split Accuracy": round(split_acc, 4),
                "Split F1 (weighted)": round(split_f1, 4),
                f"CV Accuracy (mean, {self.cv_folds}-fold)": round(np.mean(cv_scores), 4),
                f"CV Accuracy (std, {self.cv_folds}-fold)": round(np.std(cv_scores), 4),
                "Bootstrap Accuracy (mean)": round(np.mean(bs_scores), 4),
                "Bootstrap Accuracy (std)": round(np.std(bs_scores), 4),
            })
            print(f"    Train/Test split  : Acc={split_acc:.4f}, F1={split_f1:.4f}")
            print(f"    {self.cv_folds}-Fold CV         : Acc={np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            print(f"    Bootstrap ({bootstrap_iters} iter): Acc={np.mean(bs_scores):.4f} ± {np.std(bs_scores):.4f}")

        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "model_selection_validation_strategies.csv", index=False)

        print(f"\n  Recommended validation strategy: Stratified {self.cv_folds}-Fold Cross-Validation")
        print(f"  Justification: provides lower-variance estimates than a single split,")
        print(f"  preserves class balance in each fold, and is computationally feasible")
        print(f"  for the dataset size. Bootstrap is reserved for confidence intervals.")

        print("\n" + "=" * 70 + "\n")
        return df

    # ---------------------------
    # Phase 3.3 — Model comparison table
    # ---------------------------
    def compare_candidate_models(
        self,
        max_samples: int = 50_000,
    ) -> pd.DataFrame:
        """Evaluate all candidate models with stratified CV and produce a comparison table.

        Args:
            max_samples: Subsample size for speed (set to None to use full dataset).

        Returns:
            pd.DataFrame: Ranked model comparison table sorted by CV accuracy.
        """
        print("\n" + "=" * 70)
        print("PHASE 3 — CANDIDATE MODEL COMPARISON TABLE")
        print("=" * 70)

        X_arr = self.X.values
        y_arr = self.y_encoded

        # Subsample for speed during model selection phase.
        if max_samples and len(X_arr) > max_samples:
            idx = np.random.RandomState(self.random_state).choice(len(X_arr), max_samples, replace=False)
            X_arr = X_arr[idx]
            y_arr = y_arr[idx]
            print(f"  (Using {max_samples:,} samples for speed — full dataset used in Phase 4)\n")

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        results = []

        for name, info in self.CANDIDATE_MODELS.items():
            model = info["model"]
            print(f"  Evaluating: {name}...")
            try:
                cv_acc = cross_val_score(model, X_arr, y_arr, cv=skf, scoring="accuracy", n_jobs=-1)
                cv_f1  = cross_val_score(model, X_arr, y_arr, cv=skf, scoring="f1_weighted", n_jobs=-1)
                results.append({
                    "Model": name,
                    "CV Accuracy (mean)": round(np.mean(cv_acc), 4),
                    "CV Accuracy (std)": round(np.std(cv_acc), 4),
                    "CV F1 Weighted (mean)": round(np.mean(cv_f1), 4),
                    "CV F1 Weighted (std)": round(np.std(cv_f1), 4),
                    "Recommended for Phase 4": info["suitable"],
                    "Justification (brief)": info["justification"][:80] + "...",
                })
                print(f"    Acc={np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f} | "
                      f"F1={np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
            except Exception as exc:
                print(f"    Error: {exc}")
                results.append({
                    "Model": name,
                    "CV Accuracy (mean)": np.nan,
                    "CV Accuracy (std)": np.nan,
                    "CV F1 Weighted (mean)": np.nan,
                    "CV F1 Weighted (std)": np.nan,
                    "Recommended for Phase 4": False,
                    "Justification (brief)": str(exc),
                })

        df = pd.DataFrame(results).sort_values("CV Accuracy (mean)", ascending=False).reset_index(drop=True)
        df.to_csv(self.output_dir / "model_selection_comparison.csv", index=False)

        print("\n  Model ranking (by CV accuracy):")
        for _, row in df.iterrows():
            print(f"    {row['Model']:35s} Acc={row['CV Accuracy (mean)']:.4f}")

        print(f"\n  Results saved: model_selection_comparison.csv")
        print("=" * 70 + "\n")

        self._plot_model_comparison(df)
        return df

    def _plot_model_comparison(self, df: pd.DataFrame) -> None:
        """Plot a bar chart comparing model CV accuracy."""
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(df))
        bars = ax.bar(x, df["CV Accuracy (mean)"], yerr=df["CV Accuracy (std)"],
                      capsize=4, color="#378ADD", alpha=0.85, edgecolor="#185FA5")
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["Model"], rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("CV Accuracy (mean ± std)", fontsize=11)
        ax.set_title("Phase 3 — Candidate Model Comparison (Stratified 5-Fold CV)", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=df.loc[df["Model"] == "Baseline (majority class)", "CV Accuracy (mean)"].values[0]
                   if "Baseline (majority class)" in df["Model"].values else 0,
                   color="#E24B4A", linestyle="--", linewidth=1.2, label="Baseline")
        ax.legend()
        ax.grid(axis="y", alpha=0.4)
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_selection_comparison.png", dpi=100,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        print("  Plot saved: model_selection_comparison.png")

    # ---------------------------
    # Phase 3 — Full run
    # ---------------------------
    def run_model_selection(self) -> Dict[str, pd.DataFrame]:
        """Run the complete Phase 3 model selection workflow.

        Steps:
            1. Print candidate model catalogue with justifications.
            2. Assess and compare validation strategies.
            3. Produce ranked model comparison table.

        Returns:
            Dict with 'validation_strategies' and 'model_comparison' DataFrames.
        """
        print("\n" + "=" * 70)
        print("STARTING PHASE 3 — MODEL SELECTION")
        print("=" * 70)

        self.print_candidate_models()
        validation_df = self.assess_validation_strategies()
        comparison_df = self.compare_candidate_models()

        print("\n" + "=" * 70)
        print("PHASE 3 — MODEL SELECTION COMPLETE")
        print("Models recommended for Phase 4:")
        print("  → kNN (implemented from scratch)")
        print("  → Logistic Regression + Decision Tree (supervised learning)")
        print("  → Random Forest (bagging ensemble)")
        print("  → Gradient Boosting (boosting ensemble)")
        print("  → Deep Learning model (Phase 4 requirement)")
        print("  Validation strategy: Stratified 5-Fold Cross-Validation")
        print("=" * 70 + "\n")

        return {
            "validation_strategies": validation_df,
            "model_comparison": comparison_df,
        }