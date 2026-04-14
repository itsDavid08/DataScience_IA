import argparse
import logging
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    project_root = Path(__file__).resolve()
    while project_root != project_root.parent and project_root.name != "DataScience_IA":
        project_root = project_root.parent
    if project_root.name != "DataScience_IA":
        raise FileNotFoundError("Could not locate project root folder 'DataScience_IA'.")
    return project_root


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Project_Code.PythonCode.DataPreProcessor.FlightDataCleaner import FlightDataCleaner
from Project_Code.PythonCode.EDA.FlightEDA import FlightEDA
from Project_Code.PythonCode.FeatureEngeneering.FlightFeatureEngineer import FlightFeatureEngineer
from Project_Code.PythonCode.HypothesisTesting.HypothesisTester import HypothesisTester
from Project_Code.PythonCode.Util.ModelSelector import ModelSelector
from Project_Code.PythonCode.Util.DataLoader import DataLoader
from Project_Code.PythonCode.Util.DataVisualization import DataVisualization


def _build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("part1_pipeline")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DataScience_IA Part 1 pipeline.")
    parser.add_argument("--nrows", type=int, default=10000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset-path", type=str, default="DataSet/flights_sample_3m.csv")
    parser.add_argument("--output-dir", type=str, default="Output_Files")
    parser.add_argument("--balance-method", type=str, default="smote",
                        choices=["smote", "oversample", "undersample"],
                        help="Balancing strategy for the cleaner.")
    parser.add_argument("--scale-method", type=str, default="both",
                        choices=["standard", "minmax", "both"],
                        help="Feature scaling strategy.")
    parser.add_argument("--skip-umap-tsne", action="store_true")
    parser.add_argument("--skip-hypothesis", action="store_true")
    parser.add_argument("--skip-model-selection", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = _build_logger(output_dir / "pipeline_part1_notebook.log")
    logger.info("=" * 80)
    logger.info("STARTING PART 1 PIPELINE")
    logger.info("=" * 80)

    ############################################################
    ##            PHASE 2 — DATA LOADING AND CLEANSING        ##
    ############################################################
    logger.info("\nPHASE 2: DATA LOADING AND CLEANSING")

    logger.info("[STEP 1] Loading dataset and splitting train/test...")
    loader = DataLoader(str(dataset_path), test_size=args.test_size, random_state=args.random_state)
    data_train, data_test, target_train, target_test = loader.load_data(nrows=args.nrows)
    logger.info("Train shape: %s | Test shape: %s", data_train.shape, data_test.shape)

    logger.info("[STEP 2] Cleaning raw data...")
    cleaner = FlightDataCleaner(file_path=str(dataset_path))
    # FIX: balance_method now defaults to 'smote' using imblearn.
    df_clean = cleaner.load_and_clean(nrows=args.nrows, balance_method=args.balance_method)
    df_clean.to_csv(PROJECT_ROOT / "DataSet/cleaned_flight_data.csv", index=False)
    logger.info("Clean shape: %s", df_clean.shape)

    logger.info("[STEP 3] Creating classification target (DELAY_CLASS)...")
    df_clean = cleaner.classify_target()
    logger.info("DELAY_CLASS distribution:\n%s", df_clean["DELAY_CLASS"].value_counts().to_string())

    ############################################################
    ##                    PHASE 4 — EDA                       ##
    ############################################################
    logger.info("\nPHASE 4: EDA AND DIMENSIONALITY REDUCTION")

    logger.info("[STEP 4] Running full EDA...")
    eda = FlightEDA(df_clean, target_col="ARR_DELAY", output_dir=output_dir, group_col="DELAY_CLASS")
    eda_report = eda.perform_eda()
    logger.info("Missing values total: %s", int(eda_report["quality"]["missing_values"].sum()))
    logger.info("Duplicate rows: %s", int(eda_report["quality"]["duplicate_count"]))

    logger.info("[STEP 5] Running PCA (linear reduction)...")
    pca_components = eda.run_pca(n_components=2)
    logger.info("PCA shape: %s", pca_components.shape)

    if not args.skip_umap_tsne:
        logger.info("[STEP 6] Running UMAP/t-SNE (non-linear reduction)...")
        umap_components = eda.run_umap_or_tsne(n_components=2, use_umap=True)
        logger.info("UMAP/t-SNE shape: %s", umap_components.shape)

    logger.info("[STEP 7] Generating visual diagnostics...")
    viz = DataVisualization(df_clean, output_dir=output_dir)
    viz.plot_heatmap_top_correlations(top_n=20)
    if "DELAY_CLASS" in df_clean.columns:
        focus_cols = ["DISTANCE", "CRS_ELAPSED_TIME", "PLANNED_SPEED_MPM", "ARR_DELAY"]
        focus_cols = [c for c in focus_cols if c in df_clean.columns]
        if focus_cols:
            viz.plot_grouped_feature_distributions(
                columns=focus_cols, group_col="DELAY_CLASS",
                filename="viz_grouped_distributions_delay_class.png",
            )
            viz.plot_grouped_boxplots(
                columns=focus_cols, group_col="DELAY_CLASS",
                filename="viz_grouped_boxplots_delay_class.png",
            )

    ############################################################
    ##             PHASE 3 — FEATURE ENGINEERING              ##
    ############################################################
    logger.info("\nPHASE 3: FEATURE ENGINEERING")

    logger.info("[STEP 8] Generating engineered features...")
    engineer = FlightFeatureEngineer(df_clean)
    df_features = engineer.generate_features()

    # FIX: Use appropriate encoding per variable type.
    # Nominal variables (AIRLINE, AIRLINE_CODE, TIME_PERIOD) → One-Hot Encoding.
    # Ordinal / high-cardinality variables → Label Encoding.
    logger.info("[STEP 9] Encoding categorical variables (OHE + Label Encoding)...")
    df_features = engineer.encode_categorical()

    # FIX: Apply both StandardScaler and MinMaxScaler; use StandardScaler as default.
    logger.info("[STEP 10] Normalizing numeric features (StandardScaler + MinMaxScaler)...")
    df_features = engineer.normalize_features(method=args.scale_method)
    logger.info("Features shape: %s | DELAY_CLASS present: %s",
                df_features.shape, "DELAY_CLASS" in df_features.columns)

    logger.info("[STEP 11] Saving cleaned+features checkpoint...")
    loader.data = df_features
    checkpoint_clean_path = dataset_path.parent / "checkpoint_cleaned_features.pkl"
    loader.save_checkpoint(str(checkpoint_clean_path))

    ############################################################
    ##          PHASE 3 — MODEL SELECTION                     ##
    ############################################################
    if not args.skip_model_selection:
        logger.info("\nPHASE 3: MODEL SELECTION")
        logger.info("[STEP 12] Running model selection and validation strategy assessment...")
        selector = ModelSelector(
            data=df_features,
            target_col="DELAY_CLASS",
            output_dir=output_dir,
            random_state=args.random_state,
            cv_folds=5,
        )
        selection_report = selector.run_model_selection()

        for name, df_result in selection_report.items():
            out_csv = output_dir / f"model_selection_{name}.csv"
            df_result.to_csv(out_csv, index=False)
            logger.info("Saved: %s", out_csv.name)

    ############################################################
    ##            PHASE 5 — HYPOTHESIS TESTING                ##
    ############################################################
    if not args.skip_hypothesis:
        logger.info("\nPHASE 5: HYPOTHESIS TESTING")
        logger.info("[STEP 13] Running statistical tests with formal H0/H1 formulation...")

        # Main battery: normality, ANOVA, Kruskal-Wallis, Levene, pairwise t-tests.
        hypothesis_tester = HypothesisTester(
            data=df_features,
            target_col="DELAY_CLASS",
            verbose=True,
        )
        summary_report = hypothesis_tester.generate_summary_report()

        # Specific hypothesis: time of day impact on delays.
        logger.info("   -> Hypothesis: Time of Day Impact on Delays")
        if "TIME_PERIOD" in df_features.columns and "DEP_DELAY" in df_features.columns:
            tester_time = HypothesisTester(data=df_features, target_col="TIME_PERIOD", verbose=True)
            summary_report["time_period_impact"] = tester_time.test_time_of_day_impact()
        else:
            logger.info("   (TIME_PERIOD or DEP_DELAY not found — skipping time-of-day test)")

        # Specific hypothesis: airline systematic delays.
        logger.info("   -> Hypothesis: Airline Systematic Delays")
        if "AIRLINE" in df_features.columns and "ARR_DELAY" in df_features.columns:
            tester_airline = HypothesisTester(data=df_features, target_col="AIRLINE", verbose=True)
            summary_report["airline_systemic_delays"] = tester_airline.test_airline_delays()
        else:
            logger.info("   (AIRLINE or ARR_DELAY not found — skipping airline test)")

        logger.info("[STEP 14] Exporting statistical reports...")
        for test_name, results_df in summary_report.items():
            if results_df is None or results_df.empty:
                continue
            out_csv = output_dir / f"hypothesis_test_{test_name}.csv"
            results_df.to_csv(out_csv, index=False)
            logger.info("Report saved: %s", out_csv.name)

    logger.info("[STEP 15] Saving final checkpoint...")
    loader.data = df_features
    checkpoint_final_path = dataset_path.parent / "checkpoint_part1_complete.pkl"
    loader.save_checkpoint(str(checkpoint_final_path))

    logger.info("=" * 80)
    logger.info("PIPELINE PART 1 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()