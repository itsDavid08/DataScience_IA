import argparse
import logging
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    """Locate the DataScience_IA root without hardcoded absolute paths."""
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
    parser.add_argument("--nrows", type=int, default=10000, help="Rows to load from dataset (None not supported via CLI).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Train/test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split and deterministic ops.")
    parser.add_argument("--dataset-path", type=str, default="DataSet/flights_sample_3m.csv", help="Dataset CSV path, relative to project root unless absolute.")
    parser.add_argument("--output-dir", type=str, default="Output_Files", help="Artifacts output directory.")
    parser.add_argument("--skip-umap-tsne", action="store_true", help="Skip non-linear reduction stage.")
    parser.add_argument("--skip-hypothesis", action="store_true", help="Skip hypothesis testing and CSV exports.")
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
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Dataset path: %s", dataset_path)
    logger.info("Output dir: %s", output_dir)


    ############################################################
    ##                 DATA LOADING AND CLEANSING             ##
    ############################################################

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: DATA LOADING AND CLEANSING")
    logger.info("=" * 80)

    logger.info("[STEP 1] Loading dataset and splitting train/test...")
    loader = DataLoader(str(dataset_path), test_size=args.test_size, random_state=args.random_state)
    data_train, data_test, target_train, target_test = loader.load_data(nrows=args.nrows)
    logger.info("Train shape: %s", data_train.shape)
    logger.info("Test shape: %s", data_test.shape)
    logger.info("Target train shape: %s", target_train.shape if target_train is not None else None)
    logger.info("Target test shape: %s", target_test.shape if target_test is not None else None)

    logger.info("[STEP 2] Cleaning raw data...")
    cleaner = FlightDataCleaner(file_path=str(dataset_path))
    df_clean = cleaner.load_and_clean(nrows=args.nrows)
    df_clean.to_csv("DataSet/cleaned_flight_data.csv", index=False)
    logger.info("Clean shape: %s", df_clean.shape)
    numeric_missing = int(df_clean.select_dtypes(include=["number"]).isnull().sum().sum())
    logger.info("Numeric missing values after cleaning: %s", numeric_missing)


    ############################################################
    ##                          EDA                           ##
    ############################################################

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: EDA AND DIMENSIONALITY REDUCTION")
    logger.info("=" * 80)

    logger.info("[STEP 7] Running full EDA...")
    eda = FlightEDA(df_features, target_col="ARR_DELAY", output_dir=output_dir, group_col="DELAY_CLASS")
    eda_report = eda.perform_eda()
    logger.info("Missing values total: %s", int(eda_report["quality"]["missing_values"].sum()))
    logger.info("Duplicate rows: %s", int(eda_report["quality"]["duplicate_count"]))

    logger.info("[STEP 8] Running PCA...")
    pca_components = eda.run_pca(n_components=2, explained_variance_threshold=0.8)
    logger.info("PCA shape: %s", pca_components.shape)

    if not args.skip_umap_tsne:
        logger.info("[STEP 9] Running UMAP/t-SNE...")
        umap_components = eda.run_umap_or_tsne(n_components=2, use_umap=True)
        logger.info("UMAP/t-SNE shape: %s", umap_components.shape)

    logger.info("[STEP 10] Generating additional visual diagnostics...")
    viz = DataVisualization(df_features, output_dir=output_dir)
    viz.plot_heatmap_top_correlations(top_n=20)

    if "DELAY_CLASS" in df_features.columns:
        focus_cols = ["DISTANCE", "CRS_ELAPSED_TIME", "PLANNED_SPEED_MPM", "ARR_DELAY"]
        viz.plot_grouped_feature_distributions(
            columns=focus_cols,
            group_col="DELAY_CLASS",
            filename="viz_grouped_distributions_delay_class.png",
        )
        viz.plot_grouped_boxplots(
            columns=focus_cols,
            group_col="DELAY_CLASS",
            filename="viz_grouped_boxplots_delay_class.png",
        )

    ############################################################
    ##                  Feature Engineering                   ##
    ############################################################

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("=" * 80)

    logger.info("[STEP 3] Generating engineered features...")
    engineer = FlightFeatureEngineer(df_clean)
    df_features = engineer.generate_features()

    logger.info("[STEP 4] Encoding categorical variables...")
    df_features = engineer.encode_categorical()

    logger.info("[STEP 5] Normalizing numeric features...")
    df_features = engineer.normalize_features()
    logger.info("Features shape: %s", df_features.shape)
    logger.info("Total columns: %s", len(df_features.columns))
    logger.info("DELAY_CLASS present: %s", "DELAY_CLASS" in df_features.columns)

    logger.info("[STEP 6] Saving cleaned+features checkpoint...")
    loader.data = df_features
    checkpoint_clean_path = dataset_path.parent / "checkpoint_cleaned_features.pkl"
    loader.save_checkpoint(str(checkpoint_clean_path))

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: HYPOTHESIS TESTING AND FINAL CHECKPOINT")
    logger.info("=" * 80)

    if not args.skip_hypothesis:
        logger.info("[STEP 11] Running statistical tests...")
        hypothesis_tester = HypothesisTester(
            data=df_features,
            labels=df_features["DELAY_CLASS"],
            target_col="DELAY_CLASS",
            verbose=False,
        )
        summary_report = hypothesis_tester.generate_summary_report()

        logger.info("[STEP 12] Exporting statistical reports...")
        for test_name, results_df in summary_report.items():
            if results_df is None:
                continue
            out_csv = output_dir / f"hypothesis_testing_{test_name}.csv"
            results_df.to_csv(out_csv, index=False)
            logger.info("Report saved: %s", out_csv.name)

    logger.info("[STEP 13] Saving final checkpoint...")
    loader.data = df_features
    checkpoint_final_path = dataset_path.parent / "checkpoint_part1_complete.pkl"
    loader.save_checkpoint(str(checkpoint_final_path))

    logger.info("=" * 80)
    logger.info("PIPELINE PART 1 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
