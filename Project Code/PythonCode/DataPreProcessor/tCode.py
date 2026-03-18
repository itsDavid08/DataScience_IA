#!/usr/bin/env python
"""
Module: dataset_analyzer
Description: Provides classes for dataset analysis including general analysis,
numeric-specific analysis, and categorical-specific analysis. It also includes
a suite of unit tests to ensure proper functionality.
Author: Data Science
Date: 2025-01-15
"""

import pandas as pd
import numpy as np
from typing import Union, Dict
import logging
import unittest

# Set up logging for demonstration purposes.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """
    A master class for dataset analysis, providing general operations such as
    summary statistics, missing value handling, and data type validation.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the dataset analyzer.

        Parameters:
            df (pd.DataFrame): The dataset to be analyzed.

        Raises:
            ValueError: If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        # Work on a copy to avoid modifying the original DataFrame.
        self.df = df.copy()

    def __len__(self) -> int:
        """Returns the number of rows in the dataset."""
        return len(self.df)

    def __getitem__(self, column: str) -> pd.Series:
        """
        Returns a column from the dataset.

        Parameters:
            column (str): The column name to retrieve.

        Raises:
            KeyError: If the column does not exist in the dataset.

        Returns:
            pd.Series: The data for the specified column.
        """
        if column not in self.df.columns:
            raise KeyError(f"Column '{column}' not found in dataset")
        return self.df[column]

    def __setitem__(self, column: str, value) -> None:
        """
        Sets a column in the dataset.

        Parameters:
            column (str): The column name.
            value: The data to assign to the column.
        """
        self.df[column] = value

    def __repr__(self) -> str:
        """Returns a string representation of the dataset."""
        return f"DatasetAnalyzer({len(self.df)} rows, {len(self.df.columns)} columns)"

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Returns the first n rows of the dataset.

        Parameters:
            n (int, optional): Number of rows to return. Defaults to 5.

        Returns:
            pd.DataFrame: The first n rows of the dataset.
        """
        return self.df.head(n)

    def fill_missing(self, strategy: str = 'mean',
                     value: Union[int, float, str, None] = None) -> None:
        """
        Fills missing values using a specified strategy (mean, median, mode, or constant).

        Parameters:
            strategy (str): Strategy to fill missing values ('mean', 'median', 'mode', or 'constant').
            value (Union[int, float, str, None]): A custom value for the 'constant' strategy.

        Raises:
            ValueError: If an unsupported strategy is provided or if value is not provided for 'constant'.
        """
        valid_strategies = ['mean', 'median', 'mode', 'constant']
        if strategy not in valid_strategies:
            raise ValueError("Strategy must be one of 'mean', 'median', 'mode', or 'constant'")

        if strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            means = self.df[numeric_cols].mean()
            self.df[numeric_cols] = self.df[numeric_cols].fillna(means)
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            medians = self.df[numeric_cols].median()
            self.df[numeric_cols] = self.df[numeric_cols].fillna(medians)
        elif strategy == 'mode':
            for col in self.df.columns:
                mode_val = self.df[col].mode()
                if not mode_val.empty:
                    self.df[col].fillna(mode_val.iloc[0], inplace=True)
        elif strategy == 'constant':
            if value is None:
                raise ValueError("Must provide a value for constant filling")
            self.df.fillna(value, inplace=True)

    def assess_outliers(self, method: str = 'zscore', threshold: float = 3.0) -> Dict[str, int]:
        """
        Detects outliers in numerical columns using either the Z-score or IQR method.

        Parameters:
            method (str): The method to use for outlier detection ('zscore' or 'iqr').
            threshold (float): The threshold to define an outlier.

        Returns:
            dict: A dictionary with numerical column names as keys and outlier counts as values.

        Raises:
            ValueError: If the method is not 'zscore' or 'iqr'.
        """
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number])
        for col in numeric_cols:
            if method == 'zscore':
                std = self.df[col].std()
                # Avoid division by zero in case of constant columns.
                if std == 0:
                    outliers[col] = 0
                    continue
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / std)
                outliers[col] = (z_scores > threshold).sum()
            elif method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            else:
                raise ValueError("Method must be 'zscore' or 'iqr'")
        return outliers

    def handle_missing_values(self, method: str = 'drop',
                              fill_value: Union[int, float, str, None] = None) -> None:
        """
        Handles missing values using the specified method.

        Parameters:
            method (str): 'drop' to remove rows with missing values, or 'fill' to replace them.
            fill_value (int | float | str, optional): The value to replace missing values with if method is 'fill'.

        Raises:
            ValueError: If method is not 'drop' or 'fill', or if fill_value is not provided when needed.
        """
        if method not in ['drop', 'fill']:
            raise ValueError("Method must be 'drop' or 'fill'")

        if method == 'drop':
            self.df.dropna(inplace=True)
        elif method == 'fill':
            if fill_value is None:
                raise ValueError("fill_value must be provided when method='fill'")
            self.df.fillna(fill_value, inplace=True)

    def save(self, filename: str) -> None:
        """
        Saves the dataset to a CSV or Excel file.

        Parameters:
            filename (str): The file path. Must end with '.csv' or '.xlsx'.

        Raises:
            ValueError: If the file format is unsupported.
        """
        if filename.endswith('.csv'):
            self.df.to_csv(filename, index=False)
        elif filename.endswith('.xlsx'):
            self.df.to_excel(filename, index=False)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

    def get_dtype_info(self) -> pd.Series:
        """
        Returns the data types for all columns in the dataset.

        Returns:
            pd.Series: A series with column names as index and data types as values.
        """
        return self.df.dtypes

    def get_summary(self) -> pd.DataFrame:
        """
        Returns summary statistics for the dataset.

        Returns:
            pd.DataFrame: A table of summary statistics including count, mean, std, min, and percentiles.
        """
        return self.df.describe(include='all')


class NumericDatasetAnalyzer(DatasetAnalyzer):
    """
    A specialized class for analyzing numeric datasets. It filters out non-numeric data
    and provides methods such as computing the correlation matrix.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the NumericDatasetAnalyzer by filtering to keep only numeric columns.

        Parameters:
            df (pd.DataFrame): The dataset to analyze.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        super().__init__(numeric_df)

    def correlation_matrix(self) -> pd.DataFrame:
        """
        Computes the correlation matrix of the numeric columns.

        Returns:
            pd.DataFrame: The correlation matrix.
        """
        return self.df.corr()


class CategoricalDatasetAnalyzer(DatasetAnalyzer):
    """
    A specialized class for analyzing categorical datasets. It filters out numeric data
    and provides frequency distribution analysis.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the CategoricalDatasetAnalyzer by filtering to keep only non-numeric columns.

        Parameters:
            df (pd.DataFrame): The dataset to analyze.
        """
        categorical_df = df.select_dtypes(exclude=[np.number])
        super().__init__(categorical_df)

    def get_frequencies(self) -> Dict[str, pd.Series]:
        """
        Computes frequency distributions for each categorical column.

        Returns:
            dict: A dictionary with column names as keys and frequency counts as values.
        """
        return {col: self.df[col].value_counts() for col in self.df.columns}


# =============================================================================
# Unit Tests
# =============================================================================

class TestDatasetAnalyzer(unittest.TestCase):
    """Unit tests for DatasetAnalyzer and its general functionality."""

    def setUp(self) -> None:
        """Set up a sample DataFrame for testing."""
        data = {
            'A': [1, 2, 3, np.nan, 5],
            'B': [5, np.nan, 7, 8, 9],
            'C': ['foo', 'bar', 'foo', 'bar', 'foo']
        }
        self.df = pd.DataFrame(data)
        self.analyzer = DatasetAnalyzer(self.df)

    def test_len(self) -> None:
        """Test that __len__ returns the correct number of rows."""
        self.assertEqual(len(self.analyzer), 5)

    def test_getitem(self) -> None:
        """Test retrieving a column using __getitem__."""
        pd.testing.assert_series_equal(self.analyzer['A'], self.df['A'])
        with self.assertRaises(KeyError):
            _ = self.analyzer['D']

    def test_setitem(self) -> None:
        """Test setting a new column using __setitem__."""
        self.analyzer['D'] = [10, 20, 30, 40, 50]
        self.assertTrue('D' in self.analyzer.df.columns)
        pd.testing.assert_series_equal(self.analyzer['D'],
                                         pd.Series([10, 20, 30, 40, 50], name='D'))

    def test_fill_missing_mean(self) -> None:
        """Test fill_missing using the 'mean' strategy."""
        self.analyzer.fill_missing(strategy='mean')
        self.assertFalse(self.analyzer.df['A'].isna().any())
        self.assertFalse(self.analyzer.df['B'].isna().any())

    def test_fill_missing_constant(self) -> None:
        """Test fill_missing using the 'constant' strategy."""
        self.analyzer.fill_missing(strategy='constant', value=0)
        self.assertFalse(self.analyzer.df.isna().any().any())

    def test_assess_outliers_zscore(self) -> None:
        """Test assess_outliers using the 'zscore' method."""
        # Create a DataFrame with a clear outlier.
        df_outlier = pd.DataFrame({'num': [1, 2, 3, 4, 100]})
        analyzer = DatasetAnalyzer(df_outlier)
        # Lower the threshold to 1.7 to detect the outlier in this dataset.
        outliers = analyzer.assess_outliers(method='zscore', threshold=1.7)
        self.assertGreater(outliers['num'], 0)

    def test_handle_missing_drop(self) -> None:
        """Test handle_missing_values with the 'drop' method."""
        self.analyzer.handle_missing_values(method='drop')
        # Only rows with no missing values remain.
        self.assertEqual(len(self.analyzer), 3)

    def test_handle_missing_fill(self) -> None:
        """Test handle_missing_values with the 'fill' method."""
        self.analyzer.handle_missing_values(method='fill', fill_value=999)
        self.assertFalse(self.analyzer.df.isna().any().any())

    def test_save_invalid_format(self) -> None:
        """Test that saving with an unsupported file format raises an error."""
        with self.assertRaises(ValueError):
            self.analyzer.save("data.txt")

    def test_get_dtype_info(self) -> None:
        """Test that get_dtype_info returns data types for the columns."""
        dtypes = self.analyzer.get_dtype_info()
        self.assertIn('A', dtypes.index)
        self.assertIn('C', dtypes.index)

    def test_get_summary(self) -> None:
        """Test that get_summary returns a summary DataFrame."""
        summary = self.analyzer.get_summary()
        # Check that summary includes statistics for column 'A'.
        self.assertTrue('A' in summary.columns or 'A' in summary.index)


class TestNumericDatasetAnalyzer(unittest.TestCase):
    """Unit tests for NumericDatasetAnalyzer functionality."""

    def setUp(self) -> None:
        """Set up a DataFrame with numeric and non-numeric columns."""
        data = {
            'num1': [1, 2, 3, 4],
            'num2': [10, 20, 30, 40],
            'cat': ['a', 'b', 'a', 'b']
        }
        self.df = pd.DataFrame(data)
        self.numeric_analyzer = NumericDatasetAnalyzer(self.df)

    def test_numeric_columns_only(self) -> None:
        """Test that only numeric columns are retained."""
        self.assertListEqual(list(self.numeric_analyzer.df.columns), ['num1', 'num2'])

    def test_correlation_matrix(self) -> None:
        """Test that the correlation matrix is computed correctly."""
        corr_matrix = self.numeric_analyzer.correlation_matrix()
        self.assertEqual(corr_matrix.shape, (2, 2))


class TestCategoricalDatasetAnalyzer(unittest.TestCase):
    """Unit tests for CategoricalDatasetAnalyzer functionality."""

    def setUp(self) -> None:
        """Set up a DataFrame with both numeric and categorical data."""
        data = {
            'num': [1, 2, 3, 4],
            'cat1': ['a', 'b', 'a', 'b'],
            'cat2': ['x', 'x', 'y', 'y']
        }
        self.df = pd.DataFrame(data)
        self.categorical_analyzer = CategoricalDatasetAnalyzer(self.df)

    def test_categorical_columns_only(self) -> None:
        """Test that only non-numeric columns are retained."""
        self.assertListEqual(list(self.categorical_analyzer.df.columns), ['cat1', 'cat2'])

    def test_get_frequencies(self) -> None:
        """Test that frequency counts are computed correctly for categorical columns."""
        freqs = self.categorical_analyzer.get_frequencies()
        self.assertIn('cat1', freqs)
        self.assertIn('cat2', freqs)
        self.assertEqual(freqs['cat1'].loc['a'], 2)
        self.assertEqual(freqs['cat1'].loc['b'], 2)

# Note: run script to do unity tests