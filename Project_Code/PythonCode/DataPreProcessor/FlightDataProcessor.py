import pandas as pd
import numpy as np
from typing import Union, Dict
import logging
import unittest

class FlightDataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset into a pandas DataFrame.
        """
        self.df = pd.read_csv(self.filepath)
        return self.df

class FlightDataCleaner:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the FlightDataCleaner with a DataFrame.

        Parameters:
            df (pd.DataFrame): The dataset to be cleaned.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        # Work on a copy to avoid modifying the original DataFrame
        self.df = df.copy()

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

    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    def remove_cancelled_diverted(self):
        self.df = self.df[(self.df['CANCELLED'] == 0) & (self.df['DIVERTED'] == 0)]
        return self.df

    def remove_data_leak_cols(self):
        leakage_cols = [
            'DEP_DELAY', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
            'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT', 'ARR_TIME', 'DEP_TIME',
            'WHEELS_OFF', 'WHEELS_ON', 'TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME',
            'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED'
        ]
        self.df = self.df.drop(columns=leakage_cols, errors='ignore')

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

#class FlightDataTransformer: