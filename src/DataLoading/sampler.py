import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple


class PVForecastDataset(Dataset):
    """
    Dataset for supervised time series forecasting.
    Creates samples of (input_window, target_window) pairs suitable for training PyTorch models.

    Args:
        data (pd.DataFrame): DataFrame containing the time series data.
        target_col (str): The name of the target column to forecast.
        lookback (int): Length of the past input window (X).
        horizon (int): Length of the future target window (Y).
        step (int, optional): Stride between consecutive samples. Defaults to 1.

    Attributes:
        target_col_idx (int): Index of the target column in the dataframe.
        data_tensor (torch.Tensor): Tensor containing the time series data.
        lookback (int): Length of the input window.
        horizon (int): Length of the target window.
        step (int): Stride between consecutive samples.
        total_window (int): Total length of a sample (lookback + horizon).
        num_samples (int): Total number of valid samples in the dataset.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        lookback: int,
        horizon: int,
        step: int = 1,
    ):
        if target_col not in data.columns:
            raise ValueError(f"Target '{target_col}' not found.")
        self.target_col_idx = data.columns.get_loc(target_col)

        self.data_tensor = torch.tensor(data.values, dtype=torch.float32)

        self.lookback = lookback
        self.horizon = horizon
        self.step = step

        # Computing how much valid samples we can create
        self.total_window = lookback + horizon
        self.num_samples = (len(self.data_tensor) - self.total_window) // self.step + 1

    def __len__(self):
        """
        Returns the total number of valid samples in the dataset.

        Args:
            None

        Returns:
            int: The number of samples.
        """
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample of (input_window, target_window) for the given index.
        The input window has length 'lookback' and the target window has length 'horizon'.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input window and target window.
        """
        start_idx = idx * self.step

        mid_idx = start_idx + self.lookback
        end_idx = mid_idx + self.horizon

        # --- INPUT Extracting (X) ---
        # Taking from start_idx to mid_x, (all the features)
        # Shape: (lookback, n_features)
        X = self.data_tensor[start_idx:mid_idx, :]

        # --- OUTPUT Extracting (Y) ---
        # Taking from mid_idx to end_idx, (only target_columns)
        # Shape: (horizon, 1)
        y = self.data_tensor[mid_idx:end_idx, self.target_col_idx]
        y = y.unsqueeze(-1)

        return X, y
