import torch
from torch.utils.data import Dataset
import pandas as pd


class PretrainingDataset(Dataset):
    """
    Dataset for PatchTST pretraining (Masked Modeling).
    
    Unlike PVForecastDataset, this dataset returns ONLY the input sequence X.
    The target Y is not needed because the model reconstructs X itself (masked reconstruction).

    Args:
        df (pd.DataFrame): DataFrame containing the time series data (already normalized).
        lookback (int, optional): Length of the input window. Defaults to 48.
        step (int, optional): Stride between consecutive samples. Defaults to 1.

    Attributes:
        data (torch.Tensor): Tensor containing the time series data.
        lookback (int): Length of the input window.
        step (int): Stride between consecutive samples.
        n_samples (int): Total number of samples available in the dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 48,
        step: int = 1,
    ):
        self.data = torch.tensor(df.values, dtype=torch.float32)
        self.lookback = lookback
        self.step = step

        #Compute how much sample we can create from the time series
        self.n_samples = (len(self.data) - self.lookback) // self.step + 1

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Args:
            None

        Returns:
            int: Total number of samples.
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Retrieves the input sequence X for the given index.
        The sequence has length 'lookback' and contains 'num_features' features.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: The input sequence tensor of shape (lookback, num_features).
        """
        start_idx = idx * self.step #Calculating the starting index of the window
        end_idx = start_idx + self.lookback #Calculating the ending index of the window

        X = self.data[start_idx:end_idx]  # Shape: (lookback, num_features) #Extracting the window from the data
        
        return X