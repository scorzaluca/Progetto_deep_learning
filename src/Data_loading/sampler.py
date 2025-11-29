import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple

# --- LA CLASSE DATASET RESTA UGUALE ---
class PVForecastDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        target_col: str, 
        lookback: int = 168, 
        horizon: int = 24,
        step: int = 1
    ):
        
        
        if target_col not in data.columns:
            raise ValueError(f"Target '{target_col}' not found.")
        self.target_col_idx = data.columns.get_loc(target_col)

        self.data_tensor = torch.tensor(data.values, dtype=torch.float32)

        self.lookback = lookback
        self.horizon = horizon
        self.step = step
        
        # Calcoliamo quanti campioni validi possiamo generare
        self.total_window = lookback+horizon
        self.num_samples = (len(self.data_tensor) - self.total_window) // self.step + 1

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        #Questa funzione viene chiamata dal DataLoader per ogni campione del batch
        start_idx = idx * self.step
        
        
        mid_idx = start_idx + self.lookback
        end_idx = mid_idx + self.horizon  

        
        # --- INPUT Extracting (X) ---
        # Taking from start_idx to mid_x, (all the features)
        # Shape: (lookback, n_features)
        X = self.data_tensor[start_idx:mid_idx, :]

        # --- OUTPUT Extracting (Y) ---
        # Taking from mid_idx to end_idx, (only target_columns)
        # Shape: (lookback, n_features)
        y = self.data_tensor[mid_idx:end_idx, self.target_col_idx]
        y=y.unsqueeze(-1)
        
        return X, y