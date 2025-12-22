"""
Dataset per Self-Supervised Pretraining.
Restituisce solo sequenze X, senza target Y.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd


class PretrainingDataset(Dataset):
    """
    Dataset per pretraining di PatchTST.
    
    A differenza di PVForecastDataset, questo restituisce SOLO X.
    Non serve Y perché il modello ricostruisce X stesso (masked reconstruction).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 48,
        step: int = 1,
    ):
        """
        Args:
            df: DataFrame con i dati (già normalizzati).
            lookback: Lunghezza della finestra di input.
            step: Passo tra campioni consecutivi.
        """
        self.data = torch.tensor(df.values, dtype=torch.float32)
        self.lookback = lookback
        self.step = step

        # Calcola quanti campioni possiamo creare
        # Non serve spazio per horizon perché non prediciamo il futuro
        self.n_samples = (len(self.data) - self.lookback) // self.step + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Restituisce solo X (la sequenza di lookback ore).
        
        Returns:
            X: tensor di shape (lookback, num_features)
        """
        start_idx = idx * self.step
        end_idx = start_idx + self.lookback

        X = self.data[start_idx:end_idx]  # Shape: (lookback, num_features)
        
        return X