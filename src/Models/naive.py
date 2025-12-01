import torch
import torch.nn as nn

class NaivePersistence(nn.Module):
    def __init__(self, model_config: dict):
        super(NaivePersistence, self).__init__()
        self.horizon = model_config.get("horizon", 24)
        
        # IMPORTANTE: Il modello deve sapere quale feature nell'input è la potenza PV.
        # L'LSTM impara da solo quali feature pesare, ma il Naive deve copiare quella giusta.
        # Di default proviamo con 0, ma va configurato correttamente in config.py.
        self.target_idx = model_config.get("target_feature_index", 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, Lookback, Input_Size) -> es. (64, 48, 10)
        Output:  (Batch, Horizon, 1) -> es. (64, 24, 1)
        """
        
        # Strategia: "Daily Persistence"
        # L'input contiene 48 ore di storia. Le ultime 24 ore (da t-23 a t) 
        # rappresentano "oggi". Noi usiamo "oggi" per prevedere "domani".
        
        # Slicing:
        # : -> tutti i batch
        # -self.horizon: -> prendiamo solo gli ultimi 'horizon' step temporali (ultime 24h)
        # self.target_idx -> prendiamo solo la colonna della potenza PV
        
        last_day_values = x[:, -self.horizon:, self.target_idx]
        
        # Ora abbiamo shape (Batch, Horizon).
        # Il training loop si aspetta (Batch, Horizon, 1) per calcolare la Loss.
        return last_day_values.unsqueeze(-1)