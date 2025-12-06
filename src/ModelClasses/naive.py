import torch
import torch.nn as nn


class NaivePersistence(nn.Module):
    """
    Modello Naive Persistence (Daily Persistence).
    
    Predice che domani sarà uguale a oggi.
    Usato come baseline per calcolare il MASE.
    """
    
    def __init__(self, train_loader=None, target_idx: int = None):
        """
        Args:
            train_loader: DataLoader per rilevare dinamicamente target_idx se non specificato.
            target_idx: Indice della colonna target (pv_power). Se None, viene rilevato dal loader.
        """
        super(NaivePersistence, self).__init__()
        self.horizon = 24
        
        # --- TARGET INDEX DINAMICO ---
        if target_idx is not None:
            self.target_idx = target_idx
            print(f"NaivePersistence - Usando target_idx fornito: {self.target_idx}")
        elif train_loader is not None:
            # Leggiamo l'indice target dal dataset (attributo target_col_idx)
            self.target_idx = train_loader.dataset.target_col_idx
            print(f"NaivePersistence - target_idx rilevato dal loader: {self.target_idx}")
        else:
            # Fallback al valore di default per retrocompatibilità
            self.target_idx = 0
            print(f"⚠️ NaivePersistence - target_idx non specificato, usando default: {self.target_idx}")

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