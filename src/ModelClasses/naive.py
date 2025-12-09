import torch
import torch.nn as nn


class NaivePersistence(nn.Module):
    """
    Modello Naive Persistence (Daily Persistence).

    Predice che domani sarà uguale a oggi.
    Usato come baseline per calcolare il MASE.
    """

    def __init__(self, model_config: dict = None):
        """
        Args:
            model_config: Dizionario con i parametri del modello.
                         Se None, usa valori di default.
        """
        super(NaivePersistence, self).__init__()

        if model_config is None:
            model_config = {}

        # --- Lettura parametri da config (non più da train_loader) ---
        self.target_idx = model_config.get("target_idx", 23)
        self.horizon = model_config.get("horizon", 24)

        print(
            f"NaivePersistence - target_idx: {self.target_idx}, horizon: {self.horizon}"
        )

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

        last_day_values = x[:, -self.horizon :, self.target_idx]

        # Ora abbiamo shape (Batch, Horizon).
        # Il training loop si aspetta (Batch, Horizon, 1) per calcolare la Loss.
        return last_day_values.unsqueeze(-1)
