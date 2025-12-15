import torch.nn as nn
from config import LOOKBACK, HORIZON


class PatchTST(nn.Module):
    """
    PatchTST per PV Power Forecasting.

    PatchTST è un Transformer specializzato per time series che:
    1. Divide la sequenza in patch (segmenti sovrapposti) invece di elaborare ogni timestep
    2. Usa Channel Independence: ogni feature è processata separatamente con gli stessi pesi
    3. Applica self-attention per catturare dipendenze temporali a lungo raggio

    Vantaggi del patching:
    - Riduce la lunghezza della sequenza per l'attention (48 timestep → ~5 patch)
    - Ogni patch cattura pattern locali (es. mezzo ciclo giornaliero con patch_length=16)
    - L'overlap (stride < patch_length) assicura continuità tra patch
    """

    def __init__(self, model_config: dict):
        """
        Wrapper per il modello PatchTST di HuggingFace.

        Args:
            model_config: Dizionario con i parametri del modello.
                          Deve contenere 'lookback' e 'horizon'.
        """
        super().__init__()

        # LAZY IMPORT: carica transformers solo quando serve
        from transformers import PatchTSTConfig, PatchTSTForPrediction

        # Lettura parametri da model_config
        self.num_channels = model_config.get("num_channels", 25)
        self.target_idx = model_config.get("target_idx", 24)

        # LOOKBACK e HORIZON dal model_config (permette override da notebook)
        # Fallback ai valori globali solo se non specificati
        self.lookback = model_config.get("lookback", LOOKBACK)
        self.horizon = model_config.get("horizon", HORIZON)

        print(
            f"PatchTST - Features: {self.num_channels}, target_idx: {self.target_idx}, "
            f"lookback: {self.lookback}, horizon: {self.horizon}"
        )

        # Configurazione del modello HuggingFace
        # - context_length: finestra di input (lookback)
        # - prediction_length: orizzonte di previsione
        # - num_input_channels: numero di feature (Channel Independence)
        # - num_targets: predice tutte le feature, poi estraiamo solo il target
        # - patch_length: dimensione di ogni patch (16h = più di mezzo ciclo giornaliero)
        # - stride: passo tra patch (8h overlap per continuità)
        # - d_model: dimensione embedding del transformer
        # - n_heads: numero di attention heads (d_model deve essere divisibile per n_heads)
        hf_config = PatchTSTConfig(
            context_length=self.lookback,
            prediction_length=self.horizon,
            num_input_channels=self.num_channels,
            num_targets=self.num_channels,
            patch_length=model_config.get("patch_length", 16),
            stride=model_config.get("stride", 8),
            d_model=model_config.get("d_model", 128),
            n_heads=model_config.get("n_heads", 4),
            n_layers=model_config.get("n_layers", 3),
            dropout=model_config.get("dropout", 0.2),
            use_cls_token=model_config.get("use_cls_token", False),
        )

        self.model = PatchTSTForPrediction(hf_config)

    def forward(self, x):
        """
        Forward pass del PatchTST.

        Il modello HuggingFace restituisce predizioni per TUTTE le feature.
        Noi estraiamo solo la predizione per pv_power (target_idx).

        Input x: (Batch, Lookback, Num_Channels)
        Output:  (Batch, Horizon, 1) -> Solo la predizione per pv_power
        """
        # Il modello HuggingFace restituisce prediction_outputs con shape:
        # (Batch, Horizon, Num_Channels) - predizioni per TUTTE le feature
        full_output = self.model(past_values=x).prediction_outputs

        # Estraiamo solo il canale target (pv_power)
        # Lo slicing [idx : idx + 1] mantiene la dimensione 3D
        target_output = full_output[:, :, self.target_idx : self.target_idx + 1]
        return target_output
