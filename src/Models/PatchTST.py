import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction
import config


class PatchTST(nn.Module):
    # Cambiamo il nome del parametro in 'train_loader' così è chiaro cosa vuole
    def __init__(self, model_config: dict, train_loader):
        """
        Args:
            train_loader: Il DataLoader che hai creato nel data_loader.py
        """
        super().__init__()

        if train_loader is None:
            raise ValueError("Devi passare il train_loader!")

        # 1. ACCESSO AL DATASET
        # train_loader (parametro) -> .dataset (attributo PyTorch standard)
        # Questo ci dà l'oggetto PVForecastDataset che hai definito in sampler.py
        dataset = train_loader.dataset

        # 2. ACCESSO AL TENSORE DATI
        # PVForecastDataset ha l'attributo self.data_tensor (riga 20 di sampler.py)
        # Shape: [Righe, Features] -> Prendiamo l'indice 1 (Features)
        self.num_channels = dataset.data_tensor.shape[1]

        print(f"Features rilevate dal train_loader: {self.num_channels}")

        # --- Il resto rimane uguale ---
        self.lookback = config.LOOKBACK
        self.horizon = config.HORIZON

        hf_config = PatchTSTConfig(
            context_length=self.lookback,
            prediction_length=self.horizon,
            num_input_channels=self.num_channels,
            num_targets=self.num_channels,
            # ... parametri dal dizionario ...
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
        return self.model(past_values=x).logits
