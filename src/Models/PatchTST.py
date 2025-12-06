import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForPrediction
import config 


class PatchTST(nn.Module):
    def __init__(self, model_config: dict, train_loader):
        """
        Wrapper per il modello PatchTST di HuggingFace.
        
        Args:
            model_config: Dizionario con i parametri del modello.
            train_loader: Il DataLoader che hai creato nel data_loader.py.
        """
        super().__init__()
        
        if train_loader is None:
             raise ValueError("Devi passare il train_loader!")
        
        # 1. ACCESSO AL DATASET
        # train_loader (parametro) -> .dataset (attributo PyTorch standard)
        # Questo ci dà l'oggetto PVForecastDataset che hai definito in sampler.py
        dataset = train_loader.dataset
        
        # 2. ACCESSO AL TENSORE DATI
        # PVForecastDataset ha l'attributo self.data_tensor
        # Shape: [Righe, Features] -> Prendiamo l'indice 1 (Features)
        self.num_channels = dataset.data_tensor.shape[1]
        
        # 3. SALVIAMO L'INDICE DEL TARGET (pv_power)
        # Serve per estrarre solo la predizione del PV dal forward
        self.target_idx = dataset.target_col_idx
        
        print(f"PatchTST - Features rilevate: {self.num_channels}, target_idx: {self.target_idx}")

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
            use_cls_token=model_config.get("use_cls_token", False)
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
        # Output shape: (Batch, Horizon, Num_Channels)
        full_output = self.model(past_values=x).prediction_outputs
        
        # Estraiamo solo il canale target (pv_power)
        # Output shape: (Batch, Horizon, 1)
        target_output = full_output[:, :, self.target_idx:self.target_idx+1]
        
        return target_output