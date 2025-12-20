'''import torch.nn as nn
from ..config import LOOKBACK, HORIZON
class PatchTSTFinetune(nn.Module):
    """PatchTST con Transfer Learning"""
    
    def __init__(self, model_config: dict):
        super().__init__()
        
        from transformers import PatchTSTConfig, PatchTSTForPrediction
        
        # 1. Leggi parametri dal model_config (num_channels, target_idx, lookback, horizon, ecc.)
        
        # 2. Crea una PatchTSTConfig con le TUE dimensioni
        
        # 3. Crea il modello: self.model = PatchTSTForPrediction(config)
        
        # 4. Carica pesi pre-addestrati con _load_pretrained_weights()
        
        # 5. Opzionalmente congela il backbone se freeze_backbone=True
    
    def _load_pretrained_weights(self, pretrained_name: str):
        """
        Carica pesi da 'ibm-granite/granite-timeseries-patchtst'
        Solo i pesi con shape compatibile vengono copiati.
        """
        # Carica modello pre-addestrato
        # Confronta ogni peso: se shape uguale, copia
        # Ignora pesi con shape diversa (head, embedding)
    
    def forward(self, x):
        # Stesso del PatchTST normale
        # Input: (Batch, Lookback, Num_Channels)
        # Output: (Batch, Horizon, 1) - solo target'''