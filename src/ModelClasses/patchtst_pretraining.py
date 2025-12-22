"""
PatchTST per Self-Supervised Pretraining con Masked Patch Reconstruction.
"""

import torch
import torch.nn as nn
from ..config import LOOKBACK, INPUT_SIZE


class PatchTSTPretraining(nn.Module):
    """
    Wrapper per PatchTSTForPretraining di HuggingFace.
    
    Questo modello impara a ricostruire patch mascherate della sequenza.
    NON fa forecasting, serve solo per il pretraining.
    """

    def __init__(self, model_config: dict):
        super().__init__()

        # Import lazy (solo quando serve)
        from transformers import PatchTSTConfig, PatchTSTForPretraining

        # Parametri dal config
        self.num_channels = model_config.get("num_channels", INPUT_SIZE)
        self.lookback = model_config.get("lookback", LOOKBACK)

        # Config HuggingFace per PRETRAINING
        hf_config = PatchTSTConfig(
            num_input_channels=self.num_channels,
            context_length=self.lookback,
            
            # Parametri architettura (come PatchTST normale)
            patch_length=model_config.get("patch_length", 16),
            patch_stride=model_config.get("stride", 8),           # ← CORRETTO: patch_stride
            d_model=model_config.get("d_model", 128),
            num_attention_heads=model_config.get("n_heads", 4),
            num_hidden_layers=model_config.get("n_layers", 3),
            attention_dropout=model_config.get("dropout", 0.2),   # ← CORRETTO: tipo specifico
            ff_dropout=model_config.get("dropout", 0.2),
            # PARAMETRI SPECIFICI PER PRETRAINING
            mask_type="random",           # Maschera patch a caso
            random_mask_ratio=0.4,        # 40% delle patch mascherate
        )

        # Modello per pretraining (NON PatchTSTForPrediction!)
        self.model = PatchTSTForPretraining(hf_config)
        
        print(f"PatchTSTPretraining - Channels: {self.num_channels}, "
              f"Lookback: {self.lookback}, Mask ratio: 40%")

    def forward(self, x):
        """
        Forward pass per pretraining.
        
        Input x: (Batch, Lookback, Num_Channels)
        Output: loss di ricostruzione (MSE tra patch originali e ricostruite)
        """
        # PatchTSTForPretraining calcola automaticamente la loss
        output = self.model(past_values=x)
        return output.loss

    def get_encoder_state_dict(self):
        """
        Estrae i pesi dell'encoder per salvarli.
        Questi pesi verranno caricati nel PatchTST di forecasting.
        """
        return self.model.model.encoder.state_dict()

    def get_embeddings(self, x):
        """
        Estrae le rappresentazioni latenti dall'encoder (senza masking).
        
        Utile per usare l'encoder come feature extractor per altri modelli.
        
        Args:
            x: Input tensor di shape (batch, lookback, num_channels)
               Es: (64, 48, 24)
        
        Returns:
            embeddings: Tensor di shape (batch, n_patches, d_model)
                        Es: (64, 5, 128)
                        Dove n_patches dipende da patch_length e stride
        """
        self.model.eval()  # Modalità inference
        
        with torch.no_grad():
            # Ottieni gli hidden states dall'encoder
            # do_mask_input=False evita il masking (vogliamo TUTTI i dati)
            outputs = self.model.model(
                past_values=x,
                output_hidden_states=True,
            )
            
            # last_hidden_state contiene gli embeddings di tutte le patch
            embeddings = outputs.last_hidden_state
            
        return embeddings