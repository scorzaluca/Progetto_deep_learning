"""
Modello ibrido: Encoder PatchTST pretrained + LSTM Head.

L'encoder estrae embeddings dalle sequenze, l'LSTM li processa per la previsione.
"""

import torch
import torch.nn as nn
from ..config import HORIZON, INPUT_SIZE, PATCHTST_CONFIG



class EncoderLSTM(nn.Module):
    """
    Combina un encoder PatchTST pretrained (congelato) con un LSTM.


    Flusso:
    1. Input X passa attraverso l'encoder → embeddings
    2. Embeddings passano attraverso LSTM → features temporali
    3. Linear head → previsione 24h
    """

    def __init__(self, model_config: dict):
        super().__init__()


        from .patchtst_pretraining import PatchTSTPretraining


        # Parametri
        self.pretrain_path = model_config.get("pretrain_path")
        self.d_model = model_config.get("d_model", 128)
        self.projection_dim = model_config.get(
            "projection_dim", 64
        )  # Tunabile con Optuna
        self.lstm_hidden = model_config.get("lstm_hidden", 64)
        self.lstm_layers = model_config.get("lstm_layers", 1)
        self.dropout = model_config.get("dropout", 0.2)
        self.freeze_encoder = model_config.get("freeze_encoder", True)
        # 1. Encoder pretrained (congelato)
        self.encoder = PatchTSTPretraining(PATCHTST_CONFIG)

        if self.pretrain_path:
            print(f"Caricamento encoder da: {self.pretrain_path}")
            encoder_state = torch.load(self.pretrain_path, map_location="cpu")
            self.encoder.model.model.encoder.load_state_dict(
                encoder_state, strict=False
            )
            if self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        # Congela l'encoder
        self.is_pretrained = bool(self.pretrain_path) and not self.freeze_encoder

        # 2. Layer di proiezione: riduce da (d_model * num_channels) a projection_dim
        # Questo layer IMPARA come combinare le rappresentazioni dei diversi canali
        self.embedding_dim = self.d_model * INPUT_SIZE  # 128 * 24 = 3072
        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)

        # 3. LSTM che processa gli embeddings proiettati
        self.lstm = nn.LSTM(
            input_size=self.projection_dim,  # Dimensione tunabile
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
        )

        

        # 4. Head per la previsione
        self.head = nn.Linear(self.lstm_hidden, HORIZON)

        print(
            f"EncoderLSTM - embedding: {self.embedding_dim} → projection: {self.projection_dim}, "
            f"lstm_hidden: {self.lstm_hidden}, layers: {self.lstm_layers}"
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, lookback, num_channels) es. (64, 48, 24)

        Returns:
            predictions: (batch, horizon, 1) es. (64, 24, 1)
        """
        if self.freeze_encoder:
            embeddings = self.encoder.get_embeddings(x)
        else:
            outputs = self.encoder.model.model(
                past_values=x,
                output_hidden_states=True,
            )
            hidden = outputs.last_hidden_state
            batch_size, num_channels, num_patches, d_model = hidden.shape
            embeddings = hidden.permute(0, 2, 1, 3).reshape(batch_size, num_patches, -1)
        # 2. Proietta gli embeddings (impara combinazione ottimale dei canali)
        embeddings = self.projection(embeddings)  # (batch, n_patches, d_model)

        # 3. LSTM processa la sequenza di patch
        lstm_out, _ = self.lstm(embeddings)  # (batch, n_patches, lstm_hidden)

        # 4. Usa l'ultimo output per la previsione

        # 4. Usa l'ultimo output per la previsione
        last_out = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        # 5. Proietta su horizon

        # 5. Proietta su horizon
        predictions = self.head(last_out)  # (batch, horizon)


        # 5. Reshape per compatibilità con altri modelli
        predictions = predictions.unsqueeze(-1)  # (batch, horizon, 1)


        return predictions
