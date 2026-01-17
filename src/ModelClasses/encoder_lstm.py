import torch
import torch.nn as nn
from ..config import HORIZON, INPUT_SIZE, PATCHTST_CONFIG



class EncoderLSTM(nn.Module):
    """
    Hybrid model combining a pretrained PatchTST encoder (optionally frozen) with an LSTM head.

    Key Features:
    - Uses PatchTST as a feature extractor (generates embeddings).
    - Projects embeddings to a lower dimension.
    - Uses LSTM to process the temporal sequence of embeddings.
    - Final linear head for forecasting.

    Flow:
    1. Input X -> Encoder -> Embeddings
    2. Embeddings -> Projection -> LSTM -> Temporal Features
    3. Last LSTM State -> Linear Head -> Forecast

    Args:
        model_config (dict): Configuration dictionary containing:
            - pretrain_path (str, optional): Path to pretrained encoder weights.
            - d_model (int, optional): PatchTST model dimension. Defaults to 128.
            - projection_dim (int, optional): Dimension to project embeddings to. Defaults to 64.
            - lstm_hidden (int, optional): LSTM hidden state dimension. Defaults to 64.
            - lstm_layers (int, optional): Number of LSTM layers. Defaults to 1.
            - dropout (float, optional): Dropout rate. Defaults to 0.2.
            - freeze_encoder (bool, optional): Whether to freeze encoder weights. Defaults to True.

    Attributes:
        encoder (PatchTSTPretraining): The PatchTST encoder module.
        projection (nn.Linear): Linear layer to project encoder embeddings.
        lstm (nn.LSTM): LSTM layer for temporal processing.
        head (nn.Linear): Final linear layer for prediction.
    """

    def __init__(self, model_config: dict):
        super().__init__()

        
        from .patchtst_pretraining import PatchTSTPretraining

        # Extract configuration parameters
        self.pretrain_path = model_config.get("pretrain_path")
        self.d_model = model_config.get("d_model", 128)
        self.projection_dim = model_config.get("projection_dim", 64) 
        self.lstm_hidden = model_config.get("lstm_hidden", 64)
        self.lstm_layers = model_config.get("lstm_layers", 1)
        self.dropout = model_config.get("dropout", 0.2)
        self.freeze_encoder = model_config.get("freeze_encoder", True)

        # Initialize the Backbone Encoder (Pretrained PatchTST)
        self.encoder = PatchTSTPretraining(PATCHTST_CONFIG)

        # Load pretrained weights if a path is provided
        if self.pretrain_path:
            print(f"Loading encoder from: {self.pretrain_path}")
            encoder_state = torch.load(self.pretrain_path, map_location="cpu")
            
            # Load the weights into the specific encoder submodule
            # strict=False allows ignoring heads/other layers that don't match exactly
            self.encoder.model.model.encoder.load_state_dict(
                encoder_state, strict=False
            )
            
            # Freeze the encoder weights if requested
            if self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        self.is_pretrained = bool(self.pretrain_path) and not self.freeze_encoder

        #Projection Layer: Reduces dimension from (d_model * num_channels) to projection_dim
        # This layer learns how to combine representations from different channels
        self.embedding_dim = self.d_model * INPUT_SIZE  # e.g., 128 * 24 = 3072
        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)

        # LSTM Layer processing the projected embeddings
        self.lstm = nn.LSTM(
            input_size=self.projection_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
        )

        # Prediction Head: Maps from LSTM hidden state to Forecast Horizon
        self.head = nn.Linear(self.lstm_hidden, HORIZON)

        print(
            f"EncoderLSTM - embedding: {self.embedding_dim} -> projection: {self.projection_dim}, "
            f"lstm_hidden: {self.lstm_hidden}, layers: {self.lstm_layers}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the EncoderLSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Lookback, Channels).

        Returns:
            torch.Tensor: Predicted tensor of shape (Batch, Horizon, 1).
        """
        if self.freeze_encoder:
            # Extract embeddings from the frozen encoder
            # If frozen, we use the helper method to use the encoder as a feature extractor
            embeddings = self.encoder.get_embeddings(x)
        else:
            # Forward pass through the trainable encoder of HuggingFace's PatchTST
            outputs = self.encoder.model.model(
                past_values=x,
                output_hidden_states=True,
            )
            hidden = outputs.last_hidden_state # The output of the encoder. Shape: (batch, num_channels, num_patches, d_model)
            batch_size, num_channels, num_patches, d_model = hidden.shape
            
            # Reshape to combine channels for each patch
            # We want one vector per patch containing info from ALL channels
            # Shape becomes: (batch, num_patches, num_channels * d_model)
            embeddings = hidden.permute(0, 2, 1, 3).reshape(batch_size, num_patches, -1)

        # Project channel-concatenated embeddings to a smaller dimension
        # Reduces dimensionality to allow LSTM to process it efficiently
        embeddings = self.projection(embeddings)  # Shape: (batch, n_patches, projection_dim)

        # Process the sequence of patches with LSTM
        # The LSTM captures temporal dependencies between the patches
        lstm_out, _ = self.lstm(embeddings)  # Shape: (batch, n_patches, lstm_hidden)

        # Use the LSTM output from the last time step (last patch)
        # This represents the summarized context of the entire input window
        last_out = lstm_out[:, -1, :]  # Shape: (batch, lstm_hidden)

        # Generate forecasts for the horizon using the final linear head
        predictions = self.head(last_out)  # Shape: (batch, horizon)
        
        predictions = predictions.unsqueeze(-1)  # Shape: (batch, horizon, 1)

        return predictions
