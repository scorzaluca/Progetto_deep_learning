import torch
import torch.nn as nn
from ..config import LOOKBACK, HORIZON


class PatchTST(nn.Module):
    """
    PatchTST model for PV Power Forecasting.

    This model utilizes the PatchTST architecture, a Transformer-based model optimized for time series forecasting.
    It segments the input time series into patches, which are then processed as tokens by the Transformer.

    Key Features:
    1. **Patching**: Reduces sequence length and captures local semantic information.
    2. **Channel Independence**: Each channel (variable) is processed independently by the same backbone, sharing weights.
    3. **Transformer Backbone**: Captures long-range temporal dependencies using self-attention.

    This class serves as a wrapper around the HuggingFace `PatchTSTForPrediction` model.

    Args:
        model_config (dict): Configuration dictionary containing:
            - num_channels (int): Number of input variables/channels.
            - target_idx (int): Index of the target column (PV Power).
            - lookback (int): Context length (history window size).
            - horizon (int): Prediction horizon (steps to forecast).
            - patch_length (int): Length of each patch segment.
            - stride (int): Stride between consecutive patches.
            - d_model (int): Hidden dimension of the Transformer.
            - n_heads (int): Number of attention heads.
            - n_layers (int): Number of Transformer layers.
            - dropout (float): Dropout probability.
            - use_cls_token (bool): Whether to include a CLS token.

    Attributes:
        model (PatchTSTForPrediction): The underlying HuggingFace PatchTST model.
        num_channels (int): Number of input channels.
        target_idx (int): Index of the target variable.
        lookback (int): Input sequence length.
        horizon (int): Prediction horizon.
    """

    def __init__(self, model_config: dict):
        super().__init__()

        from transformers import PatchTSTConfig, PatchTSTForPrediction

        # Read parameters from model_config
        self.num_channels = model_config.get("num_channels", 25)
        self.target_idx = model_config.get("target_idx", 24)

        # Fallback to global constants (LOOKBACK, HORIZON) only if not specified
        self.lookback = model_config.get("lookback", LOOKBACK)
        self.horizon = model_config.get("horizon", HORIZON)

        print(
            f"PatchTST - Features: {self.num_channels}, target_idx: {self.target_idx}, "
            f"lookback: {self.lookback}, horizon: {self.horizon}"
        )

        hf_config = PatchTSTConfig(
            context_length=self.lookback,
            prediction_length=self.horizon,
            num_input_channels=self.num_channels,
            num_targets=self.num_channels,  # Predict all channels, select target later
            patch_length=model_config.get("patch_length", 16),
            patch_stride=model_config.get("stride", 8),
            d_model=model_config.get("d_model", 128),
            num_attention_heads=model_config.get("n_heads", 4),
            num_hidden_layers=model_config.get("n_layers", 3),
            attention_dropout=model_config.get("dropout", 0.2),
            ff_dropout=model_config.get("dropout", 0.2),
            use_cls_token=model_config.get("use_cls_token", False),
        )

        self.is_pretrained = False

        # Instantiate the model for Prediction
        self.model = PatchTSTForPrediction(hf_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the PatchTST model.

        The HuggingFace model predicts values for ALL input channels.
        We extract only the prediction for the target variable (PV power).

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Lookback, Num_Channels).

        Returns:
            torch.Tensor: Predicted tensor for the target variable only, shape (Batch, Horizon, 1).
        """
        # The HuggingFace model returns predictions for ALL input features.
        # This is because it processes each channel independently (Channel Independence).
        # output shape: (Batch, Horizon, Num_Channels)
        full_output = self.model(past_values=x).prediction_outputs

        # We slice the tensor to extract the specific channel corresponding to target_idx.
        # Result shape: (Batch, Horizon, 1)
        target_output = full_output[:, :, self.target_idx : self.target_idx + 1]

        return target_output

    def load_pretrained_encoder(self, pretrain_path: str):
        """
        Loads the pretrained encoder weights from a file.

        This method allows transferring knowledge from the self-supervised pretraining phase
        to the forecasting model. It loads the weights into the encoder.

        Args:
            pretrain_path (str): Path to the .pth file containing the pretrained encoder state dict.

        Example:
            model = PatchTST(config)
            model.load_pretrained_encoder("results/pretrained/patchtst_encoder.pth")
            # Now the model is initialized with pretrained weights
        """

        print(f"Loading pretrained weights from: {pretrain_path}")

        # Load the state dictionary from the file
        encoder_state = torch.load(pretrain_path, map_location="cpu")

        # Load the weights specifically into the ENCODER part of the model.
        # Structure:
        # self.model                -> PatchTSTForPrediction (HuggingFace wrapper)
        # self.model.model          -> PatchTSTModel (Core model)
        # self.model.model.encoder  -> The Transformer Encoder
        self.model.model.encoder.load_state_dict(encoder_state, strict=False)

        self.is_pretrained = True

        print("Encoder weights loaded successfully!")
