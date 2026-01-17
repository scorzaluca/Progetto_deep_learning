import torch
import torch.nn as nn
from ..config import LOOKBACK, INPUT_SIZE


class PatchTSTPretraining(nn.Module):
    """
    Wrapper for HuggingFace's PatchTSTForPretraining model.

    This model is designed for Self-Supervised Learning (SSL) via Masked Patch Reconstruction.
    It learns to reconstruct masked segments of the input time series, forcing the model to capture
    temporal dependencies and structural patterns without requiring labeled target data (forecasting).

    Args:
        model_config (dict): Configuration dictionary containing:
            - num_channels (int): Number of input variables/channels.
            - lookback (int): Context length (history window size).
            - mask_ratio (float): Ratio of patches to mask (e.g., 0.4 means 40% masked).
            - patch_length (int): Length of each patch.
            - stride (int): Stride between patches.
            - d_model (int): Hidden dimension of the Transformer.
            - n_heads (int): Number of attention heads.
            - n_layers (int): Number of Transformer layers.
            - dropout (float): Dropout probability.

    Attributes:
        model (PatchTSTForPretraining): The underlying HuggingFace model.
        num_channels (int): Number of input channels.
        lookback (int): Input sequence length.
        mask_ratio (float): Masking ratio used during pretraining.
    """

    def __init__(self, model_config: dict):
        super().__init__()

        
        from transformers import PatchTSTConfig, PatchTSTForPretraining

        # Configuration parameters
        self.num_channels = model_config.get("num_channels", INPUT_SIZE)
        self.lookback = model_config.get("lookback", LOOKBACK)
        self.mask_ratio = model_config.get("mask_ratio", 0.4)

        # Create the HuggingFace configuration object for PRETRAINING
        hf_config = PatchTSTConfig(
            # Input dimensions
            num_input_channels=self.num_channels,
            context_length=self.lookback,
            
            # Architecture (Transformer) parameters
            patch_length=model_config.get("patch_length", 16),
            patch_stride=model_config.get("stride", 8),
            d_model=model_config.get("d_model", 128),
            num_attention_heads=model_config.get("n_heads", 4),
            num_hidden_layers=model_config.get("n_layers", 3),
            attention_dropout=model_config.get("dropout", 0.2),
            ff_dropout=model_config.get("dropout", 0.2),
            
            # PRETRAINING SPECIFIC PARAMETERS
            # This enables the masked patch reconstruction task
            mask_type="random",
            random_mask_ratio=self.mask_ratio,
        )

        # Instantiate the model specifically for Pretraining (Reconstruction task)
        self.model = PatchTSTForPretraining(hf_config)

        print(
            f"PatchTSTPretraining - Channels: {self.num_channels}, "
            f"Lookback: {self.lookback}, Mask ratio: {self.mask_ratio:.0%}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass for self-supervised pretraining.

        The model automatically masks a portion of the input patches and calculates
        the reconstruction loss (MSE) between the original and reconstructed patches.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Lookback, Num_Channels).

        Returns:
            torch.Tensor: The reconstruction loss (scalar).
        """
        # Pass the input sequence 'x' to the HuggingFace model.
        # The model internally:
        # - Patchifies the input sequence.
        # - Applies random masking to a subset of patches (controlled by mask_ratio).
        # - Reconstructs the masked patches using the Transformer backbone.
        # - Computes the Mean Squared Error (MSE) loss between the original and reconstructed patches.
        output = self.model(past_values=x)

        # Return the calculated reconstruction loss directly (scalar tensor).
        # We minimize this loss to force the model to learn temporal dependencies.
        return output.loss

    def get_encoder_state_dict(self):
        """
        Extracts the state dictionary of the PatchTST encoder.

        This method is used to save only the learned encoder weights after pretraining,
        discarding the reconstruction head. These weights can then be loaded into
        a forecasting model (e.g., in EncoderLSTM).

        Returns:
            dict: A dictionary containing the weights of the encoder.
        """
        return self.model.model.encoder.state_dict()

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts latent representations (embeddings) from the encoder without masking.

        This method is useful when using the pretrained encoder as a feature extractor
        for downstream tasks (e.g., forecasting with an LSTM head). It runs the encoder
        in evaluation mode and disables masking to get the full representation of the input.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Lookback, Num_Channels).

        Returns:
            torch.Tensor: Embeddings tensor. The shape depends on the implementation, typically
                          (Batch, Num_Patches, Channels * D_Model) after reshaping.
        """
        # Set inference mode to disable dropout
        self.model.eval()

        with torch.no_grad():
            # Get hidden states from the encoder
            # By default, it doesn't apply masking when called this way (unless specified),
            # providing full access to the input sequence's representation.
            outputs = self.model.model(
                past_values=x,
                output_hidden_states=True,
            )

            hidden = outputs.last_hidden_state  # Shape: (Batch, Num_Channels, Num_Patches, D_Model)
            
            # Retrieve dimensions
            batch_size, num_channels, num_patches, d_model = hidden.shape
            
            # Reshape logic:
            # 1. Permute to bring channels next to d_model: (Batch, Num_Patches, Num_Channels, D_Model)
            # 2. Reshape to flatten channels and d_model: (Batch, Num_Patches, Num_Channels * D_Model)
            # This allows treating all channels of a single patch as one large feature vector.
            embeddings = hidden.permute(0, 2, 1, 3).reshape(batch_size, num_patches, -1)

        return embeddings
