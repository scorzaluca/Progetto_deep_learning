import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Standard LSTM model for PV Power Forecasting.
    
    This architecture includes:
    1. LSTM Layer (uni- or bi-directional) for temporal feature extraction.
    2. Layer Normalization on the final state to stabilize training.
    3. Fully Connected (Linear) head for projecting to the forecast horizon.

    Args:
        model_config (dict): Configuration dictionary containing:
            - input_size (int): Number of input features/channels.
            - hidden_size (int): Dimension of the LSTM hidden state.
            - output_size (int): Forecast horizon (number of steps to predict).
            - num_layers (int): Number of stacked LSTM layers.
            - dropout (float): Dropout probability.
            - bidirectional (bool): Whether to use a bidirectional LSTM.

    Attributes:
        lstm (nn.LSTM): The LSTM module.
        layer_norm (nn.LayerNorm): Layer normalization applied to the final hidden state.
        fc (nn.Linear): The final linear layer for prediction.
    """

    def __init__(self, model_config: dict):
        super(LSTM, self).__init__()

        # Read configuration parameters
        self.input_size = model_config.get("input_size", 24)
        self.hidden_size = model_config.get("hidden_size", 64)
        self.output_size = model_config.get("output_size", 24)
        self.num_layers = model_config.get("num_layers", 1)
        self.dropout = model_config.get("dropout", 0.0)
        self.bidirectional = model_config.get("bidirectional", False)

        print(
            f"LSTM - Features: {self.input_size}, Hidden: {self.hidden_size}, Layers: {self.num_layers}, Bidir: {self.bidirectional}"
        )

        # Initialize the LSTM layer
        # bias=True by default in PyTorch
        # batch_first=True ensures input/output tensors are (Batch, Seq, Feature)
        # bidirectional=True processes sequence from start to end AND end to start
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0, # Dropout is applied only between layers if num_layers > 1
            bidirectional=self.bidirectional,
            batch_first=True, 
        )

        # Determine the input size for the fully connected layer
        # If bidirectional, the hidden state includes both forward and backward states (concatenated)
        fc_input_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        # Layer Normalization to stabilize training and enable higher learning rates
        self.layer_norm = nn.LayerNorm(fc_input_size)

        # Final projection layer to map the hidden state to the forecast horizon
        self.fc = nn.Linear(fc_input_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Lookback, Input_Size).

        Returns:
            torch.Tensor: Predicted tensor of shape (Batch, Horizon, 1).
        """

        # Temporal Processing with LSTM
        # Pass the entire sequence (x) through the LSTM
        # lstm_out contains the hidden states for EACH time step.
        # Shape: (Batch_Size, Lookback, Hidden_Size) or (Batch_Size, Lookback, Hidden_Size * 2) if bidirectional
        lstm_out, _ = self.lstm(x)

        # Select Final States
        if self.bidirectional:
            # For BiLSTM, we need to concatenate the final states of both directions:
            # - Forward: The last timestep (index -1) contains info from t=0 to T
            # - Backward: The first timestep (index 0) contains info from T to 0
            out_forward = lstm_out[:, -1, : self.hidden_size]  # (Batch, Hidden)
            out_backward = lstm_out[:, 0, self.hidden_size :]  # (Batch, Hidden)
            last_time_step_feature = torch.cat((out_forward, out_backward), dim=1)
        else:
            # Standard LSTM: Take the last timestep
            # This contains the accumulated information from the entire sequence
            last_time_step_feature = lstm_out[:, -1, :] # Shape: (Batch_Size, Hidden_Size)

        # Layer Normalization
        # Normalize activations to stabilize training and allow higher learning rates
        last_time_step_feature = self.layer_norm(last_time_step_feature)

        # Prediction (Decoding)
        # Project the summarized feature vector to the forecast horizon
        prediction = self.fc(last_time_step_feature) # Shape: (Batch_Size, Horizon)

        # Output Shaping
        # Add a feature dimension to match the target shape (Batch, Horizon, 1) 
        return prediction.unsqueeze(-1)
