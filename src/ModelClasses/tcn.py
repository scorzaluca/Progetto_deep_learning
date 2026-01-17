import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    Base building block of the TCN (Temporal Convolutional Network).
    It contains two dilated causal convolutional layers with a residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (filters).
        kernel_size (int): Convolutional kernel size.
        dilation (int): Dilation factor for the convolution.
        dropout (float): Dropout probability.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()

        # Causal Padding:
        # We add zeros at the beginning of the sequence.
        # This shifts the input so that the convolution filters only see past values (t and before),
        self.padding = (kernel_size - 1) * dilation

        # First Convolutional Layer
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=self.padding, # Adds zeros to the sequence start
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels) # Batch Normalization
        self.relu1 = nn.ReLU() # Activation function
        self.dropout1 = nn.Dropout(dropout) # Regularization

        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual Connection (Skip Connection)
        # If input and output channels differ, we need a 1x1 convolution to match dimensions.
        # If they match, we can just add x directly.
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the temporal block.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Length).

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        # Main branch: Conv1 -> BN -> ReLU -> Dropout
        out = self.conv1(x)
        
        # Remove the extra padding from the END of the sequence.
        # Since we padded the start (left) to be causal, the convolution output extends
        # beyond the original length. We slice off the last 'padding' elements to
        # return to the original sequence length and alignment.
        out = out[:, :, : -self.padding] if self.padding > 0 else out
        
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        
        out = self.conv2(out)
        out = out[:, :, : -self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)


        # We add the original input 'x' to the processed output 'out'.
        # If the number of channels changed, we use 'downsample' (1x1 conv) to match dimensions.
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_out(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) for PV Power Forecasting.

    This model uses dilated causal convolutions to capture temporal dependencies at different scales.
    Causal padding ensures that the model does not violate the temporal order (no data leakage from the future).
    The receptive field grows exponentially with the network depth due to dilation.

    Args:
        model_config (dict): Configuration dictionary containing:
            - input_size (int): Number of input features.
            - output_size (int): Forecast horizon (number of steps to predict).
            - hidden_size (int): Number of filters (channels) in convolutional layers.
            - num_layers (int): Number of stacked TemporalBlocks.
            - kernel_size (int): Size of the convolutional kernel.
            - dropout (float): Dropout probability.

    Attributes:
        network (nn.Sequential): The stack of TemporalBlocks forming the TCN extraction backbone.
        fc (nn.Linear): Final linear layer mapping the extracted features to the forecast horizon.
    """

    def __init__(self, model_config: dict):
        super(TCN, self).__init__()

        # Extraction of config parameters
        self.input_size = model_config.get("input_size", 24)
        self.output_size = model_config.get("output_size", 24)
        self.hidden_size = model_config.get("hidden_size", 64)
        self.num_layers = model_config.get("num_layers", 4)
        self.kernel_size = model_config.get("kernel_size", 3)
        self.dropout = model_config.get("dropout", 0.2)

        print(f"TCN - Features: {self.input_size}")

        # --- Stack of Temporal Blocks ---
        layers = []
        for i in range(self.num_layers):
            # Exponentially increasing dilation (1, 2, 4, 8, ...)
            # This allows the receptive field to grow exponentially with depth
            dilation = 2**i

            # First layer maps input_size to hidden_size.
            # Subsequent layers map hidden_size to hidden_size.
            in_ch = self.input_size if i == 0 else self.hidden_size
            out_ch = self.hidden_size

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout,
                )
            )

        self.network = nn.Sequential(*layers)

        # Final Linear Layer:
        # Maps the extracted features (hidden_size) to the desired output prediction (horizon).
        # We take the output of the last temporal block at the last timestep and project it.
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the TCN.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Lookback, Input_Size).

        Returns:
            torch.Tensor: Forecast tensor of shape (Batch, Horizon, 1).
        """
        # Permute input for Conv1d compatibility
        # From: (Batch, Lookback, Input_Size) To: (Batch, Input_Size, Lookback)
        x = x.permute(0, 2, 1)

        # Extract features using the TCN backbone
        # Output shape: (Batch, hidden_size, Lookback)
        out = self.network(x)

        # Select the Last Timestep
        # Since convolutions are causal, the last timestep contains information
        # aggregated from the entire receptive field (history).
        # We take all features at the last time step. Shape: (Batch, hidden_size)
        out = out[:, :, -1]

        # Project feature vector to the forecast horizon
        # Shape: (Batch, Horizon)
        prediction = self.fc(out)

        # Reshape for training compatibility
        # Add a dimension to match expected shape (Batch, Horizon, 1)
        return prediction.unsqueeze(-1)
