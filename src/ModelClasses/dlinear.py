import torch
import torch.nn as nn


# ==============================================================================
# Decomposition module (from original LTSF-Linear paper)
# https://github.com/cure-lab/LTSF-Linear
# ==============================================================================


class MovingAvg(nn.Module):
    """
    Moving average block to extract the trend from the time series.
    Uses replication padding at the boundaries (as in the original paper).

    Args:
        kernel_size (int): Size of the moving average window.
        stride (int, optional): Stride of the window. Defaults to 1.

    Attributes:
        kernel_size (int): Size of the moving average window.
        avg (nn.AvgPool1d): Average pooling layer used to compute the mean.
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        super(MovingAvg, self).__init__()  # Initialize the parent class
        self.kernel_size = kernel_size
        # Initialize the average pooling layer
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the moving average of the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_len, Channels).

        Returns:
            torch.Tensor: Moving average tensor of shape (Batch, Seq_len, Channels).
        """
        # Padding replication at the boundaries
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # AvgPool1d requires (Batch, Channels, Seq_len)
        x = self.avg(x.permute(0, 2, 1))  # change the shape of the tensor
        x = x.permute(0, 2, 1)  # change the shape of the tensor
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block: separates time series into trend and seasonal components.

    Args:
        kernel_size (int): Size of the moving average window used for decomposition.

    Attributes:
        moving_avg (MovingAvg): The moving average module used to extract the trend.
    """

    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(
            kernel_size, stride=1
        )  # initialize the moving average module

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes the input time series into seasonal and trend components.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_len, Channels).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - seasonal (torch.Tensor): The seasonal component (Batch, Seq_len, Channels).
                - trend (torch.Tensor): The trend component (Batch, Seq_len, Channels).
        """
        moving_mean = self.moving_avg(
            x
        )  # using the moving average module to extract the trend
        residual = (
            x - moving_mean
        )  # subtract the trend from the input to get the seasonal component
        return residual, moving_mean


# ==============================================================================
# DLinear (Paper Originale - Channel-Independent, Weight-Shared)
# ==============================================================================


class DLinearM(nn.Module):
    """
    DLinear (Original Paper): Decomposition-Linear Model (Multivariate/Channel-Independent).

    This implementation follows the original paper "Are Transformers Effective for Time Series Forecasting?"
    (Zeng et al., AAAI 2023).

    Key Features:
    - Channel-independent: Each channel is processed separately.
    - Weight-shared: Linear layers are SHARED across all channels.
    - Decomposition: Separates trend and seasonal components using moving averages.
    - Output: Predicts all channels, then extracts only the target (pv_power).

    Args:
        model_config (dict): Configuration dictionary containing:
            - input_size (int): Number of input channels.
            - target_idx (int): Index of the target column.
            - lookback (int): Input sequence length.
            - horizon (int): Prediction horizon.
            - kernel_size (int): Moving average kernel size.

    Attributes:
        input_size (int): Number of input channels.
        target_idx (int): Index of the target column.
        lookback (int): Input sequence length.
        horizon (int): Prediction horizon.
        kernel_size (int): Moving average kernel size.
        decomposition (SeriesDecomp): Module for series decomposition.
        linear_seasonal (nn.Linear): Linear layer for the seasonal component.
        linear_trend (nn.Linear): Linear layer for the trend component.
    """

    def __init__(self, model_config: dict):
        super(DLinearM, self).__init__()

        # Configuration parameters (extracting parameters from the config dictionary)
        self.input_size = model_config.get("input_size", 24)  # num channels
        self.target_idx = model_config.get("target_idx", 23)  # indice pv_power
        self.lookback = model_config.get("lookback", 48)  # seq_len
        self.horizon = model_config.get("horizon", 24)  # pred_len
        self.kernel_size = model_config.get("kernel_size", 25)

        # Kernel size must be odd for symmetry (if the kernel size is even, it is incremented by 1)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        print(
            f"DLinearM (Paper Original) - Channels: {self.input_size}, "
            f"Target idx: {self.target_idx}, Kernel: {self.kernel_size}"
        )

        # Create the decomposition module
        self.decomposition = SeriesDecomp(self.kernel_size)

        # Create the linear layers for the seasonal and trend components
        # These layers are shared across all channels (individual=False in the paper)
        # They map: lookback -> horizon for each channel independently
        self.linear_seasonal = nn.Linear(self.lookback, self.horizon)
        self.linear_trend = nn.Linear(self.lookback, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of DLinearM.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Lookback, Channels).

        Returns:
            torch.Tensor: Predicted tensor of shape (Batch, Horizon, 1) containing only the target column.
        """
        # Decomposition: separate seasonal and trend components
        # seasonal, trend: (Batch, Lookback, Channels)
        seasonal, trend = self.decomposition(x)

        # Permute for applying Linear on lookback: (Batch, Channels, Lookback)
        # This is done because pytorch applies the linear layer on the last dimension
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        # Applying the linear layers
        seasonal_out = self.linear_seasonal(seasonal)
        trend_out = self.linear_trend(trend)

        # Summing the outputs and permuting back: (Batch, Horizon, Channels)
        output = (seasonal_out + trend_out).permute(0, 2, 1)

        # Extracting only the target (pv_power): (Batch, Horizon, 1)
        output = output[:, :, self.target_idx].unsqueeze(-1)

        return output


class DLinearI(nn.Module):
    """
    DLinear-I (Individual): Decomposition Linear Model (Univariate).

    This model focuses ONLY on the target variable (pv_power) for prediction.
    It uses the same decomposition strategy as the paper (moving average + replication padding),
    but applied only to the target channel. It is much lighter but does not capture cross-feature relationships.

    Args:
        model_config (dict): Configuration dictionary containing:
            - target_idx (int): Index of the target column.
            - lookback (int): Input sequence length.
            - horizon (int): Prediction horizon.
            - kernel_size (int): Moving average kernel size.

    Attributes:
        target_idx (int): Index of the target column.
        lookback (int): Input sequence length.
        horizon (int): Prediction horizon.
        kernel_size (int): Moving average kernel size.
        decomposition (SeriesDecomp): Module for series decomposition.
        linear_seasonal (nn.Linear): Linear layer for the seasonal component.
        linear_trend (nn.Linear): Linear layer for the trend component.
    """

    def __init__(self, model_config: dict):
        super(DLinearI, self).__init__()

        # Configuration parameters
        # input_size is NOT needed here because this model is UNIVARIATE (it only looks at the target column)
        self.target_idx = model_config.get("target_idx", 23)
        self.lookback = model_config.get("lookback", 48)
        self.horizon = model_config.get("horizon", 24)
        self.kernel_size = model_config.get("kernel_size", 25)

        # Kernel size must be odd for symmetry
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        print(
            f"DLinearI (Paper Original - Univariate) - Target idx: {self.target_idx}, "
            f"Kernel: {self.kernel_size}"
        )

        # Create the decomposition module
        self.decomposition = SeriesDecomp(self.kernel_size)

        # Create the linear layers for seasonal and trend components
        # Unlike DLinearM, these layers are applied ONLY to the target channel (1 channel)
        # There is no weight sharing across different variables because we only use one.
        self.linear_seasonal = nn.Linear(self.lookback, self.horizon)
        self.linear_trend = nn.Linear(self.lookback, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of DLinearI (Univariate).

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Lookback, Channels).
                              Note that only the target channel is used.

        Returns:
            torch.Tensor: Predicted tensor of shape (Batch, Horizon, 1).
        """
        # Extract only the target: (Batch, Lookback, 1)
        x_target = x[:, :, self.target_idx : self.target_idx + 1]

        # Decomposition: (Batch, Lookback, 1)
        seasonal, trend = self.decomposition(x_target)

        # Permute: (Batch, 1, Lookback)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        # Linear: (Batch, 1, Horizon)
        seasonal_out = self.linear_seasonal(seasonal)
        trend_out = self.linear_trend(trend)

        # Output: (Batch, Horizon, 1)
        output = (seasonal_out + trend_out).permute(0, 2, 1)

        return output
