import torch
import torch.nn as nn


class NaivePersistence(nn.Module):
    """
    Naive Persistence Model (Daily Persistence).

    This model simply predicts that the future will be identical to the same time window from the previous day.
    It serves as a baseline for evaluating the performance of more complex models (e.g., when calculating MASE).

    Args:
        model_config (dict, optional): Configuration dictionary. Defaults to None.
            - target_idx (int): Index of the target column.
            - horizon (int): Prediction horizon.

    Attributes:
        target_idx (int): Index of the target column.
        horizon (int): Prediction horizon (number of steps to predict).
    """

    def __init__(self, model_config: dict = None):
        super(NaivePersistence, self).__init__()

        # Handle case where model_config is not provided
        if model_config is None:
            model_config = {}

        # Configuration parameters
        # target_idx: The column index of the variable we want to predict (PV power)
        self.target_idx = model_config.get("target_idx", 23)
        
        # horizon: How many steps into the future we want to predict
        self.horizon = model_config.get("horizon", 24)

        print(f"NaivePersistence - target_idx: {self.target_idx}, horizon: {self.horizon}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the Naive Persistence model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Lookback, Input_Size).

        Returns:
            torch.Tensor: Predicted tensor of shape (Batch, Horizon, 1).
        """

        # Strategy: "Daily Persistence"
        # The input x contains 'lookback' hours of history.
        # The last 24 hours (from t-23 to t) represent "today".
        # We use today's profile to predict tomorrow's profile, assuming they will be identical.

        # Slicing the input tensor:
        # :                => Select all examples in the batch
        # -self.horizon:   => Select only the last 'horizon' time steps (the most recent 24h)
        # self.target_idx  => Select only the target variable (PV Power)
        last_day_values = x[:, -self.horizon :, self.target_idx]

        # Current shape: (Batch, Horizon)
        # The training loop expects shape (Batch, Horizon, 1) for loss calculation.
        return last_day_values.unsqueeze(-1)
