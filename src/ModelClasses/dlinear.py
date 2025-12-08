import torch
import torch.nn as nn


class DLinearM(nn.Module):
    """
    DLinear-M (Multivariate): Decomposition Linear Model.

    Decompone la serie temporale in Trend e Seasonal.
    Usa TUTTE le features per la predizione (cattura relazioni cross-feature).
    """

    def __init__(self, model_config: dict, train_loader):
        super(DLinearM, self).__init__()

        # Parametri da configurazione
        self.lookback = model_config.get("lookback", 48)
        self.horizon = model_config.get("horizon", 24)
        self.kernel_size = model_config.get("kernel_size", 25)

        # Lettura dinamica delle features dal loader
        self.input_size = train_loader.dataset.data_tensor.shape[1]

        print(f"DLinearM (Multivariate) - Features: {self.input_size}")

        # Kernel dispari per simmetria
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.padding = (self.kernel_size - 1) // 2

        # Average Pooling 1D per il trend
        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size, stride=1, padding=self.padding
        )

        # Linear layers: tutte le features flatten
        self.linear_trend = nn.Linear(self.lookback * self.input_size, self.horizon)
        self.linear_seasonal = nn.Linear(self.lookback * self.input_size, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, Lookback, Input_Size)
        Output:  (Batch, Horizon, 1)
        """
        batch_size = x.size(0)

        x_permuted = x.permute(0, 2, 1)  # (Batch, Input_Size, Lookback)

        # Decomposizione
        trend = self.avg_pool(x_permuted)
        seasonal = x_permuted - trend

        # Flatten
        trend = trend.permute(0, 2, 1).reshape(batch_size, -1)
        seasonal = seasonal.permute(0, 2, 1).reshape(batch_size, -1)

        # Predizione
        prediction = self.linear_trend(trend) + self.linear_seasonal(seasonal)

        return prediction.unsqueeze(-1)


class DLinearI(nn.Module):
    """
    DLinear-I (Individual): Decomposition Linear Model (paper originale).

    Usa SOLO il target (pv_power) per la predizione.
    Molto più leggero, non cattura relazioni cross-feature.
    """

    def __init__(self, model_config: dict, train_loader):
        super(DLinearI, self).__init__()

        # Parametri da configurazione
        self.lookback = model_config.get("lookback", 48)
        self.horizon = model_config.get("horizon", 24)
        self.kernel_size = model_config.get("kernel_size", 25)

        # Target index dal dataset
        self.target_idx = train_loader.dataset.target_col_idx

        print(f"DLinearI (Individual) - Target idx: {self.target_idx}")

        # Kernel dispari per simmetria
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.padding = (self.kernel_size - 1) // 2

        # Average Pooling 1D per il trend
        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size, stride=1, padding=self.padding
        )

        # Linear layers: solo lookback (1 feature)
        self.linear_trend = nn.Linear(self.lookback, self.horizon)
        self.linear_seasonal = nn.Linear(self.lookback, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, Lookback, Input_Size)
        Output:  (Batch, Horizon, 1)
        """
        # Estrai solo il target
        x_target = x[:, :, self.target_idx]  # (Batch, Lookback)

        # Decomposizione
        x_for_pool = x_target.unsqueeze(1)  # (Batch, 1, Lookback)
        trend = self.avg_pool(x_for_pool).squeeze(1)  # (Batch, Lookback)
        seasonal = x_target - trend

        # Predizione
        prediction = self.linear_trend(trend) + self.linear_seasonal(seasonal)

        return prediction.unsqueeze(-1)
