import torch
import torch.nn as nn


# ==============================================================================
# Moduli di decomposizione (dal paper originale LTSF-Linear)
# https://github.com/cure-lab/LTSF-Linear
# ==============================================================================


class MovingAvg(nn.Module):
    """
    Moving average block per estrarre il trend dalla serie temporale.
    Usa padding replicato agli estremi (come nel paper originale).
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, Seq_len, Channels)
        Output:  (Batch, Seq_len, Channels)
        """
        # Padding replicato agli estremi (come nel paper)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # AvgPool1d richiede (Batch, Channels, Seq_len)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block: separa trend e seasonal.
    """

    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        """
        Input x: (Batch, Seq_len, Channels)
        Output: (seasonal, trend) entrambi (Batch, Seq_len, Channels)
        """
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


# ==============================================================================
# DLinear (Paper Originale - Channel-Independent, Weight-Shared)
# ==============================================================================


class DLinearM(nn.Module):
    """
    DLinear (Paper Originale): Decomposition-Linear Model.

    Implementazione IDENTICA al paper "Are Transformers Effective for Time Series Forecasting?"
    (Zeng et al., AAAI 2023) - https://github.com/cure-lab/LTSF-Linear

    Caratteristiche:
    - Channel-independent: ogni canale è processato separatamente
    - Weight-shared: i layer lineari sono CONDIVISI tra tutti i canali
    - Decomposizione: separa trend e seasonal con moving average
    - Output: predice tutti i canali, poi estrae solo il target (pv_power)
    """

    def __init__(self, model_config: dict):
        super(DLinearM, self).__init__()

        # Parametri da configurazione
        self.input_size = model_config.get("input_size", 24)  # num channels
        self.target_idx = model_config.get("target_idx", 23)  # indice pv_power
        self.lookback = model_config.get("lookback", 48)  # seq_len
        self.horizon = model_config.get("horizon", 24)  # pred_len
        self.kernel_size = model_config.get("kernel_size", 25)

        # Kernel dispari per simmetria
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        print(
            f"DLinearM (Paper Original) - Channels: {self.input_size}, "
            f"Target idx: {self.target_idx}, Kernel: {self.kernel_size}"
        )

        # Decomposizione (come nel paper)
        self.decomposition = SeriesDecomp(self.kernel_size)

        # Layer lineari CONDIVISI tra tutti i canali (individual=False nel paper)
        # Mappano: lookback -> horizon per ogni canale indipendentemente
        self.linear_seasonal = nn.Linear(self.lookback, self.horizon)
        self.linear_trend = nn.Linear(self.lookback, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, Lookback, Channels)
        Output:  (Batch, Horizon, 1)  [solo il target pv_power]
        """
        # Decomposizione: separa seasonal e trend
        # seasonal, trend: (Batch, Lookback, Channels)
        seasonal, trend = self.decomposition(x)

        # Permute per applicare Linear su lookback: (Batch, Channels, Lookback)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        # Linear condiviso applicato a ogni canale: (Batch, Channels, Horizon)
        seasonal_out = self.linear_seasonal(seasonal)
        trend_out = self.linear_trend(trend)

        # Somma e permute: (Batch, Horizon, Channels)
        output = (seasonal_out + trend_out).permute(0, 2, 1)

        # Estrai solo il target (pv_power): (Batch, Horizon, 1)
        output = output[:, :, self.target_idx].unsqueeze(-1)

        return output


class DLinearI(nn.Module):
    """
    DLinear-I (Individual): Decomposition Linear Model (paper originale).

    Usa SOLO il target (pv_power) per la predizione.
    Usa la stessa decomposizione del paper con moving average e padding replicato.
    Molto più leggero, non cattura relazioni cross-feature.
    """

    def __init__(self, model_config: dict):
        super(DLinearI, self).__init__()

        # Parametri da configurazione
        self.target_idx = model_config.get("target_idx", 23)
        self.lookback = model_config.get("lookback", 48)
        self.horizon = model_config.get("horizon", 24)
        self.kernel_size = model_config.get("kernel_size", 25)

        # Kernel dispari per simmetria
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        print(
            f"DLinearI (Paper Original - Univariate) - Target idx: {self.target_idx}, "
            f"Kernel: {self.kernel_size}"
        )

        # Decomposizione (stessa del paper)
        self.decomposition = SeriesDecomp(self.kernel_size)

        # Linear layers: solo lookback (1 canale)
        self.linear_seasonal = nn.Linear(self.lookback, self.horizon)
        self.linear_trend = nn.Linear(self.lookback, self.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, Lookback, Channels)
        Output:  (Batch, Horizon, 1)
        """
        # Estrai solo il target: (Batch, Lookback, 1)
        x_target = x[:, :, self.target_idx : self.target_idx + 1]

        # Decomposizione: (Batch, Lookback, 1)
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
