import torch
import torch.nn as nn


class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear Model
    
    Questo modello decompone la serie temporale in due componenti:
    1. Trend: la componente di tendenza a lungo termine
    2. Seasonal (Stagionale): le fluttuazioni periodiche
    
    Ogni componente viene elaborata da un layer lineare separato.
    Le predizioni finali sono la somma delle due componenti.
    """
    
    def __init__(self, model_config: dict, train_loader):
        super(DLinear, self).__init__()
        
        # Parametri da configurazione
        self.lookback = model_config.get("lookback", 48)
        self.horizon = model_config.get("horizon", 24)
        self.kernel_size = model_config.get("kernel_size", 25)
        
        # --- MODIFICA: Lettura dinamica delle features dal loader ---
        # Accede al dataset dentro il loader e legge la shape del tensore dati
        self.input_size = train_loader.dataset.data_tensor.shape[1]
            
        # DECOMPOSIZIONE: Moving Average per estrarre il trend
        # Il kernel_size determina quanto è "liscia" la media mobile
        # Deve essere un numero dispari per avere simmetria
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
            
        # Padding per mantenere la stessa lunghezza temporale dopo la convoluzione
        self.padding = (self.kernel_size - 1) // 2
        
        # Average Pooling 1D per calcolare la media mobile (trend)
        # groups=input_size significa che ogni feature viene processata indipendentemente
        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding
        )
        
        # --- LINEAR LAYERS ---
        # Due layer lineari separati: uno per il trend, uno per la stagionalità
        # Input: (Batch, Lookback, Input_Size) flatten to (Batch, Lookback * Input_Size)
        # Output: (Batch, Horizon)
        
        self.linear_trend = nn.Linear(
            self.lookback * self.input_size,
            self.horizon
        )
        
        self.linear_seasonal = nn.Linear(
            self.lookback * self.input_size,
            self.horizon
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modello DLinear.
        
        Input x: (Batch, Lookback, Input_Size) -> es. (64, 48, 10)
        Output:  (Batch, Horizon, 1) -> es. (64, 24, 1)
        """
        batch_size = x.size(0)
        
        # --- PASSO 1: DECOMPOSIZIONE ---
        # Spostiamo le dimensioni per applicare AvgPool1d
        # AvgPool1d si aspetta (Batch, Channels, Length)
        # Quindi trasformiamo da (Batch, Lookback, Input_Size) a (Batch, Input_Size, Lookback)
        x_permuted = x.permute(0, 2, 1)  # (Batch, Input_Size, Lookback)
        
        # Calcoliamo il TREND usando una media mobile
        # L'AvgPool1d calcola la media su finestre temporali scorrevoli
        trend = self.avg_pool(x_permuted)  # (Batch, Input_Size, Lookback)
        
        # Calcoliamo la componente STAGIONALE sottraendo il trend dall'input
        seasonal = x_permuted - trend  # (Batch, Input_Size, Lookback)
        
        # --- PASSO 2: FLATTEN ---
        # Riportiamo le dimensioni a (Batch, Lookback, Input_Size)
        trend = trend.permute(0, 2, 1)  # (Batch, Lookback, Input_Size)
        seasonal = seasonal.permute(0, 2, 1)  # (Batch, Lookback, Input_Size)
        
        # Flatten per passare ai layer lineari
        # Da (Batch, Lookback, Input_Size) a (Batch, Lookback * Input_Size)
        trend_flat = trend.reshape(batch_size, -1)
        seasonal_flat = seasonal.reshape(batch_size, -1)
        
        # --- PASSO 3: PREDIZIONE ---
        # Ogni componente passa attraverso il proprio layer lineare
        trend_pred = self.linear_trend(trend_flat)  # (Batch, Horizon)
        seasonal_pred = self.linear_seasonal(seasonal_flat)  # (Batch, Horizon)
        
        # Sommiamo le due predizioni
        prediction = trend_pred + seasonal_pred  # (Batch, Horizon)
        
        # --- PASSO 4: RESHAPE PER IL TRAINING LOOP ---
        # Il training loop si aspetta (Batch, Horizon, 1)
        # Aggiungiamo una dimensione fittizia alla fine
        return prediction.unsqueeze(-1)  # (Batch, Horizon, 1)
