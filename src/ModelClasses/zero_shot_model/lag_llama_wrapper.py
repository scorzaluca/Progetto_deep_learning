import torch
import pandas as pd
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.common import ListDataset
from .zero_shot_wrapper import ZeroShotWrapper


class LagLlamaWrapper(ZeroShotWrapper):
    """Wrapper per Lag-Llama."""
    
    def __init__(self, context_length=48):
        """
        Inizializza Lag-Llama.
        
        Args:
            context_length: lunghezza del context (default 48 = tuo lookback)
        """
        self.model_name = "lag-llama"
        self.context_length = context_length
        self.prediction_length = 24
        
        # RoPE scaling per context > 32
        rope_scaling = None
        if context_length > 32:
            rope_scaling = {
                "type": "linear",
                "factor": float(context_length / 32)
            }
        
        # Carica modello (scarica automaticamente da HuggingFace)
        from huggingface_hub import snapshot_download
        ckpt_path = snapshot_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            allow_patterns="*.ckpt"
        )
        
        self.estimator = LagLlamaEstimator(
            ckpt_path=f"{ckpt_path}/lag-llama.ckpt",
            prediction_length=self.prediction_length,
            context_length=context_length,
            rope_scaling=rope_scaling,
            num_parallel_samples=100,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Crea predictor
        self.predictor = self.estimator.create_predictor()
    
    def predict(self, context: torch.Tensor, horizon: int, val_loader) -> torch.Tensor:
        """
        Genera predizioni zero-shot su dati normalizzati.
        
        Args:
            context: Tensor (batch_size, lookback, features) - normalizzato [0,1]
            horizon: passi da predire (24)
            val_loader: DataLoader per target_col_idx
        
        Returns:
            Tensor (batch_size, horizon, 1) - normalizzato [0,1]
        """
        # Ottieni indice target (come ChronosWrapper)
        target_idx = val_loader.dataset.target_col_idx
        
        batch_size = context.shape[0]
        context_series = context[:, :, target_idx]  # (batch_size, lookback)
        
        predictions = []
        
        for i in range(batch_size):
            # Converti in numpy
            series = context_series[i].cpu().numpy()
            
            # Converti in GluonTS ListDataset
            dataset = ListDataset(
                [{
                    "start": pd.Timestamp("2020-01-01"),
                    "target": series
                }],
                freq="H"  # dati orari
            )
            
            # Predizione
            forecast_it, _ = make_evaluation_predictions(
                dataset=dataset,
                predictor=self.predictor,
                num_samples=100
            )
            
            # Estrai la media (come Chronos)
            forecast = next(iter(forecast_it))
            pred = torch.tensor(forecast.mean, dtype=torch.float32)
            predictions.append(pred)
        
        # Stack e reshape
        pred_tensor = torch.stack(predictions, dim=0)  # (batch_size, 24)
        return pred_tensor.unsqueeze(-1)  # (batch_size, 24, 1)
    
    def get_model_info(self) -> dict:
        return {
            "model_type": "lag-llama",
            "model_name": self.model_name,
            "parameters": self._get_param_count(),
            "pretrained": True,
            "context_length": self.context_length
        }
    
    def _get_param_count(self) -> str:
        """Conta parametri del modello."""
        try:
            model = self.estimator.create_lightning_module()
            total_params = sum(p.numel() for p in model.parameters())
            
            if total_params >= 1e9:
                return f"{total_params/1e9:.1f}B"
            elif total_params >= 1e6:
                return f"{total_params/1e6:.1f}M"
            else:
                return f"{total_params/1e3:.1f}K"
        except Exception:
            return "Unknown"