import torch
from .zero_shot_wrapper import ZeroShotWrapper
from chronos import BaseChronosPipeline


# src/ModelClasses/ChronosWrapper.py
class ChronosWrapper(ZeroShotWrapper):
    """Wrapper per modelli Chronos."""
    
    def __init__(self):
        """
        Inizializza Chronos-2
        """
        self.model_name= "chronos-2"
        
        self.pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2",  # Non più f"amazon/chronos-t5-{model_size}"
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16
        )
    
    def predict(self, context: torch.Tensor, horizon: int, val_loader) -> torch.Tensor:
        """
        Genera predizioni zero-shot su dati normalizzati.
        
        Args:
            context: Tensor (batch_size, lookback, features) - dati normalizzati [0,1]
            horizon: numero di passi da predire (es. 24)
            val_loader: DataLoader per accedere a target_col_idx
        
        Returns:
            Tensor (batch_size, horizon, 1) - predizioni normalizzate [0,1]
        """
        # Ottieni l'indice della colonna target dal dataset (COME IN NAIVE.PY!)
        target_idx = val_loader.dataset.target_col_idx
        
        batch_size = context.shape[0]
        
        # Estrai la colonna target usando l'indice dinamico
        context_series = context[:, :, target_idx]  # (batch_size, lookback)
        
        predictions = []
        for i in range(batch_size):
            series = context_series[i].unsqueeze(0)  # (1, lookback)
            
            # Predizione Chronos
            quantiles, mean = self.pipeline.predict_quantiles(
                context=series,
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9]
            )
            
            pred = mean.squeeze(0)  # (horizon,)
            predictions.append(pred)
        
        # Stack e reshape
        pred_tensor = torch.stack(predictions, dim=0)  # (batch_size, horizon)
        return pred_tensor.unsqueeze(-1)  # (batch_size, horizon, 1)
        
    def get_model_info(self) -> dict:
        return {
            "model_type": "chronos-2",
            "model_name": self.model_name,  # Non più model_size
            "parameters": self._get_param_count(),
            "pretrained": True
        }

    def _get_param_count(self) -> str:
        """Conta parametri del modello Chronos-2."""
        try:
            total_params = sum(p.numel() for p in self.pipeline.model.parameters())
            
            if total_params >= 1e9:
                return f"{total_params/1e9:.1f}B"
            elif total_params >= 1e6:
                return f"{total_params/1e6:.1f}M"
            else:
                return f"{total_params/1e3:.1f}K"
        except Exception:
            return "Unknown"