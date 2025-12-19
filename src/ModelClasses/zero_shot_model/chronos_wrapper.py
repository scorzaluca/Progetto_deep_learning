import torch
from .zero_shot_wrapper import ZeroShotWrapper
from chronos import Chronos2Pipeline


# src/ModelClasses/ChronosWrapper.py
class ChronosWrapper(ZeroShotWrapper):
    """Wrapper for Chronos model."""
    
    def __init__(self):
        """
        Initialize Chronos-2
        """
        self.model_name = "chronos-2"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",  # Non più f"amazon/chronos-t5-{model_size}"
            device_map=device,
            torch_dtype=torch_dtype,
        )
    
    def predict(self, context: torch.Tensor, horizon: int, val_loader) -> torch.Tensor:
        """
        Make predictions zero-shot on normalized data.
        
        Args:
            context: Tensor (batch_size, lookback, features) - normalized data [0,1]
            horizon: step to predict (es. 24)
            val_loader: validation dataloader
        Returns:
            Tensor (batch_size, horizon, 1) - normalized predictions [0,1]
        """
        # Ottieni l'indice della colonna target dal dataset (COME IN NAIVE.PY!) 
        target_idx = val_loader.dataset.target_col_idx
        
        batch_size = context.shape[0]
        
        predictions = []
        for i in range(batch_size):
            # Target: solo la colonna target dal lookback
            target_series = context[i, :, target_idx].cpu().numpy()  # shape: (48,)
            
            # Past covariates: tutte le altre colonne dal lookback
            past_covs = {}
            num_features = context.shape[2]
            for feat_idx in range(num_features):
                if feat_idx != target_idx:
                    past_covs[f"feat_{feat_idx}"] = context[i, :, feat_idx].cpu().numpy()  # shape: (48,)
            
            # Input strutturato per Chronos-2
            input_dict = {
                "target": target_series,  # (48,)
                "past_covariates": past_covs if past_covs else None,
                # NO future_covariates -> niente data leakage!
            }
            
            # Chiamata Chronos-2
            preds = self.pipeline.predict(
                inputs=[input_dict],
                prediction_length=horizon
            )
            
            # Output: preds è una lista, preds[0] ha shape (1, n_quantiles, horizon)
            # Prendiamo la mediana (quantile centrale, indice 1 se [0.1, 0.5, 0.9])
            pred_raw = preds[0][0, 1, :]
            pred = pred_raw if isinstance(pred_raw, torch.Tensor) else torch.tensor(pred_raw)
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