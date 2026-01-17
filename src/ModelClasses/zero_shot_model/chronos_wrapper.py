import torch
from chronos import Chronos2Pipeline


class ChronosWrapper:
    """
    Wrapper for the Chronos-2 foundation model (Zero-Shot Forecasting).

    This class handles the interface between our data structures (PyTorch Tensors)
    and the Chronos pipeline, which expects specific dictionary formats.
    It supports zero-shot inference, meaning the model forecasts without being trained on our specific data.

    Attributes:
        pipeline (Chronos2Pipeline): The pre-trained Chronos pipeline loaded from HuggingFace.
        model_name (str): Identifier for the model version (e.g., 'chronos-2').
    """
    
    def __init__(self):
        self.model_name = "chronos-2"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Loading {self.model_name} on {device} with {torch_dtype}...")

        # Load the pipeline directly from HuggingFace
        self.pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=device,
            torch_dtype=torch_dtype,
        )
    
    def predict(self, context: torch.Tensor, horizon: int, val_loader) -> torch.Tensor:
        """
        Performs zero-shot forecasting using Chronos.

        This method adapts our input data format (Tensor) to the dictionary format expected
        by Chronos. It handles one sample at a time, collects predictions,
        and reassembles them into a tensor.

        Args:
            context (torch.Tensor): Input tensor of shape (Batch, Lookback, Num_Features).
                                    These are the past observed values.
            horizon (int): The number of future steps to predict.
            val_loader (DataLoader): Validation dataloader (used to access dataset metadata
                                     like target_col_idx).

        Returns:
            torch.Tensor: Predicted values of shape (Batch, Horizon, 1).
                          Note: Chronos returns probabilistic distributions; here we
                          return the MEDIAN (0.5 quantile) as the point forecast.
        """
        # Retrieve the target column index from the dataset
        target_idx = val_loader.dataset.target_col_idx
        
        batch_size = context.shape[0]
        
        predictions = []
        # Loop through each sample in the batch individually.
        # Chronos expects a list of dictionaries, one per time series.
        for i in range(batch_size):
            # Extract Target Series:
            # Take the specific column we want to predict (e.g., PV Power) from the context window.
            # Convert to numpy array as expected by Chronos.
            target_series = context[i, :, target_idx].cpu().numpy()  # shape: (Lookback,)
            
            # Extract Past Covariates (Optional but helpful):
            # All OTHER columns in the input are treated as "past covariates" (context features).
            # Iterate through features and add them if they are not the target.
            past_covs = {}
            num_features = context.shape[2]
            for feat_idx in range(num_features):
                if feat_idx != target_idx:
                    past_covs[f"feat_{feat_idx}"] = context[i, :, feat_idx].cpu().numpy()  # shape: (Lookback,)
            
            # Construct Input Dictionary:
            # Formatting the data for the pipeline.
            # We do NOT provide 'future_covariates' here to strictly avoid data leakage
            input_dict = {
                "target": target_series,
                "past_covariates": past_covs if past_covs else None,
            }
            
            # Run Inference
            # The pipeline handles tokenization, inference, and de-tokenization.
            preds = self.pipeline.predict(
                inputs=[input_dict],
                prediction_length=horizon
            )
            
            # Extract Prediction
            # The output 'preds' is a list (one per input).
            # Shape is (Num_Samples, Quantiles, Horizon).
            # We select:
            # - [0]: The first (and only) input sample
            # - [0, 1, :]: Sample 0, Quantile index 1 (the Median), all time steps.
            pred_raw = preds[0][0, 1, :]
            
            # Ensure it's a tensor
            pred = pred_raw if isinstance(pred_raw, torch.Tensor) else torch.tensor(pred_raw)
            predictions.append(pred)
        
        # Stack predictions back into a batch tensor
        # Shape:From list of (Horizon,) -> (Batch, Horizon)
        pred_tensor = torch.stack(predictions, dim=0)
        
        # Add the feature dimension to match (Batch, Horizon, 1) output format
        return pred_tensor.unsqueeze(-1)
        
    def get_model_info(self) -> dict:
        """
        Returns data about the loaded Chronos model.
        Useful for logging and tracking during experiments.

        Returns:
            dict: Dictionary containing model type, name, usage type ('pretrained'),
                  and parameter count.
        """
        return {
            "model_type": "chronos-2",
            "model_name": self.model_name,
            "parameters": self._get_param_count(),
            "pretrained": True
        }

    def _get_param_count(self) -> str:
        """
        Calculates and formats the total number of parameters in the model.
        
        Returns:
            str: string representing parameter count 
                 
        """
        try:
            # Sum up all elements (weights/biases) in the model parameters
            total_params = sum(p.numel() for p in self.pipeline.model.parameters())
            
            # Format the number based on magnitude (Billions, Millions, Thousands)
            if total_params >= 1e9:
                return f"{total_params/1e9:.1f}B"
            elif total_params >= 1e6:
                return f"{total_params/1e6:.1f}M"
            else:
                return f"{total_params/1e3:.1f}K"
        except Exception:
            return "Unknown"