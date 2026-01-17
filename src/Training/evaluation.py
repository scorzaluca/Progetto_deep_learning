import torch
import numpy as np


def evaluate_model(model, val_loader, device, scaler=None, target_idx=None):
    """
    Runs inference on the entire validation/test set and optionally Denormalizes the results.

    Unlike `validate_one_epoch` which only returns aggregated metrics (loss, mae),
    this function returns the **raw predictions** and **targets** (arrays).

    It is used at the END of training (or during inference) to:
    1. Collect all predictions.
    2. Denormalize them back to real units (Watts) if a scaler is provided.
    3. Enable plotting and detailed error analysis.

    Args:
        model: Trained model to evaluate.
        val_loader: DataLoader for validation/test set.
        device: Device (cuda/cpu).
        scaler: sklearn MinMaxScaler used for preprocessing (optional).
        target_idx: Index of the target column (pv_power) in the scaler (optional).

    Returns:
        tuple: (all_preds, all_targets)
               - If scaler is None: Normalized values (0-1).
               - If scaler is provided: Values in original units (Watts).
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Disable gradient calculation (Inference Mode)
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)

            # Forward pass: Generate predictions
            outputs = model(batch_x)
            
            # Move to CPU and convert to NumPy for easier handling
            preds = outputs.cpu().numpy()
            targets = batch_y.numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    # Concatenate all batches into single huge arrays
    # Shape: (Total_Samples, Horizon)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Optional Denormalization 
    # If the scaler is provided, we convert the predictions back to the original scale (Watts).
    # This is crucial because metrics like MAE should be interpretable in the real world.
    if scaler is not None and target_idx is not None:
        # The MinMaxScaler formula is: X_scaled = (X - X_min) / (X_max - X_min)
        # To invert it: X = X_scaled * (X_max - X_min) + X_min
        
        # We extract min and max only for the target column (pv_power)
        data_min = scaler.data_min_[target_idx]
        data_max = scaler.data_max_[target_idx]
        data_range = data_max - data_min

        # Apply the inverse transformation
        all_preds = all_preds * data_range + data_min
        all_targets = all_targets * data_range + data_min

    return all_preds, all_targets
