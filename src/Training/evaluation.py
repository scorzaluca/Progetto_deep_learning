import torch
import numpy as np


def evaluate_model(model, val_loader, device, scaler=None, target_idx=None):
    """
    Esegue predizioni su tutto il validation set e opzionalmente denormalizza.

    Args:
        model: Modello da valutare.
        val_loader: DataLoader del validation set.
        device: Device (cuda/cpu).
        scaler: MinMaxScaler usato per normalizzare i dati (opzionale).
        target_idx: Indice della colonna target (pv_power) (opzionale).

    Returns:
        tuple: (all_preds, all_targets)
               - Se scaler è None: valori normalizzati (0-1)
               - Se scaler è fornito: valori in unità reali (Watt)
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)

            outputs = model(batch_x)
            preds = outputs.cpu().numpy()
            targets = batch_y.numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalizzazione (opzionale)
    if scaler is not None and target_idx is not None:
        # MinMaxScaler: X_scaled = (X - X_min) / (X_max - X_min)
        # Inverse: X = X_scaled * (X_max - X_min) + X_min
        data_min = scaler.data_min_[target_idx]
        data_max = scaler.data_max_[target_idx]
        data_range = data_max - data_min

        all_preds = all_preds * data_range + data_min
        all_targets = all_targets * data_range + data_min

    return all_preds, all_targets
