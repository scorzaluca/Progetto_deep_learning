import torch
import numpy as np


def evaluate_model(model, val_loader, device, scaler=None, target_idx=None):
    """
    Esegue predizioni su tutto il validation set e denormalizza.

    Args:
        model: Modello da valutare.
        val_loader: DataLoader del validation set.
        device: Device (cuda/cpu).
        scaler: MinMaxScaler usato per normalizzare i dati.
        target_idx: Indice della colonna target (pv_power).

    Returns:
        tuple: (all_preds, all_targets) in unità reali (Watt).
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

    # Denormalizzazione
    if scaler is not None and target_idx is not None:
        scale_factor = scaler.scale_[target_idx]
        min_factor = scaler.min_[target_idx]
        all_preds = (all_preds - min_factor) / scale_factor
        all_targets = (all_targets - min_factor) / scale_factor

    return all_preds, all_targets
