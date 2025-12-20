"""
Script per valutare modelli zero-shot usando la stessa metodologia dello screening.
"""

import json
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
import numpy as np

from .config import TARGET_COL, SAMPLING_CONFIG, NAIVE_MAE_PER_FOLD, RESULTS_DIR
from .DataLoading import TS_Cross_Validator
from .ModelClasses import ChronosWrapper
from .Utils import plot_predictions

# Configurazione
FOLD_INDEX = 2  # Stesso fold dello screening (il più grande)
DATA_PATH = "data/processed/preprocessed_ds.csv"

# Modelli zero-shot da valutare
MODELS_TO_EVALUATE = {
    "chronos-2": ChronosWrapper,
}


def load_data_and_fold():
    """
    Carica dati e seleziona fold 2 (come screening).
    
    Returns:
        tuple: (train_loader, val_loader, scaler) per il fold selezionato
    """
    print(f"\nCaricamento dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    
    # Crea cross-validator
    validator = TS_Cross_Validator(df, target_col=TARGET_COL, cfg_dict=SAMPLING_CONFIG)
    folds = list(validator.get_folds())
    
    # Seleziona fold 2 (stesso dello screening)
    if len(folds) <= FOLD_INDEX:
        raise ValueError(f"Fold {FOLD_INDEX} non disponibile. Totale fold: {len(folds)}")
    
    train_loader, val_loader, scaler = folds[FOLD_INDEX]
    
    print(f"\nFold {FOLD_INDEX + 1} selezionato:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Baseline MAE (Naive): {NAIVE_MAE_PER_FOLD[FOLD_INDEX]:.6f}")
    
    return train_loader, val_loader, scaler


def evaluate_model(wrapper, val_loader, scaler,device, fold_idx, model_name):
    """
    Valuta un modello zero-shot sul validation set.
    IDENTICO alla logica di optuna_optimizer.py righe 155-166.
    
    Args:
        wrapper: Wrapper del modello zero-shot
        val_loader: DataLoader per validazione (SOLO VALIDATION!)
        device: torch.device (cuda/cpu)
        fold_idx: Indice del fold (per baseline MAE)
    
    Returns:
        float: MASE score
    """
    # MAE function e baseline
    mae_fn = nn.L1Loss()
    baseline_mae = NAIVE_MAE_PER_FOLD[fold_idx]
    
    print(f"\n  Inferenza su validation set ({len(val_loader)} batch)...")
    running_mae = 0.0

    all_preds = []
    all_targets = []
    
    target_idx = val_loader.dataset.target_col_idx
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
            # Passa batch_x e batch_y al device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Predizione zero-shot (passa val_loader per target_col_idx)
            pred = wrapper.predict(batch_x, horizon=24, val_loader=val_loader)
            pred = pred.to(device)
            
            # Calcola MAE
            running_mae += mae_fn(pred, batch_y).item()

            all_preds.append(pred[:, 0, 0].cpu().numpy())
            all_targets.append(batch_y[:, 0, 0].cpu().numpy())
            
            # Progress ogni 50 batch
            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx + 1}/{len(val_loader)}")
    
    avg_mae = running_mae / len(val_loader)
    mase = avg_mae / baseline_mae
    
    print(f"  MAE: {avg_mae:.6f}")
    print(f"  MASE: {mase:.4f}")

    # Concatena e denormalizza per plot
    preds_flat = np.concatenate(all_preds)
    targets_flat = np.concatenate(all_targets)
    
    # Denormalizza (target è l'unica colonna, usa inverse_transform parziale)
    # Crea array dummy per inverse_transform
    n_features = scaler.n_features_in_
    preds_full = np.zeros((len(preds_flat), n_features))
    targets_full = np.zeros((len(targets_flat), n_features))
    preds_full[:, target_idx] = preds_flat
    targets_full[:, target_idx] = targets_flat
    
    preds_denorm = scaler.inverse_transform(preds_full)[:, target_idx]
    targets_denorm = scaler.inverse_transform(targets_full)[:, target_idx]

    plot_predictions(preds_denorm, targets_denorm, model_name, fold_idx, RESULTS_DIR)
    
    return mase



def main():
    print("\n" + "=" * 70)
    print("ZERO-SHOT MODEL EVALUATION (SCREENING METHODOLOGY)")
    print("=" * 70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Carica dati (NOTA: usiamo solo val_loader!)
    train_loader, val_loader, scaler = load_data_and_fold()
    
    # Risultati
    results = {}
    
    # Evalua ogni modello
    for model_name, wrapper_class in MODELS_TO_EVALUATE.items():
        print(f"\n{'=' * 70}")
        print(f"EVALUATING: {model_name.upper()}")
        print(f"{'=' * 70}")
        
        try:
            # Istanzia wrapper
            print(f"\nCaricamento modello {model_name}...")
            wrapper = wrapper_class()
            
            # Ottieni info modello
            model_info = wrapper.get_model_info()
            print(f"  Tipo: {model_info.get('model_type', 'N/A')}")
            print(f"  Parametri: {model_info.get('parameters', 'N/A')}")
            
            # Valuta (SOLO SU VAL_LOADER!)
            mase = evaluate_model(wrapper, val_loader, scaler, device, FOLD_INDEX, model_name)
            
            # Salva risultati
            results[model_name] = {
                "best_mase": float(mase),
                "model_info": model_info,
                "fold_used": FOLD_INDEX + 1,
            }
            
            print(f"\n✓ {model_name.upper()}: MASE = {mase:.4f}")
            
        except Exception as e:
            print(f"\n✗ {model_name.upper()}: ERRORE - {str(e)}")
            import traceback
            traceback.print_exc()
            
            results[model_name] = {
                "best_mase": float("inf"),
                "error": str(e),
            }
    
    # Crea ranking
    valid_models = [m for m in results.keys() if results[m]["best_mase"] < float("inf")]
    ranking = sorted(valid_models, key=lambda m: results[m]["best_mase"])
    results["ranking"] = ranking
    
    # Salva risultati
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = f"{RESULTS_DIR}/zero_shot_results.json"
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    if ranking:
        print(f"\nRanking:")
        for i, model in enumerate(ranking, 1):
            mase = results[model]["best_mase"]
            print(f"  {i}. {model}: MASE = {mase:.4f}")
    
    print(f"\nRisultati salvati in: {results_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()