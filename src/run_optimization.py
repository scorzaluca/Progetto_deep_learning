"""
Script per l'ottimizzazione degli iperparametri con Optuna.
Configurazione manuale in src/config/tuning_config.py.

Uso:
    python -m src.run_optimization

Il flusso:
1. Carica configurazione da tuning_config.py
2. Crea/riprende studio Optuna
3. Esegue ottimizzazione (media MASE sui fold)
4. Salva best params in JSON
5. Riaddestra su TUTTO il dataset con best params
6. Salva checkpoint .pth
"""

import os
import torch
import pandas as pd

from .config import (
    TARGET_COL,
    SEED,
    # Tuning config
    MODEL_NAME,
    N_TRIALS,
    N_FOLDS,
    TUNING_EPOCHS,
    PATIENCE,
    STUDY_NAME,
    NEW_STUDY,
    RESULTS_DIR,
    # Training config
    NAIVE_MAE_FINAL_FOLD,
)
from .Tuning import OptunaOptimizer
from .Utils import set_seed, get_device, load_data_and_folds, save_results


def train_final_model(model_name: str, best_params: dict, df: pd.DataFrame, device):
    """
    Riaddestra il modello con i best params su 22 mesi (train) + 2 mesi (val).
    Usa early stopping per evitare overfitting/underfitting.
    Salva il checkpoint in results/checkpoints/.

    Args:
        model_name: nome del modello
        best_params: iperparametri ottimali
        df: DataFrame con tutti i dati
        device: torch device

    Returns:
        str: percorso del checkpoint salvato
    """
    from .Training.engine import create_model, fit_model
    from .DataLoading import create_final_train_val_loaders

    print("\n" + "=" * 60)
    print("RETRAINING FINALE (22 mesi train + 2 mesi val)")
    print("=" * 60)

    # Crea train/val loaders (22 mesi train, 2 mesi val con early stopping)
    train_loader, val_loader, scaler = create_final_train_val_loaders(
        df, target_col=TARGET_COL
    )

    # Crea modello
    model = create_model(model_name, best_params)
    model.to(device)

    # Estrai parametri
    lr = best_params.get("lr", 0.001)
    grad_clip_norm = best_params.get("grad_clip_norm", 1.0)

    print(f"Training {model_name.upper()} con early stopping (patience={PATIENCE})...")

    # Usa fit_model con early stopping (stessa logica dell'ottimizzazione)
    # Passa baseline_mae direttamente invece di fold_idx per il final training
    model, history, best_epoch = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=TUNING_EPOCHS,
        lr=lr,
        device=device,
        fold_idx=None,  # Non è un fold, usa baseline_mae
        patience=PATIENCE,
        trial=None,  # Nessun pruning Optuna
        grad_clip_norm=grad_clip_norm,
        verbose=True,  # Verbose per vedere il progresso
        baseline_mae=NAIVE_MAE_FINAL_FOLD,  # MAE naive per calcolo MASE
    )

    # Estrai il best MASE dalla history (all'epoca migliore)
    best_mase = history["val_mase"][best_epoch]
    best_rmse = history["val_rmse"][best_epoch]

    # Salva checkpoint
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{STUDY_NAME}.pth")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nCheckpoint salvato: {checkpoint_path}")
    print(f"Best MASE su validation finale: {best_mase:.4f}")
    print(f"Epoche effettive: {len(history['train_loss'])}")

    return checkpoint_path, best_rmse


def main():
    """Funzione principale."""
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Modello: {MODEL_NAME.upper()}")
    print(f"Studio: {STUDY_NAME} ({'NUOVO' if NEW_STUDY else 'RIPRENDE'})")
    print(f"Trials: {N_TRIALS}, Folds: {N_FOLDS}, Epochs: {TUNING_EPOCHS}")
    print("=" * 60 + "\n")

    set_seed(SEED)
    device = get_device()
    df, folds = load_data_and_folds()

    # Crea directory risultati
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Configura quale fold usare
    config = {
        "n_trials": N_TRIALS,
        "n_folds": N_FOLDS,
        "epochs": TUNING_EPOCHS,
        "patience": PATIENCE,
    }

    # Storage SQLite per persistenza
    storage_path = os.path.join(RESULTS_DIR, "optuna_studies.db")

    # Crea optimizer
    optimizer = OptunaOptimizer(
        model_name=MODEL_NAME,
        folds=folds,
        device=device,
        config=config,
        storage_path=storage_path,
        verbose=False,
    )

    # Esegui ottimizzazione
    result = optimizer.optimize(
        study_name=STUDY_NAME,
        n_trials=N_TRIALS,
        new_study=NEW_STUDY,
    )

    print("\n" + "=" * 60)
    print("OTTIMIZZAZIONE COMPLETATA")
    print("=" * 60)
    print(f"Best MASE: {result['best_mase']:.4f}")
    print(f"Best params: {result['best_params']}")

    
    # Retraining finale su tutto il dataset
    checkpoint_path, best_rmse = train_final_model(MODEL_NAME, result["best_params"], df, device)

    # Salva risultati
    save_results(MODEL_NAME, STUDY_NAME, result["best_params"], result["best_mase"], best_rmse)

    print("\n" + "=" * 60)
    print("RIEPILOGO FINALE")
    print("=" * 60)
    print(f"Modello: {MODEL_NAME.upper()}")
    print(f"Studio: {STUDY_NAME}")
    print(f"Best MASE (validazione): {result['best_mase']:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print("\nPer recuperare lo studio:")
    print(f'  optuna.load_study("{STUDY_NAME}", storage="sqlite:///{storage_path}")')
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
