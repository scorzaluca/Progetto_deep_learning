"""
Script principale per il training dei modelli con cross-validation.
Include valutazione finale con plotting e salvataggio delle predizioni.
"""

import pandas as pd
import torch
import numpy as np
import random
import os
from .config import (
    TARGET_COL,
    SAMPLING_CONFIG,
    SEED,
    EPOCHS,
    LEARNING_RATE,
    PATCHTST_CONFIG,
    LSTM_CONFIG,
    DLINEAR_CONFIG,
    RESULTS_DIR,
)
from .DataLoading import TS_Cross_Validator
from .Training.engine import fit_model
from .Training.evaluation import evaluate_model
from .ModelClasses import PatchTST, LSTM, DLinearM
from .Utils import plot_predictions, plot_training_history

DATA_PATH = "data/processed/preprocessed_ds.csv"


def set_seed(seed: int):
    """Imposta il seed per riproducibilita."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    validator = TS_Cross_Validator(df, target_col=TARGET_COL, cfg_dict=SAMPLING_CONFIG)
    folds = list(validator.get_folds())  # Converti a lista per poter usare len()

    models_to_train = ["LSTM", "DLinearM", "PatchTST"]

    # Crea directory per risultati
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    preds_dir = os.path.join(RESULTS_DIR, "predictions")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    for fold_idx, (train_loader, val_loader, scaler) in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'=' * 60}")

        target_idx = train_loader.dataset.target_col_idx

        for model_name in models_to_train:
            print(f"\nTraining {model_name}...")

            # I modelli ora prendono solo model_config, non train_loader
            if model_name == "PatchTST":
                model = PatchTST(model_config=PATCHTST_CONFIG)
            elif model_name == "LSTM":
                model = LSTM(model_config=LSTM_CONFIG)
            elif model_name == "DLinearM":
                model = DLinearM(model_config=DLINEAR_CONFIG)

            model.to(device)

            trained_model, history, best_epoch = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                device=device,
                fold_idx=fold_idx,
            )

            print(
                f"  Best epoch: {best_epoch + 1}, Best MASE: {min(history['val_mase']):.4f}"
            )

            # Valutazione finale (predizioni denormalizzate)
            preds, targets = evaluate_model(
                model=trained_model,
                val_loader=val_loader,
                device=device,
                scaler=scaler,
                target_idx=target_idx,
            )

            # Salvataggio modello
            os.makedirs(RESULTS_DIR, exist_ok=True)
            save_path = os.path.join(
                RESULTS_DIR, f"{model_name}_fold_{fold_idx + 1}.pth"
            )
            torch.save(trained_model.state_dict(), save_path)
            print(f"  Modello salvato: {save_path}")

            # Salvataggio predizioni
            preds_path = os.path.join(
                preds_dir, f"{model_name}_fold_{fold_idx + 1}_predictions.npz"
            )
            np.savez(preds_path, predictions=preds, targets=targets)
            print(f"  Predizioni salvate: {preds_path}")

            # Plotting (funzioni da Utils.plotting)
            plot_predictions(preds, targets, model_name, fold_idx, plots_dir)
            plot_training_history(history, model_name, fold_idx, plots_dir)

    print("\nTraining completato.")


if __name__ == "__main__":
    main()
