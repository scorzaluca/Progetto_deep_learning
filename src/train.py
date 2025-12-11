"""
Script principale per il training dei modelli con cross-validation.
"""

import pandas as pd
import torch
import numpy as np
import random
import os
from .config import (
    TARGET_COL,
    SAMPLING_CONFIG,
    EPOCHS,
    LEARNING_RATE,
    PATCHTST_CONFIG,
    LSTM_CONFIG,
    DLINEAR_CONFIG,
    RESULTS_DIR,
)
from DataLoading import TS_Cross_Validator
from Training import fit_model
from Training import evaluate_model
from ModelClasses import PatchTST, LSTM, DLinear

SEED = 42
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
    folds = validator.get_folds()

    models_to_train = ["LSTM", "DLinearM", "PatchTST"]

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

            trained_model, history = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                device=device,
                fold_idx=fold_idx,
            )

            # Valutazione finale
            preds, targets = evaluate_model(
                model=trained_model,
                val_loader=val_loader,
                device=device,
                scaler=scaler,
                target_idx=target_idx,
            )

            # Salvataggio
            os.makedirs(RESULTS_DIR, exist_ok=True)
            save_path = os.path.join(
                RESULTS_DIR, f"{model_name}_fold_{fold_idx + 1}.pth"
            )
            torch.save(trained_model.state_dict(), save_path)
            print(f"Salvato: {save_path}")

    print("\nTraining completato.")


if __name__ == "__main__":
    main()
