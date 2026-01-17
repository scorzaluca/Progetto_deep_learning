"""
Main Training Script for Basic Model Evaluation.

This script executes a standard training pipeline using Cross-Validation.
It trains LSTM, DLinear, and PatchTST models using default configurations
found in `src/config.py`, then evaluates them and saves plots/predictions.

Usage:
    python -m src.Utils.train
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
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main execution function.
    
    Steps:
    1. Setup environment (Seed, Device).
    2. Load Data.
    3. Initialize Cross-Validation.
    4. Loop through Folds and Models.
    5. Train, Evaluate, and Save results for each combination.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Initialize Validator
    validator = TS_Cross_Validator(df, target_col=TARGET_COL, cfg_dict=SAMPLING_CONFIG)
    folds = list(validator.get_folds())  # Convert to list to use len()

    models_to_train = ["LSTM", "DLinearM", "PatchTST"]

    # Create directories for results
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

            # Models are initialized using default configs from src.config
            if model_name == "PatchTST":
                model = PatchTST(model_config=PATCHTST_CONFIG)
            elif model_name == "LSTM":
                model = LSTM(model_config=LSTM_CONFIG)
            elif model_name == "DLinearM":
                model = DLinearM(model_config=DLINEAR_CONFIG)

            model.to(device)

            # Train the model
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

            # Final Evaluation (Denormalized predictions)
            preds, targets = evaluate_model(
                model=trained_model,
                val_loader=val_loader,
                device=device,
                scaler=scaler,
                target_idx=target_idx,
            )

            # Save Model Weights
            os.makedirs(RESULTS_DIR, exist_ok=True)
            save_path = os.path.join(
                RESULTS_DIR, f"{model_name}_fold_{fold_idx + 1}.pth"
            )
            torch.save(trained_model.state_dict(), save_path)
            print(f"  Model saved: {save_path}")

            # Save Predictions
            preds_path = os.path.join(
                preds_dir, f"{model_name}_fold_{fold_idx + 1}_predictions.npz"
            )
            np.savez(preds_path, predictions=preds, targets=targets)
            print(f"  Predictions saved: {preds_path}")

            # Generate Plots (using functions from Utils.plotting)
            plot_predictions(preds, targets, model_name, fold_idx, plots_dir)
            plot_training_history(history, model_name, fold_idx, plots_dir)

    print("\nTraining Completed.")


if __name__ == "__main__":
    main()
