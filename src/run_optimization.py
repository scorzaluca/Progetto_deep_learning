"""
Script for Hyperparameter Optimization using Optuna.

Manages the full lifecycle of the hyperparameter tuning process:
1. Loads configuration from `src/config/tuning_config.py`.
2. Creates or Resumes an Optuna Study (SQLite persistence).
3. Executes the optimization loop (Cross-Validation Mean MASE).
4. Retrains the best model on the extended dataset (Training + Validation).
5. Saves the final checkpoint and results.

Usage:
    python -m src.run_optimization
"""

from src.config import LOOKBACK
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
    EPOCHS,
)
from .Tuning import OptunaOptimizer
from .Utils import set_seed, get_device, load_data_and_folds, save_results


def train_final_model(model_name: str, best_params: dict, df: pd.DataFrame, device):
    """
    Retrains the model with the best parameters on the Final Fold (22 months Train + 2 months Val).

    It uses Early Stopping based on the Validation MASE to prevent overfitting.
    The final weights are saved to `results/checkpoints/`.

    Args:
        model_name (str): Name of the model architecture (e.g., 'encoderlstm').
        best_params (dict): Dictionary of optimal hyperparameters found by Optuna.
        df (pd.DataFrame): The full dataset.
        device (torch.device): The computing device (CPU/GPU).

    Returns:
        tuple: (checkpoint_path, best_rmse) - Path to saved weights and final RMSE.
    """
    from .Training.engine import create_model, fit_model
    from .DataLoading import create_final_train_val_loaders

    print("\n" + "=" * 60)
    print("FINAL RETRAINING (22 months Train + 2 months Val)")
    print("=" * 60)

    # Create Final DataLoaders
    # This allows the model to learn from the maximum amount of history while still having a safety validation check.
    train_loader, val_loader, scaler = create_final_train_val_loaders(
        df, TARGET_COL, LOOKBACK
    )

    # Re-create the Model Architecture
    # Using the same factory function as during optimization
    model = create_model(model_name, best_params)
    model.to(device)

    # Extract Training Hyperparameters
    lr = best_params.get("lr", 0.001)
    grad_clip_norm = best_params.get("grad_clip_norm", 1.0)
    weight_decay = best_params.get("weight_decay", 0.001)

    # Scheduler Parameters
    scheduler_factor = best_params.get("scheduler_factor", 0.5)
    scheduler_patience = best_params.get("scheduler_patience", 3)
    scheduler_min_lr = best_params.get("scheduler_min_lr", 1e-6)

    print(f"Training {model_name.upper()} with Early Stopping (Patience=10)...")

    # Train with Early Stopping
    # We re-use the generic 'fit_model' engine.
    # - fold_idx=None: Indicates this is not a CV fold.
    # - trial=None: Disables Optuna pruning (we want to complete training).
    # - baseline_mae=NAIVE_MAE_FINAL_FOLD: We use the specific baseline error for this 2-month period.
    model, history, best_epoch = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,  
        lr=lr,
        device=device,
        fold_idx=None,
        patience=10,  
        trial=None,
        optimizer_kwargs={"weight_decay": weight_decay},
        grad_clip_norm=grad_clip_norm,
        verbose=True,
        baseline_mae=NAIVE_MAE_FINAL_FOLD,
        scheduler_kwargs={
            "mode": "min",
            "factor": scheduler_factor,
            "patience": scheduler_patience,
            "min_lr": scheduler_min_lr,
        },
    )

    # Extract Final Metrics
    # 'best_epoch' is the index where the validation loss was lowest
    best_mase = history["val_mase"][best_epoch]
    best_rmse = history["val_rmse"][best_epoch]

    # Save Checkpoint
    checkpoint_dir = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{STUDY_NAME}_100_epochs.pth")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")
    print(f"Final Validation MASE: {best_mase:.4f}")
    print(f"Effective Epochs: {len(history['train_loss'])}")

    return checkpoint_path, best_rmse


def main():
    """
    Main Execution Pipeline for Hyperparameter Optimization.

    Steps:
    1. Setup: Seed, Device, Directories.
    2. Data Loading: Loads time series and CV folds.
    3. Tuning: Runs Optuna optimization (Cross-Validation).
    4. Reporting: Displays best MASE and hyperparameters.
    5. Final Training: Retrains the best model configuration on the full dataset.
    6. Saving: Persists checkpoint, metrics, and study data.
    """
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Model: {MODEL_NAME.upper()}")
    print(f"Study: {STUDY_NAME} ({'NEW' if NEW_STUDY else 'RESUME'})")
    print(f"Trials: {N_TRIALS}, Folds: {N_FOLDS}, Epochs: {TUNING_EPOCHS}")
    print("=" * 60 + "\n")

    # Setup Environment
    set_seed(SEED)
    device = get_device()
    
    # Load Data and Cross-Validation Folds
    df, folds = load_data_and_folds()

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Tuning Configuration
    config = {
        "n_trials": N_TRIALS,
        "n_folds": N_FOLDS,
        "epochs": TUNING_EPOCHS,
        "patience": PATIENCE,
    }

    # SQLite Storage for persistence (allows resuming studies)
    storage_path = os.path.join(RESULTS_DIR, "optuna_studies.db")

    # Initialize Optimizer
    # The Generic 'OptunaOptimizer' handles the loop over folds for each trial
    optimizer = OptunaOptimizer(
        model_name=MODEL_NAME,
        folds=folds,
        device=device,
        config=config,
        storage_path=storage_path,
        verbose=False,
    )

    # Execute Optimization
    # Minimizes the average MASE across folds
    result = optimizer.optimize(
        study_name=STUDY_NAME,
        n_trials=N_TRIALS,
        new_study=NEW_STUDY,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETED")
    print("=" * 60)
    print(f"Best MASE: {result['best_mase']:.4f}")
    print(f"Best params: {result['best_params']}")

    # Final Retraining
    # We take the best parameters found and retrain or the specific validation fold to produce the final weight file.
    
    # Retrieve fixed parameters (if any) stored in the best trial user attributes
    best_params = {**result["best_params"], **result["study"].best_trial.user_attrs}
    
    # Final Retraining
    # We take the best parameters found and retrain or the specific validation fold to produce the final weight file.
    checkpoint_path, best_rmse = train_final_model(MODEL_NAME, best_params, df, device)

    # Save Results
    # Saves a summary JSON with all metrics and parameters
    save_results(MODEL_NAME, STUDY_NAME, best_params, result["best_mase"], best_rmse)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Model: {MODEL_NAME.upper()}")
    print(f"Study: {STUDY_NAME}")
    print(f"Best MASE (CV): {result['best_mase']:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    

if __name__ == "__main__":
    main()
