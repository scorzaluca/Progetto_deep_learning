"""
Script to evaluate Zero-Shot models using the same methodology as the main screening.

This script isolates the validation logic for models that do not require training (like Chronos).
It ensures fair comparison by using the exact same Data Loading, customized Cross-Validation,
and Metric Calculation (MASE) as the trained models.
"""

import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from .config import TARGET_COL, SAMPLING_CONFIG, NAIVE_MAE_PER_FOLD, RESULTS_DIR
from .DataLoading import TS_Cross_Validator
from .ModelClasses import ChronosWrapper
from .Utils import plot_predictions

# Configuration
FOLD_INDEX = 2 
DATA_PATH = "data/processed/preprocessed_ds.csv"

# Models to evaluate
MODELS_TO_EVALUATE = {
    "chronos-2": ChronosWrapper,
}


def load_data_and_fold():
    """
    Loads the preprocessed dataset and retrieves the specific Cross-Validation fold.

    We aim to evaluate on the SAME exact data split used during the screening of trained models
    to ensure the results are directly comparable.

    Returns:
        tuple: (train_loader, val_loader, scaler) for the selected fold.
    """
    print(f"\nLoading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Initialize Cross-Validator with the same config as training
    validator = TS_Cross_Validator(df, target_col=TARGET_COL, cfg_dict=SAMPLING_CONFIG)
    folds = list(validator.get_folds())

    # Verify that the requested fold exists
    if len(folds) <= FOLD_INDEX:
        raise ValueError(
            f"Fold {FOLD_INDEX} not available. Total folds: {len(folds)}"
        )

    # Extract the loaders and scaler for the target fold
    train_loader, val_loader, scaler = folds[FOLD_INDEX]

    print(f"\nFold {FOLD_INDEX + 1} selected:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Baseline MAE (Naive): {NAIVE_MAE_PER_FOLD[FOLD_INDEX]:.6f}")

    return train_loader, val_loader, scaler


def evaluate_model(wrapper, val_loader, scaler, device, fold_idx, model_name):
    """
    Evaluates a Zero-Shot model on the validation set.

    It replicates the logic used in `optuna_optimizer.py` to ensure consistency.
    1. Iterates over validation batches.
    2. Generates predictions using the wrapper (which handles context).
    3. Computes MAE and MASE.
    4. Plots the results.

    Args:
        wrapper: Zero-shot model wrapper (e.g., ChronosWrapper).
        val_loader: DataLoader for validation (used for iteration and context).
        scaler: Scaler used for denormalization (for plotting).
        device: torch.device (cuda/cpu).
        fold_idx: Index of the fold (used to retrieve baseline MAE).
        model_name: Name of the model (for plot title).

    Returns:
        float: MASE score.
    """
    # Initialize metric function and retrieve baseline
    mae_fn = nn.L1Loss()
    baseline_mae = NAIVE_MAE_PER_FOLD[fold_idx]

    print(f"\n  Inference on validation set ({len(val_loader)} batches)...")
    running_mae = 0.0

    all_preds = []
    all_targets = []

    # Get target column index for denormalization
    target_idx = val_loader.dataset.target_col_idx

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
            # Move data to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Zero-Shot Prediction
            # We pass `val_loader` because ChronosWrapper needs access to the 
            # dataset context (history) to generate forecasts, not just the current batch_x.
            pred = wrapper.predict(batch_x, horizon=24, val_loader=val_loader)
            pred = pred.to(device)

            # Calculate MAE
            running_mae += mae_fn(pred, batch_y).item()

            # Store predictions and targets for plotting (take first feature/step)
            all_preds.append(pred[:, 0, 0].cpu().numpy())
            all_targets.append(batch_y[:, 0, 0].cpu().numpy())

            # Log progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx + 1}/{len(val_loader)}")

    # Calculate aggregate metrics
    avg_mae = running_mae / len(val_loader)
    mase = avg_mae / baseline_mae

    print(f"  MAE: {avg_mae:.6f}")
    print(f"  MASE: {mase:.4f}")

    # Concatenate results for plotting
    preds_flat = np.concatenate(all_preds)
    targets_flat = np.concatenate(all_targets)

    # --- Denormalization for Plotting ---
    # We need to invert the scaling to get real units (Watts).
    # Since scaler expects shape (N, n_features), we create dummy arrays
    # and fill only the target column.
    n_features = scaler.n_features_in_
    preds_full = np.zeros((len(preds_flat), n_features))
    targets_full = np.zeros((len(targets_flat), n_features))
    
    # Fill the target column
    preds_full[:, target_idx] = preds_flat
    targets_full[:, target_idx] = targets_flat

    # Inverse transform and extract target column back
    preds_denorm = scaler.inverse_transform(preds_full)[:, target_idx]
    targets_denorm = scaler.inverse_transform(targets_full)[:, target_idx]

    # Generate and save prediction plots
    plot_predictions(preds_denorm, targets_denorm, model_name, fold_idx, RESULTS_DIR)

    return mase


def main():
    """
    Main execution loop for Zero-Shot evaluation.

    Steps:
    1. Sets up the computation device (CUDA/CPU).
    2. Loads the validation fold and scaler.
    3. Iterates through all Zero-Shot models defined in MODELS_TO_EVALUATE.
    4. Runs evaluation for each model using `evaluate_model`.
    5. Aggregates results, creates a ranking, and saves everything to a JSON file.
    """
    print("\n" + "=" * 70)
    print("ZERO-SHOT MODEL EVALUATION")
    print("=" * 70)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load Data (we only need val_loader for Zero-Shot evaluation)
    _, val_loader, scaler = load_data_and_fold()

    # Dictionary to store results
    results = {}

    # Iterate over each Zero-Shot model defined in MODELS_TO_EVALUATE
    for model_name, wrapper_class in MODELS_TO_EVALUATE.items():
        print(f"\n{'=' * 70}")
        print(f"EVALUATING: {model_name.upper()}")
        print(f"{'=' * 70}")

        try:
            # Instantiate the Wrapper
            print(f"\nLoading model {model_name}...")
            wrapper = wrapper_class()

            # Retrieve Model Info (Parameters, Architecture)
            model_info = wrapper.get_model_info()
            print(f"  Type: {model_info.get('model_type', 'N/A')}")
            print(f"  Parameters: {model_info.get('parameters', 'N/A')}")

            # Evaluate (ONLY ON VAL_LOADER!)
            mase = evaluate_model(
                wrapper, val_loader, scaler, device, FOLD_INDEX, model_name
            )

            # Save Results
            results[model_name] = {
                "best_mase": float(mase),
                "model_info": model_info,
                "fold_used": FOLD_INDEX + 1,
            }

            print(f"\n✓ {model_name.upper()}: MASE = {mase:.4f}")

        except Exception as e:
            print(f"\n✗ {model_name.upper()}: ERROR - {str(e)}")
            import traceback

            traceback.print_exc()

            results[model_name] = {
                "best_mase": float("inf"),
                "error": str(e),
            }

    # Create Ranking
    # Filter out failed models (infinite MASE)
    valid_models = [m for m in results.keys() if results[m]["best_mase"] < float("inf")]
    # Sort by MASE (ascending)
    ranking = sorted(valid_models, key=lambda m: results[m]["best_mase"])
    results["ranking"] = ranking

    # Save to JSON
    import os

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = f"{RESULTS_DIR}/zero_shot_results.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print Final Summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    if ranking:
        print("\nRanking:")
        for i, model in enumerate(ranking, 1):
            mase = results[model]["best_mase"]
            print(f"  {i}. {model}: MASE = {mase:.4f}")

    print(f"\nResults saved to: {results_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
