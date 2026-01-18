"""
Script for Inference on Test Data.
Loads a trained model, preprocesses the test dataset, and computes performance metrics.

Usage:
    python -m src.run_inference

Configuration:
    Modify the configuration variables at the top of the file to customize paths:
    - TRAIN_DATA_PATH: Path to training data (needed for fitting the scaler).
    - TEST_DATA_PATH: Path to the new test data (csv/xlsx).
    - CHECKPOINT_PATH: Path to the trained model weights (.pth).
    - PARAMS_PATH: Path to the model hyperparameters (.json).
"""

import json
import os
import numpy as np
import pandas as pd
import torch

# Import global project constants
# LOOKBACK: Input sequence length
# HORIZON: Prediction horizon
# TARGET_IDX: Index of the target column in the tensor
# NAIVE_MAE_TEST: Baseline error on the test set for MASE calculation
from .config import LOOKBACK, HORIZON, TARGET_IDX, NAIVE_MAE_TEST, PREPROCESS_CONFIG
from .config.training_config import TARGET_COL
from .PreProcessing import Preprocesser
from .DataLoading import create_test_loader
from .Training.engine import create_model
from .Training.evaluation import evaluate_model
from .Utils import set_seed, get_device, plot_test_predictions, plot_error_distribution


# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES BEFORE RUNNING
# =============================================================================
# Path to the TRAINING dataset (Required to fit the Scaler correctly)
# We must use the same scaling parameters (mean, std) as seen during training.
TRAIN_DATA_PATH = "data/processed/preprocessed_ds.csv"

# Path to the TEST dataset (The new data we want to predict on)
TEST_DATA_PATH = "data/processed/merged_test_ds.csv"

# Path to the trained model weights (.pth file)
# This file contains the learnable parameters (weights/biases) saved after training.
CHECKPOINT_PATH = "results/checkpoints/encoderlstm_48_final_100_epochs.pth"

# Path to the hyperparameter file (.json)
# This file contains the structural config (hidden_size, n_layers, etc.) needed to rebuild the model architecture.
PARAMS_PATH = "results/params/encoderlstm_48_100_epochs_params.json"

# Name of the model architecture to instantiate (must match the one in engine.create_model)
MODEL_NAME = "encoderlstm"

# Step size for sliding window generation in Test set
STEP_SAMPLES_TEST = 1


def load_and_preprocess_test(test_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the test dataset.
    Supports both Excel (.xlsx, .xls) and CSV (.csv) formats.

    It instantiates the Preprocesser class to apply the same cleaning pipeline used for training data.

    Args:
        test_path (str): Path to the test data file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for inference.
    """
    print(f"\n Loading test data: {test_path}")

    # Determine file format and load accordingly
    if test_path.endswith((".xlsx", ".xls")):
        test_df = pd.read_excel(test_path)
    elif test_path.endswith(".csv"):
        test_df = pd.read_csv(test_path)
    else:
        raise ValueError(f"Unsupported format: {test_path}. Use .xlsx, .xls or .csv")

    print(f"   Raw shape: {test_df.shape}")

    # Initialize the Preprocesser with the raw dataframe and configuration
    preprocesser = Preprocesser(test_df, PREPROCESS_CONFIG)

    # Run the full preprocessing pipeline
    # This ensures the test data has the exact same columns/features as training data
    test_df = preprocesser.run()

    print(f"   Preprocessed shape: {test_df.shape}")

    return test_df


def load_model(
    checkpoint_path: str,
    params_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Reconstructs the model architecture and loads the trained weights.

    Args:
        checkpoint_path (str): Path to the .pth file containing model weights (state_dict).
        params_path (str): Path to the .json file containing model hyperparameters (hidden_size, layers, etc.).
        device (torch.device): The device (CPU/GPU) where the model will be loaded.

    Returns:
        torch.nn.Module: The fully initialized model in evaluation mode.
    """
    print("\n Loading model...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Params: {params_path}")

    # Load Hyperparameters
    # We need the configuration to instantiate the correct model class.
    with open(params_path, "r") as f:
        params_data = json.load(f)

    best_params = params_data.get("best_params", params_data)

    # Instantiate Architecture
    # Uses the Factory Pattern from engine.py to create the empty model structure
    model = create_model(MODEL_NAME, best_params)

    # Load Weights
    # Load the state dictionary (weights/biases) from the checkpoint file
    # map_location ensures we can load a GPU-trained model on CPU if needed
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Apply weights into the model architecture
    model.load_state_dict(state_dict)

    # Move model to target device and switch to Evaluation Mode
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")

    return model


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    naive_mae: float = None,
) -> dict:
    """
    Computes performance metrics (MAE, RMSE, MASE).

    Args:
        predictions (np.ndarray): Array of predicted values (shape: [N, horizon, 1]).
        targets (np.ndarray): Array of ground truth values (shape: [N, horizon, 1]).
        naive_mae (float, optional): Baseline MAE for MASE calculation.

    Returns:
        dict: Dictionary containing 'MAE', 'RMSE', and 'MASE' (if applicable).
    """
    # Flatten arrays to compute aggregate metrics across all samples/steps
    # From shape (N, Horizon, 1) -> (N * Horizon)
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(preds_flat - targets_flat))

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))

    # Calculate Mean Absolute Scaled Error (MASE)
    # Only possible if the baseline naive error is provided
    mase = mae / naive_mae if naive_mae else None

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MASE": mase,
    }


def save_predictions_to_excel(
    predictions: np.ndarray,
    timestamps: pd.Series,
    lookback: int,
    horizon: int,
    save_path: str,
) -> None:
    """
    Saves predictions to an Excel file with timestamps and hourly forecast columns.

    Creates a DataFrame with 25 columns:
    - Column 1: datetime (timestamp in UTC+10 Sydney timezone)
    - Columns 2-25: predictions for t+1, t+2, ..., t+24

    The first (lookback - 1) rows will have NaN for all prediction columns,
    since they don't have enough historical data for the lookback window.

    Args:
        predictions (np.ndarray): Array of predictions with shape (N, horizon) or (N, horizon, 1).
        timestamps (pd.Series): Series of datetime strings from the original test data.
        lookback (int): Lookback window size (e.g., 48).
        horizon (int): Prediction horizon (e.g., 24).
        save_path (str): Path to save the Excel file.

    Returns:
        None
    """
    # Squeeze predictions if they have an extra dimension
    if predictions.ndim == 3:
        predictions = predictions.squeeze(-1)  # (N, horizon, 1) -> (N, horizon)

    # Number of rows that cannot be predicted (not enough lookback)
    n_unpredictable = lookback - 1  # 48 - 1 = 47
    n_total_rows = len(timestamps)

    # Create column names
    columns = ["datetime"] + [f"t+{h}" for h in range(1, horizon + 1)]

    # Initialize DataFrame with NaN
    df = pd.DataFrame(index=range(n_total_rows), columns=columns)

    # Fill datetime column with timestamps (already in UTC+10 Sydney)
    df["datetime"] = timestamps.values

    # Fill predictions starting from row (lookback - 1)
    # The first valid prediction corresponds to timestep = lookback - 1
    # Because we need lookback samples [0:lookback] to predict [lookback:lookback+horizon]
    for i, pred in enumerate(predictions):
        row_idx = n_unpredictable + i
        if row_idx < n_total_rows:
            for h in range(horizon):
                df.loc[row_idx, f"t+{h + 1}"] = pred[h]

    # Convert prediction columns to float (they were object due to NaN initialization)
    for h in range(1, horizon + 1):
        df[f"t+{h}"] = pd.to_numeric(df[f"t+{h}"], errors="coerce")

    # Save to Excel
    df.to_excel(save_path, index=False)
    print(f"Predictions saved to Excel: {save_path}")
    print(f"   Total rows: {n_total_rows}")
    print(f"   Unpredictable rows (NaN): {n_unpredictable}")
    print(f"   Predicted rows: {len(predictions)}")


def main():
    """
    Main execution pipeline for Test Inference.

    Steps:
    1. Sets up the device and reproducibility (seed).
    2. Loads TRAIN data to correctly fit the scaler (ensuring consistent normalization).
    3. Loads and preprocesses TEST data.
    4. Creates a Test DataLoader with sliding window logic.
    5. Loads the trained model architecture and weights.
    6. Runs inference TWICE:
       - First on NORMALIZED data (to calculate MASE comparable with validation).
       - Second on DENORMALIZED data (to get Real World metrics in Watts).
    7. Computes and prints metrics (MAE, RMSE, MASE).
    8. Generates plots for visual inspection.
    9. Saves all results to a JSON file.
    """
    print("\n" + "=" * 60)
    print("TEST SET INFERENCE")
    print("=" * 60)

    set_seed(42)
    device = get_device()

    # Load Training Data (Required for Scaler fitting):
    # The scaler must be fitted on training data statistics (mean, std)
    # to avoid data leakage and ensure the model sees data in the expected range.
    print(f"\nLoading training data: {TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"   Shape: {train_df.shape}")

    # Load and Preprocess Test Data
    # First, load raw data to extract timestamps before preprocessing removes them
    if TEST_DATA_PATH.endswith((".xlsx", ".xls")):
        raw_test_df = pd.read_excel(TEST_DATA_PATH)
    else:
        raw_test_df = pd.read_csv(TEST_DATA_PATH)

    # Extract timestamps (dt_iso column) before preprocessing
    # These are in UTC+10 (Sydney) timezone
    timestamps = raw_test_df["dt_iso"]

    # Now preprocess the test data
    test_df = load_and_preprocess_test(TEST_DATA_PATH)

    # Verify column consistency:
    # The model expects the exact same features as during training.
    if list(train_df.columns) != list(test_df.columns):
        print("\n WARNING: Test set columns do not match Training set columns!")
        print(f"   Train columns: {list(train_df.columns)}")
        print(f"   Test columns: {list(test_df.columns)}")
        # Attempt to reorder columns to match training
        test_df = test_df[train_df.columns]

    # Create Test Loader
    # Scaler is fitted here on train_df and applied to test_df
    # STEP_SAMPLES_TEST controls the sliding window stride
    test_loader, scaler = create_test_loader(
        train_df=train_df,
        test_df=test_df,
        target_col=TARGET_COL,
        lookback=LOOKBACK,
        horizon=HORIZON,
        step=STEP_SAMPLES_TEST,
    )

    # Load Model
    model = load_model(CHECKPOINT_PATH, PARAMS_PATH, device)

    # Run Inference (NORMALIZED):
    # We pass scaler=None to keep predictions in the 0-1 range.
    # This is necessary to calculate MASE using the NAIVE_MAE_TEST constant (which is normalized).
    print("\n Running Inference (Normalized)...")
    predictions_norm, targets_norm = evaluate_model(
        model=model,
        val_loader=test_loader,
        device=device,
        scaler=None,  # No scaler -> output remains normalized
        target_idx=None,
    )
    print(f"   Predictions shape: {predictions_norm.shape}")
    print(f"   Targets shape: {targets_norm.shape}")

    # Compute Metrics (NORMALIZED)
    print("\n Computing Metrics...")
    metrics_norm = calculate_metrics(predictions_norm, targets_norm, NAIVE_MAE_TEST)

    # Run Inference (DENORMALIZED):
    # We pass the scaler to transform predictions back to real units (Watts).
    # This gives us interpretable errors.
    predictions_denorm, targets_denorm = evaluate_model(
        model=model,
        val_loader=test_loader,
        device=device,
        scaler=scaler,  # Pass scaler -> output is denormalized
        target_idx=TARGET_IDX,
    )
    metrics_denorm = calculate_metrics(
        predictions_denorm, targets_denorm, naive_mae=None
    )

    # Print Results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print("\n--- NORMALIZED Metrics (Scale 0-1) ---")
    print(f"MAE:  {metrics_norm['MAE']:.6f}")
    print(f"RMSE: {metrics_norm['RMSE']:.6f}")
    if metrics_norm["MASE"]:
        print(f"MASE: {metrics_norm['MASE']:.4f}")
    else:
        print("MASE: N/A")

    print("\n--- DENORMALIZED Metrics (Watts) ---")
    print(f"MAE:  {metrics_denorm['MAE']:.2f} W")
    print(f"RMSE: {metrics_denorm['RMSE']:.2f} W")
    print("=" * 60)

    # Save Results to JSON
    results_dir = "results/test_results"
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "model_name": MODEL_NAME,
        "checkpoint": CHECKPOINT_PATH,
        "test_path": TEST_DATA_PATH,
        "n_samples": len(predictions_norm),
        "metrics_normalized": metrics_norm,
        "metrics_denormalized": {
            "MAE": metrics_denorm["MAE"],
            "RMSE": metrics_denorm["RMSE"],
        },
    }

    results_path = os.path.join(results_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Generate Plots (on Denormalized data)
    print("\n Generating Plots...")

    # Plot random samples of predictions vs ground truth
    plot_test_predictions(
        predictions_denorm,
        targets_denorm,
        n_samples=5,
        save_path=os.path.join(results_dir, "predictions_samples.png"),
    )

    # Plot error distribution histogram
    plot_error_distribution(
        predictions_denorm,
        targets_denorm,
        save_path=os.path.join(results_dir, "error_distribution.png"),
    )

    # Save Predictions to Excel
    # Creates a file with 25 columns: datetime (UTC+10), t+1, t+2, ..., t+24
    # First 47 rows have NaN (not enough lookback data)
    print("\n Saving Predictions to Excel...")
    save_predictions_to_excel(
        predictions=predictions_denorm,
        timestamps=timestamps,
        lookback=LOOKBACK,
        horizon=HORIZON,
        save_path=os.path.join(results_dir, "predictions.xlsx"),
    )

    print("\nInference Completed!")


if __name__ == "__main__":
    main()
