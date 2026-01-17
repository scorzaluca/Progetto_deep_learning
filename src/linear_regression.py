"""
Linear Regression Baseline for PV Forecasting.

This script evaluates a simple Linear Regression model to serve as a baseline for comparison
against more complex deep learning models. It leverages sklearn and reuses the project's
existing data loading pipeline.

Two Versions Evaluated:
1. UNIVARIATE: Uses ONLY the historical target (pv_power) values.
2. MULTIVARIATE: Uses ALL available features (pv_power + weather data).

Usage:
    python -m src.linear_regression

Output:
    - MASE and RMSE for each Cross-Validation fold.
    - Weighted Average MASE (comparable to Optuna optimization metric).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import (
    TARGET_COL,
    SAMPLING_CONFIG,
    NAIVE_MAE_PER_FOLD,
    SEED,
    DATA_PATH,
    LOOKBACK,
    HORIZON,
    TARGET_IDX,
)
from .DataLoading import TS_Cross_Validator
from .Utils import set_seed


def prepare_data_from_loader(loader, univariate=False, target_idx=TARGET_IDX):
    """
    Extracts features (X) and targets (y) from a PyTorch DataLoader and converts them to NumPy.

    Since Linear Regression (sklearn) expects 2D matrices (samples, features), this function
    flattens the time dimension and feature dimension into a single feature vector.

    Args:
        loader: PyTorch DataLoader containing batches of time series.
        univariate: If True, uses only the target column (Autoregressive only).
                    If False, uses all flattened features (Multivariate).
        target_idx: Index of the target column in the input tensor.

    Returns:
        X: Numpy array of shape (n_samples, lookback) [Univariate] 
           or (n_samples, lookback * n_features) [Multivariate].
        y: Numpy array of shape (n_samples, horizon).
    """
    X_list = []
    y_list = []

    # Iterate over batches
    for batch_x, batch_y in loader:
        # batch_x shape: (batch_size, lookback, n_features)
        # batch_y shape: (batch_size, horizon, 1)

        if univariate:
            # Option Univariate (Autoregressive)
            # Take only the target column across all lookback steps.
            # Shape becomes: (batch_size, lookback)
            X_batch = batch_x[:, :, target_idx].numpy()
        else:
            # Option Multivariate
            # Take ALL features and flatten them.
            # The model treats (t-1, feat1) and (t-2, feat1) as distinct independent features.
            # Shape becomes: (batch_size, lookback * n_features)
            X_batch = batch_x.numpy().reshape(batch_x.size(0), -1)

        # Flatten targets as well: (batch_size, horizon)
        y_batch = batch_y.numpy().reshape(batch_y.size(0), -1)

        X_list.append(X_batch)
        y_list.append(y_batch)

    # Concatenate all batches into single large matrices
    X = np.vstack(X_list)
    y = np.vstack(y_list)

    return X, y


def evaluate_linear_regression(train_loader, val_loader, naive_mae, univariate=False):
    """
    Trains and evaluates a Linear Regression model on a specific Train/Val fold.

    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        naive_mae: Baseline MAE of the Naive model (used for MASE calculation).
        univariate: If True, uses only past targets. If False, uses all features.

    Returns:
        dict: Containing 'mase', 'rmse', 'mae', and 'n_features'.
    """
    # Prepare Data
    # Convert PyTorch loaders to NumPy matrices for sklearn
    X_train, y_train = prepare_data_from_loader(train_loader, univariate)
    X_val, y_val = prepare_data_from_loader(val_loader, univariate)

    # Train Model
    # Simple Ordinary Least Squares (OLS) Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on Validation Set
    y_pred = model.predict(X_val)

    # Compute Metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    
    # MASE: Scaled Error relative to the Naive Baseline
    mase = mae / naive_mae

    return {
        "mase": mase,
        "rmse": rmse,
        "mae": mae,
        "n_features": X_train.shape[1],
    }


def main():
    """
    Main execution pipeline for Linear Regression Baseline.

    Steps:
    1. Loads dataset and splits it into Cross-Validation folds.
    2. UNIVARIATE Evaluation:
       - Trains on past target values only.
       - Computes Weighted MASE across all folds.
    3. MULTIVARIATE Evaluation:
       - Trains on all available features (PV power + Weather).
       - Computes Weighted MASE.
    4. FINAL FOLD Evaluation:
       - Trains on the maximum available history (22 months) and validates on the last 2 months.
       - Serves as the ultimate performance check.
    5. Prints a summary table comparing Linear Models vs Naive Baseline.
    """
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION BASELINE")
    print("=" * 70)

    set_seed(SEED)

    # Load Data
    print(f"\nLoading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Lookback: {LOOKBACK}h, Horizon: {HORIZON}h")

    # Create Cross-Validation Folds
    validator = TS_Cross_Validator(df, TARGET_COL, SAMPLING_CONFIG)
    folds = list(validator.get_folds())

    # Calculate Fold Weights (based on training sample size)
    # This ensures consistency with the weighted metric used in Optuna
    train_sample_counts = [len(fold[0].dataset) for fold in folds]
    total_samples = sum(train_sample_counts)
    fold_weights = [n / total_samples for n in train_sample_counts]

    print(f"\nFold weights: {[f'{w:.3f}' for w in fold_weights]}")

    # ==================== UNIVARIATE EVALUATION ====================
    print("\n" + "-" * 70)
    print("UNIVARIATE VERSION (Target Only)")
    print("-" * 70)

    uni_mase_scores = []
    uni_rmse_scores = []

    for i, (train_loader, val_loader, _) in enumerate(folds):
        result = evaluate_linear_regression(
            train_loader, val_loader, NAIVE_MAE_PER_FOLD[i], univariate=True
        )
        uni_mase_scores.append(result["mase"])
        uni_rmse_scores.append(result["rmse"])

        print(
            f"  Fold {i + 1}: MASE={result['mase']:.4f}, RMSE={result['rmse']:.4f}, "
            f"Features={result['n_features']}"
        )

    # Weighted Average MASE
    uni_weighted_mase = sum(w * m for w, m in zip(fold_weights, uni_mase_scores))
    uni_avg_rmse = sum(w * r for w, r in zip(fold_weights, uni_rmse_scores))

    print(f"\n  → Weighted MASE: {uni_weighted_mase:.4f}")
    print(f"  → Weighted RMSE: {uni_avg_rmse:.4f}")

    # ==================== MULTIVARIATE EVALUATION ====================
    print("\n" + "-" * 70)
    print("MULTIVARIATE VERSION (Target + Weather)")
    print("-" * 70)

    # Re-create folds 
    validator = TS_Cross_Validator(df, TARGET_COL, SAMPLING_CONFIG)
    folds = list(validator.get_folds())

    multi_mase_scores = []
    multi_rmse_scores = []

    for i, (train_loader, val_loader, _) in enumerate(folds):
        result = evaluate_linear_regression(
            train_loader, val_loader, NAIVE_MAE_PER_FOLD[i], univariate=False
        )
        multi_mase_scores.append(result["mase"])
        multi_rmse_scores.append(result["rmse"])

        print(
            f"  Fold {i + 1}: MASE={result['mase']:.4f}, RMSE={result['rmse']:.4f}, "
            f"Features={result['n_features']}"
        )

    # Weighted Average MASE
    multi_weighted_mase = sum(w * m for w, m in zip(fold_weights, multi_mase_scores))
    multi_avg_rmse = sum(w * r for w, r in zip(fold_weights, multi_rmse_scores))

    print(f"\n  → Weighted MASE: {multi_weighted_mase:.4f}")
    print(f"  → Weighted RMSE: {multi_avg_rmse:.4f}")

    # ==================== FINAL FOLD (22 months train, 2 months val) ====================
    print("\n" + "-" * 70)
    print("FINAL FOLD (Max Training History)")
    print("-" * 70)

    from .DataLoading import create_final_train_val_loaders
    from .config import NAIVE_MAE_FINAL_FOLD

    final_train_loader, final_val_loader, _ = create_final_train_val_loaders(
        df, target_col=TARGET_COL
    )

    # Univariate Final
    final_uni_result = evaluate_linear_regression(
        final_train_loader, final_val_loader, NAIVE_MAE_FINAL_FOLD, univariate=True
    )
    print(
        f"  Univariate:   MASE={final_uni_result['mase']:.4f}, "
        f"RMSE={final_uni_result['rmse']:.4f}, Features={final_uni_result['n_features']}"
    )

    # Re-create loaders
    final_train_loader, final_val_loader, _ = create_final_train_val_loaders(
        df, target_col=TARGET_COL
    )

    # Multivariate Final
    final_multi_result = evaluate_linear_regression(
        final_train_loader, final_val_loader, NAIVE_MAE_FINAL_FOLD, univariate=False
    )
    print(
        f"  Multivariate: MASE={final_multi_result['mase']:.4f}, "
        f"RMSE={final_multi_result['rmse']:.4f}, Features={final_multi_result['n_features']}"
    )

    # ==================== SUMMARY TABLE ====================
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print(
        f"{'Model':<30} {'MASE (CV)':<12} {'MASE (Final)':<12} {'RMSE (Final)':<12}"
    )
    print("-" * 66)
    print(f"{'Naive Persistence':<30} {'1.0000':<12} {'1.0000':<12} {'-':<12}")
    print(
        f"{'Linear (Univariate)':<30} {uni_weighted_mase:<12.4f} "
        f"{final_uni_result['mase']:<12.4f} {final_uni_result['rmse']:<12.4f}"
    )
    print(
        f"{'Linear (Multivariate)':<30} {multi_weighted_mase:<12.4f} "
        f"{final_multi_result['mase']:<12.4f} {final_multi_result['rmse']:<12.4f}"
    )
    print("=" * 70)

    


if __name__ == "__main__":
    main()
