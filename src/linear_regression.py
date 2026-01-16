"""
Linear Regression Baseline per PV Forecasting.

Questo script valuta una Linear Regression come baseline per confrontare
con i modelli deep learning. Usa sklearn e i moduli esistenti del progetto.

Due versioni:
1. UNIVARIATA: usa solo il target (pv_power) storico
2. MULTIVARIATA: usa tutte le features (pv_power + meteo)

Uso:
    python -m src.linear_regression

Output:
    - MASE e RMSE per ogni fold
    - MASE medio pesato (come Optuna)
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
    Estrae X e y dal DataLoader in formato numpy.

    Args:
        loader: DataLoader PyTorch
        univariate: se True, usa solo il target; se False, tutte le features
        target_idx: indice della colonna target

    Returns:
        X: (n_samples, lookback) se univariate, (n_samples, lookback * features) se multivariate
        y: (n_samples, horizon)
    """
    X_list = []
    y_list = []

    for batch_x, batch_y in loader:
        # batch_x: (batch, lookback, features)
        # batch_y: (batch, horizon, 1)

        if univariate:
            # Solo target: (batch, lookback)
            X_batch = batch_x[:, :, target_idx].numpy()
        else:
            # Tutte le features flatten: (batch, lookback * features)
            X_batch = batch_x.numpy().reshape(batch_x.size(0), -1)

        y_batch = batch_y.numpy().reshape(batch_y.size(0), -1)

        X_list.append(X_batch)
        y_list.append(y_batch)

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    return X, y


def evaluate_linear_regression(train_loader, val_loader, naive_mae, univariate=False):
    """
    Addestra e valuta una Linear Regression.

    Args:
        train_loader: DataLoader training
        val_loader: DataLoader validation
        naive_mae: MAE del modello naive per calcolare MASE
        univariate: se True, usa solo pv_power

    Returns:
        dict: {"mase": float, "rmse": float, "mae": float}
    """
    # Prepara dati
    X_train, y_train = prepare_data_from_loader(train_loader, univariate)
    X_val, y_val = prepare_data_from_loader(val_loader, univariate)

    # Addestra modello
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predici
    y_pred = model.predict(X_val)

    # Metriche
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mase = mae / naive_mae

    return {
        "mase": mase,
        "rmse": rmse,
        "mae": mae,
        "n_features": X_train.shape[1],
    }


def main():
    """Funzione principale."""
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION BASELINE")
    print("=" * 70)

    set_seed(SEED)

    # Carica dati
    print(f"\nCaricamento dati: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {df.shape[0]} righe, {df.shape[1]} colonne")
    print(f"Lookback: {LOOKBACK}h, Horizon: {HORIZON}h")

    # Crea fold
    validator = TS_Cross_Validator(df, TARGET_COL, SAMPLING_CONFIG)
    folds = list(validator.get_folds())

    # Calcola pesi fold (come in Optuna)
    train_sample_counts = [len(fold[0].dataset) for fold in folds]
    total_samples = sum(train_sample_counts)
    fold_weights = [n / total_samples for n in train_sample_counts]

    print(f"\nFold weights: {[f'{w:.3f}' for w in fold_weights]}")

    # ==================== UNIVARIATA ====================
    print("\n" + "-" * 70)
    print("VERSIONE UNIVARIATA (solo pv_power)")
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

    # MASE ponderato
    uni_weighted_mase = sum(w * m for w, m in zip(fold_weights, uni_mase_scores))
    uni_avg_rmse = sum(w * r for w, r in zip(fold_weights, uni_rmse_scores))

    print(f"\n  → MASE ponderato: {uni_weighted_mase:.4f}")
    print(f"  → RMSE ponderato: {uni_avg_rmse:.4f}")

    # ==================== MULTIVARIATA ====================
    print("\n" + "-" * 70)
    print("VERSIONE MULTIVARIATA (pv_power + meteo)")
    print("-" * 70)

    # Ri-crea fold (necessario perché sono generator esauriti)
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

    # MASE ponderato
    multi_weighted_mase = sum(w * m for w, m in zip(fold_weights, multi_mase_scores))
    multi_avg_rmse = sum(w * r for w, r in zip(fold_weights, multi_rmse_scores))

    print(f"\n  → MASE ponderato: {multi_weighted_mase:.4f}")
    print(f"  → RMSE ponderato: {multi_avg_rmse:.4f}")

    # ==================== FINAL FOLD (22 mesi train, 2 mesi val) ====================
    print("\n" + "-" * 70)
    print("FINAL FOLD (22 mesi train + 2 mesi val)")
    print("-" * 70)

    from .DataLoading import create_final_train_val_loaders
    from .config import NAIVE_MAE_FINAL_FOLD

    final_train_loader, final_val_loader, _ = create_final_train_val_loaders(
        df, target_col=TARGET_COL
    )

    # Univariata
    final_uni_result = evaluate_linear_regression(
        final_train_loader, final_val_loader, NAIVE_MAE_FINAL_FOLD, univariate=True
    )
    print(
        f"  Univariata:   MASE={final_uni_result['mase']:.4f}, "
        f"RMSE={final_uni_result['rmse']:.4f}, Features={final_uni_result['n_features']}"
    )

    # Ri-crea loaders (consumati dal precedente)
    final_train_loader, final_val_loader, _ = create_final_train_val_loaders(
        df, target_col=TARGET_COL
    )

    # Multivariata
    final_multi_result = evaluate_linear_regression(
        final_train_loader, final_val_loader, NAIVE_MAE_FINAL_FOLD, univariate=False
    )
    print(
        f"  Multivariata: MASE={final_multi_result['mase']:.4f}, "
        f"RMSE={final_multi_result['rmse']:.4f}, Features={final_multi_result['n_features']}"
    )

    # ==================== RIEPILOGO ====================
    print("\n" + "=" * 70)
    print("RIEPILOGO BASELINE")
    print("=" * 70)
    print(
        f"{'Modello':<30} {'MASE (CV)':<12} {'MASE (Final)':<12} {'RMSE (Final)':<12}"
    )
    print("-" * 66)
    print(f"{'Naive Persistence':<30} {'1.0000':<12} {'1.0000':<12} {'-':<12}")
    print(
        f"{'Linear (Univariata)':<30} {uni_weighted_mase:<12.4f} "
        f"{final_uni_result['mase']:<12.4f} {final_uni_result['rmse']:<12.4f}"
    )
    print(
        f"{'Linear (Multivariata)':<30} {multi_weighted_mase:<12.4f} "
        f"{final_multi_result['mase']:<12.4f} {final_multi_result['rmse']:<12.4f}"
    )
    print("=" * 70)

    print("\nPer confronto, i tuoi modelli DL dovrebbero battere questi MASE!")


if __name__ == "__main__":
    main()
