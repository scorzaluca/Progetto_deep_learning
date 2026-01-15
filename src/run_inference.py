"""
Script per l'inferenza su dati di test.
Carica il modello addestrato, preprocessa i dati di test e calcola le metriche.

Uso:
    python -m src.run_inference

Modifica le variabili di configurazione all'inizio del file per personalizzare i percorsi.
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .config import (
    LOOKBACK,
    HORIZON,
    TARGET_IDX,
    NAIVE_MAE_TEST,
)
from .config.training_config import TARGET_COL
from .PreProcessing.preprocessing import Preprocesser, PREPROCESS_CONFIG
from .DataLoading import create_test_loader
from .Training.engine import create_model
from .Training.evaluation import evaluate_model
from .Utils import set_seed, get_device


# =============================================================================
# CONFIGURAZIONE - MODIFICA QUESTI VALORI PRIMA DI ESEGUIRE
# =============================================================================
TRAIN_DATA_PATH = "data/processed/preprocessed_ds.csv"
TEST_DATA_PATH = (
    "data/processed/merge_ds.xlsx"
    # "data/test/test_data.xlsx"  # Modifica questo con il path dei dati di test
)
CHECKPOINT_PATH = "results/checkpoints/encoderlstm_24_giusto_100_epochs_10_patience.pth"
PARAMS_PATH = "results/params/encoderlstm_24_giusto_params.json"
MODEL_NAME = "encoderlstm"


def load_and_preprocess_test(test_path: str) -> pd.DataFrame:
    """
    Carica e preprocessa i dati di test.
    Supporta file Excel (.xlsx, .xls) e CSV (.csv).

    Args:
        test_path: percorso al file di test

    Returns:
        pd.DataFrame: dati preprocessati
    """
    print(f"\n📂 Caricamento dati di test: {test_path}")

    # Determina il formato e carica
    if test_path.endswith((".xlsx", ".xls")):
        test_df = pd.read_excel(test_path)
    elif test_path.endswith(".csv"):
        test_df = pd.read_csv(test_path)
    else:
        raise ValueError(f"Formato non supportato: {test_path}. Usa .xlsx, .xls o .csv")

    print(f"   Shape raw: {test_df.shape}")

    # Applica preprocessing (usa run() per eseguire tutta la pipeline)
    preprocesser = Preprocesser(test_df, PREPROCESS_CONFIG)
    test_df = preprocesser.run()

    print(f"   Shape preprocessed: {test_df.shape}")

    return test_df


def load_model(
    checkpoint_path: str,
    params_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Carica il modello dai pesi salvati.

    Args:
        checkpoint_path: percorso al file .pth
        params_path: percorso al file params.json
        device: device su cui caricare il modello

    Returns:
        nn.Module: modello caricato
    """
    print("\n🔧 Caricamento modello...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Params: {params_path}")

    # Carica parametri
    with open(params_path, "r") as f:
        params_data = json.load(f)

    best_params = params_data.get("best_params", params_data)

    # Crea modello
    model = create_model(MODEL_NAME, best_params)

    # Carica pesi
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"   ✅ Modello caricato su {device}")

    return model


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    naive_mae: float = None,
) -> dict:
    """
    Calcola le metriche di valutazione.

    Args:
        predictions: predizioni denormalizzate (N, horizon, 1)
        targets: target denormalizzati (N, horizon, 1)
        naive_mae: MAE del modello naive per calcolare MASE

    Returns:
        dict: dizionario con MAE, RMSE, MASE
    """
    # Flatten per calcolo metriche
    preds_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # MAE
    mae = np.mean(np.abs(preds_flat - targets_flat))

    # RMSE
    rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))

    # MASE (se naive_mae disponibile)
    mase = mae / naive_mae if naive_mae else None

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MASE": mase,
    }


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_samples: int = 5,
    save_path: str = None,
):
    """
    Plotta alcune predizioni vs target.

    Args:
        predictions: predizioni (N, horizon, 1)
        targets: target (N, horizon, 1)
        n_samples: numero di campioni da plottare
        save_path: percorso dove salvare il plot
    """
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))

    # Seleziona campioni casuali
    indices = np.random.choice(len(predictions), n_samples, replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i] if n_samples > 1 else axes

        pred = predictions[idx].flatten()
        target = targets[idx].flatten()

        ax.plot(target, label="Target", marker="o", linewidth=2)
        ax.plot(pred, label="Prediction", marker="x", linewidth=2)
        ax.set_title(f"Sample {idx}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("PV Power")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Plot salvato: {save_path}")

    plt.show()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = None,
):
    """
    Plotta la distribuzione degli errori.
    """
    errors = predictions.flatten() - targets.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")

    # Scatter
    axes[1].scatter(targets.flatten(), predictions.flatten(), alpha=0.3, s=5)
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect")
    axes[1].set_xlabel("Target")
    axes[1].set_ylabel("Prediction")
    axes[1].set_title("Prediction vs Target")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Plot salvato: {save_path}")

    plt.show()


def main():
    """Funzione principale."""
    print("\n" + "=" * 60)
    print("🔮 INFERENZA SU DATI DI TEST")
    print("=" * 60)

    set_seed(42)
    device = get_device()

    # 1. Carica dati di training (per fittare lo scaler)
    print(f"\n📂 Caricamento dati di training: {TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"   Shape: {train_df.shape}")

    # 2. Carica e preprocessa dati di test
    test_df = load_and_preprocess_test(TEST_DATA_PATH)

    # Verifica che le colonne matchino
    if list(train_df.columns) != list(test_df.columns):
        print("\n⚠️ ATTENZIONE: Le colonne del test set non corrispondono!")
        print(f"   Train columns: {list(train_df.columns)}")
        print(f"   Test columns: {list(test_df.columns)}")
        # Prova a riordinare
        test_df = test_df[train_df.columns]

    # 3. Crea test loader (scaler fittato su training)
    test_loader, scaler = create_test_loader(
        train_df=train_df,
        test_df=test_df,
        target_col=TARGET_COL,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    # 4. Carica modello
    model = load_model(CHECKPOINT_PATH, PARAMS_PATH, device)

    # 5. Esegui inferenza (dati NORMALIZZATI per metriche coerenti con NAIVE_MAE_TEST)
    print("\n🔮 Esecuzione inferenza...")
    predictions_norm, targets_norm = evaluate_model(
        model=model,
        val_loader=test_loader,
        device=device,
        scaler=None,  # Senza scaler -> dati normalizzati
        target_idx=None,
    )
    print(f"   Predictions shape: {predictions_norm.shape}")
    print(f"   Targets shape: {targets_norm.shape}")

    # 6. Calcola metriche su dati NORMALIZZATI (coerenti con NAIVE_MAE_TEST)
    print("\n📊 Calcolo metriche...")
    metrics_norm = calculate_metrics(predictions_norm, targets_norm, NAIVE_MAE_TEST)

    # 7. Denormalizza per metriche in scala reale (Watt)
    predictions_denorm, targets_denorm = evaluate_model(
        model=model,
        val_loader=test_loader,
        device=device,
        scaler=scaler,  # Con scaler -> dati denormalizzati
        target_idx=TARGET_IDX,
    )
    metrics_denorm = calculate_metrics(
        predictions_denorm, targets_denorm, naive_mae=None
    )

    # 8. Stampa TUTTE le metriche
    print("\n" + "=" * 60)
    print("📈 RISULTATI TEST SET")
    print("=" * 60)
    print("\n--- Metriche NORMALIZZATE (scala 0-1) ---")
    print(f"MAE:  {metrics_norm['MAE']:.6f}")
    print(f"RMSE: {metrics_norm['RMSE']:.6f}")
    if metrics_norm["MASE"]:
        print(f"MASE: {metrics_norm['MASE']:.4f}")
    else:
        print("MASE: N/A")

    print("\n--- Metriche DENORMALIZZATE (Watt) ---")
    print(f"MAE:  {metrics_denorm['MAE']:.2f} W")
    print(f"RMSE: {metrics_denorm['RMSE']:.2f} W")
    print("=" * 60)

    # 9. Salva risultati
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
    print(f"\n💾 Risultati salvati: {results_path}")

    # 10. Plot (dati denormalizzati)
    print("\n📊 Generazione plots...")
    plot_predictions(
        predictions_denorm,
        targets_denorm,
        n_samples=5,
        save_path=os.path.join(results_dir, "predictions_samples.png"),
    )
    plot_error_distribution(
        predictions_denorm,
        targets_denorm,
        save_path=os.path.join(results_dir, "error_distribution.png"),
    )

    print("\n✅ Inferenza completata!")


if __name__ == "__main__":
    main()
