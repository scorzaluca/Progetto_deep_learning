"""
Funzioni di plotting per visualizzazione training e predizioni.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os


def plot_cv_indices(cv_split_generator, n_samples, n_splits):
    """
    Crea un grafico che mostra i fold della Time Series Cross Validation.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 1. Definiamo i colori ESATTI una volta per tutte
    # Usiamo la colormap 'coolwarm':
    # 0.1 è un bel Blu (Training), 0.9 è un bel Rosso (Validation)
    cmap = plt.cm.coolwarm
    c_train = cmap(0.1)
    c_val = cmap(0.9)

    # Convertiamo il generatore in lista per poterlo scorrere
    splits = list(cv_split_generator)

    # Iteriamo sui fold
    for ii, (tr, tt) in enumerate(splits):
        # tr e tt sono già gli indici (numeri di riga) che ci servono!

        # --- Disegna Training (Blu) ---
        # Usiamo direttamente 'color=c_train' invece di c=indices + cmap
        ax.scatter(tr, [ii + 0.5] * len(tr), color=c_train, marker="_", lw=10)

        # --- Disegna Validation (Rosso) ---
        ax.scatter(tt, [ii + 0.5] * len(tt), color=c_val, marker="_", lw=10)

    # Formattazione Grafico
    yticklabels = [f"Fold {i + 1}" for i in range(n_splits)]
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Indice Temporale (Ore)",
        ylabel="Iterazione",
        ylim=[n_splits + 0.2, -0.2],
        xlim=[0, n_samples],
    )

    ax.set_title(f"Strategia di Split: {n_splits} Fold (Expanding Window)", fontsize=15)

    # Legenda (Ora usa le stesse variabili colore del grafico)
    legend_elements = [
        Patch(facecolor=c_val, label="Validation Set"),
        Patch(facecolor=c_train, label="Training Set"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.show()

    # IMPORTANTE: Dato che abbiamo consumato il generatore trasformandolo in lista,
    # se questa funzione dovesse restituire qualcosa, dovremmo restituire la lista 'splits'.
    # Ma dato che nel tuo codice usi plot_cv_indices solo per visualizzare e poi ricrei
    # lo splitter nel training, va benissimo così.


def plot_predictions(preds, targets, model_name, fold_idx, save_dir):
    """
    Crea un grafico delle predizioni vs ground truth.

    Args:
        preds: Array di predizioni (denormalizzate).
        targets: Array di ground truth (denormalizzate).
        model_name: Nome del modello.
        fold_idx: Indice del fold.
        save_dir: Directory dove salvare il grafico.
    """
    # Prendi solo le prime N ore per visualizzazione leggibile
    n_hours = min(168, len(preds))  # Max 1 settimana

    # Se le predizioni sono multi-step, prendi solo il primo step
    if len(preds.shape) > 1:
        preds_plot = preds[:n_hours, 0]
        targets_plot = targets[:n_hours, 0]
    else:
        preds_plot = preds[:n_hours]
        targets_plot = targets[:n_hours]

    fig, ax = plt.subplots(figsize=(14, 5))

    hours = np.arange(len(preds_plot))
    ax.plot(hours, targets_plot, label="Ground Truth", alpha=0.8, linewidth=1.5)
    ax.plot(hours, preds_plot, label="Predizioni", alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Ore")
    ax.set_ylabel("PV Power (W)")
    ax.set_title(f"{model_name} - Fold {fold_idx + 1} - Predizioni vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Salva il grafico
    plot_path = os.path.join(
        save_dir, f"{model_name}_fold_{fold_idx + 1}_predictions.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  Grafico salvato: {plot_path}")


def plot_training_history(history, model_name, fold_idx, save_dir):
    """
    Crea un grafico della history di training.

    Args:
        history: Dizionario con train_loss, val_loss, val_mase, val_rmse.
        model_name: Nome del modello.
        fold_idx: Indice del fold.
        save_dir: Directory dove salvare il grafico.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot 1: Train vs Val Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Train vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: MASE
    axes[1].plot(epochs, history["val_mase"], label="Val MASE", color="green")
    axes[1].axhline(y=1.0, color="red", linestyle="--", label="Baseline (Naive)")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("MASE")
    axes[1].set_title("Validation MASE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: RMSE
    axes[2].plot(epochs, history["val_rmse"], label="Val RMSE", color="orange")
    axes[2].set_xlabel("Epoca")
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("Validation RMSE")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} - Fold {fold_idx + 1} - Training History", fontsize=14)
    plt.tight_layout()

    # Salva il grafico
    plot_path = os.path.join(save_dir, f"{model_name}_fold_{fold_idx + 1}_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  History salvata: {plot_path}")
