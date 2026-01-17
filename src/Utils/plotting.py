"""
Plotting Utility Functions.

This module contains functions to visualize:
1. Cross-Validation Splits (train/validation folds).
2. Prediction vs Ground Truth comparisons.
3. Training History (Loss, MASE, RMSE curves).
4. Error Distributions (Histograms, Scatter plots).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os


def plot_cv_indices(cv_split_generator, n_samples, n_splits):
    """
    Visualizes the Time Series Cross-Validation folds.

    Args:
        cv_split_generator: Generator yielding (train_idx, test_idx).
        n_samples (int): Total number of samples.
        n_splits (int): Number of folds.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors using coolwarm colormap
    # 0.1 is Blue (Training), 0.9 is Red (Validation)
    cmap = plt.cm.coolwarm
    c_train = cmap(0.1)
    c_val = cmap(0.9)

    # Convert generator to list to iterate
    splits = list(cv_split_generator)

    # Iterate over folds
    for ii, (tr, tt) in enumerate(splits):
        # Draw Training indices (Blue)
        ax.scatter(tr, [ii + 0.5] * len(tr), color=c_train, marker="_", lw=10)

        # Draw Validation indices (Red)
        ax.scatter(tt, [ii + 0.5] * len(tt), color=c_val, marker="_", lw=10)

    # Formatting the plot
    yticklabels = [f"Fold {i + 1}" for i in range(n_splits)]
    ax.set(
        yticks=np.arange(n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Time Index (Hours)",
        ylabel="Iteration",
        ylim=[n_splits + 0.2, -0.2],
        xlim=[0, n_samples],
    )

    ax.set_title(f"Split Strategy: {n_splits} Fold (Expanding Window)", fontsize=15)

    # Legend
    legend_elements = [
        Patch(facecolor=c_val, label="Validation Set"),
        Patch(facecolor=c_train, label="Training Set"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.show()




def plot_predictions(preds, targets, model_name, fold_idx, save_dir):
    """
    Plots predictions vs ground truth for a specific fold.

    Args:
        preds: Array of predictions (denormalized).
        targets: Array of ground truth values (denormalized).
        model_name (str): Name of the model.
        fold_idx (int): Index of the current fold.
        save_dir (str): Directory to save the plot.
    """
    # Take only the first N hours for readable visualization (Max 1 week)
    n_hours = min(168, len(preds))

    # If multi-step predictions, take only the first step
    if len(preds.shape) > 1:
        preds_plot = preds[:n_hours, 0]
        targets_plot = targets[:n_hours, 0]
    else:
        preds_plot = preds[:n_hours]
        targets_plot = targets[:n_hours]

    fig, ax = plt.subplots(figsize=(14, 5))

    hours = np.arange(len(preds_plot))
    ax.plot(hours, targets_plot, label="Ground Truth", alpha=0.8, linewidth=1.5)
    ax.plot(hours, preds_plot, label="Predictions", alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Hours")
    ax.set_ylabel("PV Power (W)")
    ax.set_title(f"{model_name} - Fold {fold_idx + 1} - Predictions vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(
        save_dir, f"{model_name}_fold_{fold_idx + 1}_predictions.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  Plot saved: {plot_path}")


def plot_training_history(history, model_name, fold_idx, save_dir):
    """
    Plots the training history (Loss, MASE, RMSE).

    Args:
        history (dict): Dictionary with keys 'train_loss', 'val_loss', 'val_mase', 'val_rmse'.
        model_name (str): Name of the model.
        fold_idx (int): Index of the fold.
        save_dir (str): Directory where to save the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot 1: Train vs Val Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Train vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: MASE
    axes[1].plot(epochs, history["val_mase"], label="Val MASE", color="green")
    axes[1].axhline(y=1.0, color="red", linestyle="--", label="Baseline (Naive)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MASE")
    axes[1].set_title("Validation MASE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: RMSE
    axes[2].plot(epochs, history["val_rmse"], label="Val RMSE", color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("Validation RMSE")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} - Fold {fold_idx + 1} - Training History", fontsize=14)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(save_dir, f"{model_name}_fold_{fold_idx + 1}_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  History saved: {plot_path}")


def plot_test_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_samples: int = 5,
    save_path: str = None,
):
    """
    Plots a subset of random predictions vs targets relative to the Test set.

    Args:
        predictions (np.ndarray): Predictions array (N, horizon, 1).
        targets (np.ndarray): Targets array (N, horizon, 1).
        n_samples (int): Number of random samples to plot.
        save_path (str, optional): Path where to save the plot.
    """
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))

    # Randomly select samples indices
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
        print(f" Plot saved: {save_path}")

    plt.show()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = None,
):
    """
    Plots the error distribution (Histogram) and correlation (Scatter plot).

    Args:
        predictions (np.ndarray): Predictions array.
        targets (np.ndarray): Target array.
        save_path (str, optional): Path where to save the plot.
    """
    errors = predictions.flatten() - targets.flatten()

    # Calculate statistics
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    var_err = np.var(errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with statistics
    axes[0].hist(errors, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero")
    axes[0].axvline(
        mean_err,
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_err:.4f}",
    )
    axes[0].set_xlabel("Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")

    # Text box with statistics
    stats_text = f"Mean: {mean_err:.4f}\nStd: {std_err:.4f}\nVar: {var_err:.6f}"
    axes[0].text(
        0.95,
        0.95,
        stats_text,
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[0].legend(loc="upper left")

    # Scatter - Perfect line crosses the entire data range
    targets_flat = targets.flatten()
    preds_flat = predictions.flatten()
    axes[1].scatter(targets_flat, preds_flat, alpha=0.3, s=5)

    # Perfect line crossing the whole graph
    min_val = min(targets_flat.min(), preds_flat.min())
    max_val = max(targets_flat.max(), preds_flat.max())
    axes[1].plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect"
    )

    axes[1].set_xlabel("Target")
    axes[1].set_ylabel("Prediction")
    axes[1].set_title("Prediction vs Target")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f" Plot saved: {save_path}")

    plt.show()