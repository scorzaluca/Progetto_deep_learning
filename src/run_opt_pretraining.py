"""
Script for Hyperparameter Optimization of PatchTST Pretraining.

This script manages the Self-Supervised Learning (SSL) pretraining phase for the PatchTST weights.
It is self-contained and includes:
1. Definition of the Hyperparameter Search Space (d_model, n_heads, patch_length, etc.).
2. The `fit_pretrain` loop specific to the Masked Autoencoder task (Reconstruction Loss).
3. The Optuna `PretrainOptimizer` class to manage the study.

Usage:
    python -m src.run_opt_pretraining
"""

import os
import json
import torch
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import SEED, LOOKBACK, INPUT_SIZE
from .config.tuning_pretrain_config import (
    N_TRIALS,
    EPOCHS,
    PATIENCE,
    STUDY_NAME,
    NEW_STUDY,
    VAL_SPLIT,
    DATA_PATH,
    RESULTS_DIR,
)
from .DataLoading import PretrainingDataset
from .ModelClasses import PatchTSTPretraining
from .Utils import set_seed


# =============================================================================
# HYPERPARAMETER SPACE
# =============================================================================
def get_pretrain_hyperparameter_space(trial) -> dict:
    """
    Defines the Hyperparameter Search Space for PatchTST Pretraining.

    IMPORTANT: The optimal 'd_model' found here MUST be manually updated 
    in `PATCHTST_CONFIG` and `EncoderLSTM` before running the downstream tuning tasks.

    Args:
        trial (optuna.Trial): The Optuna trial object to suggest parameters.

    Returns:
        dict: A dictionary of sampled hyperparameters.
    """
    return {
        # 'd_model': Embedding dimension size. Higher = more capacity but heavier.
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),

        # 'n_heads': Number of Attention Heads.
        # Must be a divisor of d_model
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),

        # 'n_layers': Number of Transformer Encoder layers.
        "n_layers": trial.suggest_int("n_layers", 2, 4),

        # 'patch_length': Size of each patch (token)
        # Larger patches = less global tokens but more local context per token.
        "patch_length": trial.suggest_categorical("patch_length", [8, 12, 16, 24]),

        # 'stride': Stride between patches.
        "stride": trial.suggest_categorical("stride", [4, 6, 8]),

        # 'mask_ratio': Percentage of patches to hide during pretraining (Masked Autoencoder).
        # The model tries to reconstruct these missing patches.
        "mask_ratio": trial.suggest_float("mask_ratio", 0.3, 0.6),

        # 'dropout': Regularization to prevent overfitting during reconstruction.
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),

        # 'lr': Learning Rate for the AdamW optimizer
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
    }


# =============================================================================
# DATA LOADING
# =============================================================================
def load_and_prepare_data():
    """
    Loads, splits, and scales the dataset for Pretraining.

    The split is strictly temporal (no shuffling) to avoid data leakage.
    Normalization is fitted ONLY on the training part and applied to both.

    Returns:
        tuple: (df_train_scaled, df_val_scaled) - The two normalized DataFrames.
    """
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Temporal Split:
    # We use the first (1 - VAL_SPLIT)% for training and the rest for validation.
    split_idx = int(len(df) * (1 - VAL_SPLIT))
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    # Normalization:
    scaler = MinMaxScaler()
    
    # Fit on Train, Transform Train
    df_train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train),
        columns=df.columns,
    )
    
    # Transform Validation using Train statistics
    df_val_scaled = pd.DataFrame(
        scaler.transform(df_val),
        columns=df.columns,
    )

    print(f"Train: {len(df_train_scaled)} rows, Val: {len(df_val_scaled)} rows")
    return df_train_scaled, df_val_scaled


def create_dataloaders(df_train, df_val, batch_size=64):
    """
    Creates PyTorch DataLoaders for the Pretraining phase.

    Args:
        df_train (pd.DataFrame): Normalized training dataframe.
        df_val (pd.DataFrame): Normalized validation dataframe.
        batch_size (int): Number of samples per batch.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create Datasets using the custom PretrainingDataset class
    # This class handles the sliding window logic (creating sequences of length LOOKBACK)
    train_dataset = PretrainingDataset(df=df_train, lookback=LOOKBACK, step=1)
    val_dataset = PretrainingDataset(df=df_val, lookback=LOOKBACK, step=1)

    # Train Loader:
    # - shuffle=True: for training to break temporal correlation effects within batches
    # - drop_last=True: Drops incomplete batches to ensure stable dimensions
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(), # Optimizes transfer to GPU
        num_workers=0,  
    )

    # Validation Loader:
    # - shuffle=False
    # - drop_last=False: We want to evaluate on ALL samples
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    return train_loader, val_loader


# =============================================================================
# FIT FUNCTION (PRETRAINING SPECIFIC)
# =============================================================================
def fit_pretrain(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    patience=10,
    verbose=False,
):
    """
    Executes the Pretraining loop (Masked Autoencoder).

    Unlike supervised training, the pretraining task is unsupervised:
    the model tries to reconstruct masked patches of the input time series.
    The loss function (MSE) is computed automatically inside the model's `forward` method
    when `mode='pretrain'`.

    Args:
        model (nn.Module): The PatchTST model in pretraining mode.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Max number of epochs.
        lr (float): Learning rate.
        device (torch.device): Device to train on.
        patience (int): Early stopping patience.
        verbose (bool): Whether to print progress logs.

    Returns:
        tuple: (best_encoder_state_dict, best_val_loss)
    """
    model = model.to(device)
    # AdamW optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Scheduler to reduce LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3, # Wait 3 epochs before reducing LR
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_encoder_state = None

    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            # Forward pass:
            # - The model internally masks a ratio of patches
            # - It tries to reconstruct the masked patches
            # - Returns the reconstruction MSE loss
            loss = model(batch_x)
            
            loss.backward()
            
            # Gradient Clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                # In validation, we still mask and compute reconstruction loss
                loss = model(batch_x)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Update scheduler based on validation loss
        scheduler.step(val_loss)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save only the encoder weights 
            best_encoder_state = model.get_encoder_state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    return best_encoder_state, best_val_loss


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================
class PretrainOptimizer:
    """
    Optuna Optimizer wrapper for PatchTST Pretraining.

    It manages the creation of trials, the suggestion of hyperparameters,
    and the execution of the training loop for each configuration.
    It tracks the best validation loss and saves the corresponding encoder weights.
    """

    def __init__(self, train_loader, val_loader, device, verbose=False):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.verbose = verbose
        self.best_encoder_state = None
        self.best_params = None

    def _objective(self, trial) -> float:
        """
        Optuna Objective Function.
        
        Minimizes the Reconstruction Loss on the Validation set.
        For each trial:
        1. Samples hyperparameters.
        2. Builds the PatchTST model in Pretraining mode.
        3. Trains the model (Unsupervised Masked Autoencoder).
        4. Tracks the best model found so far across trials.

        Args:
            trial (optuna.Trial): Current trial.

        Returns:
            float: Best validation loss achieved in this trial.
        """
        # Sample Hyperparameters
        params = get_pretrain_hyperparameter_space(trial)

        # Configure Model
        # We explicitly map the sampled params to the config structure expected by the model
        model_config = {
            "num_channels": INPUT_SIZE,
            "lookback": LOOKBACK,
            # Sampled params:
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "n_layers": params["n_layers"],
            "patch_length": params["patch_length"],
            "stride": params["stride"],
            "dropout": params["dropout"],
            "mask_ratio": params["mask_ratio"],
        }

        # Instantiate Model
        # PatchTSTPretraining includes the Encoder and the Projection Head for reconstruction
        model = PatchTSTPretraining(model_config)

        # Train (fit_pretrain)
        # Returns the best encoder weights (head removed) and the min validation loss
        encoder_state, val_loss = fit_pretrain(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=EPOCHS,
            lr=params["lr"],
            device=self.device,
            patience=PATIENCE,
            verbose=self.verbose,
        )

        # Track Best Trial
        # If this is the first trial OR if it beat the previous best in the study
        if trial.number == 0 or val_loss < trial.study.best_value:
            self.best_encoder_state = encoder_state
            self.best_params = params

        return val_loss

    def optimize(self, n_trials: int, study_name: str, new_study: bool = True):
        """
        Runs the Optuna optimization process.

        Handles the creation/loading of the SQLite study, manages resume logic,
        and executes the optimization loop.

        Args:
            n_trials (int): Total number of trials to run.
            study_name (str): Unique identifier for the study in the DB.
            new_study (bool): If True, deletes existing study with same name.

        Returns:
            optuna.Study: The completed study object.
        """
        # Define storage path for SQLite database
        # This allows persistence: we can stop and resume optimization later.
        storage_path = os.path.join(RESULTS_DIR, "optuna_studies.db")
        storage = f"sqlite:///{storage_path}"

        # Manage Study Creation vs Resuming
        # If new_study=True, we delete any previous study with the same name to start fresh.
        if new_study:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                print(f"Existing study '{study_name}' deleted.")
            except KeyError:
                pass  # Study did not exist, nothing to delete

        # Create or Load the Study
        # - direction='minimize': We want to minimize the Reconstruction Loss.
        # - sampler=TPESampler: Tree-structured Parzen Estimator (Bayesian optimization).
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=not new_study,
            direction="minimize",
            sampler=TPESampler(seed=SEED),
        )

        # Idempotency Check: Calculate how many trials are left
        completed_trials = len(study.trials)
        remaining_trials = max(0, n_trials - completed_trials)

        print(f"\n{'=' * 60}")
        print("\nPRETRAINING OPTIMIZATION")
        print(f"Study: {study_name}")
        print(f"Total Trials: {n_trials}, Remaining: {remaining_trials}")
        print(f"Epochs: {EPOCHS}, Patience: {PATIENCE}")
        print(f"{'=' * 60}\n")

        # Run Optimization loop
        if remaining_trials > 0:
            study.optimize(
                self._objective, n_trials=remaining_trials, show_progress_bar=True
            )

        # Update local best_params from the study
        # This ensures we have the best parameters even if we didn't run any new trials (just loaded).
        self.best_params = study.best_params

        return study


# =============================================================================
# SAVE RESULTS
# =============================================================================
def save_pretrain_results(encoder_state, best_params, best_loss, study_name):
    """
    Saves the best pretrained encoder weights and hyperparameters.

    Args:
        encoder_state (dict): State dict of the encoder (excluding projection head).
        best_params (dict): Best hyperparameters found by Optuna.
        best_loss (float): Best validation loss achieved.
        study_name (str): Name of the study (used for filenames).

    Returns:
        str: Path to the saved encoder weights file.
    """
    # Save Encoder Weights
    # We save these in a specific 'pretrained' folder to distinguish them from full models.
    pretrained_dir = os.path.join(RESULTS_DIR, "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)
    
    encoder_path = os.path.join(pretrained_dir, f"{study_name}_encoder.pth")
    torch.save(encoder_state, encoder_path)
    print(f"Encoder saved: {encoder_path}")

    # Save Hyperparameters
    # We save this JSON so we can load the correct config during fine-tuning.
    params_dir = os.path.join(RESULTS_DIR, "params")
    os.makedirs(params_dir, exist_ok=True)
    
    results = {
        "study_name": study_name,
        "best_val_loss": best_loss,
        "best_params": best_params,
    }
    
    params_path = os.path.join(params_dir, f"{study_name}_params.json")
    with open(params_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Params saved: {params_path}")

    return encoder_path


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Main Execution Pipeline for PatchTST Pretraining Optimization.

    Steps:
    1. Sets random seed for reproducibility.
    2. Checks and selects the computing device (GPU/CPU).
    3. Loads and Preprocesses the dataset (MinMax Scaling).
    4. Creates Training and Validation DataLoaders.
    5. Initializes the Optuna Optimizer Wrapper.
    6. Runs the Hyperparameter Optimization (TPE Sampler).
    7. Displays the Best Results (Loss and Parameters).
    8. Saves the Best Encoder Weights and Configuration for later use.
    """
    print("\n" + "=" * 60)
    print("PATCHTST PRETRAINING HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    # Setup Environment
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data Loading & Preparation:
    # Load raw data, split chronologically, and normalize (fit on train)
    df_train, df_val = load_and_prepare_data()

    # Create DataLoaders
    # Convert DataFrames to PyTorch DataLoaders
    train_loader, val_loader = create_dataloaders(df_train, df_val)

    # Initialize Optimizer:
    # The class that bridges Optuna with our training loop
    optimizer = PretrainOptimizer(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        verbose=False, 
    )

    # Run Optimization
    # Executes n_trials, minimizing the validation reconstruction loss
    study = optimizer.optimize(
        n_trials=N_TRIALS,
        study_name=STUDY_NAME,
        new_study=NEW_STUDY,
    )

    # Report Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETED")
    print("=" * 60)
    print(f"Best Val Loss: {study.best_value:.6f}")
    print(f"Best Params: {study.best_params}")

    # Save Best Model
    encoder_path = save_pretrain_results(
        encoder_state=optimizer.best_encoder_state,
        best_params=study.best_params,
        best_loss=study.best_value,
        study_name=STUDY_NAME,
    )

    
    


if __name__ == "__main__":
    main()
