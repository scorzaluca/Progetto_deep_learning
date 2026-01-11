"""
Script per l'ottimizzazione degli iperparametri del pretraining PatchTST.
Self-contained: contiene hyperparameter space, Optuna optimizer, fit function.

Uso:
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
    Spazio di ricerca per il pretraining PatchTST.
    NOTA: Dopo l'ottimizzazione, aggiorna d_model in PATCHTST_CONFIG e EncoderLSTM.
    """
    return {
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "patch_length": trial.suggest_categorical("patch_length", [8, 12, 16, 24]),
        "stride": trial.suggest_categorical("stride", [4, 6, 8]),
        "mask_ratio": trial.suggest_float("mask_ratio", 0.3, 0.6),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
    }


# =============================================================================
# DATA LOADING
# =============================================================================
def load_and_prepare_data():
    """Carica, splitta e normalizza i dati per il pretraining."""
    print(f"Caricamento dati da: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Split temporale (mantiene ordine cronologico, no data leakage)
    split_idx = int(len(df) * (1 - VAL_SPLIT))
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    # Normalizza: fit su train, transform su entrambi
    scaler = MinMaxScaler()
    df_train_scaled = pd.DataFrame(
        scaler.fit_transform(df_train),
        columns=df.columns,
    )
    df_val_scaled = pd.DataFrame(
        scaler.transform(df_val),
        columns=df.columns,
    )

    print(f"Train: {len(df_train_scaled)} righe, Val: {len(df_val_scaled)} righe")
    return df_train_scaled, df_val_scaled


def create_dataloaders(df_train, df_val, batch_size=64):
    """Crea DataLoaders per train e validation."""
    train_dataset = PretrainingDataset(df=df_train, lookback=LOOKBACK, step=1)
    val_dataset = PretrainingDataset(df=df_val, lookback=LOOKBACK, step=1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

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
    Training loop per il pretraining.
    Restituisce i pesi dell'encoder del best model e la best validation loss.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
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
            loss = model(batch_x)
            loss.backward()
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
                loss = model(batch_x)
                val_loss += loss.item()
        val_loss /= len(val_loader)

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
    """Optimizer Optuna per il pretraining PatchTST."""

    def __init__(self, train_loader, val_loader, device, verbose=False):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.verbose = verbose
        self.best_encoder_state = None
        self.best_params = None

    def _objective(self, trial) -> float:
        """Funzione obiettivo: minimizza validation reconstruction loss."""
        # Genera iperparametri
        params = get_pretrain_hyperparameter_space(trial)

        # Crea config per il modello
        model_config = {
            "num_channels": INPUT_SIZE,
            "lookback": LOOKBACK,
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "n_layers": params["n_layers"],
            "patch_length": params["patch_length"],
            "stride": params["stride"],
            "dropout": params["dropout"],
            "mask_ratio": params["mask_ratio"],
        }

        # Crea modello
        model = PatchTSTPretraining(model_config)

        # Training
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

        # Salva i pesi del best trial
        if trial.number == 0 or val_loss < trial.study.best_value:
            self.best_encoder_state = encoder_state
            self.best_params = params

        return val_loss

    def optimize(self, n_trials: int, study_name: str, new_study: bool = True):
        """Esegue l'ottimizzazione."""
        storage_path = os.path.join(RESULTS_DIR, "optuna_studies.db")
        storage = f"sqlite:///{storage_path}"

        # Gestione nuovo studio vs ripresa
        if new_study:
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                print(f"Studio '{study_name}' esistente eliminato.")
            except KeyError:
                pass

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=not new_study,
            direction="minimize",
            sampler=TPESampler(seed=SEED),
        )

        # Calcola trials rimanenti
        completed_trials = len(study.trials)
        remaining_trials = max(0, n_trials - completed_trials)

        print(f"\n{'=' * 60}")
        print("\nPRETRAINING OPTIMIZATION")
        print(f"Studio: {study_name}")
        print(f"Trial totali: {n_trials}, Rimanenti: {remaining_trials}")
        print(f"Epochs: {EPOCHS}, Patience: {PATIENCE}")
        print(f"{'=' * 60}\n")

        if remaining_trials > 0:
            study.optimize(
                self._objective, n_trials=remaining_trials, show_progress_bar=True
            )

        # Aggiorna best_params dal study (in caso di resume)
        self.best_params = study.best_params

        return study


# =============================================================================
# SAVE RESULTS
# =============================================================================
def save_pretrain_results(encoder_state, best_params, best_loss, study_name):
    """Salva pesi encoder e parametri."""
    # Salva pesi encoder
    pretrained_dir = os.path.join(RESULTS_DIR, "pretrained")
    os.makedirs(pretrained_dir, exist_ok=True)
    encoder_path = os.path.join(pretrained_dir, f"{study_name}_encoder.pth")
    torch.save(encoder_state, encoder_path)
    print(f"Encoder salvato: {encoder_path}")

    # Salva parametri
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
    print(f"Params salvati: {params_path}")

    return encoder_path


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("PATCHTST PRETRAINING HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    # Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Carica dati
    df_train, df_val = load_and_prepare_data()

    # Crea loaders
    train_loader, val_loader = create_dataloaders(df_train, df_val)

    # Crea optimizer
    optimizer = PretrainOptimizer(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        verbose=False,
    )

    # Esegui ottimizzazione
    study = optimizer.optimize(
        n_trials=N_TRIALS,
        study_name=STUDY_NAME,
        new_study=NEW_STUDY,
    )

    print("\n" + "=" * 60)
    print("OTTIMIZZAZIONE COMPLETATA")
    print("=" * 60)
    print(f"Best Val Loss: {study.best_value:.6f}")
    print(f"Best Params: {study.best_params}")

    # Salva risultati
    encoder_path = save_pretrain_results(
        encoder_state=optimizer.best_encoder_state,
        best_params=study.best_params,
        best_loss=study.best_value,
        study_name=STUDY_NAME,
    )

    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"Encoder pretrained: {encoder_path}")
    print("\nPer usare nel fine-tuning EncoderLSTM:")
    print(f'  pretrain_path = "{encoder_path}"')
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
