"""
Engine di training: funzioni per training, validazione e fit con early stopping.
Supporta integrazione con Optuna per hyperparameter tuning.
Include factory function per la creazione dei modelli.
"""

import copy
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # AMP per mixed precision
import optuna
from ..config import NAIVE_MAE_PER_FOLD, INPUT_SIZE, TARGET_IDX, LOOKBACK, HORIZON


def create_model(model_name: str, params: dict) -> nn.Module:
    """
    Factory function per creare istanze di modelli.

    Args:
        model_name: nome del modello (lowercase): "lstm", "dlinearm", "dlineari", "patchtst", "tcn"
        params: dizionario con iperparametri del modello.
            Parametri comuni: nessuno (vengono presi da config)
            LSTM: hidden_size, num_layers, dropout
            DLinearM/I: kernel_size
            PatchTST: patch_length, stride, d_model, n_heads, n_layers, dropout
            TCN: hidden_size, num_layers, kernel_size, dropout

    Returns:
        nn.Module: istanza del modello (non spostata su device)

    Raises:
        ValueError: se model_name non è supportato
    """
    model_name = model_name.lower()

    if model_name == "lstm":
        from ..ModelClasses import LSTM

        model_config = {
            "input_size": INPUT_SIZE,
            "hidden_size": params["hidden_size"],
            "output_size": HORIZON,
            "num_layers": params["num_layers"],
            "dropout": params["dropout"],
            "bidirectional": False,
        }
        return LSTM(model_config=model_config)

    elif model_name == "dlinearm":
        from ..ModelClasses import DLinearM

        model_config = {
            "input_size": INPUT_SIZE,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "kernel_size": params["kernel_size"],
        }
        return DLinearM(model_config=model_config)

    elif model_name == "dlineari":
        from ..ModelClasses import DLinearI

        model_config = {
            "target_idx": TARGET_IDX,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "kernel_size": params["kernel_size"],
        }
        return DLinearI(model_config=model_config)

    elif model_name == "patchtst":
        from ..ModelClasses import PatchTST

        model_config = {
            "num_channels": INPUT_SIZE,
            "target_idx": TARGET_IDX,
            "lookback": LOOKBACK,  # Passa lookback esplicitamente
            "horizon": HORIZON,  # Passa horizon esplicitamente
            "patch_length": params["patch_length"],
            "stride": params["stride"],
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "n_layers": params["n_layers"],
            "dropout": params["dropout"],
            "use_cls_token": False,
        }
        return PatchTST(model_config=model_config)

    elif model_name == "tcn":
        from ..ModelClasses import TCN

        model_config = {
            "input_size": INPUT_SIZE,
            "output_size": HORIZON,
            "hidden_size": params["hidden_size"],
            "num_layers": params["num_layers"],
            "kernel_size": params["kernel_size"],
            "dropout": params["dropout"],
        }
        return TCN(model_config=model_config)


    elif model_name == "patchtst_finetune":
        from ..ModelClasses import PatchTSTFinetune
    
        model_config = {
            "num_channels": INPUT_SIZE,
            "target_idx": TARGET_IDX,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "patch_length": params["patch_length"],
            "stride": params["stride"],
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "n_layers": params["n_layers"],
            "dropout": params["dropout"],
            "freeze_backbone": params.get("freeze_backbone", False),
        }
        return PatchTSTFinetune(model_config=model_config)

    else:
        raise ValueError(
            f"Modello '{model_name}' non supportato. "
            f"Modelli validi: lstm, dlinearm, dlineari, patchtst, tcn, patchtst_finetune"
        )


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    loss_fn,
    device,
    grad_clip_norm: float = None,
    scaler: GradScaler = None,
):
    """
    Esegue un'epoca di addestramento (Forward + Backward).
    Supporta AMP (Automatic Mixed Precision) se viene passato uno scaler.

    Args:
        model: Modello PyTorch.
        dataloader: DataLoader del training set.
        optimizer: Optimizer PyTorch.
        loss_fn: Loss function.
        device: Device (cuda/cpu).
        grad_clip_norm: Se specificato, applica gradient clipping con questa norma.
        scaler: GradScaler per AMP (opzionale).

    Returns:
        float: Loss media dell'epoca.
    """
    model.train()
    running_loss = 0.0
    use_amp = scaler is not None

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()

        # AMP: forward con autocast
        with autocast(device_type="cuda", enabled=use_amp):
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)

        if use_amp:
            # AMP: backward con scaling
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, device):
    """
    Esegue un'epoca di validazione (Solo Forward).
    Calcola MSE, MAE e RMSE.

    Args:
        model: Modello PyTorch.
        dataloader: DataLoader del validation set.
        loss_fn: Loss function.
        device: Device (cuda/cpu).

    Returns:
        tuple: (avg_loss, avg_mae, avg_rmse)
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0

    mae_fn = nn.L1Loss()

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)

            # 1. Calcolo Loss (MSE) per Early Stopping
            loss = loss_fn(predictions, batch_y)
            running_loss += loss.item()

            # 2. Calcolo MAE per metrica MASE
            mae = mae_fn(predictions, batch_y)
            running_mae += mae.item()

    avg_loss = running_loss / len(dataloader)
    avg_mae = running_mae / len(dataloader)

    # 3. Calcolo RMSE = sqrt(MSE)
    avg_rmse = math.sqrt(avg_loss)

    return avg_loss, avg_mae, avg_rmse


def fit_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    fold_idx=None,
    patience=10,
    trial=None,
    optimizer_cls=None,
    optimizer_kwargs=None,
    loss_fn=None,
    grad_clip_norm=1.0,
    verbose=True,
    baseline_mae=None,
):
    """
    Ciclo principale di addestramento con Early Stopping.
    Supporta integrazione con Optuna per pruning e reporting.

    Args:
        model: Modello PyTorch da addestrare.
        train_loader: DataLoader per il training.
        val_loader: DataLoader per la validazione.
        epochs: Numero massimo di epoche.
        lr: Learning rate.
        device: Device (cuda o cpu).
        fold_idx: Indice del fold corrente (per calcolare MASE). Ignorato se baseline_mae è fornito.
        patience: Numero di epoche senza miglioramento prima dell'early stopping.
        trial: Oggetto optuna.Trial per pruning/reporting (opzionale).
        optimizer_cls: Classe optimizer (default: torch.optim.Adam).
        optimizer_kwargs: Kwargs extra per l'optimizer (es. weight_decay).
        loss_fn: Loss function (default: nn.MSELoss()).
        grad_clip_norm: Norma massima per gradient clipping (None per disabilitare).
        verbose: Se True, stampa progress durante il training.
        baseline_mae: MAE del modello naive per calcolare MASE. Se None, usa NAIVE_MAE_PER_FOLD[fold_idx].

    Returns:
        tuple: (model, history, best_epoch)
            - model: Modello addestrato con i pesi migliori.
            - history: Dizionario con le metriche per ogni epoca.
            - best_epoch: Epoca con il miglior MASE.
    """
    # Default optimizer e loss - AdamW con weight decay per regolarizzazione
    if optimizer_cls is None:
        optimizer_cls = torch.optim.AdamW
    if optimizer_kwargs is None:
        optimizer_kwargs = {"weight_decay": 0.01}
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    # AMP: crea scaler solo se su CUDA
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    # Usa baseline_mae passato direttamente, altrimenti prendi dalla lista per fold_idx
    if baseline_mae is None:
        if fold_idx is None:
            raise ValueError("Devi fornire baseline_mae o fold_idx per calcolare MASE")
        baseline_mae = NAIVE_MAE_PER_FOLD[fold_idx]

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mase": [],
        "val_rmse": [],
    }

    best_mase = float("inf")
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0

    if verbose:
        print(f"Start Training on {device}...")

    for epoch in range(epochs):
        # --- TRAINING ---
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip_norm, scaler
        )

        # --- VALIDATION ---
        val_loss, val_mae, val_rmse = validate_one_epoch(
            model, val_loader, loss_fn, device
        )

        current_mase = val_mae / baseline_mae

        # Salviamo la storia
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mase"].append(current_mase)
        history["val_rmse"].append(val_rmse)

        # Stampa pulita
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f} | "
                f"Val MASE: {current_mase:.4f} | "
                f"Val RMSE: {val_rmse:.6f}"
            )

        # --- OPTUNA INTEGRATION ---
        if trial is not None:
            trial.report(current_mase, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # --- EARLY STOPPING CHECK (su MASE, coerente con Optuna) ---
        if current_mase < best_mase:
            best_mase = current_mase
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"Early Stopping attivato all'epoca {epoch + 1}. "
                        f"Best MASE: {best_mase:.4f}"
                    )
                break

    # IMPORTANTE: Carichiamo i pesi migliori nel modello prima di restituirlo
    model.load_state_dict(best_model_wts)
    if verbose:
        print("Training Completato.")

    return model, history, best_epoch
