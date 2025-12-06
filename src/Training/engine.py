import copy
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src import config


def train_one_epoch(model, dataloader: DataLoader, optimizer, loss_fn, device):
    """
    Esegue un'epoca di addestramento (Forward + Backward).
    """
    model.train()  # Abilita dropout/batchnorm
    running_loss = 0.0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()

        prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, device):
    """
    Esegue un'epoca di validazione (Solo Forward).
    Calcola MSE, MAE e RMSE.
    """
    model.eval()  # Disabilita dropout
    running_loss = 0.0
    running_mae = 0.0  # MAE per il MASE

    mae_fn = nn.L1Loss()  # Funzione per calcolare il MAE

    with torch.no_grad():  # Disabilita il calcolo dei gradienti (risparmia memoria)
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


def fit_model(model, train_loader, val_loader, epochs, lr, device, fold_idx, patience=10):
    """
    Ciclo principale di addestramento con Early Stopping.
    
    Args:
        model: Modello PyTorch da addestrare.
        train_loader: DataLoader per il training.
        val_loader: DataLoader per la validazione.
        epochs: Numero massimo di epoche.
        lr: Learning rate.
        device: Device (cuda o cpu).
        fold_idx: Indice del fold corrente (per calcolare MASE).
        patience: Numero di epoche senza miglioramento prima dell'early stopping.
        
    Returns:
        model: Modello addestrato con i pesi migliori.
        history: Dizionario con le metriche per ogni epoca.
    """
    # Definisci Optimizer e Loss qui (o passali come argomenti se vuoi più controllo)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    baseline_mae = config.NAIVE_MAE_PER_FOLD[fold_idx]

    history = {
        "train_loss": [], 
        "val_loss": [], 
        "val_mase": [],
        "val_rmse": []  # Aggiunta metrica RMSE
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())  # Copia iniziale

    print(f"Start Training on {device}...")

    for epoch in range(epochs):
        # --- TRAINING ---
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # --- VALIDATION ---
        val_loss, val_mae, val_rmse = validate_one_epoch(model, val_loader, loss_fn, device)

        current_mase = val_mae / baseline_mae

        # Salviamo la storia
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mase"].append(current_mase)
        history["val_rmse"].append(val_rmse)

        # Stampa pulita
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f} | "
                f"Val MASE: {current_mase:.4f} | "
                f"Val RMSE: {val_rmse:.6f}"
            )

        # --- EARLY STOPPING CHECK (su MSE più stabile) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Salviamo i pesi MIGLIORI, non gli ultimi!
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early Stopping attivato all'epoca {epoch + 1}. Best Val Loss: {best_val_loss:.6f}"
                )
                break

    # IMPORTANTE: Carichiamo i pesi migliori nel modello prima di restituirlo
    model.load_state_dict(best_model_wts)
    print("Training Completato.")

    return model, history
