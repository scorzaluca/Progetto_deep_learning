import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config


def train_one_epoch(model, dataloader: DataLoader, optimizer, loss_fn, device):
    """
    Esegue un'epoca di addestramento (Forward + Backward).
    """
    model.train()  # Abilita dropout/batchnorm
    running_loss = 0.0

    num_batches = len(dataloader)

    for i, batch_x, batch_y in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()

        prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # .size(0) ci dà il numero di campioni nel batch (es. 64 o l'ultimo che può essere minore)
        current_batch_size = batch_x.size(0)
        
        # .shape ci dà le dimensioni complete: [Batch, Time, Features]
        # Trasformiamo in list per una stampa più pulita (es. [64, 48, 12])
        input_shape = list(batch_x.shape)   
        target_shape = list(batch_y.shape)  

        # --- STAMPA DETTAGLIATA ---
        print(f"   Batch {i+1}/{num_batches} | "
              f"Loss: {running_loss:.6f} | "
              f"Samples: {current_batch_size} | "
              f"In Shape: {input_shape} | "
              f"Out Shape: {target_shape}")

    return running_loss / num_batches


def validate_one_epoch(model, dataloader, loss_fn, device):
    """
    Esegue un'epoca di validazione (Solo Forward).
    """
    model.eval()  # Disabilita dropout
    running_loss = 0.0
    running_mae = 0.0  # MAE per il MASE

    mae_fn = nn.L1Loss() # Funzione per calcolare il MAE

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

    return avg_loss, avg_mae


def fit_model(model, train_loader, val_loader, epochs, lr, device, fold_idx, patience=10):
    """
    Ciclo principale di addestramento con Early Stopping.
    """
    # Definisci Optimizer e Loss qui (o passali come argomenti se vuoi più controllo)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    baseline_mae = config.NAIVE_MAE_PER_FOLD[fold_idx]

    history = {"train_loss": [], "val_loss": [], "val_mase": []}

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())  # Copia iniziale

    print(f"Start Training on {device}...")

    for epoch in range(epochs):
        # --- TRAINING ---
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # --- VALIDATION ---
        val_loss, val_mae = validate_one_epoch(model, val_loader, loss_fn, device)

        current_mase = val_mae / baseline_mae

        # Salviamo la storia
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mase"].append(current_mase)

        # Stampa pulita
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f} | "
                f"Val MASE: {current_mase:.4f}"
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
