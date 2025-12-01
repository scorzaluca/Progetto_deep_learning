import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
    """
    model.eval()  # Disabilita dropout
    running_loss = 0.0

    with torch.no_grad():  # Disabilita il calcolo dei gradienti (risparmia memoria)
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)

            running_loss += loss.item()

    return running_loss / len(dataloader)


def fit_model(model, train_loader, val_loader, epochs, lr, device, patience=10):
    """
    Ciclo principale di addestramento con Early Stopping.
    """
    # Definisci Optimizer e Loss qui (o passali come argomenti se vuoi più controllo)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())  # Copia iniziale

    print(f"Start Training on {device}...")

    for epoch in range(epochs):
        # --- TRAINING ---
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # --- VALIDATION ---
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

        # Salviamo la storia
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Stampa pulita
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

        # --- EARLY STOPPING CHECK ---
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
