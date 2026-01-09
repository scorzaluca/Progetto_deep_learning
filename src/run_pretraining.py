"""
Script per Self-Supervised Pretraining di PatchTST.

Esegue masked reconstruction sul dataset e salva i pesi dell'encoder.
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from .config import RESULTS_DIR, SEED, PATCHTST_CONFIG, LOOKBACK
from .DataLoading import PretrainingDataset
from .ModelClasses import PatchTSTPretraining
from .Utils import set_seed
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ===== CONFIGURAZIONE PRETRAINING =====
DATA_PATH = "data/processed/preprocessed_ds.csv"
PRETRAIN_EPOCHS = 50
PRETRAIN_LR = 1e-4
BATCH_SIZE = 64
PATIENCE = 10           # Early stopping
VAL_SPLIT = 0.2         # 20% per validation
SCHEDULER_FACTOR = 0.5  # Riduzione LR
SCHEDULER_PATIENCE = 3  # Epoche prima di ridurre LR
SCHEDULER_MIN_LR = 1e-6 # LR minimo


def load_and_prepare_data():
    """Carica, splitta e normalizza i dati per il pretraining."""
    print("Caricamento dati...")
    df = pd.read_csv(DATA_PATH)
    
    # Split temporale 80/20 (mantiene ordine cronologico)
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


def create_pretraining_loaders(df_train, df_val):
    """Crea i DataLoader per train e validation."""
    train_dataset = PretrainingDataset(df=df_train, lookback=LOOKBACK, step=1)
    val_dataset = PretrainingDataset(df=df_val, lookback=LOOKBACK, step=1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    
    print(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
    print(f"Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
    return train_loader, val_loader


def pretrain(model, train_loader, val_loader, device, epochs, lr, patience):
    """
    Loop di pretraining con validation, early stopping e scheduler.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Scheduler ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
    )
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    
    print(f"\nInizio pretraining per {epochs} epoche...")
    print("-" * 60)
    
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
        
        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.get_encoder_state_dict()
            print(f"Epoca {epoch + 1:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | ✓ Best")
        else:
            epochs_no_improve += 1
            print(f"Epoca {epoch + 1:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping all'epoca {epoch + 1}")
            break
    
    print("-" * 60)
    print(f"Pretraining completato. Best val loss: {best_val_loss:.6f}")
    
    return best_state


def save_encoder_weights(encoder_state, save_dir):
    """Salva i pesi dell'encoder."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "patchtst_encoder_pretrained.pth")
    torch.save(encoder_state, save_path)
    print(f"\nEncoder salvato: {save_path}")
    return save_path


def main():
    print("\n" + "=" * 60)
    print("PATCHTST SELF-SUPERVISED PRETRAINING")
    print("=" * 60)
    
    # Setup
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Carica e splitta dati
    df_train, df_val = load_and_prepare_data()
    
    # Crea loaders
    train_loader, val_loader = create_pretraining_loaders(df_train, df_val)
    
    # Crea modello
    model = PatchTSTPretraining(PATCHTST_CONFIG)
    
    # Pretraining con validation
    encoder_state = pretrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=PRETRAIN_EPOCHS,
        lr=PRETRAIN_LR,
        patience=PATIENCE,
    )
    
    # Salva pesi
    save_dir = os.path.join(RESULTS_DIR, "pretrained")
    save_path = save_encoder_weights(encoder_state, save_dir)
    
    print("\n" + "=" * 60)
    print("PRETRAINING COMPLETATO")
    print("=" * 60)
    print(f"\nPer usare i pesi nel fine-tuning:")
    print(f'  model.load_pretrained_encoder("{save_path}")')
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()