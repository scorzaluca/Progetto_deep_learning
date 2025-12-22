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


# ===== CONFIGURAZIONE PRETRAINING =====
DATA_PATH = "data/processed/preprocessed_ds.csv"
PRETRAIN_EPOCHS = 50
PRETRAIN_LR = 1e-4
BATCH_SIZE = 64
PATIENCE = 10  # Early stopping


def load_and_prepare_data():
    """Carica e normalizza i dati per il pretraining."""
    print("Caricamento dati...")
    df = pd.read_csv(DATA_PATH)
    
    # Normalizza tutto il dataset
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    print(f"Dataset shape: {df_scaled.shape}")
    return df_scaled


def create_pretraining_loader(df_scaled):
    """Crea il DataLoader per pretraining."""
    dataset = PretrainingDataset(
        df=df_scaled,
        lookback=LOOKBACK,
        step=1,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    
    print(f"Samples totali: {len(dataset)}")
    print(f"Batches per epoca: {len(loader)}")
    return loader


def pretrain(model, loader, device, epochs, lr, patience):
    """
    Loop di pretraining con early stopping.
    
    Il modello PatchTSTForPretraining calcola automaticamente
    la loss di ricostruzione nel forward pass.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    best_state = None
    
    print(f"\nInizio pretraining per {epochs} epoche...")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_x in loader:
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            
            # Forward: il modello restituisce direttamente la loss
            loss = model(batch_x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_state = model.get_encoder_state_dict()
            print(f"Epoca {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.6f} | ✓ Best")
        else:
            epochs_no_improve += 1
            print(f"Epoca {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.6f}")
            
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping all'epoca {epoch + 1}")
                break
    
    print("-" * 50)
    print(f"Pretraining completato. Best loss: {best_loss:.6f}")
    
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
    
    # Carica dati
    df_scaled = load_and_prepare_data()
    
    # Crea loader
    loader = create_pretraining_loader(df_scaled)
    
    # Crea modello
    model = PatchTSTPretraining(PATCHTST_CONFIG)
    
    # Pretraining
    encoder_state = pretrain(
        model=model,
        loader=loader,
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