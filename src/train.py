import pandas as pd
import torch
import numpy as np
import random
import os
import config
from data_loader import TS_Cross_Validator
from engine import fit_model
from PatchTST import PatchTST
from lstm import LSTM
from dlinear import DLinear

# --- SEED PER RIPRODUCIBILITÀ ---
SEED = 42
DATA_PATH = "data/processed/adjusted_ds.csv"


def set_seed(seed: int):
    """Imposta il seed per garantire riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. CARICAMENTO DATASET PULITO ---
df = pd.read_csv(DATA_PATH)

# --- 2. GENERAZIONE FOLD (Usando il Data Loader) ---
validator = TS_Cross_Validator(
    df, target_col=config.TARGET_COL, cfg_dict=config.SAMPLING_CONFIG
)
folds = validator.get_folds()

models_to_train = ["PatchTST", "LSTM", "DLinear"]

# --- 3. TRAINING LOOP ---
for fold_idx, (train_loader, val_loader, scaler) in enumerate(folds):
    print(f"\n=== FOLD {fold_idx + 1} ===")

    for model_name in models_to_train:
        print(f"Training {model_name}...")

        # --- ISTANZIAZIONE CORRETTA ---
        if model_name == "PatchTST":
            # Passiamo il loader così legge le feature da solo
            model = PatchTST(
                model_config=config.PATCHTST_CONFIG, train_loader=train_loader
            )

        elif model_name == "LSTM":
            # Passiamo il config statico
            model = LSTM(model_config=config.LSTM_CONFIG, train_loader=train_loader)

        elif model_name == "DLinear":
            # Passiamo il config e il loader per input_size dinamico
            model = DLinear(
                model_config=config.DLINEAR_CONFIG, train_loader=train_loader
            )

        model.to(device)

        # --- TRAINING (Usando il tuo Engine) ---
        trained_model, history = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.EPOCHS,
            lr=config.LEARNING_RATE,
            device=device,
            fold_idx=fold_idx,  # Aggiunto per calcolare MASE correttamente
        )

        # --- SALVATAGGIO ---
        save_path = os.path.join(
            config.RESULTS_DIR, f"{model_name}_fold_{fold_idx + 1}.pth"
        )
        torch.save(trained_model.state_dict(), save_path)
        print(f"Salvato: {save_path}")

print("Training completato.")
