import pandas as pd
import torch
import os
import config
from data_loader import TS_Cross_Validator
from engine import fit_model
from PatchTST import PatchTST 
from lstm import LSTM  

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(config.RESULTS_DIR):
    os.makedirs(config.RESULTS_DIR)

# --- 1. CARICAMENTO DATASET PULITO ---
# Assumiamo che il file indicato in config sia già quello pronto/pulito o che venga caricato così.
# Non faccio nessuna modifica, drop o dummy.
if config.DATA_PATH.endswith('.csv'):
    df = pd.read_csv(config.DATA_PATH)
else:
    df = pd.read_excel(config.DATA_PATH)


# --- 2. GENERAZIONE FOLD (Usando il tuo Data Loader) ---
validator = TS_Cross_Validator(df, target_col=config.TARGET_COL, cfg_dict=config.SAMPLING_CONFIG)
folds = validator.get_folds()

models_to_train = ["PatchTST", "LSTM"]

# --- 3. TRAINING LOOP ---
for fold_idx, (train_loader, val_loader, scaler) in enumerate(folds):
    print(f"\n=== FOLD {fold_idx+1} ===")
    
    for model_name in models_to_train:
        print(f"Training {model_name}...")
        
        # --- ISTANZIAZIONE CORRETTA ---
        if model_name == "PatchTST":
            # Passiamo il loader così legge le feature da solo (Modifica fatta prima)
            model = PatchTST(
                model_config=config.PATCHTST_CONFIG, 
                train_loader=train_loader 
            )
            
        elif model_name == "LSTM":
            # Passiamo il config statico (Modifica fatta prima)
            model = LSTM(
                model_config=config.LSTM_CONFIG,
                train_loader=train_loader
            )
        
        model.to(device)
        
        # --- TRAINING (Usando il tuo Engine) ---
        trained_model, history = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.EPOCHS,
            lr=config.LEARNING_RATE,
            device=device
        )
        
        # --- SALVATAGGIO ---
        save_path = os.path.join(config.RESULTS_DIR, f"{model_name}_fold_{fold_idx+1}.pth")
        torch.save(trained_model.state_dict(), save_path)
        print(f"Salvato: {save_path}")

print("Training completato.")