import pandas as pd
import torch
import os
import config
from Data_loading import TS_Cross_Validator
from Training import fit_model
from Training import evaluate_model
from Models import LSTM, PatchTST


# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''if not os.path.exists(config.RESULTS_DIR):
    os.makedirs(config.RESULTS_DIR)'''

# --- 1. CARICAMENTO DATASET PULITO ---
# Assumiamo che il file indicato in config sia già quello pronto/pulito o che venga caricato così.
# Non faccio nessuna modifica, drop o dummy.
if config.ADJUSTED_DF.endswith('.csv'):
    df = pd.read_csv(config.ADJUSTED_DF)
else:
    df = pd.read_excel(config.ADJUSTED_DF)


target_idx = df.columns.get_loc(config.TARGET_COL)


# --- 2. GENERAZIONE FOLD (Usando il tuo Data Loader) ---
validator = TS_Cross_Validator(df, target_col=config.TARGET_COL, cfg_dict=config.SAMPLING_CONFIG)
validator.visualize_splits()
for fold_idx, (train_loader, val_loader, scaler) in enumerate(validator.get_folds()):
    
    print(f"\n{'='*40}")
    print(f"🚀 TRAINING FOLD {fold_idx + 1}/{config.N_SPLITS}")
    print(f"{'='*40}")

    # --- A. INIZIALIZZA IL MODELLO ---
    # Usiamo config.LSTM_CONFIG.
    # Passiamo train_loader perché la tua classe LSTM calcola self.input_size dinamicamente da lì.
    model = LSTM(config.LSTM_CONFIG, train_loader).to(device)
    
    # --- B. AVVIA IL TRAINING (fit_model) ---
    # fit_model gestisce training loop, validation loop, early stopping e MASE.
    # Passiamo fold_idx così pesca il NAIVE_MAE corretto dal config.
    trained_model, history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        device=device,
        fold_idx=fold_idx, 
        patience=10  # Puoi parametrizzarlo nel config se vuoi
    )

    # --- C. VALUTAZIONE FINALE (evaluate_model) ---
    print(f"\nValutazione finale Fold {fold_idx + 1}...")
    
    # Questa funzione fa predizioni, denormalizza (grazie a scaler e target_idx)
    # e restituisce i valori in Watt reali.
    preds, targets = evaluate_model(
        model=trained_model,
        val_loader=val_loader,
        device=device,
        scaler=scaler,
        target_idx=target_idx
    )
    
    # --- D. SALVATAGGIO DEL MODELLO ---
    # Salviamo i pesi del modello migliore di questo fold
    save_path = os.path.join(config.RESULTS_DIR, f"lstm_fold_{fold_idx}.pth")
    torch.save(trained_model.state_dict(), save_path)
    print(f"Modello salvato in: {save_path}")

print("\n--- TUTTI I FOLD COMPLETATI ---")

'''models_to_train = ["PatchTST", "LSTM"]

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

print("Training completato.")'''