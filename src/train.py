import pandas as pd
import torch
import numpy as np
import random
import os
from config import (
    TARGET_COL,
    SAMPLING_CONFIG,
    EPOCHS,
    LEARNING_RATE,
    PATCHTST_CONFIG,
    LSTM_CONFIG,
    DLINEAR_CONFIG,
)
from DataLoading import TS_Cross_Validator
from Training.engine import fit_model
from ModelClasses import PatchTST, LSTM, DLinear

SEED = 42
DATA_PATH = "data/processed/adjusted_ds.csv"
RESULTS_DIR = "./results/"


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(DATA_PATH)

validator = TS_Cross_Validator(df, target_col=TARGET_COL, cfg_dict=SAMPLING_CONFIG)
folds = validator.get_folds()

models_to_train = ["PatchTST", "LSTM", "DLinear"]

for fold_idx, (train_loader, val_loader, scaler) in enumerate(folds):
    print(f"\n=== FOLD {fold_idx + 1} ===")

    for model_name in models_to_train:
        print(f"Training {model_name}...")

        if model_name == "PatchTST":
            model = PatchTST(model_config=PATCHTST_CONFIG, train_loader=train_loader)

        elif model_name == "LSTM":
            model = LSTM(model_config=LSTM_CONFIG, train_loader=train_loader)

        elif model_name == "DLinear":
            model = DLinear(model_config=DLINEAR_CONFIG, train_loader=train_loader)

        model.to(device)

        trained_model, history = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            device=device,
            fold_idx=fold_idx,
        )

        save_path = os.path.join(RESULTS_DIR, f"{model_name}_fold_{fold_idx + 1}.pth")
        torch.save(trained_model.state_dict(), save_path)
        print(f"Salvato: {save_path}")

print("Training completato.")
