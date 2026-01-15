"""
Configurazione per il training.
"""

from .model_config import LOOKBACK, HORIZON

# --- SAMPLING ---
STEP_TRAIN = 1  # Passo tra campioni per training (tutti i campioni)
STEP_VAL = 1  # Passo tra campioni per validation (campioni meno correlati)
BATCH_SIZE = 64
N_SPLITS = 3
TARGET_COL = "pv_power"

SAMPLING_CONFIG = {
    "lookback": LOOKBACK,
    "horizon": HORIZON,
    "batch_size": BATCH_SIZE,
    "step_train": STEP_TRAIN,
    "step_val": STEP_VAL,
    "n_splits": N_SPLITS,
}

# --- TRAINING ---
SEED = 42
EPOCHS = 100
LEARNING_RATE = 0.001

# --- BENCHMARK (per MASE) ---
# NOTA: I validation set sono identici per expanding e sliding windows,
# quindi il MAE naive è lo stesso per entrambe le strategie.
NAIVE_MAE_PER_FOLD = [0.06228089907571026, 0.08621430409181377, 0.06193031994221003]

# MAE del modello naive sul final training fold (22 mesi train + 2 mesi val)
NAIVE_MAE_FINAL_FOLD = 0.04919686427582865

# MAE del modello naive sul test set (da calcolare con naive_run.ipynb)
# Questo valore verrà aggiornato dopo aver ricevuto i dati di test
NAIVE_MAE_TEST = 0.06225388054566009  # TODO: Calcolare con naive_run.ipynb sul test set
