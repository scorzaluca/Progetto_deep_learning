"""
Configurazione per il training.
"""

from .model_config import LOOKBACK, HORIZON

# --- SAMPLING ---
STEP_TRAIN = 1  # Passo tra campioni per training (tutti i campioni)
STEP_VAL = 6  # Passo tra campioni per validation (campioni meno correlati)
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
EPOCHS = 50
LEARNING_RATE = 0.001

# --- BENCHMARK (per MASE) ---
NAIVE_MAE_PER_FOLD = [0.06228089907571026, 0.08621430409181377, 0.06193031994221003]

# MAE del modello naive sul final training fold (22 mesi train + 2 mesi val)
NAIVE_MAE_FINAL_FOLD = 0.04919686427582865
