"""
Configurazione per il training.
"""

from .model_config import LOOKBACK, HORIZON

# --- SAMPLING ---
STEP = 1
BATCH_SIZE = 64
N_SPLITS = 3
TARGET_COL = "pv_power"

SAMPLING_CONFIG = {
    "lookback": LOOKBACK,
    "horizon": HORIZON,
    "batch_size": BATCH_SIZE,
    "step": STEP,
    "n_splits": N_SPLITS,
}

# --- TRAINING ---
SEED = 42
EPOCHS = 50
LEARNING_RATE = 0.001

# --- BENCHMARK (per MASE) ---
NAIVE_MAE_PER_FOLD = [0.06171385527493945, 0.07321843994187488, 0.06725753733105418]
