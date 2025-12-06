"""
Configurazione per il training e l'ottimizzazione.
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
EPOCHS = 50
LEARNING_RATE = 0.001

# --- BENCHMARK (per MASE) ---
NAIVE_MAE_PER_FOLD = [0.061884590465089546, 0.07340711300544765, 0.06745997751536577]

# --- PERCORSI ---
RESULTS_DIR = "./results/"

# --- OPTUNA ---
SCREENING_CONFIG = {
    "n_trials": 15,
    "n_folds": 1,  # Usa fold più grande (indice 2)
    "patience": 5,
    "epochs": 30,
}

INTENSIVE_CONFIG = {
    "n_trials": 50,
    "n_folds": 3,  # Tutti i fold
    "patience": 10,
    "epochs": 50,
}
