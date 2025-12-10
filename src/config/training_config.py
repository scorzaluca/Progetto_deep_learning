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
SEED = 42
EPOCHS = 50
LEARNING_RATE = 0.001

# --- BENCHMARK (per MASE) ---
NAIVE_MAE_PER_FOLD = [0.06171385527493945, 0.07321843994187488, 0.06725753733105418]

# --- PERCORSI ---
RESULTS_DIR = "./results/"

# --- OPTUNA ---
# SCREENING: 8 trials × 5 modelli × ~20 epoche = ~2.5h su GTX 1660 Ti
SCREENING_CONFIG = {
    "n_trials": 8,
    "n_folds": 1,  # Usa fold più grande (indice 2)
    "patience": 5,
    "epochs": 20,
}

INTENSIVE_CONFIG = {
    "n_trials": 50,
    "n_folds": 3,  # Tutti i fold
    "patience": 10,
    "epochs": 50,
}
