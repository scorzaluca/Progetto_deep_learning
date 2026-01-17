"""
Training Configuration.

Contains parameters for data sampling, training loop, and benchmark metrics.
"""

from .model_config import LOOKBACK, HORIZON


# --- SAMPLING ---
STEP_TRAIN = 1  # Step between samples for training (use all samples)
STEP_VAL = 1  # Step between samples for validation
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


# --- BENCHMARK (for MASE) ---
# Naive models were run with lookback=48 and STEP_TRAIN=STEP_VAL=1
NAIVE_MAE_PER_FOLD = [0.06228089907571026, 0.08621430409181377, 0.06193031994221003]

# Naive Model MAE on the final training fold (22 months train + 2 months val)
NAIVE_MAE_FINAL_FOLD = 0.04919686427582865

# Naive Model MAE on the Test Set (calculated via naive_run.ipynb)
NAIVE_MAE_TEST = 0.062445086238714045
