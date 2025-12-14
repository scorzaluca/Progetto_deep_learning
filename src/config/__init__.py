"""
Export di tutte le configurazioni dal package config.
"""

from .model_config import (
    LOOKBACK,
    HORIZON,
    INPUT_SIZE,
    TARGET_IDX,
    LSTM_CONFIG,
    DLINEAR_CONFIG,
    PATCHTST_CONFIG,
    TCN_CONFIG,
    NAIVE_CONFIG,
)

from .training_config import (
    TARGET_COL,
    SAMPLING_CONFIG,
    SEED,
    EPOCHS,
    LEARNING_RATE,
    NAIVE_MAE_PER_FOLD,
    NAIVE_MAE_FINAL_FOLD,
)

from .tuning_config import (
    MODEL_NAME,
    N_TRIALS,
    N_FOLDS,
    EPOCHS as TUNING_EPOCHS,
    PATIENCE,
    STUDY_NAME,
    NEW_STUDY,
    DATA_PATH,
    RESULTS_DIR,
)

__all__ = [
    # Model config
    "LOOKBACK",
    "HORIZON",
    "INPUT_SIZE",
    "TARGET_IDX",
    "LSTM_CONFIG",
    "DLINEAR_CONFIG",
    "PATCHTST_CONFIG",
    "TCN_CONFIG",
    "NAIVE_CONFIG",
    # Training config
    "TARGET_COL",
    "SAMPLING_CONFIG",
    "SEED",
    "EPOCHS",
    "LEARNING_RATE",
    "NAIVE_MAE_PER_FOLD",
    "NAIVE_MAE_FINAL_FOLD",
    # Tuning config
    "MODEL_NAME",
    "N_TRIALS",
    "N_FOLDS",
    "TUNING_EPOCHS",
    "PATIENCE",
    "STUDY_NAME",
    "NEW_STUDY",
    "DATA_PATH",
    "RESULTS_DIR",
]
