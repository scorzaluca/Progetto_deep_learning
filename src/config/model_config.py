"""
Deep Learning Models Configuration.

Contains default parameters for each model architecture.
"""

# Shared Temporal Parameters
LOOKBACK = 48  # Lookback window size (Input X)
HORIZON = 24  # Forecast horizon size (Target Y)

# --- DATASET DERIVED PARAMETERS ---
# Calculated via compute_params.ipynb on preprocessed_ds.csv
# After dropping weather_description_other (dummy trap)
INPUT_SIZE = 24  # Number of features in the dataset (columns)
TARGET_IDX = 23  # Index of the pv_power column (target)
GHI_IDX = 10  # Index of the Ghi column (for night detection)

# --- LSTM ---
# Bidirectional processes lookback in both directions
LSTM_CONFIG = {
    "input_size": INPUT_SIZE,
    "hidden_size": 64,
    "output_size": HORIZON,
    "num_layers": 1,
    "dropout": 0.0,
    "bidirectional": False,  # True doubles hidden_size
}

# --- DLinear ---
DLINEAR_CONFIG = {
    "input_size": INPUT_SIZE,
    "target_idx": TARGET_IDX,
    "lookback": LOOKBACK,
    "horizon": HORIZON,
    "kernel_size": 25,
}

# --- PatchTST ---
# These parameters MUST match those used in pretraining for EncoderLSTM
PATCHTST_CONFIG = {
    "num_channels": INPUT_SIZE,
    "target_idx": TARGET_IDX,
    "patch_length": 16,
    "stride": 4,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 4,
    "dropout": 0.2,
    "use_cls_token": False,
}

# --- TCN ---
TCN_CONFIG = {
    "input_size": INPUT_SIZE,
    "output_size": HORIZON,
    "hidden_size": 64,
    "num_layers": 4,
    "kernel_size": 3,
    "dropout": 0.2,
}

# --- Naive ---
NAIVE_CONFIG = {
    "target_idx": TARGET_IDX,
    "horizon": HORIZON,
}

# --- EncoderLSTM ---
ENCODER_WEIGHTS_PATH = "results/pretrained/patchtst_pretrain_48_encoder.pth"
FREEZE_ENCODER = False
D_MODEL = 128
