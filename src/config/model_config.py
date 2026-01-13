"""
Configurazione dei modelli di deep learning.
Contiene i parametri di default per ogni architettura.
"""

# Parametri temporali condivisi
LOOKBACK = 24  # Quante ore guardo indietro (Input X)
HORIZON = 24  # Quante ore prevedo avanti (Target Y)

# --- PARAMETRI DERIVATI DAL DATASET ---
# Calcolati dal notebook compute_params.ipynb sul dataset preprocessed_ds.csv
# Dopo drop di weather_description_other (dummy trap)
INPUT_SIZE = 24  # Numero di feature nel dataset (colonne)
TARGET_IDX = 23  # Indice della colonna pv_power (target)

# --- LSTM ---
# bidirectional processa il lookback in entrambe le direzioni (valido per forecasting)
LSTM_CONFIG = {
    "input_size": INPUT_SIZE,
    "hidden_size": 64,
    "output_size": HORIZON,
    "num_layers": 1,
    "dropout": 0.0,
    "bidirectional": False,  # True raddoppia hidden_size, spesso non migliora molto
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
# NOTA: Questi parametri DEVONO matchare quelli usati nel pretraining per EncoderLSTM
PATCHTST_CONFIG = {
    "num_channels": INPUT_SIZE,
    "target_idx": TARGET_IDX,
    "patch_length": 12,  # Deve matchare il pretraining!
    "stride": 4,  # Deve matchare il pretraining!
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.1,
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
ENCODER_WEIGHTS_PATH = "results/pretrained/patchtst_pretrain_24_encoder.pth"
FREEZE_ENCODER = False
D_MODEL = 128
