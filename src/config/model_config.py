"""
Configurazione dei modelli di deep learning.
Contiene i parametri di default per ogni architettura.
"""

# Parametri temporali condivisi
LOOKBACK = 48  # Quante ore guardo indietro (Input X)
HORIZON = 24  # Quante ore prevedo avanti (Target Y)

# --- LSTM ---
# Nota: bidirectional non è configurabile (forzato a False per forecasting)
LSTM_CONFIG = {
    "hidden_size": 64,
    "output_size": HORIZON,
    "num_layers": 1,
    "dropout": 0.0,
}

# --- DLinear ---
DLINEAR_CONFIG = {
    "lookback": LOOKBACK,
    "horizon": HORIZON,
    "kernel_size": 25,
}

# --- PatchTST ---
PATCHTST_CONFIG = {
    "patch_length": 16,
    "stride": 8,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.2,
    "use_cls_token": False,
}
