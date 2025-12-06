# -----------------------UPLOADING AND PREPROCESS ---------------------------
# DATASET PARAMETERS
DATA_PATH = "data/raw/wx_dataset.xlsx"
LABEL_PATH = "data/raw/pv_dataset.xlsx"
TARGET_COL = "pv_power"
DATE_COL = "dt_iso"
COLUMNS_TO_REMOVE = ["lat", "lon", DATE_COL]
DUMMY_COLUMN = "weather_description"
FILLNA_COLUMN = ["rain_1h"]

# --- PREPROCESS CONFIG DICT ---
PREPROCESS_CONFIG = {
    "date_col": DATE_COL,
    "columns_to_remove": COLUMNS_TO_REMOVE,
    "dummy_column": DUMMY_COLUMN,
    "fill_na_column": FILLNA_COLUMN,
}


# -----------------------------SAMPLING----------------------------------
# --- SAMPLING Parameters (I più importanti) ---
LOOKBACK = 48  # Quante ore guardo indietro (Input X)
HORIZON = 24  # Quante ore prevedo avanti (Target Y)
STEP = 1  # Scorri di 1 ora alla volta
BATCH_SIZE = 64
N_SPLITS = 3  # Numero di fold per la Cross Validation
# --- SAMPLING CONFIG DICT---
SAMPLING_CONFIG = {
    "lookback": LOOKBACK,
    "horizon": HORIZON,
    "batch_size": BATCH_SIZE,
    "step": STEP,
    "n_splits": N_SPLITS,
}


# ------------------------MODELS CONFIGURATION-----------------------------

# --- LSTM PARAMETERS ---

LSTM_CONFIG = {
    "hidden_size": 64,
    "output_size": HORIZON,
    "num_layers": 1,
    "dropout": 0.0,
    "bidirectional": False,
    "batch_first": True,
}

# --- DLINEAR PARAMETERS ---

DLINEAR_CONFIG = {
    "lookback": LOOKBACK,  # 48 ore di storia
    "horizon": HORIZON,  # 24 ore di previsione
    "kernel_size": 25,  # Dimensione della finestra per la media mobile (trend)
}


PATCHTST_CONFIG = {
    "patch_length": 16,
    "stride": 8,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "dropout": 0.2,
    "use_cls_token": False,
}


EPOCHS = 50  # Numero di epoche per il training
LEARNING_RATE = 0.001

# --- BENCHMARK VALUES ---
# Valori MAE del modello Naive calcolati sui 3 fold di validazione.
# Servono per calcolare la metrica MASE durante il training.
NAIVE_MAE_PER_FOLD = [0.061884590465089546, 0.07340711300544765, 0.06745997751536577]

# --- PERCORSI FILE ---
RESULTS_DIR = "./results/"
CHECKPOINTS_DIR = "./results/checkpoints/"

# --- OPTUNA CONFIGURATION ---
SCREENING_CONFIG = {
    "n_trials": 15,  # Trial per modello nella fase screening
    "n_folds": 1,  # Fold più grande (indice 2, il terzo)
    "patience": 5,  # Early stopping più aggressivo
    "epochs": 30,  # Meno epoche per velocità
}

INTENSIVE_CONFIG = {
    "n_trials": 50,  # Più trial per ottimizzazione fine
    "n_folds": 3,  # Tutti i fold per robustezza
    "patience": 10,  # Più pazienza per convergenza
    "epochs": 50,  # Più epoche per performance
}
