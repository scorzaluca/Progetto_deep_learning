DATA_PATH = 'data/raw/wx_dataset.xlsx'
LABEL_PATH='data/raw/pv_dataset.xlsx'


# --- PARAMETRI STRUTTURALI DEL DATASET ---
TARGET_COL = 'pv_power'
# Definisci qui le colonne che vuoi escludere o includere se serve
#EXCLUDE_COLS = ['dt_iso', 'city_name'] 



DATE_COL = 'dt_iso'
COLUMNS_TO_REMOVE = ['lat', 'lon', DATE_COL]
DUMMY_COLUMN='weather_description'
FILLNA_COLUMN=['rain_1h']
GHI_COLUMN = ['Ghi']


# --- PREPROCESS CONFIG DICT ---
PREPROCESS_CONFIG = {
    'date_col': DATE_COL,
    'columns_to_remove': COLUMNS_TO_REMOVE,
    'dummy_column': DUMMY_COLUMN,
    'fill_na_column': FILLNA_COLUMN,
    'Ghi': GHI_COLUMN,
    'pv_power' : TARGET_COL,

}



# --- SAMPLING CONFIG (I più importanti) ---
LOOKBACK = 48       # Quante ore guardo indietro (Input X)
HORIZON = 24        # Quante ore prevedo avanti (Target Y)
STEP = 1            # Scorri di 1 ora alla volta
BATCH_SIZE = 64
N_SPLITS = 3        # Numero di fold per la Cross Validation


# --- SAMPLING CONFIG DICT---
# Questo è comodissimo da passare alla tua classe TimeSeriesCrossValidator
# così non devi passare 10 argomenti separati.
SAMPLING_CONFIG = {
    'lookback': LOOKBACK,
    'horizon': HORIZON,
    'batch_size': BATCH_SIZE,
    'step': STEP,
    'n_splits': N_SPLITS
}

EPOCHS = 50         # Numero di epoche per il training
LEARNING_RATE = 0.001

# --- PERCORSI FILE ---

RESULTS_DIR = './results/'