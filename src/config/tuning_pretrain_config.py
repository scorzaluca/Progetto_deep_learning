"""
Configurazione per l'ottimizzazione degli iperparametri del pretraining PatchTST.
Modifica questi parametri manualmente prima di ogni run.
"""

# =============================================================================
# PARAMETRI OTTIMIZZAZIONE
# =============================================================================
N_TRIALS = 30  # Numero di trial Optuna
EPOCHS = 50  # Epoche massime per trial
PATIENCE = 8  # Early stopping patience

# =============================================================================
# GESTIONE STUDIO OPTUNA
# =============================================================================
STUDY_NAME = "patchtst_pretrain_48"
NEW_STUDY = True  # True = nuovo studio, False = riprendi esistente

# =============================================================================
# SPLIT DATI
# =============================================================================
VAL_SPLIT = 0.2  # 80% train, 20% validation (temporale, no shuffle)

# =============================================================================
# PERCORSI
# =============================================================================
DATA_PATH = "data/processed/preprocessed_ds.csv"
RESULTS_DIR = "./results/"
