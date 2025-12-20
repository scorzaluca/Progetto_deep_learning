"""
Configurazione per l'ottimizzazione degli iperparametri.
Modifica questi parametri manualmente prima di ogni run di ottimizzazione.
"""

# =============================================================================
# MODELLO DA OTTIMIZZARE
# =============================================================================
# Opzioni: "lstm", "dlinearm", "dlineari", "patchtst", "tcn", "patchtst_finetune"
MODEL_NAME = "lstm"

# =============================================================================
# PARAMETRI OTTIMIZZAZIONE
# =============================================================================
N_TRIALS = 3  # Numero di trial Optuna
N_FOLDS = 1  # 1 = solo fold più grande (idx 2), 2 o 3 = più fold
EPOCHS = 30  # Epoche massime per trial
PATIENCE = 7  # Early stopping patience

# =============================================================================
# GESTIONE STUDIO OPTUNA
# =============================================================================
# Nome dello studio (usato per salvare/riprendere)
# Cambia questo nome per ogni nuova ottimizzazione dello stesso modello
STUDY_NAME = "dlineari_prova3"

# Se True, crea un nuovo studio (cancella eventuale studio esistente con stesso nome)
# Se False, riprende lo studio esistente (per continuare ottimizzazione interrotta)
NEW_STUDY = True

# =============================================================================
# PERCORSI
# =============================================================================
DATA_PATH = "data/processed/preprocessed_ds.csv"
RESULTS_DIR = "./results/"
