"""
Hyperparameter Optimization Configuration for PatchTST Pretraining.

Manually modify these parameters before each pretraining run.
"""

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================
N_TRIALS = 30  # Number of Optuna trials
EPOCHS = 50  # Maximum epochs per trial
PATIENCE = 8  # Early stopping patience

# =============================================================================
# OPTUNA STUDY MANAGEMENT
# =============================================================================
STUDY_NAME = "patchtst_pretrain_48"
NEW_STUDY = True  # True = New Study, False = Resume Existing Study

# =============================================================================
# DATA SPLIT
# =============================================================================
VAL_SPLIT = 0.2  # 80% Training, 20% Validation (Temporal, No Shuffle)

# =============================================================================
# PATHS
# =============================================================================
DATA_PATH = "data/processed/preprocessed_ds.csv"
RESULTS_DIR = "./results/"
