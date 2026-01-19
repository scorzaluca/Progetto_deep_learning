"""
Hyperparameter Tuning Configuration.

Manually modify these parameters before each optimization run.
"""

# =============================================================================
# MODEL TO OPTIMIZE
# =============================================================================
# Options: "lstm", "dlinearm", "dlineari", "patchtst", "tcn", "encoderlstm"
MODEL_NAME = "encoderlstm"

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================
N_TRIALS = 30  # Number of Optuna trials
N_FOLDS = 3  # 1 = largest fold only (idx 2), 2 or 3 = multiple folds
EPOCHS = 50  # Maximum epochs per trial
PATIENCE = 8  # Early stopping patience

# =============================================================================
# OPTUNA STUDY MANAGEMENT
# =============================================================================
# Study Name (used for saving/resuming)
# Change this name for each new optimization run of the same model
STUDY_NAME = "encoderlstm_48_run_2"

# If True, creates a new study (deletes existing study with same name)
# If False, resumes existing study (to continue interrupted optimization)
NEW_STUDY = False

# =============================================================================
# PATHS
# =============================================================================
DATA_PATH = "data/processed/preprocessed_ds.csv"
RESULTS_DIR = "./results/"
