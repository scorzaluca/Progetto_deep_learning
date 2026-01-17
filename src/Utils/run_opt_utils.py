"""
Utility functions for run_optimization.py.

Includes helper functions for:
1. Setting random seeds for reproducibility.
2. Detecting and selecting the computing device (GPU/CPU).
3. Loading data and creating Cross-Validation folds.
4. Saving optimization results.
"""

import os
import json
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across all libraries.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If standard CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Automatically detects and returns the available computing device (CUDA GPU or CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU detected: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    return device


def load_data_and_folds(data_path: str = None):
    """
    Loads the dataset and generates cross-validation folds.

    Args:
        data_path (str, optional): Path to the CSV dataset. If None, uses DATA_PATH from config.

    Returns:
        tuple: (DataFrame, list of folds) - Where folds are tuples of (train_idx, val_idx).
    """
    
    from ..config import TARGET_COL, SAMPLING_CONFIG, DATA_PATH
    from ..DataLoading import TS_Cross_Validator

    # Use provided path or default from config
    path = data_path if data_path is not None else DATA_PATH

    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")

    # Initialize Validator and Generate Folds
    validator = TS_Cross_Validator(df, target_col=TARGET_COL, cfg_dict=SAMPLING_CONFIG)
    folds = list(validator.get_folds())
    print(f"Available folds: {len(folds)}")

    return df, folds


def save_results(
    model_name: str,
    study_name: str,
    best_params: dict,
    best_mase: float,
    best_rmse: float,
):
    """
    Saves the best optimization results (parameters and metrics) to a JSON file.

    Args:
        model_name (str): Name of the model.
        study_name (str): Name of the Optuna study.
        best_params (dict): Dictionary of best hyperparameters found.
        best_mase (float): Best MASE achieved.
        best_rmse (float): Best RMSE achieved.

    Returns:
        str: Path to the saved JSON file.
    """
    
    from ..config import RESULTS_DIR

    params_dir = os.path.join(RESULTS_DIR, "params")
    os.makedirs(params_dir, exist_ok=True)

    results = {
        "model_name": model_name,
        "study_name": study_name,
        "best_mase": best_mase,
        "best_rmse": best_rmse,
        "best_params": best_params,
    }

    filepath = os.path.join(params_dir, f"{model_name}_{study_name}_params.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Best params saved: {filepath}")
    return filepath
