"""
Funzioni di utilità per run_optimization.py.
"""

import os
import json
import random
import numpy as np
import torch
import pandas as pd


def set_seed(seed: int):
    """Imposta il seed per riproducibilita."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Rileva automaticamente GPU (CUDA) o fallback a CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU rilevata: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("GPU non disponibile, usando CPU")
    return device


def load_data_and_folds(data_path: str = None):
    """
    Carica il dataset e crea i fold per la cross-validation.

    Args:
        data_path: Path al dataset CSV. Se None, usa DATA_PATH da config.

    Returns:
        Tuple[pd.DataFrame, list]: (dataframe, lista di folds)
    """
    # Import lazy per evitare import circolare
    from ..config import TARGET_COL, SAMPLING_CONFIG, DATA_PATH
    from ..DataLoading import TS_Cross_Validator

    # Usa path passato o default da config
    path = data_path if data_path is not None else DATA_PATH

    print(f"Caricamento dataset: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Shape: {df.shape}")

    validator = TS_Cross_Validator(df, target_col=TARGET_COL, cfg_dict=SAMPLING_CONFIG)
    folds = list(validator.get_folds())
    print(f"Fold disponibili: {len(folds)}")

    return df, folds


def save_results(model_name: str, study_name: str, best_params: dict, best_mase: float):
    """Salva i best params in un file JSON."""
    # Import lazy per evitare import circolare
    from ..config import RESULTS_DIR

    params_dir = os.path.join(RESULTS_DIR, "params")
    os.makedirs(params_dir, exist_ok=True)

    results = {
        "model_name": model_name,
        "study_name": study_name,
        "best_mase": best_mase,
        "best_params": best_params,
    }

    filepath = os.path.join(params_dir, f"{model_name}_{study_name}_params.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Best params salvati: {filepath}")
    return filepath
