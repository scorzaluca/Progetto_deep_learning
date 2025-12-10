"""
Script per l'ottimizzazione degli iperparametri con Optuna.
"""

import random
import numpy as np
import torch
import pandas as pd
from .config import (
    TARGET_COL,
    SAMPLING_CONFIG,
    SCREENING_CONFIG,
    INTENSIVE_CONFIG,
    SEED,
)
from .DataLoading import TS_Cross_Validator
from .Tuning import run_screening, run_intensive

# Modalita: "screening" o "intensive"
MODE = "screening"

# Modello per intensive (usato solo se MODE = "intensive")
MODEL_INTENSIVE = "lstm"

DATA_PATH = "data/processed/preprocessed_ds.csv"
RESULTS_DIR = "/results/"


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


def load_data_and_folds():
    """Carica il dataset e crea i fold per la cross-validation."""
    print(f"Caricamento dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")

    validator = TS_Cross_Validator(df, target_col=TARGET_COL, cfg_dict=SAMPLING_CONFIG)
    folds = list(validator.get_folds())  # Converti generatore in lista
    print(f"Fold creati: {len(folds)}")

    return folds


def main():
    """Funzione principale."""
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Modalita: {MODE.upper()}")
    if MODE == "intensive":
        print(f"Modello: {MODEL_INTENSIVE.upper()}")
    print("=" * 60 + "\n")

    set_seed(SEED)
    device = get_device()
    folds = load_data_and_folds()

    if MODE == "screening":
        results = run_screening(
            folds=folds,
            device=device,
            config=SCREENING_CONFIG,
            results_dir=RESULTS_DIR,
        )

        print("\n" + "=" * 60)
        print("RIEPILOGO SCREENING")
        print("=" * 60)
        for model in ["lstm", "dlineari", "dlinearm", "patchtst", "tcn"]:
            if model in results:
                mase = results[model].get("best_mase", "N/A")
                if isinstance(mase, float):
                    print(f"{model.upper()}: MASE = {mase:.4f}")
                else:
                    print(f"{model.upper()}: {mase}")

        ranking = results.get("ranking", [])
        if ranking:
            print(f"\nRanking: {' > '.join(ranking)}")

        print(f"\nRisultati salvati in: {RESULTS_DIR}screening_results.json")
        print("\nProssimo passo:")
        print(f'Modifica MODEL_INTENSIVE = "{ranking[0] if ranking else "lstm"}"')
        print('Modifica MODE = "intensive"')
        print("Riesegui lo script")

    elif MODE == "intensive":
        results = run_intensive(
            model_name=MODEL_INTENSIVE,
            folds=folds,
            device=device,
            config=INTENSIVE_CONFIG,
            results_dir=RESULTS_DIR,
        )

        print("\n" + "=" * 60)
        print(f"RIEPILOGO INTENSIVE - {MODEL_INTENSIVE.upper()}")
        print("=" * 60)
        print(f"Best MASE: {results['best_mase']:.4f}")
        print(f"Best params: {results['best_params']}")
        print(f"Checkpoint: {results['checkpoint_path']}")

    else:
        print(f"Modalita '{MODE}' non valida. Usa 'screening' o 'intensive'.")


if __name__ == "__main__":
    main()
