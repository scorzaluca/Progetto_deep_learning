"""
Definizione degli spazi di ricerca per gli iperparametri di ogni modello.
Ogni funzione ritorna un dizionario con gli iperparametri suggeriti da Optuna.

NOTA: Spazi ridotti per screening veloce (<3h su GTX 1660 Ti).
Per l'ottimizzazione intensiva, estendere i range.
"""


def get_lstm_space(trial) -> dict:
    """
    Spazio di ricerca per LSTM (screening).
    Focus su hidden_size e num_layers come parametri chiave.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "grad_clip_norm": 1.0,  # Fisso per screening
    }


def get_dlinear_space(trial) -> dict:
    """
    Spazio di ricerca per DLinear (screening).
    DLinear ha pochi iperparametri, kernel_size è il più importante.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "kernel_size": trial.suggest_categorical("kernel_size", [25, 49]),
        "grad_clip_norm": 1.0,  # Fisso per screening
    }


def get_patchtst_space(trial) -> dict:
    """
    Spazio di ricerca per PatchTST (screening).
    Focus su d_model e patch_length come parametri chiave.
    n_heads=4 fisso per compatibilità con tutti i d_model.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "d_model": trial.suggest_categorical("d_model", [64, 128]),
        "n_heads": 4,  # Fisso: compatibile con d_model 64 e 128
        "n_layers": trial.suggest_int("n_layers", 2, 3),
        "patch_length": trial.suggest_categorical("patch_length", [12, 16]),
        "stride": 8,  # Fisso per screening
        "dropout": 0.2,  # Fisso per screening
        "grad_clip_norm": 1.0,  # Fisso per screening
    }


def get_tcn_space(trial) -> dict:
    """
    Spazio di ricerca per TCN (screening).
    Focus su hidden_size e num_layers.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128]),
        "num_layers": trial.suggest_int("num_layers", 3, 4),
        "kernel_size": 3,  # Fisso per screening
        "dropout": 0.2,  # Fisso per screening
        "grad_clip_norm": 1.0,  # Fisso per screening
    }


# Mapping nome modello -> funzione spazio
SPACE_REGISTRY = {
    "lstm": get_lstm_space,
    "dlinearm": get_dlinear_space,
    "dlineari": get_dlinear_space,  # DLinear-I usa stesso spazio di DLinear
    "patchtst": get_patchtst_space,
    "tcn": get_tcn_space,
}


def get_hyperparameter_space(model_name: str, trial) -> dict:
    """
    Ritorna lo spazio di ricerca per il modello specificato.

    Args:
        model_name: nome del modello (lowercase)
        trial: oggetto Optuna Trial

    Returns:
        dict con iperparametri suggeriti

    Raises:
        ValueError: se il modello non è supportato
    """
    model_name = model_name.lower()
    if model_name not in SPACE_REGISTRY:
        raise ValueError(
            f"Modello '{model_name}' non supportato. "
            f"Scegli tra: {list(SPACE_REGISTRY.keys())}"
        )
    return SPACE_REGISTRY[model_name](trial)
