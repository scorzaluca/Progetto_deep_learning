"""
Definizione degli spazi di ricerca per gli iperparametri di ogni modello.
Ogni funzione ritorna un dizionario con gli iperparametri suggeriti da Optuna.
"""


def get_lstm_space(trial) -> dict:
    """
    Spazio di ricerca per LSTM.

    Args:
        trial: oggetto Optuna Trial

    Returns:
        dict con iperparametri suggeriti
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }


def get_dlinear_space(trial) -> dict:
    """
    Spazio di ricerca per DLinear.
    DLinear ha pochi iperparametri da ottimizzare.

    Args:
        trial: oggetto Optuna Trial

    Returns:
        dict con iperparametri suggeriti
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "kernel_size": trial.suggest_categorical("kernel_size", [13, 25, 37, 49]),
    }


def get_patchtst_space(trial) -> dict:
    """
    Spazio di ricerca per PatchTST.

    Args:
        trial: oggetto Optuna Trial

    Returns:
        dict con iperparametri suggeriti
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "patch_length": trial.suggest_categorical("patch_length", [8, 12, 16, 24]),
        "stride": trial.suggest_categorical("stride", [4, 6, 8, 12]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
    }


# Mapping nome modello -> funzione spazio
SPACE_REGISTRY = {
    "lstm": get_lstm_space,
    "dlinearm": get_dlinear_space,
    "dlineari": get_dlinear_space,  # DLinear-I usa stesso spazio di DLinear
    "patchtst": get_patchtst_space,
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
