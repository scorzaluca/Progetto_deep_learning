"""
Hyperparameter Search Spaces Definition.

This module defines the search spaces for Optuna hyperparameter optimization.
Each function corresponds to a specific model and returns a dictionary of
suggested hyperparameters (sampled from defined ranges or distributions).

Needs to adjust these ranges manually before running extensive optimization.
For quick screening, use narrower ranges. For deep tuning, expand them.
"""


def get_lstm_space(trial) -> dict:
    """
    Search space for LSTM model.

    Key Parameters:
    - lr: Learning Rate (log-scale).
    - hidden_size: Number of features in the hidden state.
    - num_layers: Number of recurrent layers.
    - dropout: Dropout probability (ignored if num_layers=1).
    - grad_clip_norm: Max norm for gradient clipping.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.8, 1.7),
    }


def get_dlinear_space(trial) -> dict:
    """
    Search space for DLinear model.

    Key Parameters:
    - kernel_size: Moving average kernel size for trend/seasonal decomposition.
    - grad_clip_norm: Max norm for gradient clipping.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "kernel_size": trial.suggest_categorical("kernel_size", [13, 25, 37, 49]),
        "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.5, 2.0),
    }


def get_patchtst_space(trial) -> dict:
    """
    Search space for PatchTST model (Training from Scratch).

    Key Parameters:
    - d_model: Embedding dimension (total). Must be divisible by n_heads.
    - n_heads: Number of attention heads.
    - patch_length: Size of each patch (sub-series).
    - stride: Stride between patches.
    - dropout: Dropout probability.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "n_layers": trial.suggest_int("n_layers", 2, 4),
        "patch_length": trial.suggest_categorical("patch_length", [8, 12, 16, 24]),
        "stride": trial.suggest_categorical("stride", [4, 6, 8, 12]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.5, 2.0),
        "pretrain_path": "results/pretrained/patchtst_encoder_pretrained.pth",
    }


def get_tcn_space(trial) -> dict:
    """
    Search space for Temporal Convolutional Network (TCN).

    Key Parameters:
    - hidden_size: Number of channels in temporal blocks.
    - num_layers: Number of dilated blocks (dilation grows exponentially: 1, 2, 4...).
    - kernel_size: Convolutional kernel size.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 4, 5]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.5, 2.0),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
    }


def get_patchtst_finetune_space(trial) -> dict:
    """
    Search space for PatchTST FINE-TUNING (using a pretrained encoder).

    The architectural parameters (d_model, n_heads, n_layers, patch_length, stride)
    MUST MATCH exactly the ones used during Pretraining.
    We only optimize Learning Rate here.
    """
    return {
        "lr": trial.suggest_float(
            "lr", 1e-5, 1e-3, log=True
        ),  # Lower LR for fine-tuning to preserve pretrained knowledge
        
        # FIXED ARGUMENTS - Must match pretraining config
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 3,
        "patch_length": 16,
        "stride": 8,
        "pretrain_path": "results/pretrained/patchtst_encoder_pretrained.pth",
    }


def get_encoderlstm_space(trial) -> dict:
    """
    Search space for EncoderLSTM.

    The Pretrained Encoder parameters are FIXED (imported from config).
    We optimize only the LSTM Head parameters:
    - projection_dim: Bottleneck dimension between Encoder and LSTM.
    - lstm_hidden: Memory size of the LSTM.
    - lstm_layers: Depth of the LSTM.
    """
    from ..config import ENCODER_WEIGHTS_PATH, D_MODEL, FREEZE_ENCODER

    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "projection_dim": trial.suggest_categorical("projection_dim", [64, 128, 256]),
        "lstm_hidden": trial.suggest_categorical("lstm_hidden", [32, 64, 128]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.5, 2.0),
        "d_model": D_MODEL,
        "pretrain_path": ENCODER_WEIGHTS_PATH,
        "freeze_encoder": FREEZE_ENCODER
    }


# =============================================================================
# REGISTRY - Mapping model name -> space function
# =============================================================================
SPACE_REGISTRY = {
    "lstm": get_lstm_space,
    "dlinearm": get_dlinear_space,
    "dlineari": get_dlinear_space,
    "patchtst": get_patchtst_space,
    "tcn": get_tcn_space,
    "encoderlstm": get_encoderlstm_space,
    "patchtst_finetune": get_patchtst_finetune_space,
}


def get_hyperparameter_space(model_name: str, trial) -> dict:
    """
    Retrieves the correct hyperparameter search space for the specified model.

    Args:
        model_name (str): Name of the model (case-insensitive).
        trial: Optuna Trial object used to sample hyperparameters.

    Returns:
        dict: A dictionary of sampled hyperparameters ready for model creation.

    Raises:
        ValueError: If `model_name` is not found in the registry.
    """
    model_name = model_name.lower()
    
    # Check if the model is supported
    if model_name not in SPACE_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Choose from: {list(SPACE_REGISTRY.keys())}"
        )
    
    # Execute the corresponding space function
    return SPACE_REGISTRY[model_name](trial)
