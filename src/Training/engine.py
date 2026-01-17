"""
Training Engine Module.

This module provides the core functions for the training pipeline:
- `create_model`: Factory function to instantiate models based on configuration.
- `train_one_epoch`: Handles the training loop for a single epoch.
- `validate_one_epoch`: Handles validation and metric calculation.
- `fit_model`: Orchestrates the full training process with Early Stopping and Optuna integration.
"""

import copy
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from ..config import NAIVE_MAE_PER_FOLD, INPUT_SIZE, TARGET_IDX, LOOKBACK, HORIZON


def create_model(model_name: str, params: dict) -> nn.Module:
    """
    Factory function to instantiate PyTorch models based on name and configuration.

    Args:
        model_name (str): Name of the model (case-insensitive).
                          Supported: "lstm", "dlinearm", "dlineari", "patchtst", "tcn", "encoderlstm".
        params (dict): Dictionary containing model hyperparameters.
            - Common parameters are pulled from global config (INPUT_SIZE, etc.).
            - Specific parameters come from this dict.

    Returns:
        nn.Module: The instantiated model class (not moved to device yet).

    Raises:
        ValueError: If `model_name` is not supported.
    """
    model_name = model_name.lower()

    if model_name == "lstm":
        from ..ModelClasses import LSTM

        model_config = {
            "input_size": INPUT_SIZE,
            "hidden_size": params["hidden_size"],
            "output_size": HORIZON,
            "num_layers": params["num_layers"],
            "dropout": params["dropout"],
            "bidirectional": False,
        }
        return LSTM(model_config=model_config)

    elif model_name == "dlinearm":
        from ..ModelClasses import DLinearM

        model_config = {
            "input_size": INPUT_SIZE,
            "target_idx": TARGET_IDX, # Required to extract pv_power from the multivariate output
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "kernel_size": params["kernel_size"],
        }
        return DLinearM(model_config=model_config)

    elif model_name == "dlineari":
        from ..ModelClasses import DLinearI

        model_config = {
            "target_idx": TARGET_IDX,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
            "kernel_size": params["kernel_size"],
        }
        return DLinearI(model_config=model_config)

    elif model_name == "patchtst" or model_name == "patchtst_finetune":
        from ..ModelClasses import PatchTST

        model_config = {
            "num_channels": INPUT_SIZE,
            "target_idx": TARGET_IDX,
            "lookback": LOOKBACK,  
            "horizon": HORIZON,  
            "patch_length": params["patch_length"],
            "stride": params["stride"],
            "d_model": params["d_model"],
            "n_heads": params["n_heads"],
            "n_layers": params["n_layers"],
            "dropout": params["dropout"],
            "use_cls_token": False,
        }
        model = PatchTST(model_config=model_config)

        # Load pretrained weights if a path is provided in params
        # This is crucial for the PatchTSTFine-Tuning phase
        pretrain_path = params.get("pretrain_path", None)
        if pretrain_path:
            model.load_pretrained_encoder(pretrain_path)

        return model

    elif model_name == "tcn":
        from ..ModelClasses import TCN

        model_config = {
            "input_size": INPUT_SIZE,
            "output_size": HORIZON,
            "hidden_size": params["hidden_size"],
            "num_layers": params["num_layers"],
            "kernel_size": params["kernel_size"],
            "dropout": params["dropout"],
        }
        return TCN(model_config=model_config)

    elif model_name == "encoderlstm":
        from ..ModelClasses import EncoderLSTM

        model_config = {
            # Path to the pretrained encoder weights (PatchTST encoder)
            "pretrain_path": params.get(
                "pretrain_path", "results/pretrained/patchtst_pretrain_48_encoder.pth"
            ),
            "d_model": params.get("d_model", 128),
            "projection_dim": params.get("projection_dim", 64),
            "lstm_hidden": params.get("lstm_hidden", 64),
            "lstm_layers": params.get("lstm_layers", 1),
            "dropout": params.get("dropout", 0.2),
            "freeze_encoder": params.get("freeze_encoder", False),
        }
        return EncoderLSTM(model_config=model_config)

    else:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Valid models: lstm, dlinearm, dlineari, patchtst, tcn, encoderlstm"
        )


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    loss_fn,
    device,
    grad_clip_norm: float = None,
    scaler: GradScaler = None,
):
    """
    Executes one training epoch (Forward + Backward Pass).

    Args:
        model: PyTorch model to train.
        dataloader: DataLoader for the training set.
        optimizer: PyTorch optimizer.
        loss_fn: Loss function (e.g., MSELoss).
        device: Device to run on (cuda/cpu).
        grad_clip_norm: Max norm for gradient clipping (optional).
        scaler: GradScaler for AMP (optional).

    Returns:
        float: Average loss for the epoch.
    """
    # Set the model to training mode
    # This enables layers like Dropout and BatchNorm that behave differently during training.
    model.train()
    
    running_loss = 0.0
    
    # Check if we are using Automatic Mixed Precision (AMP)
    use_amp = scaler is not None

    # Iterate over the training DataLoader
    for batch_x, batch_y in dataloader:
        # Move inputs and targets to the configured device (CPU or GPU)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Reset Gradients
        # We must clear previous gradients before calculating new ones.
        # set_to_none=True is more efficient than setting to 0 (saves memory operations).
        optimizer.zero_grad(set_to_none=True)

        # Forward Pass (Managed by AMP if enabled)
        # 'autocast' automatically chooses between float16 and float32 precision
        # for different operations to maximize speed without losing accuracy.
        with autocast(device_type="cuda", enabled=use_amp):
            prediction = model(batch_x)          # Generate predictions
            loss = loss_fn(prediction, batch_y)  # Calculate the loss (error)

        # Backward Pass and Optimization
        if use_amp:
            # --- AMP Flow ---
            # We scale the loss (multiply it) to prevent "underflow" (values becoming 0)
            # which can happen with small gradients in float16.
            scaler.scale(loss).backward()
            
            if grad_clip_norm is not None:
                # Before clipping, we must "unscale" the gradients back to their original magnitude
                scaler.unscale_(optimizer)
                
                # Apply Gradient Clipping:
                # Caps the gradient magnitude to prevent "Exploding Gradients"
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
            
            # Update the model parameters (weights)
            scaler.step(optimizer)
            
            # Update the scaler for the next iteration (adjusts the scaling factor)
            scaler.update()
        else:
            # Standard Flow (Full Precision)
            loss.backward()  # Calculate gradients via backpropagation
            
            if grad_clip_norm is not None:
                # Apply Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip_norm
                )
            
            # Update the model parameters
            optimizer.step()

        # Accumulate the total loss for the epoch
        running_loss += loss.item()

    # Return the average loss over all batches
    return running_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, device):
    """
    Executes one validation epoch (Forward Pass Only).
    Calculates key metrics: MSE (Loss), MAE and RMSE.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader for the validation set.
        loss_fn: Loss function .
        device: Device to run on (cuda/cpu).

    Returns:
        tuple: (avg_loss, avg_mae, avg_rmse)
           - avg_loss: Mean Squared Error (MSE), used for loss tracking.
           - avg_mae: Mean Absolute Error (MAE), used to calculate MASE later.
           - avg_rmse: Root Mean Squared Error (RMSE), for human-readable error.
    """
    # Set model to Evaluation Mode
    # This disables Dropout and switches Batch Normalization to use
    # running stats instead of batch stats.
    model.eval()
    
    running_loss = 0.0
    running_mae = 0.0

    # We use L1Loss (Mean Absolute Error) as an additional metric
    mae_fn = nn.L1Loss()

    # Disable Gradient Calculation
    # We do NOT need gradients for validation
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward Pass
            # Generate predictions. No backward pass here.
            predictions = model(batch_x)

            # Calculate Main Loss (Target: MSE)
            # This is typically MSE, used for tracking general convergence (and Early Stopping).
            loss = loss_fn(predictions, batch_y)
            running_loss += loss.item()

            # Calculate Auxiliary Metric (MAE)
            # Required for calculating MASE
            mae = mae_fn(predictions, batch_y)
            running_mae += mae.item()

    # Calculate average metrics over all batches
    avg_loss = running_loss / len(dataloader)  # MSE
    avg_mae = running_mae / len(dataloader)    # MAE

    # Calculate RMSE (Root Mean Squared Error)
    avg_rmse = math.sqrt(avg_loss)

    return avg_loss, avg_mae, avg_rmse


def fit_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    fold_idx=None,
    patience=10,
    trial=None,
    epoch_offset=0,
    optimizer_cls=None,
    optimizer_kwargs=None,
    loss_fn=None,
    grad_clip_norm=1.0,
    verbose=True,
    baseline_mae=None,
    scheduler_cls=ReduceLROnPlateau,
    scheduler_kwargs=None,
    scheduler_metric="val_mase",
):
    """
    Main training loop managing the full lifecycle of model fitting.
    Integrates Early Stopping, Learning Rate Scheduling, and Optuna pruning.

    Args:
        model: PyTorch model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Maximum number of epochs.
        lr: Initial learning rate.
        device: Device to run on (cuda/cpu).
        fold_idx: Index of the current Cross-Validation fold (used for MASE calculation).
        patience: Epochs to wait without improvement before Early Stopping.
        trial: Optuna Trial object (for pruning/reporting). Optional.
        epoch_offset: Offset for global step logging (useful for multi-stage training).
        optimizer_cls: Optimizer class (default: AdamW).
        optimizer_kwargs: Extra kwargs for the optimizer (e.g., weight_decay).
        loss_fn: Loss function (default: MSELoss).
        grad_clip_norm: Max norm for gradient clipping (None to disable).
        verbose: Function verbosity (print progress).
        baseline_mae: Naive model MAE for MASE calc. Defaults to global config val if None.
        scheduler_cls: Scheduler class (default: ReduceLROnPlateau).
        scheduler_kwargs: Extra kwargs for scheduler.
        scheduler_metric: Metric to monitor for scheduler ("val_mase" or "val_loss").

    Returns:
        tuple: (model, history, best_epoch)
            - model: Trained model with the best weights restored.
            - history: Dictionary containing training metrics history.
            - best_epoch: The epoch number where the best performance was achieved.
    """
    # Default optimizer and loss
    # Default Optimizer: AdamW is generally better than Adam
    # as it decouples weight decay from gradient updates (better regularization).
    if optimizer_cls is None:
        optimizer_cls = torch.optim.AdamW
    if optimizer_kwargs is None:
        optimizer_kwargs = {"weight_decay": 0.001}
    
    # Default Loss Function: Mean Squared Error
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    # Differential Learning Rate Configuration (This block is crucial for Fine-Tuning pretrained models.)
    # We want to train the "Head" (new layers) fast, but the "Encoder" (pretrained layers) slow,
    # so we don't destroy the knowledge it has already learned.
    if hasattr(model, "is_pretrained") and model.is_pretrained:
        # Identify Encoder Parameters based on model type
        if hasattr(model, "encoder"):
            # EncoderLSTM -> model.encoder
            encoder_params = set(model.encoder.parameters())
        elif hasattr(model, "model") and hasattr(model.model, "model"):
            # PatchTST -> model.model.model.encoder (nested structure from Hugging Face)
            encoder_params = set(model.model.model.encoder.parameters())
        else:
            encoder_params = set()

        # Identify "Other" Parameters (It's everything that is NOT in the encoder set.)
        other_params = [p for p in model.parameters() if p not in encoder_params]

        # Create Parameter Groups for the Optimizer
        param_groups = [
            # Encoder gets 10% of the base Learning Rate
            {"params": list(encoder_params), "lr": lr * 0.1},
            # Head gets the full Learning Rate
            {"params": other_params, "lr": lr},
        ]
        # Use groups instead of flat parameters
        optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
    else:
        # Standard Case: One LR for the whole model
        optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    # Learning Rate Scheduler
    # Dynamically adjusts LR when the metric stops improving.
    if scheduler_kwargs is None:
        scheduler_kwargs = {
            "mode": "min",         # Want to minimize the metric (Loss/MASE)
            "factor": 0.5,         # Halve the LR when triggered
            "patience": 3,         # Wait 3 epochs before triggering
            "min_lr": 1e-6,        # Lower bound for LR
        }
    scheduler = scheduler_cls(optimizer, **scheduler_kwargs) if scheduler_cls else None
    
    # AMP Scaler
    # Create the GradScaler only if we are running on a CUDA device (GPU)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    # Baseline MASE Setup
    # To calculate MASE, we need the performance of the Naive model on this specific fold.
    if baseline_mae is None:
        if fold_idx is None:
            raise ValueError("You must provide either baseline_mae or fold_idx to calculate MASE.")
        # Retrieve the pre-calculated Naive MAE from global config
        baseline_mae = NAIVE_MAE_PER_FOLD[fold_idx]

    # Initialize History Dictionary
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mase": [],
        "val_rmse": [],
    }

    # Best Model Tracking Variables
    best_mase = float("inf")     # Initialize with infinity (we want to minimize MASE)
    epochs_no_improve = 0        # Counter for Early Stopping
    best_model_wts = copy.deepcopy(model.state_dict()) # Save initial weights as backup
    best_epoch = 0

    # Logic to deciding whether to print LR 
    log_lr = fold_idx is None and baseline_mae is not None

    if verbose:
        print(f"Start Training on {device}...")

    # Main Training Loop
    for epoch in range(epochs):
        # Training Phase (One Epoch)
        # Runs forward/backward pass on the entire training set
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, grad_clip_norm, scaler
        )

        # Validation Phase
        # Evaluates the model on unseen validation data to prevent overfitting
        val_loss, val_mae, val_rmse = validate_one_epoch(
            model, val_loader, loss_fn, device
        )

        # Calculate MASE (Mean Absolute Scaled Error)
        # It compares our error (MAE) against the Naive Baseline.
        current_mase = val_mae / baseline_mae

        # Update History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mase"].append(current_mase)
        history["val_rmse"].append(val_rmse)

        # Step the Learning Rate Scheduler
        # If the metric hasn't improved, the scheduler will reduce the LR.
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                # We monitor either MASE or Loss depending on config
                metric = current_mase if scheduler_metric == "val_mase" else val_loss
                scheduler.step(metric)
            else:
                scheduler.step()

        # Print Progress
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            if log_lr:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train MSE: {train_loss:.6f} | "
                    f"Val MSE: {val_loss:.6f} | "
                    f"Val MASE: {current_mase:.4f} | "
                    f"Val RMSE: {val_rmse:.6f} | "
                    f"LR: {current_lr:.6g}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train MSE: {train_loss:.6f} | "
                    f"Val MSE: {val_loss:.6f} | "
                    f"Val MASE: {current_mase:.4f} | "
                    f"Val RMSE: {val_rmse:.6f}"
                )

        # Optuna Integration (Pruning)
        # If we are hyperparameter tuning, we report the intermediate result to Optuna.
        # Optuna decides if this trial is promising. If not, it raises TrialPruned.
        if trial is not None:
            global_step = epoch_offset + epoch
            # Reporting MASE to Optuna
            trial.report(current_mase, global_step)
            # Check if we should stop this trial early (Pruning)
            # (In optuna_optimizer we don't use pruning as the actual configuration) 
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early Stopping Logic (based on MASE)
        # We save the model only if it beats the previous best MASE.
        if current_mase < best_mase:
            best_mase = current_mase
            epochs_no_improve = 0
            # Deepcopy is essential to save the actual weights, not just a reference
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(
                        f"Early Stopping activated at epoch {epoch + 1}. "
                        f"Best MASE: {best_mase:.4f}"
                    )
                break

    # Restore Best Weights
    # Load the weights from the epoch with the lowest MASE
    model.load_state_dict(best_model_wts)
    if verbose:
        print("Training Completed.")

    return model, history, best_epoch
