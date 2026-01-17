import optuna
from optuna.trial import Trial
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
import torch

from .hyperparameter_spaces import get_hyperparameter_space
from ..Training.engine import create_model, fit_model
from ..config import SEED

import os
import shutil


class OptunaOptimizer:
    """
    Manages hyperparameter optimization for a single model using Optuna.

    Sequential Version:
    It trains the model on each Cross-Validation fold sequentially (one after the other).
    This ensures that each trial is fully evaluated on all folds before moving to the next trial,
    providing a robust estimate of performance (Average MASE).

    Args:
            model_name: Name of the model to optimize (e.g., "lstm", "patchtst").
            folds: List of tuples (train_loader, val_loader, scaler) for Cross-Validation.
            device: Torch device (cuda/cpu).
            config: Dictionary containing 'n_trials', 'n_folds', 'patience', 'epochs'.
            storage_path: Path to the SQLite database for persistence (optional).
            verbose: If True, prints debug messages during optimization.
    """

    def __init__(
        self,
        model_name: str,
        folds: list,
        device: torch.device,
        config: dict,
        storage_path: str = None,
        verbose: bool = False,
    ):
        
        self.model_name = model_name.lower()
        self.folds = folds
        self.device = device
        self.config = config
        self.storage_path = storage_path
        self.verbose = verbose

        # Pre-calculate Fold weights based on number of training samples
        # (This is useful if folds are unbalanced)
        self.train_sample_counts = [len(fold[0].dataset) for fold in folds]
        total_samples = sum(self.train_sample_counts)
        self.fold_weights = [n / total_samples for n in self.train_sample_counts]

    def _save_callback(self, study, trial):
        """
        Callback to save the Optuna DB after each trial.
        
        This is specifically for Kaggle/Colab environments where the working directory
        might be ephemeral, so we back up the DB to a safe location.
        """
        if os.path.exists("/kaggle/working"):
            shutil.copy("results/optuna_studies.db", "/kaggle/working/optuna_backup.db")

    def _create_model(self, params: dict):
        """
        Helper method to instantiate the model using the factory function from engine.py.

        Args:
            params: Dictionary of hyperparameters.

        Returns:
            nn.Module: Instantiated PyTorch model.
        """
        return create_model(self.model_name, params)

    def _objective(self, trial: Trial) -> float:
        """
        Optuna Objective Function (SEQUENTIAL, NO PRUNING).

        This function is called by Optuna for every Trial.
        It trains the model on each Cross-Validation fold sequentially,
        and then returns the average performance (Weighted MASE).

        Args:
            trial: Optuna Trial object used to sample hyperparameters.

        Returns:
            float: Weighted average MASE across all folds.
        """
        # Hyperparameter Sampling
        # Get the search space for the current model
        params = get_hyperparameter_space(self.model_name, trial)

        # Extract training-specific parameters (popping them removes them from model params dict)
        lr = params.pop("lr")
        grad_clip_norm = params.pop("grad_clip_norm", 1.0)
        weight_decay = params.pop("weight_decay", 0.001)

        # Extract Scheduler parameters
        scheduler_factor = params.pop("scheduler_factor", 0.5)
        scheduler_patience = params.pop("scheduler_patience", 3)
        scheduler_min_lr = params.pop("scheduler_min_lr", 1e-6)

        # Log Fixed Parameters (User Attributes)
        # Optuna only tracks 'suggested' parameters by default.
        # We manually add fixed parameters (e.g., d_model=128 in finetuning) to the trial user_attrs
        # so they are saved in the DB and we can retrieve them later for retraining.
        for key, value in params.items():
            if key not in trial.params:  # If it wasn't sampled by Optuna
                trial.set_user_attr(key, value)

        # 'params' now contains ONLY the model arguments

        # Fold Selection
        # Determine which folds to use
        n_folds_to_use = self.config["n_folds"]
        if n_folds_to_use == 1:
            # If we only want 1 fold (fast debug), pick the last one
            fold_indices = [2] if len(self.folds) > 2 else [len(self.folds) - 1]
        else:
            # Use all available folds
            fold_indices = list(range(len(self.folds)))

        # Sequential Training Loop
        mase_scores = []
        
        for fold_idx in fold_indices:
            # Retrieve data for this fold
            train_loader, val_loader, _ = self.folds[fold_idx]
            
            # Create a fresh model instance (weights initialized from scratch)
            model = self._create_model(params)
            model.to(self.device)

            # Train the model using the standard engine
            # We pass `trial=None` to disable internal pruning in fit_model.
            # In this Sequential approach, we want to finish the training to get a solid metric.
            _, history, _ = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config["epochs"],
                lr=lr,
                device=self.device,
                fold_idx=fold_idx,
                patience=self.config["patience"],
                trial=None,  # DISABLE INTERNAL PRUNING
                optimizer_kwargs={"weight_decay": weight_decay},
                grad_clip_norm=grad_clip_norm,
                verbose=self.verbose,
                scheduler_kwargs={
                    "mode": "min",
                    "factor": scheduler_factor,
                    "patience": scheduler_patience,
                    "min_lr": scheduler_min_lr,
                },
            )
            
            # Get best validation score for this fold
            best_mase = min(history["val_mase"])
            mase_scores.append(best_mase)

        # Weighted Average Calculation
        # Calculate the final objective value (Weighted MASE)
        weights_to_use = [self.fold_weights[i] for i in fold_indices]
        weight_sum = sum(weights_to_use)
        
        weighted_mase = sum(
            (w / weight_sum) * mase for w, mase in zip(weights_to_use, mase_scores)
        )
        
        return weighted_mase

    def optimize(
        self, study_name: str, n_trials: int = None, new_study: bool = True
    ) -> dict:
        """
        Executes the Hyperparameter Optimization process.

        Args:
            study_name: Name of the Optuna Study.
            n_trials: Number of trials to run (overrides config if provided).
            new_study: If True, creates a new study (deleting old one).
                       If False, resumes an existing study.

        Returns:
            dict: Dictionary containing:
                - "best_params": Best hyperparameters found.
                - "best_mase": Best objective value.
                - "study": The full Optuna Study object.
        """
        if n_trials is None:
            n_trials = self.config["n_trials"]

        # SQLite storage definition for persistence
        # We need this to save progress in case of crash
        storage = f"sqlite:///{self.storage_path}" if self.storage_path else None

        # Study Management
        if new_study and storage:
            # Delete existing study to start fresh
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                print(f"Existing study '{study_name}' deleted.")
            except KeyError:
                pass  # Study did not exist, safe to proceed

        # Create or Load the Study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=not new_study,  # Resume if new_study=False
            direction="minimize",          # We want to MINIMIZE MASE
            sampler=TPESampler(seed=SEED), # Tree-structured Parzen Estimator (Bayesian optimization)
            pruner=NopPruner(),            # No Pruning (Sequential Strategy)
        )

        # Calculate remaining trials (if resuming)
        completed_trials = len(study.trials)
        remaining_trials = max(0, n_trials - completed_trials)

        # Logging
        print(f"\n{'=' * 60}")
        print(f"Optimization Start: {self.model_name.upper()}")
        print(f"Study Name: {study_name}")
        if completed_trials > 0:
            print(
                f"Trials Completed: {completed_trials}, Remaining: {remaining_trials}"
            )
        else:
            print(f"Total Trials: {n_trials}")
        print(
            f"Folds: {self.config['n_folds']}, "
            f"Epochs: {self.config['epochs']}, Patience: {self.config['patience']}"
        )

        # Log effective fold weights (for debugging purposes)
        n_folds_to_use = self.config["n_folds"]
        if n_folds_to_use == 1:
            fold_indices = [2] if len(self.folds) > 2 else [len(self.folds) - 1]
        else:
            fold_indices = list(range(len(self.folds)))

        weights_to_use = [self.fold_weights[i] for i in fold_indices]
        weight_sum = sum(weights_to_use)
        effective_weights = [
            (self.fold_weights[i] / weight_sum) if i in fold_indices else 0.0
            for i in range(len(self.folds))
        ]

        print(
            f"Active Fold Weights: {[f'{w:.3f}' for w in effective_weights]} (Sample Counts: {self.train_sample_counts})"
        )
        print(f"{'=' * 60}\n")

        # Run Optimization
        if remaining_trials > 0:
            study.optimize(
                self._objective,
                n_trials=remaining_trials,
                show_progress_bar=True,
                callbacks=[self._save_callback],
            )
            print("Optimization Finished.")
        else:
            print("All trials completed. No further optimization needed.")

        return {
            "best_params": study.best_params,
            "best_mase": study.best_value,
            "study": study,
        }
