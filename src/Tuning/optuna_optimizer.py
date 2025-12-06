"""
OptunaOptimizer: Classe per l'ottimizzazione degli iperparametri con Optuna.
Gestisce la creazione di studi, obiettivi e pruning.
"""

import os
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn

from .hyperparameter_spaces import get_hyperparameter_space


class OptunaOptimizer:
    """
    Gestisce l'ottimizzazione degli iperparametri per un singolo modello.
    """

    def __init__(
        self,
        model_name: str,
        folds: list,
        device: torch.device,
        config: dict,
        storage_path: str = None,
    ):
        """
        Args:
            model_name: nome del modello ("lstm", "dlinear", "patchtst")
            folds: lista di tuple (train_loader, val_loader, scaler)
            device: torch device (cuda/cpu)
            config: dizionario con n_trials, n_folds, patience, epochs
            storage_path: percorso del database SQLite per persistenza
        """
        self.model_name = model_name.lower()
        self.folds = folds
        self.device = device
        self.config = config
        self.storage_path = storage_path

        # Importa config per baseline MASE
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import NAIVE_MAE_PER_FOLD, LOOKBACK, HORIZON

        self.naive_mae_per_fold = NAIVE_MAE_PER_FOLD
        self.lookback = LOOKBACK
        self.horizon = HORIZON

    def _create_model(self, params: dict, train_loader):
        """
        Crea un'istanza del modello con i parametri specificati.

        Args:
            params: dizionario iperparametri
            train_loader: DataLoader per rilevare input_size dinamicamente

        Returns:
            nn.Module: istanza del modello
        """
        if self.model_name == "lstm":
            from ModelClasses import LSTM

            model_config = {
                "hidden_size": params["hidden_size"],
                "output_size": self.horizon,
                "num_layers": params["num_layers"],
                "dropout": params["dropout"],
                "bidirectional": False,
                "batch_first": True,
            }
            return LSTM(model_config=model_config, train_loader=train_loader)

        elif self.model_name == "dlinear":
            from ModelClasses import DLinear

            model_config = {
                "lookback": self.lookback,
                "horizon": self.horizon,
                "kernel_size": params["kernel_size"],
            }
            return DLinear(model_config=model_config, train_loader=train_loader)

        elif self.model_name == "patchtst":
            from ModelClasses import PatchTST

            model_config = {
                "patch_length": params["patch_length"],
                "stride": params["stride"],
                "d_model": params["d_model"],
                "n_heads": params["n_heads"],
                "n_layers": params["n_layers"],
                "dropout": params["dropout"],
                "use_cls_token": False,
            }
            return PatchTST(model_config=model_config, train_loader=train_loader)
        else:
            raise ValueError(f"Modello '{self.model_name}' non supportato")

    def _train_and_evaluate(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        lr: float,
        fold_idx: int,
        trial: Trial = None,
    ) -> float:
        """
        Addestra il modello e ritorna il best MASE sul validation set.
        Supporta pruning Optuna.

        Args:
            model: modello da addestrare
            train_loader: DataLoader training
            val_loader: DataLoader validation
            lr: learning rate
            fold_idx: indice del fold (per baseline MASE)
            trial: oggetto Trial per reporting/pruning

        Returns:
            float: best MASE raggiunto
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        mae_fn = nn.L1Loss()

        baseline_mae = self.naive_mae_per_fold[fold_idx]
        epochs = self.config["epochs"]
        patience = self.config["patience"]

        best_mase = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            # --- TRAINING ---
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

            # --- VALIDATION ---
            model.eval()
            running_mae = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    pred = model(batch_x)
                    running_mae += mae_fn(pred, batch_y).item()

            avg_mae = running_mae / len(val_loader)
            current_mase = avg_mae / baseline_mae

            # Update best
            if current_mase < best_mase:
                best_mase = current_mase
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Optuna reporting e pruning
            if trial is not None:
                trial.report(current_mase, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # Early stopping
            if epochs_no_improve >= patience:
                break

        return best_mase

    def _objective(self, trial: Trial) -> float:
        """
        Funzione obiettivo per Optuna.
        Addestra il modello sui fold specificati e ritorna la media MASE.

        Args:
            trial: oggetto Optuna Trial

        Returns:
            float: media MASE sui fold
        """
        # 1. Genera iperparametri
        params = get_hyperparameter_space(self.model_name, trial)
        lr = params.pop("lr")  # LR è gestito separatamente

        # 2. Determina quali fold usare
        n_folds_to_use = self.config["n_folds"]
        if n_folds_to_use == 1:
            # Fase screening: usa fold più grande (indice 2, il terzo)
            fold_indices = [2] if len(self.folds) > 2 else [len(self.folds) - 1]
        else:
            # Fase intensive: usa tutti i fold
            fold_indices = list(range(len(self.folds)))

        # 3. Addestra su ogni fold
        mase_scores = []
        for fold_idx in fold_indices:
            train_loader, val_loader, _ = self.folds[fold_idx]

            # Crea modello
            model = self._create_model(params, train_loader)
            model.to(self.device)

            # Addestra e valuta
            mase = self._train_and_evaluate(
                model, train_loader, val_loader, lr, fold_idx, trial
            )
            mase_scores.append(mase)

        return sum(mase_scores) / len(mase_scores)

    def optimize(self, study_name: str, n_trials: int = None) -> dict:
        """
        Esegue l'ottimizzazione.

        Args:
            study_name: nome dello studio Optuna
            n_trials: numero di trial (default da config)

        Returns:
            dict: {
                "best_params": dict,
                "best_mase": float,
                "study": optuna.Study
            }
        """
        if n_trials is None:
            n_trials = self.config["n_trials"]

        # Crea o carica studio
        storage = f"sqlite:///{self.storage_path}" if self.storage_path else None

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )

        print(f"\n{'=' * 60}")
        print(f"Ottimizzazione: {self.model_name.upper()}")
        print(
            f"Trial: {n_trials}, Fold: {self.config['n_folds']}, "
            f"Epochs: {self.config['epochs']}, Patience: {self.config['patience']}"
        )
        print(f"{'=' * 60}\n")

        study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)

        return {
            "best_params": study.best_params,
            "best_mase": study.best_value,
            "study": study,
        }
