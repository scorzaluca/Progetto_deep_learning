"""
OptunaOptimizer: Classe per l'ottimizzazione degli iperparametri con Optuna.
Gestisce la creazione di studi, obiettivi e pruning.
Utilizza fit_model da engine.py per evitare duplicazione di logica.
"""

import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from .hyperparameter_spaces import get_hyperparameter_space
from ..Training.engine import fit_model, create_model
from ..config import SEED


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
        verbose: bool = False,
    ):
        """
        Args:
            model_name: nome del modello ("lstm", "dlinearm", "dlineari", "patchtst", "tcn")
            folds: lista di tuple (train_loader, val_loader, scaler)
            device: torch device (cuda/cpu)
            config: dizionario con n_trials, n_folds, patience, epochs
            storage_path: percorso del database SQLite per persistenza
            verbose: se True, stampa messaggi di debug
        """
        self.model_name = model_name.lower()
        self.folds = folds
        self.device = device
        self.config = config
        self.storage_path = storage_path
        self.verbose = verbose

    def _create_model(self, params: dict):
        """
        Crea un'istanza del modello con i parametri specificati.
        Utilizza la factory function create_model() da engine.py.

        Args:
            params: dizionario iperparametri

        Returns:
            nn.Module: istanza del modello
        """
        return create_model(self.model_name, params)

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

        # Estrai grad_clip_norm se presente (altrimenti default)
        grad_clip_norm = params.pop("grad_clip_norm", 1.0)

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
            model = self._create_model(params)
            model.to(self.device)

            # Usa fit_model da engine.py (consolidato)
            _, history, _ = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config["epochs"],
                lr=lr,
                device=self.device,
                fold_idx=fold_idx,
                patience=self.config["patience"],
                trial=trial,  # Passa trial per pruning
                grad_clip_norm=grad_clip_norm,
                verbose=self.verbose,  # Controllato da parametro classe
            )

            # Prendi il miglior MASE dalla history
            best_mase = min(history["val_mase"])
            mase_scores.append(best_mase)

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
            sampler=TPESampler(seed=SEED),
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
