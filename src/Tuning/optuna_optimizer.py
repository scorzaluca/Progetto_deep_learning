"""
OptunaOptimizer: Classe per l'ottimizzazione degli iperparametri con Optuna.
Gestisce la creazione di studi e obiettivi.
VERSIONE SEQUENZIALE SENZA PRUNING.
"""

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
    Gestisce l'ottimizzazione degli iperparametri per un singolo modello.
    Versione sequenziale: addestra ogni fold completamente prima del successivo.
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

        # Pre-calcola i pesi dei fold (basati sul numero di campioni di training)
        self.train_sample_counts = [len(fold[0].dataset) for fold in folds]
        total_samples = sum(self.train_sample_counts)
        self.fold_weights = [n / total_samples for n in self.train_sample_counts]

    def _save_callback(self, study, trial):
        """Callback che salva il DB dopo ogni trial su Kaggle."""
        if os.path.exists('/kaggle/working'):
            shutil.copy('results/optuna_studies.db', '/kaggle/working/optuna_backup.db')

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
        Funzione obiettivo per Optuna (SEQUENZIALE, SENZA PRUNING).
        Addestra il modello su ogni fold completamente in sequenza,
        poi calcola la media ponderata MASE alla fine.
        Args:
            trial: oggetto Optuna Trial
        Returns:
            float: media ponderata del miglior MASE sui fold
        """
        # 1. Genera iperparametri (tutti opzionali hanno fallback default)
        params = get_hyperparameter_space(self.model_name, trial)
        # Parametri di training (estratti con fallback default)
        lr = params.pop("lr")
        grad_clip_norm = params.pop("grad_clip_norm", 1.0)
        weight_decay = params.pop("weight_decay", 0.001)
        # Parametri scheduler (opzionali, con default come fit_model)
        scheduler_factor = params.pop("scheduler_factor", 0.5)
        scheduler_patience = params.pop("scheduler_patience", 3)
        scheduler_min_lr = params.pop("scheduler_min_lr", 1e-6)
        # params ora contiene solo iperparametri del modello
        # 2. Determina quali fold usare
        n_folds_to_use = self.config["n_folds"]
        if n_folds_to_use == 1:
            fold_indices = [2] if len(self.folds) > 2 else [len(self.folds) - 1]
        else:
            fold_indices = list(range(len(self.folds)))
        # 3. Addestra su ogni fold SEQUENZIALMENTE
        mase_scores = []
        for fold_idx in fold_indices:
            train_loader, val_loader, _ = self.folds[fold_idx]
            model = self._create_model(params)
            model.to(self.device)
            # Usa fit_model da engine.py (SENZA trial per disabilitare pruning interno)
            _, history, _ = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config["epochs"],
                lr=lr,
                device=self.device,
                fold_idx=fold_idx,
                patience=self.config["patience"],
                trial=None,  # NESSUN PRUNING
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
            best_mase = min(history["val_mase"])
            mase_scores.append(best_mase)
        # 4. Calcola media ponderata MASE (usa pesi pre-calcolati)
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
        Esegue l'ottimizzazione.

        Args:
            study_name: nome dello studio Optuna
            n_trials: numero di trial (default da config)
            new_study: se True, crea nuovo studio. Se False, riprende esistente.

        Returns:
            dict: {
                "best_params": dict,
                "best_mase": float,
                "study": optuna.Study
            }
        """
        if n_trials is None:
            n_trials = self.config["n_trials"]

        # Storage SQLite per persistenza
        storage = f"sqlite:///{self.storage_path}" if self.storage_path else None

        # Gestione nuovo studio vs ripresa
        if new_study and storage:
            # Elimina studio esistente se presente
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
                print(f"Studio '{study_name}' esistente eliminato.")
            except KeyError:
                pass  # Studio non esiste, ok

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=not new_study,  # Carica se non è nuovo
            direction="minimize",
            sampler=TPESampler(seed=SEED),
            pruner=NopPruner(),
        )

        # Calcola trials rimanenti se si riprende
        completed_trials = len(study.trials)
        remaining_trials = max(0, n_trials - completed_trials)

        print(f"\n{'=' * 60}")
        print(f"Ottimizzazione: {self.model_name.upper()}")
        print(f"Studio: {study_name}")
        if completed_trials > 0:
            print(
                f"Trial completati: {completed_trials}, Rimanenti: {remaining_trials}"
            )
        else:
            print(f"Trial totali: {n_trials}")
        print(
            f"Fold: {self.config['n_folds']}, "
            f"Epochs: {self.config['epochs']}, Patience: {self.config['patience']}"
        )

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
            f"Fold weights: {[f'{w:.3f}' for w in effective_weights]} (samples: {self.train_sample_counts})"
        )
        print(f"{'=' * 60}\n")

        if remaining_trials > 0:
            study.optimize(
                self._objective, n_trials=remaining_trials, show_progress_bar=True, callbacks=[self._save_callback]
            )
        else:
            print("Tutti i trial già completati. Nessuna ottimizzazione necessaria.")

        return {
            "best_params": study.best_params,
            "best_mase": study.best_value,
            "study": study,
        }
