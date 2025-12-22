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

        # Pre-calcola i pesi dei fold (basati sul numero di campioni di training)
        self.train_sample_counts = [len(fold[0].dataset) for fold in folds]
        total_samples = sum(self.train_sample_counts)
        self.fold_weights = [n / total_samples for n in self.train_sample_counts]

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
        Addestra il modello sui fold specificati e ritorna la media ponderata MASE.
        I pesi sono proporzionali al numero di campioni di training per fold.

        Args:
            trial: oggetto Optuna Trial

        Returns:
            float: media ponderata MASE sui fold
        """
        # 1. Genera iperparametri
        params = get_hyperparameter_space(self.model_name, trial)
        lr = params.pop("lr")  # LR è gestito separatamente

        # Estrai grad_clip_norm se presente (altrimenti default)
        grad_clip_norm = params.pop("grad_clip_norm", 1.0)

        # Estrai weight_decay se presente (altrimenti default)
        weight_decay = params.pop("weight_decay", 0.01)

        # 2. Determina quali fold usare
        n_folds_to_use = self.config["n_folds"]
        if n_folds_to_use == 1:
            # Fase screening: usa fold più grande (indice 2, il terzo)
            fold_indices = [2] if len(self.folds) > 2 else [len(self.folds) - 1]
        else:
            # Fase intensive: usa tutti i fold
            fold_indices = list(range(len(self.folds)))

        # 3. Addestra su ogni fold e raccogli risultati
        mase_scores = []
        epochs = self.config["epochs"]

        for i, fold_idx in enumerate(fold_indices):
            train_loader, val_loader, _ = self.folds[fold_idx]

            # Crea modello
            model = self._create_model(params)
            model.to(self.device)

            # Calcola epoch_offset per step globale Optuna
            # Fold 0: step 0-29, Fold 1: step 30-59, Fold 2: step 60-89
            epoch_offset = i * epochs

            # Usa fit_model da engine.py (consolidato)
            _, history, _ = fit_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                device=self.device,
                fold_idx=fold_idx,
                patience=self.config["patience"],
                trial=trial,
                epoch_offset=epoch_offset,
                grad_clip_norm=grad_clip_norm,
                verbose=self.verbose,
            )

            # Prendi il miglior MASE dalla history
            best_mase = min(history["val_mase"])
            mase_scores.append(best_mase)

        # 4. Calcola media ponderata MASE (usa pesi pre-calcolati)
        weights_to_use = [self.fold_weights[i] for i in fold_indices]
        # Normalizza i pesi se non si usano tutti i fold
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
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
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
                self._objective, n_trials=remaining_trials, show_progress_bar=True
            )
        else:
            print("Tutti i trial già completati. Nessuna ottimizzazione necessaria.")

        return {
            "best_params": study.best_params,
            "best_mase": study.best_value,
            "study": study,
        }
