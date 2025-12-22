"""
OptunaOptimizer: Classe per l'ottimizzazione degli iperparametri con Optuna.
Gestisce la creazione di studi, obiettivi e pruning.
"""

import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from .hyperparameter_spaces import get_hyperparameter_space
from ..Training.engine import create_model
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
        Funzione obiettivo per Optuna con training INTERLEAVED sui fold.

        Invece di addestrare ogni fold completamente in sequenza, addestra
        tutti i fold un'epoca alla volta e calcola il MASE ponderato dopo
        ogni epoca. Questo permette un pruning più accurato basato sulla
        metrica aggregata (quella che conta davvero).

        Caratteristiche (come fit_model):
        - AMP (Automatic Mixed Precision) su CUDA
        - AdamW con weight_decay
        - ReduceLROnPlateau scheduler
        - Gradient clipping
        - Early stopping globale
        - Best weights saving per fold

        Args:
            trial: oggetto Optuna Trial

        Returns:
            float: media ponderata del miglior MASE sui fold
        """
        import copy
        import torch.nn as nn
        from torch.amp import GradScaler
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from ..Training.engine import train_one_epoch, validate_one_epoch
        from ..config import NAIVE_MAE_PER_FOLD

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

        epochs = self.config["epochs"]
        patience = self.config["patience"]

        # 3. Setup: crea modelli, optimizer, scheduler, scaler per ogni fold
        fold_states = []
        for fold_idx in fold_indices:
            train_loader, val_loader, _ = self.folds[fold_idx]

            model = self._create_model(params)
            model.to(self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            # Scheduler: riduce LR quando MASE non migliora
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=scheduler_min_lr,
            )

            use_amp = self.device.type == "cuda"
            scaler = GradScaler(enabled=use_amp) if use_amp else None

            fold_states.append(
                {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "scaler": scaler,
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "baseline_mae": NAIVE_MAE_PER_FOLD[fold_idx],
                    "best_mase": float("inf"),
                    "best_weights": copy.deepcopy(model.state_dict()),
                }
            )

        loss_fn = nn.MSELoss()

        # Pesi per media ponderata
        # NOTA: rinormalizziamo perché se n_folds < len(folds), i pesi non sommano a 1
        weights = [self.fold_weights[i] for i in fold_indices]
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]

        # 4. Training loop interleaved
        best_weighted_mase = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            epoch_mase_scores = []

            # Train + validate ogni fold per questa epoca
            for state in fold_states:
                # Training
                train_one_epoch(
                    state["model"],
                    state["train_loader"],
                    state["optimizer"],
                    loss_fn,
                    self.device,
                    grad_clip_norm,
                    state["scaler"],
                )

                # Validation
                _, val_mae, _ = validate_one_epoch(
                    state["model"],
                    state["val_loader"],
                    loss_fn,
                    self.device,
                )

                # Calcola MASE per questo fold
                fold_mase = val_mae / state["baseline_mae"]
                epoch_mase_scores.append(fold_mase)

                # Scheduler step (riduce LR se MASE non migliora)
                state["scheduler"].step(fold_mase)

                # Aggiorna best per questo fold
                if fold_mase < state["best_mase"]:
                    state["best_mase"] = fold_mase
                    state["best_weights"] = copy.deepcopy(state["model"].state_dict())

            # Calcola MASE ponderato per questa epoca
            weighted_mase = sum(
                w * mase for w, mase in zip(normalized_weights, epoch_mase_scores)
            )

            # Report a Optuna per pruning
            trial.report(weighted_mase, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping globale (sulla media ponderata)
            if weighted_mase < best_weighted_mase:
                best_weighted_mase = weighted_mase
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # 5. Ritorna la media ponderata dei MIGLIORI MASE di ogni fold
        best_mase_scores = [state["best_mase"] for state in fold_states]
        final_weighted_mase = sum(
            w * mase for w, mase in zip(normalized_weights, best_mase_scores)
        )

        return final_weighted_mase

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

        # Stampa i pesi dei fold (pre-calcolati nel costruttore)
        print(
            f"Fold weights: {[f'{w:.3f}' for w in self.fold_weights]} (samples: {self.train_sample_counts})"
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
