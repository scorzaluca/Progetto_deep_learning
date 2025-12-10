"""
Pipeline di ottimizzazione: funzioni per eseguire screening e intensive optimization.
Utilizza fit_model da engine.py per evitare duplicazione di logica.
"""

import os
import json
import torch
from datetime import datetime

from .optuna_optimizer import OptunaOptimizer
from ..Training.engine import fit_model, create_model


def run_screening(
    folds: list,
    device: torch.device,
    config: dict,
    results_dir: str,
) -> dict:
    """
    Esegue lo screening su tutti i modelli (LSTM, DLinear, PatchTST, TCN).

    Args:
        folds: lista di tuple (train_loader, val_loader, scaler)
        device: torch device
        config: dizionario SCREENING_CONFIG
        results_dir: directory per salvare i risultati

    Returns:
        dict: {
            "lstm": {"best_mase": float, "best_params": dict},
            "dlinearm": {"best_mase": float, "best_params": dict},
            "patchtst": {"best_mase": float, "best_params": dict},
            "ranking": ["patchtst", "dlinear", "lstm"]  # ordinato per MASE
        }
    """
    models = ["lstm", "dlinearm", "dlineari", "patchtst", "tcn"]
    results = {}

    n_trials = config["n_trials"]
    storage_path = os.path.join(results_dir, "optuna_studies.db")

    print("\n" + "=" * 60)
    print("FASE 1: SCREENING")
    print(f"Modelli: {models}")
    print(f"Trial per modello: {n_trials}")
    print("Fold utilizzati: 1 (più grande)")
    print("=" * 60 + "\n")

    for model_name in models:
        try:
            optimizer = OptunaOptimizer(
                model_name=model_name,
                folds=folds,
                device=device,
                config=config,
                storage_path=storage_path,
                verbose=True,
            )

            study_name = (
                f"screening_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            result = optimizer.optimize(study_name=study_name, n_trials=n_trials)

            results[model_name] = {
                "best_mase": result["best_mase"],
                "best_params": result["best_params"],
            }

            print(f"\n✓ {model_name.upper()}: MASE = {result['best_mase']:.4f}")
            print(f"  Best params: {result['best_params']}")

        except Exception as e:
            print(f"\n✗ {model_name.upper()}: ERRORE - {str(e)}")
            results[model_name] = {
                "best_mase": float("inf"),
                "best_params": {},
                "error": str(e),
            }

    # Crea ranking
    ranking = sorted(
        [m for m in models if results[m]["best_mase"] < float("inf")],
        key=lambda m: results[m]["best_mase"],
    )
    results["ranking"] = ranking

    # Salva risultati
    results_path = os.path.join(results_dir, "screening_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SCREENING COMPLETATO")
    print(f"Ranking: {' > '.join(ranking)}")
    print(f"Risultati salvati in: {results_path}")
    print("=" * 60 + "\n")

    return results


def run_intensive(
    model_name: str,
    folds: list,
    device: torch.device,
    config: dict,
    results_dir: str,
) -> dict:
    """
    Esegue l'ottimizzazione intensiva su UN singolo modello.
    Salva i pesi del modello migliore.

    Args:
        model_name: nome del modello ("lstm", "dlinearm", "dlineari", "patchtst", "tcn")
        folds: lista di tuple (train_loader, val_loader, scaler)
        device: torch device
        config: dizionario INTENSIVE_CONFIG
        results_dir: directory per salvare i risultati

    Returns:
        dict: {
            "best_params": dict,
            "best_mase": float,
            "checkpoint_path": str
        }
    """
    model_name = model_name.lower()
    n_trials = config["n_trials"]
    storage_path = os.path.join(results_dir, "optuna_studies.db")

    print("\n" + "=" * 60)
    print(f"FASE 2: OTTIMIZZAZIONE INTENSIVA - {model_name.upper()}")
    print(f"Trial: {n_trials}")
    print(f"Fold utilizzati: {config['n_folds']} (tutti)")
    print("=" * 60 + "\n")

    optimizer = OptunaOptimizer(
        model_name=model_name,
        folds=folds,
        device=device,
        config=config,
        storage_path=storage_path,
    )

    study_name = f"intensive_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = optimizer.optimize(study_name=study_name, n_trials=n_trials)

    # Salva best params
    params_path = os.path.join(
        results_dir, "intensive_results", f"{model_name}_best_params.json"
    )
    os.makedirs(os.path.dirname(params_path), exist_ok=True)

    best_result = {
        "best_params": result["best_params"],
        "best_mase": result["best_mase"],
        "study_name": study_name,
    }

    with open(params_path, "w") as f:
        json.dump(best_result, f, indent=2)

    # Riaddestra con i best params su tutti i fold e salva il modello migliore
    print("\n" + "-" * 40)
    print("Riaddestramento finale con best params...")
    print("-" * 40)

    checkpoint_path = _train_final_model(
        model_name=model_name,
        best_params=result["best_params"],
        folds=folds,
        device=device,
        config=config,
        results_dir=results_dir,
    )

    best_result["checkpoint_path"] = checkpoint_path

    # Aggiorna file params con checkpoint path
    with open(params_path, "w") as f:
        json.dump(best_result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"OTTIMIZZAZIONE COMPLETATA - {model_name.upper()}")
    print(f"Best MASE: {result['best_mase']:.4f}")
    print(f"Best params: {result['best_params']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Params salvati: {params_path}")
    print("=" * 60 + "\n")

    return best_result


def _train_final_model(
    model_name: str,
    best_params: dict,
    folds: list,
    device: torch.device,
    config: dict,
    results_dir: str,
) -> str:
    """
    Addestra il modello con i best params e salva i pesi.
    Usa il fold con il miglior MASE per salvare i pesi.
    Utilizza create_model() e fit_model() da engine.py.

    Returns:
        str: percorso del checkpoint salvato
    """
    import copy

    lr = best_params.get("lr", 0.001)
    grad_clip_norm = best_params.get("grad_clip_norm", 1.0)

    best_mase_overall = float("inf")
    best_model_state = None

    for fold_idx, (train_loader, val_loader, _) in enumerate(folds):
        print(f"  Training fold {fold_idx + 1}/{len(folds)}...")

        # Crea modello usando factory function
        model = create_model(model_name, best_params)
        model.to(device)

        # Usa fit_model da engine.py (consolidato)
        trained_model, history, _ = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["epochs"],
            lr=lr,
            device=device,
            fold_idx=fold_idx,
            patience=config["patience"],
            grad_clip_norm=grad_clip_norm,
            verbose=False,
        )

        best_fold_mase = min(history["val_mase"])
        print(f"    Fold {fold_idx + 1} - Best MASE: {best_fold_mase:.4f}")

        if best_fold_mase < best_mase_overall:
            best_mase_overall = best_fold_mase
            best_model_state = copy.deepcopy(trained_model.state_dict())

    # Salva checkpoint
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")

    torch.save(best_model_state, checkpoint_path)
    print(f"  Checkpoint salvato: {checkpoint_path}")

    return checkpoint_path
