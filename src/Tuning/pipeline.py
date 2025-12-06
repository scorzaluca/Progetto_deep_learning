"""
Pipeline di ottimizzazione: funzioni per eseguire screening e intensive optimization.
"""

import os
import json
import torch
from datetime import datetime

from .optuna_optimizer import OptunaOptimizer


def run_screening(
    folds: list,
    device: torch.device,
    config: dict,
    results_dir: str,
    dry_run: bool = False,
) -> dict:
    """
    Esegue lo screening su tutti i modelli (LSTM, DLinear, PatchTST).

    Args:
        folds: lista di tuple (train_loader, val_loader, scaler)
        device: torch device
        config: dizionario SCREENING_CONFIG
        results_dir: directory per salvare i risultati
        dry_run: se True, esegue solo 2 trial per test

    Returns:
        dict: {
            "lstm": {"best_mase": float, "best_params": dict},
            "dlinear": {"best_mase": float, "best_params": dict},
            "patchtst": {"best_mase": float, "best_params": dict},
            "ranking": ["patchtst", "dlinear", "lstm"]  # ordinato per MASE
        }
    """
    models = ["lstm", "dlinear", "patchtst"]
    results = {}

    n_trials = 2 if dry_run else config["n_trials"]
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
    dry_run: bool = False,
) -> dict:
    """
    Esegue l'ottimizzazione intensiva su UN singolo modello.
    Salva i pesi del modello migliore.

    Args:
        model_name: nome del modello ("lstm", "dlinear", "patchtst")
        folds: lista di tuple (train_loader, val_loader, scaler)
        device: torch device
        config: dizionario INTENSIVE_CONFIG
        results_dir: directory per salvare i risultati
        dry_run: se True, esegue solo 2 trial per test

    Returns:
        dict: {
            "best_params": dict,
            "best_mase": float,
            "checkpoint_path": str
        }
    """
    model_name = model_name.lower()
    n_trials = 2 if dry_run else config["n_trials"]
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

    Returns:
        str: percorso del checkpoint salvato
    """
    import copy
    import torch.nn as nn

    # Import config per parametri
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import NAIVE_MAE_PER_FOLD, LOOKBACK, HORIZON

    lr = best_params.get("lr", 0.001)

    best_mase_overall = float("inf")
    best_model_state = None

    for fold_idx, (train_loader, val_loader, scaler) in enumerate(folds):
        print(f"  Training fold {fold_idx + 1}/{len(folds)}...")

        # Crea modello
        if model_name == "lstm":
            from ModelClasses import LSTM

            model_config = {
                "hidden_size": best_params["hidden_size"],
                "output_size": HORIZON,
                "num_layers": best_params["num_layers"],
                "dropout": best_params["dropout"],
                "bidirectional": False,
                "batch_first": True,
            }
            model = LSTM(model_config=model_config, train_loader=train_loader)

        elif model_name == "dlinear":
            from ModelClasses import DLinear

            model_config = {
                "lookback": LOOKBACK,
                "horizon": HORIZON,
                "kernel_size": best_params["kernel_size"],
            }
            model = DLinear(model_config=model_config, train_loader=train_loader)

        elif model_name == "patchtst":
            from ModelClasses import PatchTST

            model_config = {
                "patch_length": best_params["patch_length"],
                "stride": best_params["stride"],
                "d_model": best_params["d_model"],
                "n_heads": best_params["n_heads"],
                "n_layers": best_params["n_layers"],
                "dropout": best_params["dropout"],
                "use_cls_token": False,
            }
            model = PatchTST(model_config=model_config, train_loader=train_loader)

        model.to(device)

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        mae_fn = nn.L1Loss()

        baseline_mae = NAIVE_MAE_PER_FOLD[fold_idx]
        best_fold_mase = float("inf")
        best_fold_state = None
        epochs_no_improve = 0

        for epoch in range(config["epochs"]):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            running_mae = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    pred = model(batch_x)
                    running_mae += mae_fn(pred, batch_y).item()

            avg_mae = running_mae / len(val_loader)
            current_mase = avg_mae / baseline_mae

            if current_mase < best_fold_mase:
                best_fold_mase = current_mase
                best_fold_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config["patience"]:
                break

        print(f"    Fold {fold_idx + 1} - Best MASE: {best_fold_mase:.4f}")

        if best_fold_mase < best_mase_overall:
            best_mase_overall = best_fold_mase
            best_model_state = best_fold_state

    # Salva checkpoint
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")

    torch.save(best_model_state, checkpoint_path)
    print(f"  Checkpoint salvato: {checkpoint_path}")

    return checkpoint_path
