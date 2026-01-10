import optuna

DB_PATH = "sqlite:///results/optuna_studies.db"

studies = optuna.get_all_study_names(storage=DB_PATH)
print("Studi:", studies)

for name in studies:
    study = optuna.load_study(study_name=name, storage=DB_PATH)
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    print(f"\n{name}: {len(completed)} trial COMPLETATI (su {len(study.trials)} totali)")
    if completed:
        print(f"  Miglior MASE: {study.best_value:.4f}")