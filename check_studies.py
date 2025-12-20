import optuna
studies = optuna.study.get_all_study_names(storage="sqlite:///results/optuna_studies.db")
print("Studi nel DB:")
for s in studies:
    print(f"  - {s}")