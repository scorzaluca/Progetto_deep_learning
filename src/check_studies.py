"""
import optuna
import pandas as pd
from pathlib import Path

DB_PATH = "sqlite:///results/optuna_studies.db"
OUTPUT_PATH = Path(__file__).parent / "optuna_results.xlsx"

def get_study_data():
    
    studies = optuna.get_all_study_names(storage=DB_PATH)
    
    if not studies:
        print("Nessuno studio trovato nel database.")
        return None
    
    rows = []
    
    for study_name in studies:
        study = optuna.load_study(study_name=study_name, storage=DB_PATH)
        
        # Conteggio trial per stato
        trials_by_state = {}
        for trial in study.trials:
            state = trial.state.name
            trials_by_state[state] = trials_by_state.get(state, 0) + 1
        
        # Dati base dello studio
        row = {
            "Studio": study_name,
            "Direzione": study.direction.name,
            "Trial Totali": len(study.trials),
            "Completati": trials_by_state.get("COMPLETE", 0),
            "Pruned": trials_by_state.get("PRUNED", 0),
            "Failed": trials_by_state.get("FAIL", 0),
        }
        
        # Info sul miglior trial (se esiste)
        completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        if completed_trials:
            best_trial = study.best_trial
            row["Miglior MASE"] = study.best_value
            row["Best Trial #"] = best_trial.number
            
            # Aggiungi tutti i parametri del miglior trial
            for param_name, param_value in best_trial.params.items():
                row[f"param_{param_name}"] = param_value
            
            # Aggiungi user attributes del miglior trial (es. RMSE)
            for attr_name, attr_value in best_trial.user_attrs.items():
                row[f"attr_{attr_name}"] = attr_value
            
            # Durata del miglior trial
            if best_trial.datetime_start and best_trial.datetime_complete:
                duration = best_trial.datetime_complete - best_trial.datetime_start
                row["Durata Best Trial"] = str(duration)
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def main():
    print(f"Caricamento studi da: {DB_PATH}")
    
    df = get_study_data()
    
    if df is not None:
        # Salva in Excel
        df.to_excel(OUTPUT_PATH, index=False, sheet_name="Studi Optuna")
        print(f"✅ File Excel salvato in: {OUTPUT_PATH}")
        print(f"   Studi esportati: {len(df)}")
    else:
        print("❌ Nessun dato da esportare.")

if __name__ == "__main__":
    main()



"""


import optuna

DB_PATH = "sqlite:///results/optuna_studies.db"

studies = optuna.get_all_study_names(storage=DB_PATH)
print("Studi trovati:", studies)

for name in studies:
    study = optuna.load_study(study_name=name, storage=DB_PATH)
    completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    print(f"  - {name}: {completed} trial completati (su {len(study.trials)} totali)")