import pandas as pd

def analyze_csv_direct(csv_file_name):
    """
    Legge un file CSV e analizza le colonne di tipo object e int.
    Non utilizza blocchi try-except: se il file non esiste, lo script si fermerà con un errore.
    """
    print(f"Lettura del file: {csv_file_name} ...")
    
    # Legge direttamente il CSV. Se il file non c'è, Python solleverà FileNotFoundError qui.
    df = pd.read_csv(csv_file_name)
    
    print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne.\n")
    
    # Seleziona le colonne di interesse
    cols_to_analyze = df.select_dtypes(include=['object', 'int', 'int64', 'int32']).columns
    
    results = {}
    
    for col in cols_to_analyze:
        # Calcolo valori unici
        num_unique = df[col].nunique()
        
        # Calcolo frequenze percentuali
        frequencies = df[col].value_counts(normalize=True) * 100
        
        # Salvataggio nel dizionario
        results[col] = {
            'num_unique': num_unique,
            'frequencies': frequencies
        }
        
        # Stampa dei risultati a video
        print(f"--- Colonna: {col} ---")
        print(f"Valori Unici: {num_unique}")
        print(f"Frequenze (% - Top 5):\n{frequencies.head(5).to_string()}\n")
        
    return results

# --- Esempio di utilizzo ---
# Assicurati che 'merge_ds.csv' sia nella stessa cartella, altrimenti il codice crasherà (come richiesto)
dati_analisi = analyze_csv_direct('Data/merge_ds.csv')

