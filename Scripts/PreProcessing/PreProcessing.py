import numpy as np
import pandas as pd

def carica_e_combina_csv(file_list, date_col):
    """
    Carica una lista di file CSV, li concatena, li ordina per data
    e resetta l'indice.

    Parametri:
    - file_list (list): Una lista di stringhe con i percorsi dei file CSV.
    - date_col (str): Il nome della colonna da interpretare come datetime.

    Ritorna:
    - pd.DataFrame: Un singolo DataFrame combinato, ordinato e indicizzato.
    - O un DataFrame vuoto in caso di errore.
    """
    
    lista_df = [] # Lista per contenere i DataFrame caricati
    
    print("--- Inizio Processo di Caricamento e Unione ---")
    
    try:
        # --- 1. Caricamento ---
        # Motivazione: Iteriamo sulla lista 'file_list' passata come parametro.
        # Questo rende la funzione flessibile: puoi passare 2 o 20 file.
        for file_path in file_list:
            df_temp = pd.read_csv(file_path, parse_dates=[date_col])
            print(f"Caricato '{file_path}' con {len(df_temp)} righe.")
            lista_df.append(df_temp)
        
        # --- 2. Controllo e Concatenazione ---
        # Motivazione: Verifichiamo se abbiamo effettivamente caricato qualcosa
        # prima di tentare di concatenare, per evitare errori.
        if not lista_df:
            print("Errore: Nessun file è stato caricato. La lista è vuota.")
            return pd.DataFrame() # Ritorna un DataFrame vuoto
            
        # Motivazione: pd.concat è l'operazione corretta per impilare
        # i DataFrame (axis=0). 'ignore_index=True' è fondamentale
        # per creare un nuovo indice pulito.
        df_combinato = pd.concat(lista_df, ignore_index=True)
        
        # --- 3. Ordinamento ---
        # Motivazione: Fondamentale per le serie temporali.
        # Assicura l'integrità cronologica del dataset combinato.
        df_combinato = df_combinato.sort_values(by=date_col)
        
        # --- 4. Reset Indice ---
        # Motivazione: Dopo l'ordinamento, l'indice è "mescolato".
        # reset_index(drop=True) lo ricrea da 0 a N-1 nell'ordine corretto.
        df_combinato = df_combinato.reset_index(drop=True)
        
        print(f"\nDataFrame combinato creato con successo.")
        print(f"Dimensioni totali: {df_combinato.shape}")
        
        return df_combinato

    except FileNotFoundError as e:
        print(f"ERRORE CRITICO: File non trovato. Impossibile procedere.")
        print(f"Dettagli: {e.filename}")
        return pd.DataFrame() # Ritorna un DataFrame vuoto
    except Exception as e:
        print(f"ERRORE GENERICO: Si è verificato un errore: {e}")
        return pd.DataFrame() # Ritorna un DataFrame vuoto



def cyclical_encoding(df, date_col="dt_iso"):
    """
    Applica l'encoding ciclico per i cicli GIORNALIERO e ANNUALE.

    Questa versione semplificata ignora la gestione degli anni bisestili
    per il ciclo annuale, assumendo sempre un anno di 365 giorni.

    Motivazione:
    - Il ciclo giornaliero (ora) determina la presenza/assenza del sole.
    - Il ciclo annuale (giorno dell'anno) determina la stagionalità (inverno/estate).
    """
    
    # 1. Assicurati che la colonna sia in formato datetime
    # errors='coerce' gestisce eventuali errori di parsing
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # --- 2. Ciclo Giornaliero (basato sull'ora) ---
    # Calcoliamo la frazione del giorno (da 0.0 a 0.999...)
    # Questo calcolo è preciso al secondo.
    day_fraction = (
        df[date_col].dt.hour / 24.0 + 
        df[date_col].dt.minute / 1440.0 + 
        df[date_col].dt.second / 86400.0
    )
    
    # Applichiamo sin/cos
    df['day_sin'] = np.sin(2 * np.pi * day_fraction)
    df['day_cos'] = np.cos(2 * np.pi * day_fraction)
    
    
    # --- 3. Ciclo Annuale (basato sul giorno dell'anno) ---
    # Usiamo .dt.dayofyear (es. 1 per 1° Gen, 365 per 31 Dic)
    # Normalizziamo dividendo semplicemente per 365.
    
    # Motivazione della Semplificazione:
    # Si assume che il ciclo si ripeta ogni 365 giorni.
    # L'attributo .dt.dayofyear va da 1 a 365 (o 366 nei bisestili).
    # Dividendo per 365, il valore sarà circa [0.0027, 1.0] in un anno normale
    # e circa [0.0027, 1.0027] in un anno bisestile.
    # L'impatto di questa piccola differenza è trascurabile per il modello.
    
    year_fraction = df[date_col].dt.dayofyear / 365.0
    
    # Applichiamo sin/cos
    df['year_sin'] = np.sin(2 * np.pi * year_fraction)
    df['year_cos'] = np.cos(2 * np.pi * year_fraction)
    
    return df    


# --- ONE-HOT ENCODING per 'weather_description' 
def cat_encoding(df, column_name, prefix_name):

    if column_name not in df.columns:
        print(f"Errore: La colonna '{column_name}' non è presente nel DataFrame.")
        return df 

    print(f"Dimensioni prima del One-Hot Encoding per '{column_name}': {df.shape}")

    df = pd.get_dummies(df, 
                        columns=[column_name], 
                        prefix=prefix_name)

    print(f"Dimensioni dopo il One-Hot Encoding: {df.shape}")
    print(f"\nTesta del DataFrame dopo OHE su '{column_name}':")
    print(df.head())
    
    return df


