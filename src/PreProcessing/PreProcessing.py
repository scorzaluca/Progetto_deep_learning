import numpy as np
import pandas as pd
import os
from functools import reduce


def carica_e_combina_xlsx(file_list, date_col, output_path=None):
    """
    Crea un unico DataFrame mergiando DUE file Excel (.xlsx) per la colonna datetime
    e salva anche il risultato in un file .xlsx.

    Per ciascun file:
    - legge TUTTI i fogli e li concatena (dal primo all'ultimo);
    - usa la PRIMA colonna come chiave temporale e la converte in datetime;
    - per il SECONDO file in lista elimina la PRIMA riga del DataFrame concatenato;
    - alla fine esegue il merge (outer) tra i due DataFrame sulla chiave temporale.

    Parametri:
    - file_list (list[str]): Lista di 2 percorsi dei file da caricare (solo .xlsx).
    - date_col (str): Nome canonico assegnato alla prima colonna (chiave di merge).
    - output_path (str|None): Percorso del file .xlsx finale da scrivere. Se None,
      viene creato accanto al primo file con nome 'merged_dataset.xlsx'.

    Ritorna:
    - pd.DataFrame: DataFrame finale mergiato su `date_col`, ordinato e indicizzato.
    - In caso di problemi/blocchi, ritorna un DataFrame vuoto.
    """

    print("--- Merge di 2 file .xlsx per datetime e salvataggio risultato ---")

    try:
        if not isinstance(file_list, (list, tuple)) or len(file_list) != 2:
            print("Errore: 'file_list' deve contenere esattamente 2 percorsi di file .xlsx.")
            return pd.DataFrame()

        per_file_dfs = []  # DataFrame concatenato per ciascun file

        for idx, file_path in enumerate(file_list):
            if not os.path.exists(file_path):
                print(f"Avviso: File non trovato e saltato: {file_path}")
                return pd.DataFrame()

            est = os.path.splitext(file_path)[1].lower()
            if est != ".xlsx":
                print(f"Errore: Estensione non supportata ({est}). Sono accettati solo file .xlsx.")
                return pd.DataFrame()

            print(f"\n>> Elaboro file: {file_path}")

            # Legge tutti i fogli in un dict {sheet_name: df}
            sheets_dict = pd.read_excel(file_path, sheet_name=None)
            sheet_dfs = []
            for nome_foglio, df_sheet in sheets_dict.items():
                if df_sheet is None or df_sheet.shape[1] == 0:
                    print(f"  - Foglio '{nome_foglio}': vuoto o senza colonne, saltato.")
                    continue
                df_sheet = df_sheet.copy()
                key_col = df_sheet.columns[0]
                # Converte la prima colonna a datetime
                df_sheet[key_col] = pd.to_datetime(df_sheet[key_col], errors='coerce', utc=True)
                # Rimuove timezone rendendola naive (in UTC)
                if hasattr(df_sheet[key_col].dt, 'tz'):
                    try:
                        df_sheet[key_col] = df_sheet[key_col].dt.tz_localize(None)
                    except Exception:
                        pass
                # Rinomina la prima colonna con il nome canonico 'date_col'
                if key_col != date_col:
                    df_sheet = df_sheet.rename(columns={key_col: date_col})
                sheet_dfs.append(df_sheet)

            if not sheet_dfs:
                print("  Nessun foglio valido trovato (manca la colonna data/ora). File saltato.")
                return pd.DataFrame()

            df_file = pd.concat(sheet_dfs, ignore_index=True)

            # Per il SECONDO file, elimina la prima riga del DataFrame del file
            if idx == 1 and not df_file.empty:
                df_file = df_file.iloc[1:, :]

            # Normalizza di nuovo il tipo datetime in caso di eterogeneità tra fogli
            df_file[date_col] = pd.to_datetime(df_file[date_col], errors='coerce', utc=True)
            df_file[date_col] = df_file[date_col].dt.tz_localize(None)

            # Pulisce righe con data non interpretabile dalla prima colonna (ora rinominata in date_col)
            n_before = len(df_file)
            df_file = df_file.dropna(subset=[date_col])
            n_after = len(df_file)
            if n_after < n_before:
                print(f"  Rimosse {n_before - n_after} righe con data/ora non valida.")

            # Ordina e rimuove eventuali duplicati sulla chiave temporale
            df_file = df_file.sort_values(by=date_col)
            df_file = df_file.drop_duplicates(subset=[date_col], keep='last')

            print(f"  OK: {df_file.shape[0]} righe dopo concatenazione/cleaning.")
            per_file_dfs.append(df_file)

        # Merge progressivo su date_col (outer per non perdere timestamp)
        df_merged = pd.merge(per_file_dfs[0], per_file_dfs[1], on=date_col, how='outer')

        # Ordina e resetta indice finale
        df_merged = df_merged.sort_values(by=date_col).reset_index(drop=True)

        # Salvataggio su Excel
        if output_path is None:
            base_dir = os.path.dirname(os.path.abspath(file_list[0]))
            output_path = os.path.join(base_dir, 'merged_dataset.xlsx')

        try:
            df_merged.to_excel(output_path, index=False)
            print(f"\nFile Excel finale scritto in: {output_path}")
        except Exception as e_save:
            print(f"Avviso: impossibile salvare il file Excel finale in '{output_path}': {e_save}")

        print("\nDataFrame finale creato con successo.")
        print(f"Dimensioni totali: {df_merged.shape}")

        return df_merged

    except Exception as e:
        print(f"ERRORE GENERICO: Si è verificato un errore: {e}")
        return pd.DataFrame()




 


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


df = carica_e_combina_xlsx(['Data/pv_dataset.xlsx','Data/wx_dataset.xlsx'], date_col='datetime', output_path='Data/final_dataset.xlsx')
