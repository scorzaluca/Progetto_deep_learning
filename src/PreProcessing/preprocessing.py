import pandas as pd
import numpy as np


DATE_COL = "dt_iso"
COLUMNS_TO_REMOVE = ["lat", "lon", DATE_COL]
DUMMY_COLUMN = "weather_description"
FILLNA_COLUMN = ["rain_1h"]

PREPROCESS_CONFIG = {
    "date_col": DATE_COL,
    "columns_to_remove": COLUMNS_TO_REMOVE,
    "dummy_column": DUMMY_COLUMN,
    "fill_na_column": FILLNA_COLUMN,
}


class Preprocesser:
    def __init__(self, df: pd.DataFrame, preprocess_config: dict):
        self.df = df
        self.date_col = preprocess_config.get("date_col")
        self.columns_to_remove = preprocess_config.get("columns_to_remove")
        self.dummy_column = preprocess_config.get("dummy_column")
        self.fillna_column = preprocess_config.get("fill_na_column")

    def cyclical_encoding(self):
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
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")

        # --- 2. Ciclo Giornaliero (basato sull'ora) ---
        # Calcoliamo la frazione del giorno (da 0.0 a 0.999...)
        # Questo calcolo è preciso al secondo.
        day_fraction = (
            self.df[self.date_col].dt.hour / 24.0
            + self.df[self.date_col].dt.minute / 1440.0
            + self.df[self.date_col].dt.second / 86400.0
        )

        # Applichiamo sin/cos
        self.df["day_sin"] = np.sin(2 * np.pi * day_fraction)
        self.df["day_cos"] = np.cos(2 * np.pi * day_fraction)

        # --- 3. Ciclo Annuale (basato sul giorno dell'anno) ---
        # Usiamo .dt.dayofyear (es. 1 per 1° Gen, 365 per 31 Dic)
        # Normalizziamo dividendo semplicemente per 365.

        # Motivazione della Semplificazione:
        # Si assume che il ciclo si ripeta ogni 365 giorni.
        # L'attributo .dt.dayofyear va da 1 a 365 (o 366 nei bisestili).
        # Dividendo per 365, il valore sarà circa [0.0027, 1.0] in un anno normale
        # e circa [0.0027, 1.0027] in un anno bisestile.
        # L'impatto di questa piccola differenza è trascurabile per il modello.

        year_fraction = self.df[self.date_col].dt.dayofyear / 365.0

        # Applichiamo sin/cos
        self.df["year_sin"] = np.sin(2 * np.pi * year_fraction)
        self.df["year_cos"] = np.cos(2 * np.pi * year_fraction)

        return self.df

    def remove_columns(self):
        self.df.drop(self.columns_to_remove, axis=1, inplace=True)
        return self.df

    def dummy_variable(self):
        categorie_principali = [
            "sky is clear",
            "light rain",
            "overcast clouds",
            "scattered clouds",
            "broken clouds",
            "few clouds",
            "moderate rain",
            "haze",
        ]
        # Crea una condizione Booleana: True per i valori che SONO tra i principali
        condizione_principale = self.df[self.dummy_column].isin(categorie_principali)

        # Applica .mask():
        # Dove la condizione è *False* (cioè i valori *non* sono principali),
        # sostituisci il valore con 'other'. L'assegnazione è fatta in-place.
        self.df[self.dummy_column] = self.df[self.dummy_column].mask(
            ~condizione_principale, other="other"
        )
        # ~ è l'operatore NOT, quindi seleziona i NON-principali

        # Applica la codifica One-Hot alla colonna modificata
        dummy_cols = pd.get_dummies(
            self.df[self.dummy_column], prefix=self.dummy_column, dtype=int
        )

        # Unisci le nuove colonne dummy al DataFrame
        df = pd.concat([self.df, dummy_cols], axis=1)

        self.df = df.drop(columns=self.dummy_column)

        # Se non ti serve più la colonna originale (che ora ha i valori raggruppati), puoi eliminarla:
        # df.drop(columns=[COLONNA], inplace=True)

        print(
            "La colonna 'weather_description' è stata sovrascritta con 'other' per i valori rari e le dummy create."
        )

        return self.df

    def fillnan(self):
        self.df[self.fillna_column] = self.df[self.fillna_column].fillna(0)
        return self.df

    def run(self):
        self.cyclical_encoding()
        self.remove_columns()
        self.dummy_variable()
        self.fillnan()
        return self.df


if __name__ == "__main__":
    dataset = pd.read_csv("data/raw/merge_ds.csv")
    preprocess = Preprocesser(dataset, PREPROCESS_CONFIG)
    adjusted_df = preprocess.run()
    print(list(adjusted_df.columns).index("pv_power"))
    adjusted_df.to_csv("data/processed/Adjusted_ds.csv")
    adjusted_df.to_excel("data/processed/Adjusted_ds.xlsx")
