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
        """Applica l'encoding ciclico per ciclo giornaliero e annuale."""
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")

        # Ciclo giornaliero
        day_fraction = (
            self.df[self.date_col].dt.hour / 24.0
            + self.df[self.date_col].dt.minute / 1440.0
            + self.df[self.date_col].dt.second / 86400.0
        )
        self.df["day_sin"] = np.sin(2 * np.pi * day_fraction)
        self.df["day_cos"] = np.cos(2 * np.pi * day_fraction)

        # Ciclo annuale
        year_fraction = self.df[self.date_col].dt.dayofyear / 365.0
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
        condizione_principale = self.df[self.dummy_column].isin(categorie_principali)
        self.df[self.dummy_column] = self.df[self.dummy_column].mask(
            ~condizione_principale, other="other"
        )

        dummy_cols = pd.get_dummies(
            self.df[self.dummy_column], prefix=self.dummy_column, dtype=int
        )
        df = pd.concat([self.df, dummy_cols], axis=1)
        self.df = df.drop(columns=self.dummy_column)

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
    adjusted_df.to_csv("data/processed/adjusted_ds.csv", index=False)
