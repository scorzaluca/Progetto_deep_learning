"""
Data Preprocessing Configuration.

Defines input/output paths and column handling for the preprocessing pipeline.
"""

# --- FILE PATHS ---
INPUT_PATH = "data/processed/merge_ds.csv"
OUTPUT_PATH_CSV = "data/processed/preprocessed_ds.csv"
OUTPUT_PATH_EXCEL = "data/processed/preprocessed_ds.xlsx"

# --- COLUMNS ---
DATE_COL = "dt_iso"
COLUMNS_TO_REMOVE = ["lat", "lon", DATE_COL]
DUMMY_COLUMN = "weather_description"
FILLNA_COLUMN = ["rain_1h"]

# --- CONFIG DICTIONARY ---
PREPROCESS_CONFIG = {
    "date_col": DATE_COL,
    "columns_to_remove": COLUMNS_TO_REMOVE,
    "dummy_column": DUMMY_COLUMN,
    "fill_na_column": FILLNA_COLUMN,
}
