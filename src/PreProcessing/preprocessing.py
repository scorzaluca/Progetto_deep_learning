import pandas as pd
import numpy as np




class Preprocesser:
    """
    Handles data preprocessing steps for PV power forecasting.

    This class encapsulates the entire preprocessing pipeline, including:
    - Cyclical encoding of time features (day, year).
    - Removal of unnecessary columns.
    - One-hot encoding of categorical weather variables.
    - Handling missing values.
    - Physical consistency checks (Solar Irradiance filtering).
    - Reordering columns for training compatibility.

    Args:
        df (pd.DataFrame): The raw input dataframe.
        preprocess_config (dict): Configuration dictionary specifying columns to process.

    Attributes:
        df (pd.DataFrame): The dataframe being processed.
        date_col (str): Name of the date/time column.
        columns_to_remove (list): List of column names to drop.
        dummy_column (str): Categorical column to one-hot encode.
        fillna_column (list): List of columns to fill NaNs with 0.
    """
    def __init__(self, df: pd.DataFrame, preprocess_config: dict):
        self.df = df
        
        # Extract configuration parameters from the dictionary
        self.date_col = preprocess_config.get("date_col")
        self.columns_to_remove = preprocess_config.get("columns_to_remove")
        self.dummy_column = preprocess_config.get("dummy_column")
        self.fillna_column = preprocess_config.get("fill_na_column")

    def cyclical_encoding(self):
        """
        Applies cyclical encoding to temporal features (daily and yearly cycles).

        Converts time (hour/day) into Sine and Cosine components.
        This preserves the cyclical nature of time
        - Daily cycle: Encodes time of day.
        - Yearly cycle: Encodes day of year (handling leap years).

        Returns:
            pd.DataFrame: The dataframe with added sin/cos columns.
        """
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors="coerce")

        # --- Daily Cycle Encoding ---
        # Convert time of day into a fraction of 24 hours (0 to 1).
        day_fraction = (
            self.df[self.date_col].dt.hour / 24.0
            + self.df[self.date_col].dt.minute / 1440.0
            + self.df[self.date_col].dt.second / 86400.0
        )
        # Create Sine/Cosine features for the day cycle
        self.df["day_sin"] = np.sin(2 * np.pi * day_fraction)
        self.df["day_cos"] = np.cos(2 * np.pi * day_fraction)

        # --- Yearly Cycle Encoding ---
        # Determine the length of the year (366 for leap years, 365 otherwise)
        days_in_year = self.df[self.date_col].dt.is_leap_year.map(
            {True: 366, False: 365}
        )
        # Create Sine/Cosine features for the year cycle
        year_fraction = self.df[self.date_col].dt.dayofyear / days_in_year
        self.df["year_sin"] = np.sin(2 * np.pi * year_fraction)
        self.df["year_cos"] = np.cos(2 * np.pi * year_fraction)

        return self.df

    def remove_columns(self):
        """
        Drops uninformative or redundant columns specified in the configuration.
        Example: Latitude and Longitude (static values)

        Returns:
            pd.DataFrame: The dataframe without the removed columns.
        """
        # Drop columns in-place for efficiency
        self.df.drop(self.columns_to_remove, axis=1, inplace=True)
        return self.df

    def dummy_variable(self):
        """
        Creates One-Hot Encoded (Dummy) variables for categorical weather data.

        It implements a strategy to reduce cardinality:
        1. Keeps only the main weather categories.
        2. Groups rare categories into 'other'.
        3. Creates dummy variables.
        4. Drops the 'other' column to avoid multicollinearity (Dummy Variable Trap).

        Returns:
            pd.DataFrame: Dataframe with new dummy columns and without the original categorical column.
        """
        # Define main categories to keep explicit
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
        
        # Group rare categories into 'other'
        # If the value is NOT in the main list, replace it with 'other'
        condizione_principale = self.df[self.dummy_column].isin(categorie_principali)
        self.df[self.dummy_column] = self.df[self.dummy_column].mask(
            ~condizione_principale, other="other"
        )

        # Create dummy variables
        dummy_cols = pd.get_dummies(
            self.df[self.dummy_column], prefix=self.dummy_column, dtype=int
        )

        # Avoid Multicollinearity:
        # We drop one column, which acts as the 'baseline' reference.
        # Here we choose to drop the 'other' class if it exists.
        column_to_drop = f"{self.dummy_column}_other"
        if column_to_drop in dummy_cols.columns:
            dummy_cols = dummy_cols.drop(columns=column_to_drop)

        # Concatenate dummy columns and remove the original categorical column
        df = pd.concat([self.df, dummy_cols], axis=1)
        self.df = df.drop(columns=self.dummy_column)

        return self.df

    def fillnan(self):
        """
        Fills missing values (NaN) with zeros for specified columns (e.g., rain).
        """

        self.df[self.fillna_column] = self.df[self.fillna_column].fillna(0)
        return self.df

    def reorder_columns(self, target_col: str = "pv_power"):
        """
        Reorders the dataframe columns, placing the target column at the end.
        This simplifies slicing operations during dataset creation.

        Args:
            target_col (str): The name of the target column to move.
        """
        # Create a list of all columns except the target
        cols = [c for c in self.df.columns if c != target_col]
        # Append the target column to the end of the list
        cols.append(target_col)
        # Reorder the dataframe
        self.df = self.df[cols]
        return self.df

    def night_filter(self):
        """
        Enforces physical consistency by setting PV power to 0 when Solar Irradiance (GHI) is 0.
        This corrects small sensor noise/errors recorded during the night.
        """
        # Create a boolean mask where GHI (Global Horizontal Irradiance) is 0 (Night time)
        mask = self.df["Ghi"] == 0
        # Force pv_power to 0 for these rows
        self.df.loc[mask, "pv_power"] = 0
        return self.df

    def run(self):
        """
        Executes the full preprocessing pipeline in the correct order.

        Returns:
            pd.DataFrame: The fully processed dataframe ready for training.
        """
        # 1. Encode cyclical time features (daily/yearly)
        self.cyclical_encoding()
        # 2. Remove static or unused columns
        self.remove_columns()
        # 3. One-hot encode categorical weather description
        self.dummy_variable()
        # 4. Fill missing values (e.g. rain)
        self.fillnan()
        # 5. Apply physical filters (night time zero solar)
        self.night_filter()
        # 6. Reorder columns (target last)
        self.reorder_columns()
        
        return self.df


if __name__ == "__main__":
    # Configuration for Standalone Execution
    # Define input and output file paths
    INPUT_PATH = "data/processed/merge_ds.csv"
    OUTPUT_PATH_CSV = "data/processed/preprocessed_ds.csv"
    OUTPUT_PATH_EXCEL = "data/processed/preprocessed_ds.xlsx"

    # Define column names for preprocessing steps
    DATE_COL = "dt_iso"
    COLUMNS_TO_REMOVE = ["lat", "lon", DATE_COL]
    DUMMY_COLUMN = "weather_description"
    FILLNA_COLUMN = ["rain_1h"]

    # Assemble the preprocessing configuration dictionary
    PREPROCESS_CONFIG = {
        "date_col": DATE_COL,
        "columns_to_remove": COLUMNS_TO_REMOVE,
        "dummy_column": DUMMY_COLUMN,
        "fill_na_column": FILLNA_COLUMN,
    }

    # Load the dataset
    dataset = pd.read_csv(INPUT_PATH)
    # Initialize the Preprocesser with the dataset and configuration
    preprocess = Preprocesser(dataset, PREPROCESS_CONFIG)
    # Run the full preprocessing pipeline
    preprocessed_df = preprocess.run()
    
    # Save results
    # Save the preprocessed dataframe to a CSV file
    preprocessed_df.to_csv(OUTPUT_PATH_CSV, index=False)
    print(f"Preprocessed dataset saved in csv format: {OUTPUT_PATH_CSV}")
    # Save the preprocessed dataframe to an Excel file
    preprocessed_df.to_excel(OUTPUT_PATH_EXCEL, index=False)
    print(f"Preprocessed dataset saved in excel format: {OUTPUT_PATH_EXCEL}")
