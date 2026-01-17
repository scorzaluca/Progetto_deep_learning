import pandas as pd
import datetime





class Uploader:
    """
    Handles the initial data ingestion and merging of Raw Datasets.

    This class reads the raw Excel files,
    concatenates sheets if necessary, merges them into a single dataframe,
    fixes timezone issues, and saves the result to CSV/Excel.
    """
    def __init__(self):
        pass

    def _upload_dataset(self, path: str):
        """
        Reads all sheets from an Excel file and concatenates them.

        Args:
            path (str): File path to the Excel dataset.

        Returns:
            pd.DataFrame: A single DataFrame containing data from all sheets.
        """
        xls = pd.ExcelFile(path)
        dfs = []

        # Iterate over all sheets in the Excel file
        for sh in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sh)
            dfs.append(df)

        # Concatenate all sheets into one, ignoring the original row indices
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df

    def merge_dataset(self, wx_path, pv_path, output_path_csv, output_path_excel):
        """
        Merges Weather and PV Power datasets aligned by timestamp.

        It performs the following operations:
        1. Loads both datasets.
        2. Aligns them horizontally (assuming row-wise correspondence).
        3. Fixes timezone inconsistencies (Standardizes to UTC+10 Sydney).
        4. Saves the merged dataset to CSV and Excel.

        Args:
            wx_path (str): Path to Weather dataset.
            pv_path (str): Path to PV dataset.
            output_path_csv (str): Output string path for CSV.
            output_path_excel (str): Output string path for Excel.
        """
        # Load and rename columns for PV dataset
        df_pv = self._upload_dataset(pv_path)
        print(f"Columns found in PV: {df_pv.columns.tolist()}")
        df_pv.columns = ["datetime", "pv_power"]
        
        # Load Weather dataset
        df_wx = self._upload_dataset(wx_path)
        print(f"Columns found in WX: {df_wx.columns.tolist()}")
        
        # Merge datasets side-by-side (axis=1)
        df_list = [df_wx, df_pv]
        df_merged = pd.concat(df_list, axis=1)
        
        # Drop the duplicate datetime column from the PV dataset
        df_merged.drop(
            columns=["datetime"], inplace=True
        )  
        
        # --- Timezone Standardization ---
        # Convert the main ISO date column to datetime objects
        df_merged["dt_iso"] = pd.to_datetime(
            df_merged["dt_iso"], utc=True, errors="raise"
        )
        
        # Define Sydney Timezone object (UTC+10)
        tz_fixed = datetime.timezone(
            datetime.timedelta(hours=10)
        )
        
        # Convert all timestamps to this fixed timezone
        df_merged["dt_iso"] = df_merged["dt_iso"].dt.tz_convert(
            tz_fixed
        )
        
        # Save to CSV
        df_merged.to_csv(output_path_csv, index=False)
        print(f"Merged dataset saved in csv format: {output_path_csv}")

        # Save to Excel
        df_merged["dt_iso"] = df_merged["dt_iso"].dt.tz_localize(None)
        df_merged.to_excel(output_path_excel, index=False)
        print(f"Merged dataset saved in excel format: {output_path_excel}")
        return df_merged


if __name__ == "__main__":
    # Configuration for Standalone Execution
    DATA_PATH = "data/raw/wx_dataset.xlsx"
    LABEL_PATH = "data/raw/pv_dataset.xlsx"
    OUTPUT_PATH_CSV = "data/processed/merge_ds.csv"
    OUTPUT_PATH_EXCEL = "data/processed/merge_ds.xlsx"

    df_uploader = Uploader()
    dataset = df_uploader.merge_dataset(
        DATA_PATH, LABEL_PATH, OUTPUT_PATH_CSV, OUTPUT_PATH_EXCEL
    )
