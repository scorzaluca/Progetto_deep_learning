import pandas as pd
import datetime


DATA_PATH = "data/raw/wx_dataset.xlsx"
LABEL_PATH = "data/raw/pv_dataset.xlsx"

class Uploader:
    def __init__(self):
        pass

    def _upload_dataset(self, path:str):
        xls = pd.ExcelFile(path)
        dfs = []

        for sh in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sh)
            dfs.append(df)

        # Concatena ignorando l'indice originale
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df
    
    def merge_dataset(self,wx_path,pv_path):
        df_pv=self._upload_dataset(pv_path)
        print(f"Colonne trovate nel PV: {df_pv.columns.tolist()}")
        df_pv.columns=['datetime','pv_power']
        df_wx=self._upload_dataset(wx_path)
        df_list= [df_wx, df_pv]
        df_merged = pd.concat(df_list, axis=1)
        df_merged.drop(columns=['datetime'], inplace=True)  # levo quella del df_pv perchè aveva gli orari non
                                                            # precisi anche se dal file excel non si vedeva
        df_merged['dt_iso'] = pd.to_datetime(df_merged['dt_iso'], utc=True, errors='raise')
        tz_fixed = datetime.timezone(datetime.timedelta(hours=10))    # Sydney (erano misti +10/+11)
        df_merged['dt_iso'] = df_merged['dt_iso'].dt.tz_convert(tz_fixed)  # tutti i record con lo stesso fuso
                                                                        # orario +10 (c'era l'ora legale +11)
        df_merged.to_csv('data/processed/merge_ds.csv')
        return df_merged
    
if __name__ == "__main__":
    df_uploader = Uploader()
    dataset = df_uploader.merge_dataset(DATA_PATH, LABEL_PATH)
    dataset.to_csv("data/raw/merge_ds.csv")
    dataset.to_excel("data/raw/merge_ds.xlsx")