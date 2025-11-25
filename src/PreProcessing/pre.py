import pandas as pd
import datetime

def upload_dataset(path):
    xls = pd.ExcelFile(path)
    dfs = []

    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh)
        dfs.append(df)

    # Concatena ignorando l'indice originale
    result_df = pd.concat(dfs, ignore_index=True)
    return result_df

pv_path='Data/pv_dataset.xlsx'
wx_path='Data/wx_dataset.xlsx'

df_pv=upload_dataset(pv_path)
df_pv.columns=['datetime','pv_power']
#df_pv.to_csv('Data/pv_ds.csv')
#print(df_pv)

df_wx=upload_dataset(wx_path)
#df_wx.to_csv('Data/wx_ds.csv')
#print(df_wx)

# df_merged = pd.merge(df_wx, df_pv)
df_list= [df_wx, df_pv]
df_merged = pd.concat(df_list, axis=1)
df_merged.drop(columns=['datetime'], inplace=True)  # levo quella del df_pv perchè aveva gli orari non
                                                    # precisi anche se dal file excel non si vedeva
df_merged['dt_iso'] = pd.to_datetime(df_merged['dt_iso'], utc=True, errors='raise')
tz_fixed = datetime.timezone(datetime.timedelta(hours=10))    # Sydney (erano misti +10/+11)
df_merged['dt_iso'] = df_merged['dt_iso'].dt.tz_convert(tz_fixed)  # tutti i record con lo stesso fuso
                                                                   # orario +10 (c'era l'ora legale +11)
df_merged.to_csv('Data/merge_ds.csv')
#df_merged.to_excel('Data/merge_ds.xlsx') # il tipo datetime rompe il cazzo con excel
print(df_merged)
print(df_merged.loc[2260][0])
print(df_merged.dtypes)
