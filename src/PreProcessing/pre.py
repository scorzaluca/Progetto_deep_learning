import pandas as pd
import numpy as np








class preprocesser():
    def __init__(self, df: pd.DataFrame, preprocess_config: dict):
        self.df=df
        self.date_col=preprocess_config.get('date_col')
        self.columns_to_remove=preprocess_config.get('columns_to_remove')
        self.dummy_column=preprocess_config.get('dummy_column')
        self.fillna_column=preprocess_config.get('fill_na_column')


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
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
        
        
        # --- 2. Ciclo Giornaliero (basato sull'ora) ---
        # Calcoliamo la frazione del giorno (da 0.0 a 0.999...)
        # Questo calcolo è preciso al secondo.
        day_fraction = (
            self.df[self.date_col].dt.hour / 24.0 + 
            self.df[self.date_col].dt.minute / 1440.0 + 
            self.df[self.date_col].dt.second / 86400.0
        )
        
        # Applichiamo sin/cos
        self.df['day_sin'] = np.sin(2 * np.pi * day_fraction)
        self.df['day_cos'] = np.cos(2 * np.pi * day_fraction)
        
        
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
        self.df['year_sin'] = np.sin(2 * np.pi * year_fraction)
        self.df['year_cos'] = np.cos(2 * np.pi * year_fraction)
        
        return self.df   
    
    def remove_columns(self):
        self.df.drop(self.columns_to_remove, axis=1, inplace=True)
        return self.df
    
    def dummy_variable(self):
        categorie_principali = [
            'sky is clear', 
            'light rain', 
            'overcast clouds', 
            'scattered clouds', 
            'broken clouds', 
            'few clouds', 
            'moderate rain', 
            'haze'
        ]
        # Crea una condizione Booleana: True per i valori che SONO tra i principali
        condizione_principale = self.df[self.dummy_column].isin(categorie_principali)
        # Applica .mask():
        # Dove la condizione è *False* (cioè i valori *non* sono principali), 
        # sostituisci il valore con 'other'. L'assegnazione è fatta in-place.
        self.df[self.dummy_column] = self.df[self.dummy_column].mask(
            ~condizione_principale,  # ~ è l'operatore NOT, quindi seleziona i NON-principali
            other='other'
        )
        # Applica la codifica One-Hot alla colonna modificata
        dummy_cols = pd.get_dummies(self.df[self.dummy_column], 
                                    prefix=self.dummy_column, 
                                    dtype=int)

        # Unisci le nuove colonne dummy al DataFrame
        df = pd.concat([self.df, dummy_cols], axis=1)

        self.df=df.drop(columns=self.dummy_column)


        # Se non ti serve più la colonna originale (che ora ha i valori raggruppati), puoi eliminarla:
        # df.drop(columns=[COLONNA], inplace=True) 

        print("✅ La colonna 'weather_description' è stata sovrascritta con 'other' per i valori rari e le dummy create.")

        return self.df
    
    def fillnan(self):
        self.df[self.fillna_column]=self.df[self.fillna_column].fillna(0)
        return self.df


    
    def run(self):
        self.cyclical_encoding()
        self.remove_columns()
        self.dummy_variable()
        self.fillnan()
        return self.df

        
        
    















    


    



'''#-----------main standalone--------------------------------
pv_path='data/raw/pv_dataset.xlsx'
wx_path='data/raw/wx_dataset.xlsx'


df_uploader=uploader()
df_pv=df_uploader._upload_dataset(pv_path)
print(df_pv.info())
df_pv.columns=['datetime','pv_power']
#df_pv.to_csv('Data/pv_ds.csv')
#print(df_pv)
'''
'''df_wx=upload_dataset(wx_path)
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
df_merged.to_csv('data/processed/merge_ds.csv')


# ENCODING CICLICO
cyclical_df= cyclical_encoding(df_merged)
#cyclical_df.to_csv('data/processed/cyclical_ds.csv')
###############################################


valori_unici_lat = cyclical_df['lat'].unique()
print(valori_unici_lat)
valori_unici_lon = cyclical_df['lon'].unique()
print(valori_unici_lon)

#RIMOZIONE COLONNE LAT E LON
columns_to_remove = ['lat', 'lon','dt_iso']
adjusted_df = remove_columns(cyclical_df, columns_to_remove)
##################################################################

# CREAZIONE DUMMY VARIABLE PER COLONNA WEATHER_DESCRIPTION
COLONNA = 'weather_description'
# Lista delle categorie che DEVI mantenere
categorie_principali = [
    'sky is clear', 
    'light rain', 
    'overcast clouds', 
    'scattered clouds', 
    'broken clouds', 
    'few clouds', 
    'moderate rain', 
    'haze'
]
# Crea una condizione Booleana: True per i valori che SONO tra i principali
condizione_principale = adjusted_df[COLONNA].isin(categorie_principali)
# Applica .mask():
# Dove la condizione è *False* (cioè i valori *non* sono principali), 
# sostituisci il valore con 'other'. L'assegnazione è fatta in-place.
adjusted_df[COLONNA] = adjusted_df[COLONNA].mask(
    ~condizione_principale,  # ~ è l'operatore NOT, quindi seleziona i NON-principali
    other='other'
)
adjusted_df=dummy_variable(adjusted_df, COLONNA)
#############################################################################



adjusted_df['rain_1h'] = adjusted_df['rain_1h'].fillna(0)





# Sostituisci i valori NaN nella colonna 'rain_1h' con 0.
# Questo è utile perché la mancanza di dati sulla pioggia (NaN)
# spesso significa che non c'è stata pioggia (0 mm).
adjusted_df['rain_1h'] = adjusted_df['rain_1h'].fillna(0)



adjusted_df.to_csv('data/processed/adjusted_ds.csv')

adjusted_df.to_excel('data/processed/adjusted_ds.xlsx')


print(adjusted_df.values)'''