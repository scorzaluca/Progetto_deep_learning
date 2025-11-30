from Uploader import uploader
from pre import preprocesser
import sys
import os

# 1. Trovo la cartella dove sono ora (src/preprocessing)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Torno indietro di due cartelle per arrivare alla root (src -> root)
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 3. Aggiungo la root al percorso dove Python cerca i file
sys.path.append(root_dir)

# 4. ORA posso importare config senza errori
import config


import numpy as np
import pandas as pd

df_uploader= uploader()
dataset= df_uploader.merge_dataset(config.DATA_PATH, config.LABEL_PATH)
#df_pv.to_csv('Data/pv_ds.csv')
#print(df_pv)
dataset.to_csv('data/raw/merge_ds.csv')
print(dataset)



preprocess= preprocesser(dataset, config.PREPROCESS_CONFIG)

adjusted_df=preprocess.run()


adjusted_df.to_csv('data/processed/Adjusted_ds.csv')

adjusted_df.to_excel('data/processed/Adjusted_ds.xlsx')

