
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from .sampler import PVForecastDataset

class TS_Cross_Validator():
    def __init__(self,df:pd.DataFrame, target_col:str, cfg_dict:dict):
        self.df=df
        self.target_col=target_col
        self.lookback=cfg_dict.get('lookback',48)
        self.horizon=cfg_dict.get('horizon',24)
        self.batch_size=cfg_dict.get('batch_size',64)
        self.step=cfg_dict.get('step',1)
        self.n_splits=cfg_dict.get('n_splits',3)

        self.folds_data = []

    def _index(self):
        tscv = TimeSeriesSplit(self.n_splits)
        print(f"\n=== INIZIO ISPEZIONE ({self.n_splits} splits) ===")
        print(f"Totale righe dataset: {len(self.df)}")
        return tscv.split(self.df)
    
    def split_and_normalize(self,train_indices, val_indices,experiment_idx):
        # Split Dataframes
        train_df = self.df.iloc[train_indices]
        val_df = self.df.iloc[val_indices]
        # --- Logging ---
        print(f"\n---------------- FOLD {experiment_idx} ----------------")
        # Dimensions
        print(f"SHAPE -> Train: {train_df.shape} | Val: {val_df.shape}")
        # Temporal window (Date)
        print(f"TRAIN -> Da: {train_df.index[0]}  A: {train_df.index[-1]}")
        print(f"VAL   -> Da: {val_df.index[0]}  A: {val_df.index[-1]}")
        # L'ultimo istante del train deve essere subito prima del primo del val
        print(f"Contiguità: Fine Train ({train_df.index[-1]}) -> Inizio Val ({val_df.index[0]})")
        # Dimension Control
        if len(train_df) <= self.lookback + self.horizon:
            print(f"Skipping fold {experiment_idx}: Train set troppo piccolo.")
            return None, None, None

        # scaling
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df) #array numpy
        val_scaled = scaler.transform(val_df) #array numpy

        #returning to dataframe
        train_scaled_df = pd.DataFrame(
            train_scaled, 
            columns=train_df.columns, 
            index=train_df.index
        )
        #returning to dataframe
        val_scaled_df = pd.DataFrame(
            val_scaled, 
            columns=val_df.columns, 
            index=val_df.index
        )

        #overlapping
        last_train_samples = train_scaled_df.iloc[-self.lookback:]
        val_scaled_extended = pd.concat([last_train_samples, val_scaled_df], axis=0)

        return train_scaled_df, val_scaled_extended, scaler


    def get_folds(self):
        index=self._index()
        for experiment_idx, (train_indices, val_indices) in enumerate(index):
            splitting_result = self.split_and_normalize(train_indices, val_indices,experiment_idx)

            if splitting_result[0] is None:
                continue

            train_scaled_df, val_scaled_extended, scaler= splitting_result
            
            # Creating Datasets
            train_dataset = PVForecastDataset(train_scaled_df, self.target_col, self.lookback, self.horizon, self.step)
            val_dataset = PVForecastDataset(val_scaled_extended, self.target_col, self.lookback, self.horizon, self.step)


            # Creating DataLoaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                drop_last=True
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                drop_last=False
            )

            self.folds_data.append((train_loader, val_loader,scaler))
        
            print(f"Fold {experiment_idx+1} creato -> Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
                
        return self.folds_data
