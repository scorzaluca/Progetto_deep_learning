from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from .sampler import PVForecastDataset
from ..Utils import plot_cv_indices


class TS_Cross_Validator:
    def __init__(self, df: pd.DataFrame, target_col: str, cfg_dict: dict):
        self.df = df
        self.target_col = target_col
        self.lookback = cfg_dict.get("lookback", 48)
        self.horizon = cfg_dict.get("horizon", 24)
        self.batch_size = cfg_dict.get("batch_size", 64)
        self.step = cfg_dict.get("step", 1)
        self.n_splits = cfg_dict.get("n_splits", 3)

        # 1. FIX IMPORTANTE: Inizializzalo qui!
        # Così puoi chiamare visualize_splits() SUBITO, senza aspettare il training.
        self.tscv = TimeSeriesSplit(self.n_splits)

        # self.folds_data = []  <-- RIMOSSO (Inutile spreco di RAM)

    def visualize_splits(self):
        """
        Chiama questo metodo per vedere il grafico dei fold PRIMA del training.
        """
        print("Generazione grafico dei fold in corso...")
        # Creiamo un generatore temporaneo solo per il plot
        plot_cv_indices(self.tscv.split(self.df), len(self.df), self.n_splits)
        print("Grafico chiuso. Pronto per il training.")

    def split_and_normalize(self, train_indices, val_indices, experiment_idx):
        # (Questa parte va bene, è logica pura)
        train_df = self.df.iloc[train_indices]
        val_df = self.df.iloc[val_indices]

        # Scaling
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df)
        val_scaled = scaler.transform(val_df)

        train_scaled_df = pd.DataFrame(
            train_scaled, columns=train_df.columns, index=train_df.index
        )
        val_scaled_df = pd.DataFrame(
            val_scaled, columns=val_df.columns, index=val_df.index
        )

        last_train_samples = train_scaled_df.iloc[-self.lookback :]
        val_scaled_extended = pd.concat([last_train_samples, val_scaled_df], axis=0)

        # --- Logging ---
        print(f"\n---------------- FOLD {experiment_idx + 1} ----------------")
        print(f"TRAIN: {train_scaled_df.index[0]} -> {train_scaled_df.index[-1]}")
        print(
            f"VAL  : {val_scaled_extended.index[0]} -> {val_scaled_extended.index[-1]}"
        )

        return train_scaled_df, val_scaled_extended, scaler

    def get_folds(self):
        # Usiamo lo splitter già pronto
        split_generator = self.tscv.split(self.df)

        print(f"\n=== INIZIO CROSS-VALIDATION ({self.n_splits} splits) ===")

        for experiment_idx, (train_indices, val_indices) in enumerate(split_generator):
            splitting_result = self.split_and_normalize(
                train_indices, val_indices, experiment_idx
            )

            if splitting_result[0] is None:
                continue

            train_scaled_df, val_scaled_extended, scaler = splitting_result

            # Datasets
            train_dataset = PVForecastDataset(
                train_scaled_df, self.target_col, self.lookback, self.horizon, self.step
            )
            val_dataset = PVForecastDataset(
                val_scaled_extended,
                self.target_col,
                self.lookback,
                self.horizon,
                self.step,
            )

            # DataLoaders
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
            )

            print(f"Fold {experiment_idx + 1} pronto. Yielding...")

            # 2. FIX: Yield corretto (3 oggetti). Niente liste, niente append.
            yield train_loader, val_loader, scaler


def create_full_dataloader(
    df: pd.DataFrame,
    target_col: str,
    lookback: int = 48,
    horizon: int = 24,
    batch_size: int = 64,
    step: int = 1,
) -> DataLoader:
    """
    Crea un DataLoader con TUTTI i dati (per retraining finale).
    Applica MinMaxScaler su tutto il dataset.

    Args:
        df: DataFrame con i dati
        target_col: nome colonna target
        lookback: finestra di input
        horizon: finestra di output
        batch_size: dimensione batch
        step: passo tra campioni

    Returns:
        DataLoader con tutti i dati normalizzati
    """
    # Normalizza tutto il dataset
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    # Crea dataset
    dataset = PVForecastDataset(scaled_df, target_col, lookback, horizon, step)

    # Crea DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    print(f"Full DataLoader creato: {len(dataset)} campioni, {len(dataloader)} batch")

    return dataloader
