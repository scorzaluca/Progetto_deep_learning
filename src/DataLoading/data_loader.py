import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .sampler import PVForecastDataset
from ..Utils import plot_cv_indices

# Expanding Window Split Configuration
# fold_1: 12 mesi training (0-8783) + 4 mesi validation (8784-11703)
# fold_2: 16 mesi training (0-11703) + 4 mesi validation (11704-14623)
# fold_3: 20 mesi training (0-14623) + 4 mesi validation (14624-17543)
EXPANDING_WINDOW_SPLITS = [
    (range(0, 8784), range(8784, 11704)),  # Fold 1: 12 mesi train, 4 mesi val
    (range(0, 11704), range(11704, 14624)),  # Fold 2: 16 mesi train, 4 mesi val
    (range(0, 14624), range(14624, 17544)),  # Fold 3: 20 mesi train, 4 mesi val
]


class TS_Cross_Validator:
    def __init__(self, df: pd.DataFrame, target_col: str, cfg_dict: dict):
        self.df = df
        self.target_col = target_col
        self.lookback = cfg_dict.get("lookback", 48)
        self.horizon = cfg_dict.get("horizon", 24)
        self.batch_size = cfg_dict.get("batch_size", 64)
        self.step_train = cfg_dict.get("step_train", 1)
        self.step_val = cfg_dict.get("step_val", 1)
        self.n_splits = cfg_dict.get("n_splits", 3)

        # Custom expanding window splits
        self.splits = EXPANDING_WINDOW_SPLITS[: self.n_splits]

    def visualize_splits(self):
        """
        Chiama questo metodo per vedere il grafico dei fold PRIMA del training.
        """
        print("Generazione grafico dei fold in corso...")
        # Creiamo un generatore per il plot
        split_generator = (
            (list(train_idx), list(val_idx)) for train_idx, val_idx in self.splits
        )
        plot_cv_indices(split_generator, len(self.df), self.n_splits)
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
        # Usiamo gli split custom
        print(f"\n=== INIZIO CROSS-VALIDATION ({self.n_splits} splits) ===")

        for experiment_idx, (train_indices, val_indices) in enumerate(self.splits):
            splitting_result = self.split_and_normalize(
                list(train_indices), list(val_indices), experiment_idx
            )

            if splitting_result[0] is None:
                continue

            train_scaled_df, val_scaled_extended, scaler = splitting_result

            # Datasets (step diversi per train e validation)
            train_dataset = PVForecastDataset(
                train_scaled_df,
                self.target_col,
                self.lookback,
                self.horizon,
                self.step_train,
            )
            val_dataset = PVForecastDataset(
                val_scaled_extended,
                self.target_col,
                self.lookback,
                self.horizon,
                self.step_val,
            )

            # DataLoaders con pin_memory per GPU speedup
            use_pin_memory = torch.cuda.is_available()
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=use_pin_memory,
                num_workers=0,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=use_pin_memory,
                num_workers=0,
            )

            print(f"Fold {experiment_idx + 1} pronto. Yielding...")

            # 2. FIX: Yield corretto (3 oggetti). Niente liste, niente append.
            yield train_loader, val_loader, scaler


# Final Training Split Configuration
# 22 mesi training (0-16080) + 2 mesi validation (16081-17543)
FINAL_TRAINING_SPLIT = (range(0, 16081), range(16081, 17544))


def create_final_train_val_loaders(
    df: pd.DataFrame,
    target_col: str,
    lookback: int = 48,
    horizon: int = 24,
    batch_size: int = 64,
    step_train: int = 1,
    step_val: int = 1,
):
    """
    Crea train/val DataLoaders per il retraining finale con early stopping.
    Training: 22 mesi (0-16080), Validation: 2 mesi (16081-17543)

    Args:
        df: DataFrame con i dati
        target_col: nome colonna target
        lookback: finestra di input
        horizon: finestra di output
        batch_size: dimensione batch
        step_train: passo tra campioni per training
        step_val: passo tra campioni per validation

    Returns:
        Tuple[DataLoader, DataLoader, MinMaxScaler]: train_loader, val_loader, scaler
    """
    train_range, val_range = FINAL_TRAINING_SPLIT

    # Split data
    train_df = df.iloc[list(train_range)]
    val_df = df.iloc[list(val_range)]

    # Scaling (fit su train, transform su val)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)

    train_scaled_df = pd.DataFrame(
        train_scaled, columns=train_df.columns, index=train_df.index
    )
    val_scaled_df = pd.DataFrame(val_scaled, columns=val_df.columns, index=val_df.index)

    # Estendi validation con ultimi lookback campioni del training
    # (necessario per creare i primi campioni di validation)
    last_train_samples = train_scaled_df.iloc[-lookback:]
    val_scaled_extended = pd.concat([last_train_samples, val_scaled_df], axis=0)

    # Crea datasets (step diversi per train e validation)
    train_dataset = PVForecastDataset(
        train_scaled_df, target_col, lookback, horizon, step_train
    )
    val_dataset = PVForecastDataset(
        val_scaled_extended, target_col, lookback, horizon, step_val
    )

    # Crea DataLoaders
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=use_pin_memory,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_pin_memory,
        num_workers=0,
    )

    print("\n=== FINAL TRAINING SPLIT ===")
    print(f"Train: {len(train_range)} rows -> {len(train_dataset)} samples")
    print(f"Val:   {len(val_range)} rows -> {len(val_dataset)} samples")

    return train_loader, val_loader, scaler


def create_test_loader(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    lookback: int = 24,
    horizon: int = 24,
    batch_size: int = 64,
):
    """
    Crea DataLoader per il test set con scaler fittato su tutti i dati di training.

    IMPORTANTE: Lo scaler viene fittato su TUTTO il training data (train+val originali),
    non solo sul train set. Questo è il modo corretto per fare inferenza su dati nuovi.

    Args:
        train_df: DataFrame con TUTTI i dati di training (22+2 mesi = tutto il dataset originale)
        test_df: DataFrame con i dati di test (già preprocessati)
        target_col: nome colonna target
        lookback: finestra di input
        horizon: finestra di output
        batch_size: dimensione batch

    Returns:
        Tuple[DataLoader, MinMaxScaler]: test_loader, scaler (per denormalizzazione)
    """
    # Fit scaler su TUTTO il training data
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    # Transform test data
    test_scaled = scaler.transform(test_df)
    test_scaled_df = pd.DataFrame(
        test_scaled, columns=test_df.columns, index=test_df.index
    )

    # Crea dataset
    test_dataset = PVForecastDataset(
        test_scaled_df, target_col, lookback, horizon, step=1
    )

    # Crea DataLoader
    use_pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_pin_memory,
        num_workers=0,
    )

    print("\n=== TEST DATA LOADER ===")
    print(f"Test: {len(test_df)} rows -> {len(test_dataset)} samples")

    return test_loader, scaler
