import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .sampler import PVForecastDataset
from ..Utils import plot_cv_indices

# Expanding Window Split Configuration
# fold_1: 12 training months (0-8783) + 4 validation months (8784-11703)
# fold_2: 16 training months (0-11703) + 4 validation months (11704-14623)
# fold_3: 20 training months (0-14623) + 4 validation months (14624-17543)
EXPANDING_WINDOW_SPLITS = [
    (range(0, 8784), range(8784, 11704)),
    (range(0, 11704), range(11704, 14624)),
    (range(0, 14624), range(14624, 17544)),
]


class TS_Cross_Validator:
    """
    Manages time series cross-validation using an expanding window strategy.
    Handles data splitting, scaling, and DataLoader creation for multiple folds.

    Args:
        df (pd.DataFrame): The input dataframe containing the time series data.
        target_col (str): The name of the target column to forecast.
        cfg_dict (dict): Configuration dictionary containing parameters like lookback, horizon, batch_size, etc.

    Attributes:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The target column name.
        lookback (int): Number of past time steps to use as input.
        horizon (int): Number of future time steps to predict.
        batch_size (int): Size of batches for DataLoaders.
        step_train (int): Stride for sampling training sequences.
        step_val (int): Stride for sampling validation sequences.
        n_splits (int): Number of cross-validation splits to perform.
        splits (list): List of (train_indices, val_indices) tuples for the expanding window splits.
    """

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
        self.splits = EXPANDING_WINDOW_SPLITS[
            : self.n_splits
        ]  # it takes the first "n_splits" from the defined global list

    def visualize_splits(self):
        """
        Visualizes the cross-validation folds using a plot.

        This method generates a plot showing the training and validation indices
        for each fold in the expanding window cross-validation scheme.

        Args:
            None

        Returns:
            None: Displays a matplotlib plot.
        """
        print("Generating folds plot...")

        split_generator = (
            (list(train_idx), list(val_idx)) for train_idx, val_idx in self.splits
        )
        plot_cv_indices(split_generator, len(self.df), self.n_splits)
        print("Generation completed")

    def split_and_normalize(self, train_indices, val_indices, experiment_idx):
        """
        Splits the data into training and validation sets based on provided indices and scales them.

        Args:
            train_indices (list): List of indices for the training set.
            val_indices (list): List of indices for the validation set.
            experiment_idx (int): The index of the current fold/experiment.

        Returns:
            tuple: A tuple containing:
                - train_scaled_df (pd.DataFrame): Scaled training dataframe.
                - val_scaled_extended (pd.DataFrame): Scaled validation dataframe, prepended with the last 'lookback' samples from training.
                - scaler (MinMaxScaler): The scaler fitted on the training data.
        """
        # Using .iloc to select the rows corresponding to the indices
        train_df = self.df.iloc[train_indices]
        val_df = self.df.iloc[val_indices]

        # Scaling
        scaler = MinMaxScaler()  # creating a MinMaxScaler object
        train_scaled = scaler.fit_transform(
            train_df
        )  # learning the scaling parameters from the training data
        val_scaled = scaler.transform(
            val_df
        )  # applying the same scaling to the validation data

        # Creating DataFrames from the scaled data, because the scaler returns a numpy array
        train_scaled_df = pd.DataFrame(
            train_scaled, columns=train_df.columns, index=train_df.index
        )
        val_scaled_df = pd.DataFrame(
            val_scaled, columns=val_df.columns, index=val_df.index
        )

        # Adding the last 'lookback' samples from training to the validation set
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
        """
        Iterates through the cross-validation splits and yields DataLoaders for each fold.

        Args:
            None

        Yields:
            tuple: A tuple containing:
                - train_loader (DataLoader): DataLoader for the training set of the current fold.
                - val_loader (DataLoader): DataLoader for the validation set of the current fold.
                - scaler (MinMaxScaler): The scaler fitted on the training set of the current fold.
        """

        print(f"\n=== STARTING CROSS-VALIDATION ({self.n_splits} splits) ===")

        for experiment_idx, (train_indices, val_indices) in enumerate(self.splits):
            splitting_result = self.split_and_normalize(
                list(train_indices), list(val_indices), experiment_idx
            )  # preparing raw data and scaling it for a specific fold

            if splitting_result[0] is None:
                continue

            train_scaled_df, val_scaled_extended, scaler = splitting_result

            # Creating PVForecastDataset instances for train and validation
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

            use_pin_memory = torch.cuda.is_available()  # GPU support check
            # Creating PyTorch DataLoaders for training setting shuffle=True
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=use_pin_memory,
                num_workers=0,
            )
            # Creating PyTorch DataLoaders for validation setting shuffle=False
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=use_pin_memory,
                num_workers=0,
            )

            print(f"Fold {experiment_idx + 1} ready. Yielding...")

            # Returning to the caller the complete package to start training on this fold.
            yield train_loader, val_loader, scaler


# Final Training Split Configuration
# 22 training months (0-16080) + 2 validation months (16081-17543)
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
    Creates training and validation DataLoaders for the final retraining with early stopping.
    The split is fixed: Training on the first 22 months, Validation on the subsequent 2 months.

    Args:
        df (pd.DataFrame): The input dataframe containing the time series data.
        target_col (str): The name of the target column to forecast.
        lookback (int, optional): Number of past time steps to use as input. Defaults to 48.
        horizon (int, optional): Number of future time steps to predict. Defaults to 24.
        batch_size (int, optional): Size of batches for DataLoaders. Defaults to 64.
        step_train (int, optional): Stride for sampling training sequences. Defaults to 1.
        step_val (int, optional): Stride for sampling validation sequences. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the final training set.
            - val_loader (DataLoader): DataLoader for the final validation set.
            - scaler (MinMaxScaler): The scaler fitted on the training data.
    """
    train_range, val_range = FINAL_TRAINING_SPLIT

    # Using .iloc to select the rows corresponding to the indices
    train_df = df.iloc[list(train_range)]
    val_df = df.iloc[list(val_range)]

    scaler = MinMaxScaler()  # creating a MinMaxScaler object
    train_scaled = scaler.fit_transform(
        train_df
    )  # learning the scaling parameters from the training data
    val_scaled = scaler.transform(
        val_df
    )  # applying the same scaling to the validation data

    # Creating DataFrames from the scaled data, because the scaler returns a numpy array
    train_scaled_df = pd.DataFrame(
        train_scaled, columns=train_df.columns, index=train_df.index
    )
    val_scaled_df = pd.DataFrame(val_scaled, columns=val_df.columns, index=val_df.index)

    # Adding the last 'lookback' samples from training to the validation set
    last_train_samples = train_scaled_df.iloc[-lookback:]
    val_scaled_extended = pd.concat([last_train_samples, val_scaled_df], axis=0)

    # Creating PVForecastDataset instances for train and validation
    train_dataset = PVForecastDataset(
        train_scaled_df, target_col, lookback, horizon, step_train
    )
    val_dataset = PVForecastDataset(
        val_scaled_extended, target_col, lookback, horizon, step_val
    )

    use_pin_memory = torch.cuda.is_available()  # GPU support check
    # Creating PyTorch DataLoaders for training setting shuffle=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=use_pin_memory,
        num_workers=0,
    )
    # Creating PyTorch DataLoaders for validation setting shuffle=False
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_pin_memory,
        num_workers=0,
    )

    print("\n FINAL TRAINING SPLIT")
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
    step: int = 1,
):
    """
    Creates a DataLoader for the test set, using a scaler fitted on the entire training data.

    Args:
        train_df (pd.DataFrame): DataFrame containing ALL training data (22+2 months = entire original dataset).
        test_df (pd.DataFrame): DataFrame containing the test data (already preprocessed).
        target_col (str): The name of the target column to forecast.
        lookback (int, optional): Number of past time steps to use as input. Defaults to 24.
        horizon (int, optional): Number of future time steps to predict. Defaults to 24.
        batch_size (int, optional): Size of batches for DataLoaders. Defaults to 64.
        step (int, optional): Stride for sampling test sequences. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - test_loader (DataLoader): DataLoader for the test set.
            - scaler (MinMaxScaler): The scaler fitted on the entire training data (used for denormalization).
    """
    # Fit scaler on ALL training data
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    # Transform test data
    test_scaled = scaler.transform(test_df)
    test_scaled_df = pd.DataFrame(
        test_scaled, columns=test_df.columns, index=test_df.index
    )

    # Create dataset
    test_dataset = PVForecastDataset(
        test_scaled_df, target_col, lookback, horizon, step
    )

    use_pin_memory = torch.cuda.is_available()  # GPU support check
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=use_pin_memory,
        num_workers=0,
    )

    print("\n TEST DATA LOADER")
    print(f"Test: {len(test_df)} rows -> {len(test_dataset)} samples")

    return test_loader, scaler
