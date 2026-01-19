# ⚡ PV Power Forecasting - Deep Learning Project

A comprehensive **Time Series Forecasting** pipeline for predicting Photovoltaic (PV) power generation 24 hours ahead. This project implements state-of-the-art deep learning architectures, **Self-Supervised Pre-training** via Masked Autoencoders, **Transfer Learning**, and **Bayesian Hyperparameter Optimization** with Optuna.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

---

## Table of Contents

- [Problem Definition](#-problem-definition)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Preprocessing](#-preprocessing)
- [Data Loading](#-data-loading)
- [Model Architectures](#-model-architectures)
- [Self-Supervised Pre-training](#-self-supervised-pre-training)
- [Training & Optimization](#-training--optimization)
- [Evaluation Metrics](#-evaluation-metrics)
- [Baselines](#-baselines)
- [Inference](#-inference)
- [Results](#-results)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Notebooks](#-notebooks)
- [Installation](#-installation)

---

## Problem Definition

### Task: Day-Ahead PV Power Forecasting

Given historical data, predict the **next 24 hours** of PV power production for a solar installation in **Sydney, Australia**.

| Parameter | Value |
|-----------|-------|
| **Location** | Sydney, Australia (-33.87°, 151.21°) |
| **Lookback Window** | 48 hours (selected via ablation study) |
| **Forecast Horizon** | 24 hours (1 day ahead) |
| **Resolution** | Hourly |
| **Target Variable** | `pv_power` (kW) |

### Formal Definition

```
Input:  X = [x_{t-47}, x_{t-46}, ..., x_{t}]   → 48 timesteps × 24 features
Output: Y = [y_{t+1}, y_{t+2}, ..., y_{t+24}]  → 24 timesteps × 1 target (pv_power)
```

### Challenges
- **Seasonality**: Daily and yearly cycles in solar irradiance
- **Weather Dependence**: Cloud cover, rain, and atmospheric conditions
- **Night Periods**: Zero production during nighttime hours
- **Distribution Shift**: Test data may have different statistics than training data

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Support** | LSTM, DLinear, TCN, PatchTST, EncoderLSTM |
| **Self-Supervised Pre-training** | Masked Patch Reconstruction with PatchTST |
| **Transfer Learning** | Fine-tuning pre-trained encoders for forecasting |
| **Bayesian Optimization** | TPE sampler with Optuna + SQLite persistence |
| **Expanding Window CV** | Time Series Cross-Validation (no data leakage) |
| **Physical Constraints** | Night detection + Non-negativity in model output |
| **Zero-Shot Evaluation** | Foundation model evaluation (Chronos) |
| **Comprehensive Baselines** | Linear Regression, Naive Persistence |

---

## 📂 Project Structure

```
Progetto_deep_learning/
│
├── 📁 data/
│   ├── raw/                          # Original datasets
│   └── processed/                    # Preprocessed datasets
│       ├── preprocessed_ds.csv           # Training data (24 months)
│       └── preprocessed_test_ds.csv      # Test data (12 months)
│
├── 📁 notebooks/
│   ├── EDA.ipynb                     # Exploratory Data Analysis
│   ├── naive_run.ipynb               # Naive baseline MAE calculation
│   ├── colab_pretraining.ipynb       # Self-supervised pretraining (Colab GPU)
│   ├── colab_optimization.ipynb      # Hyperparameter tuning (Colab GPU)
│   ├── retrieve_study.ipynb          # Optuna study analysis & visualization
│   └── uploading_test_data.ipynb     # Test data preparation
│
├── 📁 results/
│   ├── checkpoints/                  # Trained model weights (.pth)
│   ├── params/                       # Best hyperparameters (.json)
│   ├── pretrained/                   # Pre-trained encoder weights
│   ├── test_results/                 # Inference outputs (metrics, plots, Excel)
│   └── optuna_studies.db             # Optuna SQLite database
│
├── 📁 src/
│   │
│   ├── 📁 config/                    # Configuration files
│   │   ├── __init__.py                   # Exports all config variables
│   │   ├── model_config.py               # Model architectures, LOOKBACK, HORIZON
│   │   ├── training_config.py            # Training params, Naive MAE values
│   │   ├── tuning_config.py              # Optuna settings for forecasting
│   │   ├── tuning_pretrain_config.py     # Optuna settings for pretraining
│   │   └── preprocessing_config.py       # Columns to drop, encoding settings
│   │
│   ├── 📁 DataLoading/               # Data pipeline
│   │   ├── __init__.py
│   │   ├── data_loader.py                # TS_Cross_Validator, DataLoader creation
│   │   ├── sampler.py                    # PVForecastDataset (sliding window)
│   │   └── pretraining_loader.py         # Dataset for masked patch pretraining
│   │
│   ├── 📁 PreProcessing/             # Data preprocessing
│   │   ├── __init__.py
│   │   ├── preprocessing.py              # Preprocesser class (cleaning, encoding)
│   │   └── uploading.py                  # Data upload utilities
│   │
│   ├── 📁 ModelClasses/              # Neural network architectures
│   │   ├── __init__.py                   # Exports all model classes
│   │   ├── lstm.py                       # LSTM model
│   │   ├── dlinear.py                    # DLinear (Decomposition Linear)
│   │   ├── tcn.py                        # Temporal Convolutional Network
│   │   ├── PatchTST.py                   # PatchTST (Transformer) for forecasting
│   │   ├── patchtst_pretraining.py       # PatchTST for self-supervised learning
│   │   ├── encoder_lstm.py               # EncoderLSTM (Pretrained encoder + LSTM)
│   │   ├── naive.py                      # Naive persistence baseline
│   │   └── 📁 zero_shot_model/           # Chronos foundation model wrapper
│   │
│   ├── 📁 Training/                  # Training logic
│   │   ├── __init__.py
│   │   ├── engine.py                     # Training loop, model factory, fit_model
│   │   └── evaluation.py                 # evaluate_model, inference helpers
│   │
│   ├── 📁 Tuning/                    # Hyperparameter optimization
│   │   ├── __init__.py
│   │   ├── optuna_optimizer.py           # OptunaOptimizer class
│   │   └── hyperparameter_spaces.py      # Search spaces per model
│   │
│   ├── 📁 Utils/                     # Utilities
│   │   ├── __init__.py
│   │   ├── plotting.py                   # Visualization functions
│   │   ├── run_opt_utils.py              # set_seed, get_device, save_results
│   │   └── train.py                      # Manual training script (no Optuna)
│   │
│   ├── __init__.py
│   ├── run_optimization.py           # Hyperparameter tuning + final training
│   ├── run_opt_pretraining.py        # Self-supervised pre-training
│   ├── run_inference.py              # Test set inference & Excel export
│   ├── linear_regression.py          # Linear regression baseline
│   └── evaluation_zero_shot.py       # Zero-shot evaluation (Chronos)
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Data Pipeline

### Dataset Overview

| Dataset | Period | Rows | Purpose |
|---------|--------|------|---------|
| **Training** | Jul 2020 - Jun 2022 | 17,544 | Model training & validation |
| **Test** | Jul 2012 - Jun 2013 | 8,760 | Final evaluation |

---

## Exploratory Data Analysis

The `notebooks/EDA.ipynb` notebook performs comprehensive analysis:

### Key Analyses
1. **Time Series Visualization**: PV power patterns across days, months, years
2. **Seasonality Analysis**: Daily and yearly cyclical patterns
3. **Correlation Analysis**: Feature correlations with target
4. **Distribution Analysis**: Histograms and box plots
5. **Missing Data Analysis**: NaN patterns and handling strategies
6. **Outlier Detection**: Identifying anomalous readings

### Key Insights
- Strong **daily seasonality** (peak at noon, zero at night)
- **Yearly seasonality** (higher production in summer)
- High correlation between **GHI** (irradiance) and **pv_power**
- Night hours have **GHI = 0** → used for night detection

---

## Preprocessing

The `Preprocesser` class (`src/PreProcessing/preprocessing.py`) applies:

| Step | Method | Description |
|------|--------|-------------|
| 1 | `cyclical_encoding()` | Convert time to sin/cos (daily + yearly cycles) |
| 2 | `remove_columns()` | Drop `lat`, `lon`, `dt_iso` |
| 3 | `dummy_variable()` | One-hot encode `weather_description` |
| 4 | `fillnan()` | Fill `rain_1h` NaN with 0 |
| 5 | `night_filter()` | Set `pv_power=0` when `Ghi=0` |
| 6 | `reorder_columns()` | Move `pv_power` to last column |

### Processed features (24 columns)

| Category | Features |
|----------|----------|
| **Weather** | temp, dew_point, pressure, humidity, wind_speed, wind_deg, rain_1h, clouds_all |
| **Solar Irradiance** | Ghi, Dni, Dhi |
| **Time Encoding** | day_sin, day_cos, year_sin, year_cos |
| **Weather Category** | weather_description_* (one-hot encoded) |
| **Target** | pv_power |

### Normalization
- **MinMaxScaler** fitted on **training data only** (no data leakage)
- Applied to both training and test data
- Target column (`pv_power`) scaled along with features

---

## Data Loading

### PVForecastDataset (`src/DataLoading/sampler.py`)

Creates sliding window samples for time series forecasting:

```python
class PVForecastDataset(Dataset):
    def __getitem__(self, idx):
        X = data[start:start+lookback, :]     # (48, 24) - All features
        y = data[start+lookback:end, target]  # (24, 1)  - Only pv_power
        return X, y
```

### Time Series Cross-Validation

**Expanding Window Strategy** to prevent data leakage:

```
Fold 1: Train [12 months] Val [4 months]
Fold 2: Train [16 months] Val [4 months]
Fold 3: Train [20 months] Val [4 months]
```

Each fold uses all previous data for training and the next 4 months for validation.

---

## Model Architectures

### 1. LSTM (`src/ModelClasses/lstm.py`)

Standard Long Short-Term Memory network with:
- Configurable hidden size and layers
- Optional bidirectional processing
- Dropout regularization

### 2. DLinear (`src/ModelClasses/dlinear.py`)

Decomposition-based linear model:
- Decomposes input into **trend** and **seasonal** components
- Applies separate linear layers to each
- Lightweight and interpretable

### 3. TCN (`src/ModelClasses/tcn.py`)

Temporal Convolutional Network:
- Dilated causal convolutions
- Residual connections
- Efficient parallelization

### 4. PatchTST (`src/ModelClasses/patchtst.py`)

Patch Time Series Transformer (HuggingFace integration):
- Segments the input into patches
- Transformer encoder processes patches
- Channel-independent processing

### 5. EncoderLSTM (`src/ModelClasses/encoder_lstm.py`) ⭐ **Best Model**

Hybrid architecture combining pre-trained PatchTST encoder with LSTM head:

```
┌─────────────────────────────────────────────────────────────┐
│  Input: (Batch, 48, 24) - 48 timesteps × 24 features        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  PatchTST Encoder (Pre-trained)                             │
│  - Patch length: 16, Stride: 4 → 9 patches                  │
│  - d_model: 128, 4 Transformer layers                       │
│  - Output: (Batch, 24, 9, 128)                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Permute + Reshape                                          │
│  - Concatenate all channels per patch                       │
│  - Output: (Batch, 9, 3072)                                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Projection Layer: Linear(3072, 256)                        │
│  - Reduces dimensionality                                   │
│  - Output: (Batch, 9, 256)                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  LSTM: 3 layers, input=256, hidden=64                       │
│  - Captures temporal dependencies between patches           │
│  - Output: (Batch, 9, 64)                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Last Hidden State: [:, -1, :]                              │
│  - Takes only the final timestep                            │
│  - Output: (Batch, 64)                                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Prediction Head: Linear(64, 24)                            │
│  - Generates 24-hour forecast                               │
│  - Output: (Batch, 24)                                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Physical Constraints:                                      │
│  1. Night Detection: Zero out if GHI was 0 yesterday at     │
│     the same hour                                           │
│  2. Non-negativity: ReLU to force predictions ≥ 0           │
│  - Output: (Batch, 24, 1)                                   │
└─────────────────────────────────────────────────────────────┘
```

#### Physical Constraints (Built into Forward Pass)

1. **Night Detection**: Uses GHI from the last 24h of input. If `GHI < 0.01` for hour `h` yesterday, prediction for hour `h` tomorrow is set to 0.

2. **Non-negativity**: ReLU activation ensures all predictions are ≥ 0 (PV power cannot be negative).

```python
# Night Detection
ghi_last_24h = x[:, -24:, GHI_IDX]
night_mask = ghi_last_24h < 0.01
predictions = predictions * (~night_mask).float()

# Non-negativity
predictions = torch.relu(predictions)
```

---

## Self-Supervised Pre-training

### Masked Patch Reconstruction

The PatchTST encoder is pre-trained using a **Masked Autoencoder** approach:

1. **Patchify** input time series into segments
2. **Randomly mask** 40% of patches
3. **Reconstruct** the masked patches using the Transformer
4. **Save encoder weights** for downstream fine-tuning

### Pre-training Configuration
```python
PRETRAIN_CONFIG = {
    "mask_ratio": 0.4,      # 40% of patches are masked
    "patch_length": 16,
    "stride": 4,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 4,
}
```

**Output**: `results/pretrained/patchtst_pretrain_48_encoder.pth`

---

## Training & Optimization

### Hyperparameter Optimization with Optuna

Uses **TPE (Tree-structured Parzen Estimator)** for Bayesian optimization:

1. **Cross-Validation**: 3-fold expanding window
2. **Metric**: Weighted MASE across folds
3. **Persistence**: SQLite database for study resumption

#### Search Space (EncoderLSTM)
```python
{
    "lr": [1e-5, 1e-2] (log scale),
    "d_model": 128,
    "projection_dim": [64, 128, 256],
    "lstm_hidden": [32, 64, 128],
    "lstm_layers": [1, 2, 3],
    "dropout": [0.1, 0.5],
    "freeze_encoder": False
}
```

#### Weighted MASE Calculation
```python
# Weights based on validation set size (more data = more weight)
weights = [len(fold) for fold in folds]
weighted_mase = sum(mase * w for mase, w in zip(mases, weights)) / sum(weights)
```

### Final Model Training

After optimization, the best parameters are used to train on:
- **Train set**: First 22 months
- **Validation set**: Next 2 months (for early stopping)

```bash
python -m src.run_optimization
```

**Outputs**:
- `results/checkpoints/encoderlstm_48_final_100_epochs.pth`
- `results/params/encoderlstm_48_100_epochs_params.json`

---

## Evaluation Metrics

### MASE (Mean Absolute Scaled Error)

Primary metric that compares model error to a **Naive baseline**:

```
MASE = MAE_model / MAE_naive
```

| MASE Value | Interpretation |
|------------|----------------|
| MASE < 1 | ✅ Model beats naive baseline |
| MASE = 1 | Model equals naive baseline |
| MASE > 1 | ❌ Model is worse than naive |

### Naive Baseline Calculation

The **Naive Persistence** model predicts tomorrow = today:

```python
# For each hour h in horizon [1, 24]:
prediction[h] = actual[h - 24]  # Same hour yesterday
```

**Pre-calculated Naive MAE values** (`training_config.py`):
```python
NAIVE_MAE_PER_FOLD = [0.0622, 0.0862, 0.0619]  # Per CV fold
NAIVE_MAE_FINAL_FOLD = 0.0492                  # Final train/val split
NAIVE_MAE_TEST = 0.0624                        # Test set
```

### Additional Metrics
- **MAE (Normalized)**: Mean Absolute Error on [0,1] scale
- **MAE (Denormalized)**: MAE in Watts (interpretable)
- **RMSE**: Root Mean Squared Error

---

## Baselines

### 1. Naive Persistence (`notebooks/naive_run.ipynb`)
Predicts same value as 24 hours ago.

### 2. Linear Regression (`src/linear_regression.py`)
Sklearn Ridge regression on flattened lookback window:
```bash
python -m src.linear_regression
```

---

## Inference

### Running Inference
```bash
python -m src.run_inference
```

### Inference Pipeline

1. **Load Training Data** → Fit scaler (for normalization consistency)
2. **Load & Preprocess Test Data** → Apply same preprocessing
3. **Create Test DataLoader** → Sliding window with step=1
4. **Load Model** → Architecture + weights from checkpoint
5. **Run Inference** → For MASE calculation
6. **Calculate Metrics** → MAE, RMSE, MASE
7. **Save Results** → JSON, plots, Excel predictions

### Outputs
- `results/test_results/test_results.json` - Metrics summary
- `results/test_results/predictions_samples.png` - Sample predictions vs actuals
- `results/test_results/error_distribution.png` - Error histogram
- `results/test_results/predictions.xlsx` - All predictions with timestamps

### Excel Output Format

| datetime | t+1 | t+2 | ... | t+24 |
|----------|-----|-----|-----|------|
| 2012-07-01 00:00:00+10:00 | NaN | NaN | ... | NaN |
| ... | NaN | NaN | ... | NaN |
| 2012-07-03 00:00:00+10:00 | 0.0 | 0.0 | ... | 123.5 |

*First 47 rows have NaN (insufficient lookback data)*

---

## 📈 Results

### Test Set Performance

| Metric | Value |
|--------|-------|
| **MASE** | 0.82 ✅ |
| **MAE (Normalized)** | 0.051 |
| **MAE (Watts)** | 3.54 W |
| **RMSE (Watts)** | 7.14 W |

---

## Configuration

### Key Configuration Files

| File | Purpose |
|------|---------|
| `model_config.py` | Model architectures, LOOKBACK, HORIZON |
| `training_config.py` | EPOCHS, LR, Naive MAE values |
| `tuning_config.py` | Optuna settings, MODEL_NAME, N_TRIALS |
| `tuning_pretrain_config.py` | Pre-training optimization |
| `preprocessing_config.py` | Columns to drop, encoding settings |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `EDA.ipynb` | Exploratory Data Analysis with visualizations |
| `naive_run.ipynb` | Calculate Naive MAE for all folds and test set |
| `colab_pretraining.ipynb` | Run pre-training on Google Colab with GPU |
| `colab_optimization.ipynb` | Run hyperparameter tuning on Colab |
| `retrieve_study.ipynb` | Analyze Optuna studies, view best params |
| `uploading_test_data.ipynb` | Prepare and preprocess test dataset |

---

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Progetto_deep_learning

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- `torch` - Deep Learning framework
- `transformers` - HuggingFace (PatchTST)
- `optuna` - Hyperparameter optimization
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Preprocessing, metrics
- `matplotlib` - Visualization
- `openpyxl` - Excel export

---

## Author

Deep Learning Project - PV Power Forecasting

Federico Principi | Luca Scorza | Gianluca Barnaba
---

*Last updated: January 2026*
