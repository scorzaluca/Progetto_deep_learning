# PV Power Forecasting - Deep Learning Project

This project implements a complete **Time Series Forecasting** pipeline for predicting Photovoltaic (PV) power generation. It leverages advanced Deep Learning architectures, including **LSTMs**, **DLinear**, and the state-of-the-art **PatchTST**, incorporating **Transfer Learning** (via Masked Autoencoder Pretraining) and **Hyperparameter Optimization** with Optuna.

## Key Features

*   **Multi-Model Support:** Implementations of LSTM, DLinear, and PatchTST (Transformer-based).
*   **Self-Supervised Pretraining:** A masked autoencoder approach to pretrain the PatchTST encoder on unlabeled historical data, improving downstream performance via **Transfer Learning**.
*   **Hyperparameter Optimization:** Automated tuning using **Optuna** with SQLite persistence and Cross-Validation.
*   **Robust Evaluation:** Time Series Cross-Validation (Expanding Window) to ensure realistic performance metrics.
*   **Modular Architecture:** Clean separation of concerns (Data Loading, Modeling, Training, Configuration).
*   **Zero-Shot Analysis:** Exploration of Foundation Models for zero-shot inference.

---

## 📂 Project Structure

The source code is organized in `src/` as follows:

```
src/
├── config/                 # Configuration Files (Parameters & Hyperparams)
│   ├── model_config.py         # Architectures (Layers, Heads, Hidden Sizes)
│   ├── preprocessing_config.py # Data Cleaning (Columns, NaNs)
│   ├── training_config.py      # Training Loop (Epochs, Seeds, Benchmarks)
│   ├── tuning_config.py        # Forecasting Optimization (DLinear, LSTM)
│   └── tuning_pretrain_config.py # Pretraining Optimization (PatchTST)
│
├── DataLoading/            # Data Pipeline
│   ├── DataLoader.py           # Standard PyTorch Dataset classes
│   ├── PretrainingDataset.py   # Masked Autoencoder Dataset logic
│   └── Sampler.py              # Time Series Cross-Validator
│
├── ModelClasses/           # Neural Network Architectures (PyTorch)
│   ├── DLinear.py             # DLinear (Decomposition Linear)
│   ├── LSTM.py                # LSTM (Standard & Encoder-Decoder)
│   └── PatchTST.py            # PatchTST (Transformer-based)
│
├── PreProcessing/          # ETL Pipeline
│   └── preprocessing.py        # Cleaning, Feature Engineering, Normalization
│
├── Training/               # Training Logic
│   ├── engine.py              # Generic Training/Validation Loop
│   └── evaluation.py          # Metrics & Inference helpers
│
├── Tuning/                 # Optuna Integration
│   ├── OptunaOptimizer.py     # Forecasting Optimization Wrapper
│   └── PretrainOptimizer.py   # Pretraining Optimization Wrapper
│
├── Utils/                  # Helper Scripts
│   ├── plotting.py            # Visualization Functions
│   ├── train.py               # Manual Training Script
│   └── run_opt_utils.py       # optimization helpers
│
├── run_inference.py        # MAIN: Run Final Inference on Test Set
├── run_opt_pretraining.py  # MAIN: Run Self-Supervised Pretraining
└── run_optimization.py     # MAIN: Run Hyperparameter Tuning
```

---

## 🛠️ Installation


1.  **Clone the repository** (if applicable).
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🌊 Data Flow & Workflow

The typical workflow covers **Preprocessing - Pretraining - Optimization - Inference**.

### 1. Preprocessing (`src/PreProcessing/`)
Cleans raw data, handles missing values, encodes categorical variables (weather), and performs scaling.
*   **Config:** `src/config/preprocessing_config.py`
*   **Input:** `data/processed/merge_ds.csv`
*   **Output:** `data/processed/preprocessed_ds.csv`

### 2. Self-Supervised Pretraining (`src/run_opt_pretraining.py`)
Trains the **PatchTST Encoder** to reconstruct masked time series patches. This learns robust feature representations without labels.
*   **Config:** `src/config/tuning_pretrain_config.py`
*   **Output:** The Best Encoder weights are saved to `results/pretrained/`.

### 3. Hyperparameter Optimization (`src/run_optimization.py`)
Fines-tunes models (LSTM, DLinear, or models using the Pretrained Encoder) using **Optuna**.
*   It performs **Cross-Validation** to find the minimal **Mean Absolute Scaled Error (MASE)**.
*   The best model is then retrained on the full dataset (Train + Validation).
*   **Config:** `src/config/tuning_config.py`
*   **Output:** Best parameters (`results/params/`) and Checkpoints (`results/checkpoints/`).

### 4. Final Inference (`src/run_inference.py`)
Loads the best trained model and generates predictions for the held-out Test Set.
*   Metrics: MASE, RMSE, MAE (Normalized & Denormalized).
*   **Output:** Predictions (`results/predictions/`) and Plots (`results/plots/`).

---

## ⚙️ Configuration Guide

Parameters are strictly separated from logic. Edit these files to control experiments:

*   **`model_config.py`**:
    *   `LOOKBACK`: Input window size (hours). **Default: 48**.
    *   `HORIZON`: Prediction horizon (hours). **Default: 24**.
    *   `PATCHTST_CONFIG`: Defines transformer depth, heads, and patch size.
    *   `LSTM_CONFIG`: Hidden size, layers, bidirectional flag.

*   **`tuning_config.py`**:
    *   `MODEL_NAME`: Select model to tune (`"encoderlstm"`, `"dlinearm"`, etc.).
    *   `N_TRIALS`: Number of Optuna experiments.
    *   `STUDY_NAME`: Defines the database entry name.

*   **`training_config.py`**:
    *   Global `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`.
    *   `NAIVE_MAE_...`: Pre-calculated baselines for MASE.

---

## ▶️ Usage Examples

### 1. Run Pretraining Optimization
Optimize the self-supervised task to find the best encoder structure.
```bash
python -m src.run_opt_pretraining
```

### 2. Run Forecasting Optimization
Tune the downstream model (e.g., EncoderLSTM using the pretrained weights).
1.  Open `src/config/tuning_config.py` and set `MODEL_NAME = "encoderlstm"`.
2.  Run:
```bash
python -m src.run_optimization
```

### 3. Run Inference
Evaluate the optimal model on the Test Set.
```bash
python -m src.run_inference
```

### 4. Simple Training (No Optuna)
Train a single model using default config parameters.
```bash
python -m src.Utils.train
```

---


### Evaluation Metric: MASE
We use **MASE (Mean Absolute Scaled Error)** as the primary metric. It compares the model's error against a "Naive" baseline (predicting the last observed value).
*   **MASE < 1**: Model is better than naive.
*   **MASE > 1**: Model is worse than naive.
*   **MASE = 0.5**: Model error is half that of the naive method.

---


