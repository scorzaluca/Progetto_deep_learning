from .lstm import LSTM
from .naive import NaivePersistence
from .dlinear import DLinearM, DLinearI
from .PatchTST import PatchTST
from .tcn import TCN
from .encoder_lstm import EncoderLSTM
# Import opzionale per ChronosWrapper (richiede chronos-forecasting)
try:
    from .zero_shot_model import ChronosWrapper
except ImportError:
    ChronosWrapper = None

from .patchtst_pretraining import PatchTSTPretraining

# Definisco esplicitamente cosa viene esportato all'esterno
__all__ = ["LSTM", "NaivePersistence", "DLinear", "PatchTST", "PatchTSTPretraining", "ChronosWrapper", "TCN", "EncoderLSTM"]
