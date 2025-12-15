from .lstm import LSTM
from .naive import NaivePersistence
from .dlinear import DLinearM, DLinearI
from .PatchTST import PatchTST
from .tcn import TCN

# Import opzionale per ChronosWrapper (richiede chronos-forecasting)
try:
    from .zero_shot_model import ChronosWrapper
except ImportError:
    ChronosWrapper = None

# Definisco esplicitamente cosa viene esportato all'esterno
__all__ = ["LSTM", "NaivePersistence", "DLinear", "PatchTST", "ChronosWrapper", "TCN"]
