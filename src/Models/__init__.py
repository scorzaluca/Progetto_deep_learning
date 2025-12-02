from .lstm import LSTM
from .naive import NaivePersistence

# Definisco esplicitamente cosa viene esportato all'esterno
__all__ = [
    "LSTM",
    "NaivePersistence"

]