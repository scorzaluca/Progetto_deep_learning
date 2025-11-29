# File: src/__init__.py

# 1. Importo dal pacchetto 'data_loading' (la cartella)
# Nota: Qui Python va a leggere src/data_loading/__init__.py
from .Data_loading import PVForecastDataset, TS_Cross_Validator

# 2. Importo dal pacchetto 'preprocessing' (la cartella)
# Nota: Qui Python va a leggere src/preprocessing/__init__.py
from .PreProcessing import preprocesser, uploader

# (Opzionale) Se vuoi esporre anche qui tutto all'esterno
__all__ = [
    "PVForecastDataset",
    "TS_Cross_Validator",
    "preprocesser",
    "uploader"
]