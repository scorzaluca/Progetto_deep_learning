from abc import ABC, abstractmethod
import torch


# src/ModelClasses/ZeroShotWrapper.py
class ZeroShotWrapper(ABC):
    """Interfaccia base per wrapper di modelli zero-shot."""
    
    @abstractmethod
    def predict(self, context: torch.Tensor, horizon: int, val_loader) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Ritorna informazioni sul modello (size, type, ecc.)."""
        pass