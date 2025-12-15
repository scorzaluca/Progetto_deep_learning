from abc import ABC, abstractmethod
import torch


class ZeroShotWrapper(ABC):
    """Wrapper zero-shot model standard interface"""
    
    @abstractmethod
    def predict(self, context: torch.Tensor, horizon: int, val_loader) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """methods which gives back model info (size, type, ecc.)."""
        pass