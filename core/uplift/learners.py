from abc import ABC, abstractmethod
import numpy as np
from core.uplift.models import UpliftDataset

class BaseUpliftLearner(ABC):
    """Abstract base class for uplift modeling learners."""

    @abstractmethod
    def fit(self, dataset: UpliftDataset) -> None:

        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:

        pass

    @abstractmethod
    def save(self) -> bytes:
      
        pass
