#base.py --> common base class for all the ML models I'm building from scratch
# Follows a standard scikit-learn fit/ predict interface

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    def __init__(self):
        self.is_fitted = False   # Keeps track of whether the model has been trained yet

    @abstractmethod
    def fit(self, X: np.ndarray, y:np.ndarray):
        pass                     # Train the model. Needs to be implemented by every subclass

    @abstractmethod 
    def predict(self, X: np.ndarray):
        pass                     # Make prediction on new data.

    def _check_is_fitted(self):
        if not self.is_fitted: # Quick sanity check so we don't predict with a blind model
            raise RuntimeError(f"Hey! you need to call .fit() before trying to predict with {self.__class__.__name__}")
        