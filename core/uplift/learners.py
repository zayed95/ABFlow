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

import pickle
from typing import Any, Optional
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

class TLearner(BaseUpliftLearner):
    """
    Two-model learner (T-Learner) for uplift modeling.
    Trains separate models for the control and treatment groups.
    ITE = P(Y=1|X, T=1) - P(Y=1|X, T=0)
    """
    def __init__(self):
        self.control_model = None
        self.treatment_model = None
        self.is_fitted = False

    def fit(self, dataset: UpliftDataset) -> None:
        """
        Fit the control and treatment models using RandomForestClassifier.
        """
        if len(dataset.y_control) == 0 or len(dataset.y_treatment) == 0:
            raise ValueError("Both control and treatment groups must have data.")
            
        self.feature_names = dataset.feature_names
            
        self.control_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        self.control_model.fit(dataset.X_control, dataset.y_control)
        
        self.treatment_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        self.treatment_model.fit(dataset.X_treatment, dataset.y_treatment)
        
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).
        """
        if not self.is_fitted:
            raise RuntimeError("TLearner is not fitted yet.")
            
        pred_control = self.control_model.predict_proba(X)[:, 1]
        pred_treatment = self.treatment_model.predict_proba(X)[:, 1]
        
        ite = pred_treatment - pred_control
        return ite

    @property
    def treatment_feature_importance(self):
        """
        Returns the feature importances from the treatment model as a pandas Series.
        """
        import pandas as pd
        if not self.is_fitted:
            raise RuntimeError("TLearner is not fitted yet.")
            
        return pd.Series(
            self.treatment_model.feature_importances_,
            index=self.feature_names,
            name="treatment_importance"
        ).sort_values(ascending=False)

    def save(self) -> bytes:
        """
        Serialize the trained learner.
        """
        if not self.is_fitted:
            raise RuntimeError("TLearner is not fitted yet.")
        data = {
            'control': self.control_model,
            'treatment': self.treatment_model,
            'feature_names': self.feature_names
        }
        return pickle.dumps(data)

    @classmethod
    def load(cls, data: bytes) -> 'TLearner':
        """
        Load a serialized TLearner.
        """
        state = pickle.loads(data)
        instance = cls()
        instance.control_model = state['control']
        instance.treatment_model = state['treatment']
        instance.feature_names = state['feature_names']
        instance.is_fitted = True
        return instance

class SLearner(BaseUpliftLearner):
    """
    Single-model learner (S-Learner) for uplift modeling.
    Trains one model on all data with treatment_flag as an additional feature.
    ITE = P(Y=1|X, T=1) - P(Y=1|X, T=0)
    """
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_fitted = False

    def fit(self, dataset: UpliftDataset) -> None:
        """
        Fit the single model using RandomForestClassifier.
        """
        if len(dataset.y_control) == 0 or len(dataset.y_treatment) == 0:
            raise ValueError("Both control and treatment groups must have data.")
            
        self.feature_names = dataset.feature_names
        
        # Add treatment flag (0 for control, 1 for treatment)
        T_control = np.zeros((dataset.X_control.shape[0], 1))
        X_control_ext = np.hstack([dataset.X_control, T_control])
        
        T_treatment = np.ones((dataset.X_treatment.shape[0], 1))
        X_treatment_ext = np.hstack([dataset.X_treatment, T_treatment])
        
        # Combine data
        X_all = np.vstack([X_control_ext, X_treatment_ext])
        y_all = np.concatenate([dataset.y_control, dataset.y_treatment])
            
        self.model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        self.model.fit(X_all, y_all)
        
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the Individual Treatment Effect (ITE).
        """
        if not self.is_fitted:
            raise RuntimeError("SLearner is not fitted yet.")
            
        # Predict with T=0
        T_0 = np.zeros((X.shape[0], 1))
        X_0 = np.hstack([X, T_0])
        pred_control = self.model.predict_proba(X_0)[:, 1]
        
        # Predict with T=1
        T_1 = np.ones((X.shape[0], 1))
        X_1 = np.hstack([X, T_1])
        pred_treatment = self.model.predict_proba(X_1)[:, 1]
        
        ite = pred_treatment - pred_control
        return ite

    @property
    def feature_importance(self):
        """
        Returns the feature importances from the model as a pandas Series.
        Includes the treatment flag.
        """
        import pandas as pd
        if not self.is_fitted:
            raise RuntimeError("SLearner is not fitted yet.")
            
        extended_features = self.feature_names + ["treatment_flag"]
        return pd.Series(
            self.model.feature_importances_,
            index=extended_features,
            name="feature_importance"
        ).sort_values(ascending=False)

    def save(self) -> bytes:
        """
        Serialize the trained learner.
        """
        if not self.is_fitted:
            raise RuntimeError("SLearner is not fitted yet.")
        data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        return pickle.dumps(data)

    @classmethod
    def load(cls, data: bytes) -> 'SLearner':
        """
        Load a serialized SLearner.
        """
        state = pickle.loads(data)
        instance = cls()
        instance.model = state['model']
        instance.feature_names = state['feature_names']
        instance.is_fitted = True
        return instance
