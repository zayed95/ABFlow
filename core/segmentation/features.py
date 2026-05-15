import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pickle

class FeatureExtractor:
    """
    Transforms raw event data into a user-level feature matrix for segmentation.
    """

    def __init__(self, feature_schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor with an optional schema.
        """
        self.feature_schema = feature_schema or {}

    def extract(self, user_events: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates raw events per user into feature columns.
        
        Required features:
        - recency: days since last event
        - frequency: total event count
        - monetary: sum of metric_value
        - session_depth: unique sessions
        - conversion_rate: conversions / sessions
        
        Args:
            user_events: DataFrame containing [user_id, occurred_at, metric_value, session_id, event_type]
            
        Returns:
            pd.DataFrame: User-level features indexed by user_id.
        """
        if user_events.empty:
            return pd.DataFrame()

        # Ensure occurred_at is datetime
        if not pd.api.types.is_datetime64_any_dtype(user_events['occurred_at']):
            user_events['occurred_at'] = pd.to_datetime(user_events['occurred_at'])

        # Reference time for recency calculation
        now = datetime.utcnow()

        # Group by user_id and aggregate core metrics
        # We assume session_id and event_type are present as per feature requirements
        features = user_events.groupby('user_id').agg(
            last_event=('occurred_at', 'max'),
            frequency=('occurred_at', 'count'),
            monetary=('metric_value', 'sum'),
            session_depth=('session_id', 'nunique') if 'session_id' in user_events.columns else ('occurred_at', 'count'), # fallback if session_id missing
            conversions=('event_type', lambda x: (x == 'conversion').sum()) if 'event_type' in user_events.columns else ('metric_value', lambda x: (x > 0).sum())
        )

        # 1. Recency: days since last event
        features['recency'] = (now - features['last_event']).dt.total_seconds() / 86400

        # 2. Conversion Rate: conversions / sessions
        # If session_id was missing, session_depth defaults to frequency
        features['conversion_rate'] = features['conversions'] / features['session_depth']
        features['conversion_rate'] = features['conversion_rate'].fillna(0)

        # Cleanup intermediate columns
        features = features.drop(columns=['last_event', 'conversions'])

        # Robustness check: handle any potential inf or NaN
        features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Ensure column order and return
        cols_order = ['recency', 'frequency', 'monetary', 'session_depth', 'conversion_rate']
        return features.reset_index()[['user_id'] + cols_order]

    def fit_transform(self, user_events: pd.DataFrame) -> np.ndarray:
        """
        Calls extract(), then fits a StandardScaler on the result and returns the scaled matrix.
        Stores the fitted scaler as self.scaler.
        
        Args:
            user_events: DataFrame containing raw event data.
            
        Returns:
            np.ndarray: Scaled feature matrix.
        """
        features_df = self.extract(user_events)
        if features_df.empty:
            return np.empty((0, len(self.get_feature_names())))

        # Exclude user_id from the matrix to be scaled
        X = features_df.drop(columns=['user_id']).values

        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)

    def transform(self, user_events: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw events using the already fitted self.scaler.
        """
        if not hasattr(self, 'scaler'):
            raise RuntimeError("FeatureExtractor must be fitted before calling transform().")
            
        features_df = self.extract(user_events)
        if features_df.empty:
            return np.empty((0, len(self.get_feature_names())))
            
        X = features_df.drop(columns=['user_id']).values
        return self.scaler.transform(X)

    def get_feature_names(self) -> List[str]:
        """
        Returns the list of features extracted.
        """
        return ['recency', 'frequency', 'monetary', 'session_depth', 'conversion_rate']

    def save(self, path: str):
        """
        Serializes the scaler and schema to a pickle file.
        """
        if not hasattr(self, 'scaler'):
            raise RuntimeError("FeatureExtractor must be fitted before saving.")
        
        state = {
            'scaler': self.scaler,
            'feature_schema': self.feature_schema
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'FeatureExtractor':
        """
        Loads a FeatureExtractor from a pickle file.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        instance = cls(feature_schema=state.get('feature_schema'))
        instance.scaler = state.get('scaler')
        return instance
