import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.segmentation.features import FeatureExtractor

@pytest.fixture
def sample_events():
    """Provides a sample DataFrame of raw events."""
    now = datetime.utcnow()
    data = [
        {'user_id': 'u1', 'occurred_at': now - timedelta(days=1), 'metric_value': 10.0, 'session_id': 's1', 'event_type': 'view'},
        {'user_id': 'u1', 'occurred_at': now - timedelta(hours=5), 'metric_value': 20.0, 'session_id': 's2', 'event_type': 'conversion'},
        {'user_id': 'u2', 'occurred_at': now - timedelta(days=10), 'metric_value': 5.0, 'session_id': 's3', 'event_type': 'view'},
        {'user_id': 'u3', 'occurred_at': now - timedelta(days=2), 'metric_value': 100.0, 'session_id': 's4', 'event_type': 'conversion'},
    ]
    return pd.DataFrame(data)

class TestFeatureExtractor:
    
    def test_fit_transform_consistency(self, sample_events):
        """Test that transform() on the same data used in fit_transform() produces the same output."""
        extractor = FeatureExtractor()
        X_fit = extractor.fit_transform(sample_events)
        X_trans = extractor.transform(sample_events)
        
        assert np.allclose(X_fit, X_trans)
        assert isinstance(X_fit, np.ndarray)
        assert X_fit.shape == (3, 5) # 3 users, 5 features

    def test_transform_raises_if_unfitted(self, sample_events):
        """Test that transform() raises RuntimeError if called before fit."""
        extractor = FeatureExtractor()
        with pytest.raises(RuntimeError, match="must be fitted"):
            extractor.transform(sample_events)

    def test_extract_manual_verification(self):
        """Test feature extraction on a known DataFrame with manually computed expected values."""
        now = datetime.utcnow()
        # Define a single user with predictable metrics
        data = [
            {'user_id': 'u1', 'occurred_at': now - timedelta(days=2), 'metric_value': 50.0, 'session_id': 's1', 'event_type': 'view'},
            {'user_id': 'u1', 'occurred_at': now - timedelta(days=1), 'metric_value': 50.0, 'session_id': 's2', 'event_type': 'conversion'},
        ]
        df = pd.DataFrame(data)
        
        extractor = FeatureExtractor()
        features = extractor.extract(df)
        
        # Verify manually computed values
        assert len(features) == 1
        row = features.iloc[0]
        
        assert row['user_id'] == 'u1'
        assert row['frequency'] == 2
        assert row['monetary'] == 100.0
        assert row['session_depth'] == 2
        assert row['conversion_rate'] == 0.5 # 1 conversion / 2 sessions
        # Recency should be ~1 day (since the latest event was 1 day ago)
        assert pytest.approx(row['recency'], abs=0.01) == 1.0

    def test_empty_input_handling(self):
        """Test how the extractor handles empty DataFrames."""
        extractor = FeatureExtractor()
        empty_df = pd.DataFrame(columns=['user_id', 'occurred_at', 'metric_value', 'session_id', 'event_type'])
        
        features = extractor.extract(empty_df)
        assert features.empty
        
        X = extractor.fit_transform(empty_df)
        assert X.shape == (0, 5)

    def test_missing_optional_columns(self):
        """Test fallback logic when session_id or event_type are missing."""
        now = datetime.utcnow()
        data = [
            {'user_id': 'u1', 'occurred_at': now - timedelta(days=1), 'metric_value': 10.0}
        ]
        df = pd.DataFrame(data)
        
        extractor = FeatureExtractor()
        features = extractor.extract(df)
        
        # fallback: session_depth should equal frequency (1)
        # fallback: conversion_rate should be 1.0 if metric_value > 0
        assert features.iloc[0]['session_depth'] == 1
        assert features.iloc[0]['conversion_rate'] == 1.0
