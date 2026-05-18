import pytest
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from core.uplift.models import UpliftDataset
from core.uplift.learners import TLearner, SLearner

def test_tlearner_ite_correlation():
    """
    Test T-Learner identifies heterogeneous treatment effect correctly.
    Treatment has positive effect for high feature_1, no effect for low feature_1.
    """
    np.random.seed(42)
    n = 500
    
    # 250 control, 250 treatment
    X_control = np.random.rand(n // 2, 2)
    X_treatment = np.random.rand(n // 2, 2)
    
    # Base CR = 10%
    # True ITE: if feature_1 (index 0) > 0.5 then 0.6 else 0.0
    # y_control = 10%
    # y_treatment = 10% + 60% if feature_1 > 0.5
    
    y_control = (np.random.rand(n // 2) < 0.1).astype(float)
    
    treatment_probs = 0.1 + 0.6 * (X_treatment[:, 0] > 0.5)
    y_treatment = (np.random.rand(n // 2) < treatment_probs).astype(float)
    
    dataset = UpliftDataset(
        X_control=X_control,
        y_control=y_control,
        X_treatment=X_treatment,
        y_treatment=y_treatment,
        user_ids=[f"u_{i}" for i in range(n)],
        feature_names=["feature_1", "feature_2"]
    )
    
    learner = TLearner()
    learner.fit(dataset)
    
    # Predict on the combined X
    X_all = np.vstack([X_control, X_treatment])
    feature_1_all = X_all[:, 0]
    
    ite = learner.predict(X_all)
    
    # Check correlation between feature_1 and ITE
    corr, _ = pearsonr(feature_1_all, ite)
    
    assert corr > 0.5, f"Expected high positive correlation between feature_1 and ITE, got {corr:.2f}"
    
    # Also test treatment feature importance
    importance = learner.treatment_feature_importance
    assert "feature_1" in importance.index
    # feature_1 should be highly important for the treatment model
    assert importance["feature_1"] > importance["feature_2"]

def test_slearner_ite_correlation():
    """
    Test S-Learner on the same dataset.
    """
    np.random.seed(42)
    n = 500
    
    X_control = np.random.rand(n // 2, 2)
    X_treatment = np.random.rand(n // 2, 2)
    
    y_control = (np.random.rand(n // 2) < 0.1).astype(float)
    treatment_probs = 0.1 + 0.6 * (X_treatment[:, 0] > 0.5)
    y_treatment = (np.random.rand(n // 2) < treatment_probs).astype(float)
    
    dataset = UpliftDataset(
        X_control=X_control,
        y_control=y_control,
        X_treatment=X_treatment,
        y_treatment=y_treatment,
        user_ids=[f"u_{i}" for i in range(n)],
        feature_names=["feature_1", "feature_2"]
    )
    
    learner = SLearner()
    learner.fit(dataset)
    
    X_all = np.vstack([X_control, X_treatment])
    feature_1_all = X_all[:, 0]
    
    ite = learner.predict(X_all)
    
    corr, _ = pearsonr(feature_1_all, ite)
    
    assert corr > 0.3, f"Expected positive correlation for SLearner, got {corr:.2f}"
