"""
Models for uplift modeling.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List

@dataclass
class UpliftDataset:
    """Dataset for uplift modeling."""
    X_control: Any
    y_control: Any
    X_treatment: Any
    y_treatment: Any
    user_ids: List[str]
    feature_names: List[str]


@dataclass
class UpliftPrediction:
    """Prediction from an uplift model."""
    user_id: str
    ite: float
    percentile_rank: float


@dataclass
class ModelMeta:
    """Metadata for an uplift model."""
    experiment_id: str
    model_version: str
    feature_names: List[str]
    qini: float
    auuc: float
    trained_at: datetime
