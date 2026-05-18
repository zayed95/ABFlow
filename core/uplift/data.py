import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import select
from db.models import Experiment, Assignment, Event, ExperimentStatus
from core.uplift.models import UpliftDataset
from core.segmentation.features import FeatureExtractor
import uuid

def prepare_dataset(db: Session, experiment_id: str | uuid.UUID) -> UpliftDataset:
    """
    Queries assignments and events, builds separate feature matrices 
    for control and treatment groups. Only include users with final outcomes 
    (experiment must be complete).
    """
    if isinstance(experiment_id, str):
        experiment_id = uuid.UUID(experiment_id)
        
    experiment = db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    ).scalar_one_or_none()
    
    if not experiment or experiment.status != ExperimentStatus.complete:
        raise ValueError("Experiment must be complete to prepare dataset.")
        
    assignments = db.execute(
        select(Assignment).where(Assignment.experiment_id == experiment_id)
    ).scalars().all()
    
    if not assignments:
        raise ValueError("No assignments found for experiment.")
        
    assignment_df = pd.DataFrame([{
        "user_id": a.user_id,
        "variant": a.variant,
        "enrolled_at": a.enrolled_at,
        "features": a.features or {}
    } for a in assignments])
    
    events = db.execute(
        select(Event).where(Event.experiment_id == experiment_id)
    ).scalars().all()
    
    if not events:
        raise ValueError("No events found for experiment.")
        
    event_df = pd.DataFrame([{
        "user_id": e.user_id,
        "event_type": e.event_type,
        "metric_value": e.metric_value,
        "occurred_at": e.occurred_at
    } for e in events])
    
    merged = pd.merge(event_df, assignment_df, on="user_id", how="inner")
    
    outcome_window_hours = experiment.config.get("outcome_window_hours", 72)
    window_delta = pd.to_timedelta(outcome_window_hours, unit="h")
    
    # Only use post-enrollment events within the outcome window for outcomes
    post_events = merged[
        (merged["occurred_at"] > merged["enrolled_at"]) & 
        (merged["occurred_at"] <= merged["enrolled_at"] + window_delta)
    ]
    
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    
    # Build features_df from stored assignment features
    features_list = []
    for _, row in assignment_df.iterrows():
        f_dict = row["features"].copy() if row["features"] else {}
        f_dict["user_id"] = row["user_id"]
        for fn in feature_names:
            if fn not in f_dict:
                f_dict[fn] = 0.0
        features_list.append(f_dict)
    
    features_df = pd.DataFrame(features_list)[["user_id"] + feature_names]
    
    outcomes = post_events.groupby("user_id")["metric_value"].sum().reset_index()
    outcomes.rename(columns={"metric_value": "y"}, inplace=True)
    
    # Users with no post events get outcome 0
    data = pd.merge(features_df, outcomes, on="user_id", how="left")
    data["y"] = data["y"].fillna(0.0)
    data = pd.merge(data, assignment_df[["user_id", "variant"]], on="user_id", how="inner")
    
    control_data = data[data["variant"] == "control"]
    treatment_data = data[data["variant"] == "treatment"]
    
    return UpliftDataset(
        X_control=control_data[feature_names].values,
        y_control=control_data["y"].values,
        X_treatment=treatment_data[feature_names].values,
        y_treatment=treatment_data["y"].values,
        user_ids=data["user_id"].tolist(),
        feature_names=feature_names
    )
