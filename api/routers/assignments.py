from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
import hashlib
import pandas as pd
from db.session import get_db
from api.schemas import AssignmentEnroll, AssignmentResponse
from db.repositories import assignment_repo, experiment_repo
from db.models import ExperimentStatus, Event
from core.segmentation import ClusteringModel, assign_segment

assignment_router = APIRouter(prefix="/assignments", tags=["assignments"])

def get_variant(user_id: str, experiment_seed: int) -> str:
    """
    Deterministically assigns a user to a variant based on their ID and experiment seed.
    """
    hash_input = f"{user_id}:{experiment_seed}".encode()
    hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
    return "treatment" if hash_val % 2 == 1 else "control"

@assignment_router.post("/enroll", response_model=AssignmentResponse, status_code=status.HTTP_201_CREATED)
def enroll_user(enrollment: AssignmentEnroll, db: Session = Depends(get_db)):
    """
    Enrolls a user in an experiment, optionally assigning a segment if a model exists.
    """
    # 1. Check if experiment exists and is running
    experiment = experiment_repo.get_experiment(db, enrollment.experiment_id)
    if not experiment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    
    if experiment.status != ExperimentStatus.running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Experiment is in {experiment.status.value} status. Enrollment only allowed for running experiments."
        )

    # 2. Check if user is already assigned
    existing_assignment = assignment_repo.get_assignment(db, enrollment.experiment_id, enrollment.user_id)
    if existing_assignment:
        return existing_assignment

    # 3. Determine variant
    variant = get_variant(enrollment.user_id, experiment.seed)

    # 4. Optional Segment Assignment
    segment_id = None
    segment_model_record = assignment_repo.get_segment_model(db, enrollment.experiment_id)
    
    if segment_model_record:
        try:
            # Fetch historical events for this user (before experiment creation)
            user_events = db.query(Event).filter(
                Event.experiment_id == enrollment.experiment_id,
                Event.user_id == enrollment.user_id,
                Event.occurred_at < experiment.created_at
            ).all()

            user_features = {}
            if user_events:
                # Prepare raw event data for extraction
                df_events = pd.DataFrame([{
                    'user_id': e.user_id,
                    'occurred_at': e.occurred_at,
                    'metric_value': e.metric_value,
                    'event_type': e.event_type,
                    'session_id': getattr(e, 'session_id', None)
                } for e in user_events])

                # Extract features from available events
                _, extractor = ClusteringModel.load_artifacts(segment_model_record.kmeans_artifact)
                features_df = extractor.extract(df_events)
                
                if not features_df.empty:
                    # Convert to dict (excluding user_id)
                    user_features = features_df.drop(columns=['user_id']).iloc[0].to_dict()

            # Perform assignment (will use imputation if user_features is empty or incomplete)
            segment_id = assign_segment(user_features, segment_model_record)
        except Exception:
            # Fallback: segment_id remains None if processing fails catastrophically
            pass

    # 5. Create assignment
    return assignment_repo.create_assignment(
        db, 
        enrollment.experiment_id, 
        enrollment.user_id, 
        variant, 
        segment_id
    )