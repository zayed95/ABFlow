from sqlalchemy.orm import Session
from db.models import Assignment, SegmentModel
import uuid

def get_assignment(db: Session, experiment_id: uuid.UUID, user_id: str) -> Assignment:
    """
    Retrieves an assignment for a specific user and experiment.
    """
    return db.query(Assignment).filter(
        Assignment.experiment_id == experiment_id,
        Assignment.user_id == user_id
    ).first()

def create_assignment(db: Session, experiment_id: uuid.UUID, user_id: str, variant: str, segment_id: int = None, features: dict = None) -> Assignment:
    """
    Creates a new assignment record.
    """
    db_assignment = Assignment(
        experiment_id=experiment_id,
        user_id=user_id,
        variant=variant,
        segment_id=segment_id,
        features=features
    )
    db.add(db_assignment)
    db.commit()
    db.refresh(db_assignment)
    return db_assignment

def get_segment_model(db: Session, experiment_id: uuid.UUID) -> SegmentModel:
    """
    Retrieves the most recent segment model for an experiment.
    """
    return db.query(SegmentModel).filter(
        SegmentModel.experiment_id == experiment_id
    ).order_by(SegmentModel.trained_at.desc()).first()
