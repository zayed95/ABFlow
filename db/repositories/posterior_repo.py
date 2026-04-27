from sqlalchemy.orm import Session
from sqlalchemy import desc
from db.models import PosteriorSnapshot
import uuid
from datetime import datetime

def save_snapshot(
    db: Session, 
    experiment_id: uuid.UUID, 
    variant: str, 
    alpha_post: float, 
    beta_post: float, 
    n_trials: int, 
    n_conversions: int,
    last_processed_at: datetime
) -> PosteriorSnapshot:
    """
    Saves a new posterior snapshot to the database.
    This is an append-only log.
    
    :param db: SQLAlchemy session
    :param experiment_id: UUID of the experiment
    :param variant: Variant name (e.g., 'control', 'treatment')
    :param alpha_post: Posterior alpha parameter
    :param beta_post: Posterior beta parameter
    :param n_trials: Cumulative number of trials
    :param n_conversions: Cumulative number of conversions
    :param last_processed_at: Watermark of the last processed event/assignment
    :return: The created PosteriorSnapshot object
    """
    db_snapshot = PosteriorSnapshot(
        experiment_id=experiment_id,
        variant=variant,
        alpha_post=alpha_post,
        beta_post=beta_post,
        n_trials=n_trials,
        n_conversions=n_conversions,
        last_processed_at=last_processed_at
    )
    db.add(db_snapshot)
    db.commit()
    db.refresh(db_snapshot)
    return db_snapshot

def get_latest_snapshot(db: Session, experiment_id: uuid.UUID, variant: str) -> PosteriorSnapshot:
    """
    Retrieves the most recent snapshot for a specific variant of an experiment.
    
    :param db: SQLAlchemy session
    :param experiment_id: UUID of the experiment
    :param variant: Variant name
    :return: The latest PosteriorSnapshot or None
    """
    return (
        db.query(PosteriorSnapshot)
        .filter(
            PosteriorSnapshot.experiment_id == experiment_id,
            PosteriorSnapshot.variant == variant
        )
        .order_by(desc(PosteriorSnapshot.snapshot_at))
        .first()
    )

def get_all_snapshots(db: Session, experiment_id: uuid.UUID) -> list[PosteriorSnapshot]:
    """
    Retrieves the full history of snapshots for an experiment, ordered by time.
    
    :param db: SQLAlchemy session
    :param experiment_id: UUID of the experiment
    :return: List of PosteriorSnapshot objects
    """
    return (
        db.query(PosteriorSnapshot)
        .filter(PosteriorSnapshot.experiment_id == experiment_id)
        .order_by(PosteriorSnapshot.snapshot_at)
        .all()
    )
