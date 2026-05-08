import uuid
from sqlalchemy.orm import Session
from db.models import ExperimentStatus
from db.repositories import experiment_repo
from tasks.train_segmentation import train_segmentation_task

def start_experiment(db: Session, experiment_id: str):
    """
    Business logic to start an experiment:
    1. Updates the experiment status to 'running' in the database.
    2. Enqueues the train_segmentation_task to build user segments from historical data.
    
    Args:
        db: SQLAlchemy database session.
        experiment_id: String representation of the experiment UUID.
        
    Returns:
        The updated Experiment object, or None if not found.
    """
    exp_uuid = uuid.UUID(experiment_id)
    
    # 1. Update experiment status to running
    # This repo method already handles commit/refresh
    experiment = experiment_repo.update_status(db, exp_uuid, ExperimentStatus.running)
    
    if experiment:
        # 2. Trigger the segmentation training task asynchronously via Celery
        train_segmentation_task.delay(experiment_id)
        
    return experiment
