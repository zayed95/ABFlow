from sqlalchemy.orm import Session
from db.models import Experiment, ExperimentStatus
import uuid

def create_experiment(db: Session, data: dict) -> Experiment:
    """
    Creates a new experiment in the database.
    :param db: SQLAlchemy session
    :param data: Dictionary containing experiment fields (name, config, seed, etc.)
    :return: The created Experiment object
    """
    db_experiment = Experiment(**data)
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    return db_experiment

def get_experiment(db: Session, experiment_id: uuid.UUID) -> Experiment:
    """
    Retrieves an experiment by its ID.
    :param db: SQLAlchemy session
    :param experiment_id: UUID of the experiment
    :return: Experiment object or None
    """
    return db.query(Experiment).filter(Experiment.id == experiment_id).first()

def update_status(db: Session, experiment_id: uuid.UUID, status: ExperimentStatus) -> Experiment:
    """
    Updates the status of an existing experiment.
    :param db: SQLAlchemy session
    :param experiment_id: UUID of the experiment
    :param status: New status (ExperimentStatus enum)
    :return: The updated Experiment object or None
    """
    db_experiment = get_experiment(db, experiment_id)
    if db_experiment:
        db_experiment.status = status
        db.commit()
        db.refresh(db_experiment)
    return db_experiment
