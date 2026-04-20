from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
from db.session import get_db
from api.schemas import ExperimentCreate, ExperimentResponse
from db.repositories import experiment_repo

experiment_router = APIRouter(prefix="/experiments", tags=["experiments"])

@experiment_router.post("/", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_new_experiment(experiment: ExperimentCreate, db: Session = Depends(get_db)):
    """
    Creates a new experiment.
    """
    # Convert Pydantic model to dict for the repo
    experiment_data = experiment.model_dump()
    return experiment_repo.create_experiment(db, experiment_data)

@experiment_router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment_details(experiment_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Retrieves details of a specific experiment.
    Returns 404 if the experiment does not exist.
    """
    db_experiment = experiment_repo.get_experiment(db, experiment_id)
    if db_experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found"
        )
    return db_experiment