import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from .models.enums import ExperimentStatusEnum
from .models.db_schemas.experiment import ExperimentBase

class ExperimentCreate(ExperimentBase):
    pass

class ExperimentResponse(ExperimentBase):
    id: uuid.UUID
    status: ExperimentStatusEnum
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AssignmentEnroll(BaseModel):
    experiment_id: uuid.UUID
    user_id: str

class AssignmentResponse(BaseModel):
    id: uuid.UUID
    experiment_id: uuid.UUID
    user_id: str
    variant: str
    segment_id: Optional[int] = None
    enrolled_at: datetime

    class Config:
        from_attributes = True