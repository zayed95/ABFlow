import uuid
from datetime import datetime
from typing import Optional
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