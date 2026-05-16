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
class SegmentResult(BaseModel):
    segment_id: Optional[int]
    n_control: int
    n_treatment: int
    cr_control: float
    cr_treatment: float
    delta: float
    relative_lift: float
    ci_lower: float
    ci_upper: float
    corrected_p_value: Optional[float]
    significant: bool

    class Config:
        from_attributes = True

class PosteriorStateSchema(BaseModel):
    alpha_posterior: float
    beta_posterior: float
    expected_value: float
    variance: float

class OverallResults(BaseModel):
    control: PosteriorStateSchema
    treatment: PosteriorStateSchema
    prob_b_beats_a: float
    decision: str

class ResultsResponse(BaseModel):
    overall: Optional[OverallResults] = None
    segment_results: Optional[list[SegmentResult]] = None
    n_segments: Optional[int] = None
    n_significant_segments: int

    class Config:
        from_attributes = True

class SegmentProfile(BaseModel):
    segment_id: int
    label: str
    centroids: dict[str, float]

    class Config:
        from_attributes = True
