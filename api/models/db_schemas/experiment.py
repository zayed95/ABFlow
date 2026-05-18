from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field, model_validator, ConfigDict

class ExperimentConfig(BaseModel):
    prior_type: Literal['uniform', 'jeffreys', 'informed'] = Field(default='uniform')
    historical_cr: Optional[float] = None
    outcome_window_hours: int = Field(default=72)
    
    model_config = ConfigDict(extra='allow')

    @model_validator(mode='after')
    def check_informed_prior(self):
        if self.prior_type == 'informed' and self.historical_cr is None:
            raise ValueError("historical_cr is required when prior_type is 'informed'")
        return self

class ExperimentBase(BaseModel):
    name: str
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)
    seed: int = Field(default=42)