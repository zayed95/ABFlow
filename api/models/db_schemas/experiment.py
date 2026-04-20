from typing import Any, Dict
from pydantic import BaseModel, Field

class ExperimentBase(BaseModel):
    name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(default=42)