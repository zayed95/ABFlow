from enum import Enum

class VariantEnum(str, Enum):
    CONTROL = "control"
    TREATMENT = "treatment"

class ExperimentStatusEnum(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"

class ExperimentModeEnum(str, Enum):
    BAYESIAN = "bayesian"
    FREQUENTIST = "frequentist"
    