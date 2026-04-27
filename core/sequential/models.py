import datetime
from dataclasses import dataclass

@dataclass
class PosteriorState:
    alpha_prior: float
    beta_prior: float 
    alpha_post: float
    beta_post: float 
    n_trials: int 
    n_conversions: int 
    last_updated: datetime