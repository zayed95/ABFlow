"""Unit tests for core.sequential.models."""
import datetime
from core.sequential.models import PosteriorState

def test_posterior_state_instantiation():
    """Verify that PosteriorState dataclass can be instantiated."""
    now = datetime.datetime.now()
    state = PosteriorState(
        alpha_prior=1.0,
        beta_prior=1.0,
        alpha_post=10.0,
        beta_post=90.0,
        n_trials=100,
        n_conversions=9,
        last_updated=now
    )
    assert state.alpha_prior == 1.0
    assert state.n_trials == 100
    assert state.last_updated == now
