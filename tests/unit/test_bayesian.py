"""Unit tests for core.sequential.bayesian.BetaBinomialPosterior."""
import pytest

from core.sequential.bayesian import BetaBinomialPosterior, PosteriorState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def uniform_prior() -> BetaBinomialPosterior:
    """Beta(1, 1) — flat / uniform prior."""
    return BetaBinomialPosterior(alpha_prior=1, beta_prior=1)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_priors_stored(self, uniform_prior):
        assert uniform_prior.alpha_prior == 1.0
        assert uniform_prior.beta_prior == 1.0

    def test_posteriors_equal_priors_on_init(self, uniform_prior):
        assert uniform_prior.alpha_posterior == uniform_prior.alpha_prior
        assert uniform_prior.beta_posterior == uniform_prior.beta_prior

    def test_custom_priors(self):
        model = BetaBinomialPosterior(alpha_prior=2, beta_prior=5)
        assert model.alpha_prior == 2.0
        assert model.beta_prior == 5.0
        assert model.alpha_posterior == 2.0
        assert model.beta_posterior == 5.0


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_returns_posterior_state(self, uniform_prior):
        result = uniform_prior.update(n_new_conversions=10, n_new_trials=100)
        assert isinstance(result, PosteriorState)

    def test_alpha_beta_updated_correctly(self, uniform_prior):
        uniform_prior.update(n_new_conversions=10, n_new_trials=100)
        # alpha_post = 1 + 10 = 11, beta_post = 1 + (100 - 10) = 91
        assert uniform_prior.alpha_posterior == 11.0
        assert uniform_prior.beta_posterior == 91.0

    def test_cumulative_updates(self, uniform_prior):
        uniform_prior.update(n_new_conversions=5, n_new_trials=50)
        uniform_prior.update(n_new_conversions=5, n_new_trials=50)
        assert uniform_prior.alpha_posterior == 11.0
        assert uniform_prior.beta_posterior == 91.0

    # Validation errors
    def test_negative_conversions_raises(self, uniform_prior):
        with pytest.raises(ValueError, match="non-negative"):
            uniform_prior.update(n_new_conversions=-1, n_new_trials=10)

    def test_negative_trials_raises(self, uniform_prior):
        with pytest.raises(ValueError, match="non-negative"):
            uniform_prior.update(n_new_conversions=0, n_new_trials=-1)

    def test_conversions_exceed_trials_raises(self, uniform_prior):
        with pytest.raises(ValueError, match="cannot exceed"):
            uniform_prior.update(n_new_conversions=11, n_new_trials=10)

    def test_zero_trials_allowed(self, uniform_prior):
        """Edge case: zero observations should not change the posterior."""
        before_alpha = uniform_prior.alpha_posterior
        before_beta = uniform_prior.beta_posterior
        uniform_prior.update(n_new_conversions=0, n_new_trials=0)
        assert uniform_prior.alpha_posterior == before_alpha
        assert uniform_prior.beta_posterior == before_beta


# ---------------------------------------------------------------------------
# mean() — core statistical assertion
# ---------------------------------------------------------------------------

class TestMean:
    def test_posterior_mean_close_to_observed_rate(self, uniform_prior):
        """
        After 10 conversions in 100 trials from Beta(1,1),
        the posterior is Beta(11, 91) with mean ≈ 11/102 ≈ 0.1078.
        It should be within 0.02 of the observed rate of 0.10.
        """
        uniform_prior.update(n_new_conversions=10, n_new_trials=100)
        assert abs(uniform_prior.mean() - 0.10) < 0.02

    def test_mean_equals_expected_value_property(self, uniform_prior):
        uniform_prior.update(n_new_conversions=10, n_new_trials=100)
        assert uniform_prior.mean() == uniform_prior.expected_value

    def test_mean_in_unit_interval(self, uniform_prior):
        uniform_prior.update(n_new_conversions=10, n_new_trials=100)
        assert 0.0 < uniform_prior.mean() < 1.0


# ---------------------------------------------------------------------------
# credible_interval()
# ---------------------------------------------------------------------------

class TestCredibleInterval:
    def test_95_ci_contains_true_rate(self, uniform_prior):
        """
        After 10 conversions in 100 trials, the 95% credible interval
        should contain the observed rate of 0.10.
        """
        uniform_prior.update(n_new_conversions=10, n_new_trials=100)
        lower, upper = uniform_prior.credible_interval(width=0.95)
        assert lower < 0.10 < upper

    def test_ci_returns_tuple_of_two_floats(self, uniform_prior):
        uniform_prior.update(n_new_conversions=10, n_new_trials=100)
        result = uniform_prior.credible_interval()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_lower_less_than_upper(self, uniform_prior):
        uniform_prior.update(n_new_conversions=10, n_new_trials=100)
        lower, upper = uniform_prior.credible_interval()
        assert lower < upper

    def test_wider_interval_contains_narrower(self, uniform_prior):
        uniform_prior.update(n_new_conversions=10, n_new_trials=100)
        lo90, hi90 = uniform_prior.credible_interval(width=0.90)
        lo99, hi99 = uniform_prior.credible_interval(width=0.99)
        assert lo99 < lo90 and hi90 < hi99

    def test_invalid_width_zero_raises(self, uniform_prior):
        with pytest.raises(ValueError, match="width must be in"):
            uniform_prior.credible_interval(width=0)

    def test_invalid_width_one_raises(self, uniform_prior):
        with pytest.raises(ValueError, match="width must be in"):
            uniform_prior.credible_interval(width=1)

    def test_invalid_width_negative_raises(self, uniform_prior):
        with pytest.raises(ValueError, match="width must be in"):
            uniform_prior.credible_interval(width=-0.5)
