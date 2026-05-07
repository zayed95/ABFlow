"""Unit tests for core.sequential.bayesian.BetaBinomialPosterior."""
import pytest
import numpy as np
from core.sequential.bayesian import BetaBinomialPosterior, PosteriorState


@pytest.fixture()
def uniform_prior() -> BetaBinomialPosterior:
    """Beta(1, 1) — flat / uniform prior."""
    return BetaBinomialPosterior(prior_type='uniform')


class TestInit:
    def test_default_is_uniform(self):
        model = BetaBinomialPosterior()
        assert model.alpha_prior == 1.0
        assert model.beta_prior == 1.0

    def test_jeffreys_prior(self):
        model = BetaBinomialPosterior(prior_type='jeffreys')
        assert model.alpha_prior == 0.5
        assert model.beta_prior == 0.5

    def test_informed_prior(self):
        cr = 0.1
        model = BetaBinomialPosterior(prior_type='informed', historical_cr=cr)
        # Beta(2, 1/0.1 - 2) = Beta(2, 8)
        assert model.alpha_prior == 2.0
        assert model.beta_prior == 8.0

    def test_custom_priors(self):
        model = BetaBinomialPosterior(alpha_prior=2, beta_prior=5)
        assert model.alpha_prior == 2.0
        assert model.beta_prior == 5.0
        assert model.alpha_posterior == 2.0
        assert model.beta_posterior == 5.0

    def test_informed_prior_validation(self):
        with pytest.raises(ValueError, match="historical_cr must be provided"):
            BetaBinomialPosterior(prior_type='informed')
        with pytest.raises(ValueError, match="must be in"):
            BetaBinomialPosterior(prior_type='informed', historical_cr=0.7)


class TestUpdate:
    @pytest.mark.parametrize("alpha_p, beta_p, conv, trials, expected_alpha, expected_beta", [
        (1.0, 1.0, 10, 100, 11.0, 91.0),   # Uniform + 10/100
        (0.5, 0.5, 50, 100, 50.5, 50.5), # Jeffreys + 50/100
        (2.0, 8.0, 5, 10, 7.0, 13.0),     # Informed (0.1 CR) + 5/10
        (10.0, 90.0, 0, 50, 10.0, 140.0)  # Strong prior + 0/50
    ])
    def test_posterior_update_correctness(self, alpha_p, beta_p, conv, trials, expected_alpha, expected_beta):
        """
        Verify that known inputs (priors + data) lead to known posterior parameters.
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (trials - successes)
        """
        model = BetaBinomialPosterior(alpha_prior=alpha_p, beta_prior=beta_p)
        model.update(n_new_conversions=conv, n_new_trials=trials)
        assert model.alpha_posterior == expected_alpha
        assert model.beta_posterior == expected_beta

    def test_invalid_updates(self, uniform_prior):
        with pytest.raises(ValueError):
            uniform_prior.update(-1, 10)
        with pytest.raises(ValueError):
            uniform_prior.update(11, 10)

    def test_invalid_credible_interval_width(self, uniform_prior):
        with pytest.raises(ValueError, match="width must be in"):
            uniform_prior.credible_interval(width=1.5)
        with pytest.raises(ValueError, match="width must be in"):
            uniform_prior.credible_interval(width=-0.1)

    def test_invalid_prob_b_beats_a_samples(self, uniform_prior):
        with pytest.raises(ValueError, match="n_samples must be a positive integer"):
            from core.sequential.bayesian import prob_b_beats_a
            prob_b_beats_a(uniform_prior, uniform_prior, n_samples=0)


class TestStatistics:
    def test_mean_formula(self):
        """Verify that mean() equals alpha / (alpha + beta)."""
        model = BetaBinomialPosterior(alpha_prior=3, beta_prior=7)
        model.update(n_new_conversions=5, n_new_trials=10)
        # alpha = 3+5=8, beta = 7+5=12
        expected_mean = 8 / (8 + 12)  # 0.4
        assert model.mean() == pytest.approx(expected_mean)
        assert model.mean() == model.expected_value

    def test_credible_interval_coverage(self):
        """
        Simulate 1,000 posteriors with a known true rate.
        Verify that the 95% Credible Interval contains the true rate 94–96% of the time.
        """
        np.random.seed(42)
        true_rate = 0.15
        n_trials = 500
        n_simulations = 1000
        
        contained_count = 0
        for _ in range(n_simulations):
            # Use uniform prior for objective coverage test
            model = BetaBinomialPosterior(prior_type='uniform')
            
            # Generate synthetic data
            conversions = np.random.binomial(n=n_trials, p=true_rate)
            model.update(n_new_conversions=conversions, n_new_trials=n_trials)
            
            # Check if true rate is within 95% CI
            lower, upper = model.credible_interval(width=0.95)
            if lower <= true_rate <= upper:
                contained_count += 1
                
        coverage = contained_count / n_simulations
        print(f"Coverage: {coverage:.2%}")
        # With 1000 simulations, we expect coverage to be very close to 95%
        assert 0.94 <= coverage <= 0.96
