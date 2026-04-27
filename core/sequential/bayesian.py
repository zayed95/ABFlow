from typing import NamedTuple

import numpy as np
from scipy.stats import beta as beta_dist


class PosteriorState(NamedTuple):
    """Snapshot of the posterior Beta distribution after an update."""
    alpha_posterior: float
    beta_posterior: float
    expected_value: float
    variance: float


class BetaBinomialPosterior:
    """
    Beta-Binomial posterior distribution for modeling binary outcomes (success/failure).
    Uses a Beta distribution as the conjugate prior for a Binomial likelihood.
    """
    def __init__(self, alpha_prior: float = 1, beta_prior: float = 1):
        """
        Initialize the Beta prior and posterior.

        Args:
            alpha_prior: Prior alpha parameter (pseudo-successes). Defaults to 1 (Uniform prior).
            beta_prior:  Prior beta parameter (pseudo-failures).  Defaults to 1 (Uniform prior).
        """
        # Store the original priors so they can be inspected or used to reset.
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)

        # Posteriors start equal to the priors; they are updated as data arrives.
        self.alpha_posterior = self.alpha_prior
        self.beta_posterior = self.beta_prior


    def update(self, n_new_conversions: int, n_new_trials: int) -> PosteriorState:
        """
        Incorporate new observations into the posterior.

        Args:
            n_new_conversions: Number of successes (conversions) observed.
            n_new_trials:      Total number of trials observed.

        Returns:
            PosteriorState snapshot after the update.

        Raises:
            ValueError: If inputs are negative or n_new_conversions > n_new_trials.
        """
        if n_new_conversions < 0 or n_new_trials < 0:
            raise ValueError(
                "n_new_conversions and n_new_trials must both be non-negative; "
                f"got n_new_conversions={n_new_conversions}, n_new_trials={n_new_trials}."
            )
        if n_new_conversions > n_new_trials:
            raise ValueError(
                "n_new_conversions cannot exceed n_new_trials; "
                f"got n_new_conversions={n_new_conversions}, n_new_trials={n_new_trials}."
            )

        self.alpha_posterior += n_new_conversions
        self.beta_posterior += n_new_trials - n_new_conversions
        return PosteriorState(
            alpha_posterior=self.alpha_posterior,
            beta_posterior=self.beta_posterior,
            expected_value=self.expected_value,
            variance=self.variance,
        )

    def mean(self) -> float:
        """
        Return the expected conversion rate of the posterior Beta distribution.

        Computed as alpha_posterior / (alpha_posterior + beta_posterior).
        """
        return self.alpha_posterior / (self.alpha_posterior + self.beta_posterior)

    @property
    def expected_value(self) -> float:
        """
        Alias for mean(). Returns the expected conversion rate.
        """
        return self.mean()
    
    @property
    def variance(self) -> float:
        """
        Calculate the variance of the posterior Beta distribution.
        """
        a, b = self.alpha_posterior, self.beta_posterior
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def credible_interval(self, width: float = 0.95) -> tuple[float, float]:
        """
        Return the equal-tailed credible interval of the posterior Beta distribution.

        Args:
            width: Probability mass to include (default 0.95 → 95% credible interval).
                   Must be in (0, 1).

        Returns:
            (lower, upper) tuple of floats representing the credible interval bounds.

        Raises:
            ValueError: If width is not strictly between 0 and 1.
        """
        if not (0 < width < 1):
            raise ValueError(f"width must be in (0, 1), got {width}.")

        tail = (1 - width) / 2
        lower = beta_dist.ppf(tail, self.alpha_posterior, self.beta_posterior)
        upper = beta_dist.ppf(1 - tail, self.alpha_posterior, self.beta_posterior)
        return (lower, upper)


def prob_b_beats_a(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    n_samples: int = 20_000,
) -> float:
    """
    Estimate P(B > A) via Monte Carlo sampling.

    Draws n_samples from each posterior Beta distribution and returns
    the fraction of samples where B's draw exceeds A's draw.

    Args:
        posterior_a: Posterior for variant A.
        posterior_b: Posterior for variant B.
        n_samples:   Number of Monte Carlo samples (default 20,000).

    Returns:
        Estimated probability that variant B's true conversion rate
        exceeds variant A's, as a float in [0, 1].

    Raises:
        ValueError: If n_samples is not a positive integer.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples}.")

    samples_a = np.random.beta(posterior_a.alpha_posterior, posterior_a.beta_posterior, size=n_samples)
    samples_b = np.random.beta(posterior_b.alpha_posterior, posterior_b.beta_posterior, size=n_samples)
    return float(np.mean(samples_b > samples_a))