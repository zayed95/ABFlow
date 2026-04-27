"""Decision result dataclass and evaluation logic for sequential Bayesian A/B tests."""
from dataclasses import dataclass
from enum import Enum

import numpy as np
from core.sequential.bayesian import BetaBinomialPosterior, PosteriorState, prob_b_beats_a as _prob_b_beats_a


class Decision(str, Enum):
    """Possible outcomes of a sequential test evaluation."""
    CONTINUE     = "CONTINUE"      # Not enough data or no threshold crossed yet.
    STOP_WINNER  = "STOP_WINNER"   # B beats A with sufficient probability.
    STOP_NULL    = "STOP_NULL"     # A beats B with sufficient probability.


@dataclass(frozen=True)
class DecisionResult:
    """
    Immutable snapshot of a sequential test decision.

    Attributes:
        decision:       Test outcome — one of Decision.CONTINUE, Decision.STOP_WINNER,
                        or Decision.STOP_NULL.
        prob_b_beats_a: Monte Carlo estimate of P(B > A) in [0, 1].
        posterior_a:    Snapshot of variant A's posterior at decision time.
        posterior_b:    Snapshot of variant B's posterior at decision time.
        n_total:        Total real observations across both variants
                        (prior pseudo-counts excluded).
    """

    decision: Decision
    prob_b_beats_a: float
    posterior_a: PosteriorState
    posterior_b: PosteriorState
    n_total: int


def evaluate_decision(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    threshold_win: float = 0.95,
    threshold_null: float = 0.05,
    min_samples: int = 100,
) -> DecisionResult:
    """
    Apply the sequential stopping rule and return a DecisionResult.

    The decision logic is (evaluated in order):
      1. If total real observations < min_samples  → CONTINUE  (burn-in guard;
         probability is ignored to prevent early stopping on tiny samples).
      2. If P(B > A) > threshold_win               → STOP_WINNER.
      3. If P(B > A) < threshold_null              → STOP_NULL.
      4. Otherwise                                 → CONTINUE.

    Args:
        posterior_a:     Posterior for the control variant (A).
        posterior_b:     Posterior for the treatment variant (B).
        threshold_win:   P(B > A) strictly above which B is declared the winner (default 0.95).
        threshold_null:  P(B > A) strictly below which A is declared the winner (default 0.05).
        min_samples:     Minimum real observations before any stopping decision (default 100).

    Returns:
        DecisionResult with a Decision enum value, P(B > A), both posterior
        snapshots, and the total real observation count.

    Raises:
        ValueError: If threshold_null >= threshold_win, either threshold is
                    outside (0, 1), or min_samples is not positive.
    """
    if not (0 < threshold_null < threshold_win < 1):
        raise ValueError(
            f"Thresholds must satisfy 0 < threshold_null < threshold_win < 1; "
            f"got threshold_null={threshold_null}, threshold_win={threshold_win}."
        )
    if min_samples <= 0:
        raise ValueError(f"min_samples must be a positive integer, got {min_samples}.")

    # Real observations = posterior params minus the prior pseudo-counts.
    n_a = (posterior_a.alpha_posterior - posterior_a.alpha_prior) + (posterior_a.beta_posterior - posterior_a.beta_prior)
    n_b = (posterior_b.alpha_posterior - posterior_b.alpha_prior) + (posterior_b.beta_posterior - posterior_b.beta_prior)
    n_total = int(n_a + n_b)

    # Capture immutable snapshots before running the MC estimate.
    state_a = PosteriorState(
        alpha_posterior=posterior_a.alpha_posterior,
        beta_posterior=posterior_a.beta_posterior,
        expected_value=posterior_a.expected_value,
        variance=posterior_a.variance,
    )
    state_b = PosteriorState(
        alpha_posterior=posterior_b.alpha_posterior,
        beta_posterior=posterior_b.beta_posterior,
        expected_value=posterior_b.expected_value,
        variance=posterior_b.variance,
    )

    p = _prob_b_beats_a(posterior_a, posterior_b)

    if n_total < min_samples:
        decision = Decision.CONTINUE
    elif p > threshold_win:
        decision = Decision.STOP_WINNER
    elif p < threshold_null:
        decision = Decision.STOP_NULL
    else:
        decision = Decision.CONTINUE

    return DecisionResult(
        decision=decision,
        prob_b_beats_a=p,
        posterior_a=state_a,
        posterior_b=state_b,
        n_total=n_total,
    )


def expected_loss(
    posterior_a: BetaBinomialPosterior,
    posterior_b: BetaBinomialPosterior,
    n_samples: int = 20_000,
) -> float:
    """
    Estimate the expected loss of choosing the current leader via Monte Carlo sampling.

    The leader is determined by comparing the posterior means.
    If B is the leader, loss = E[max(A - B, 0)].
    If A is the leader, loss = E[max(B - A, 0)].

    Args:
        posterior_a: Posterior for variant A.
        posterior_b: Posterior for variant B.
        n_samples:   Number of Monte Carlo samples (default 20,000).

    Returns:
        Estimated expected loss as a float.
    """
    samples_a = np.random.beta(
        posterior_a.alpha_posterior, posterior_a.beta_posterior, size=n_samples
    )
    samples_b = np.random.beta(
        posterior_b.alpha_posterior, posterior_b.beta_posterior, size=n_samples
    )

    if posterior_b.mean() >= posterior_a.mean():
        # Loss of choosing B
        return float(np.mean(np.maximum(samples_a - samples_b, 0)))
    else:
        # Loss of choosing A
        return float(np.mean(np.maximum(samples_b - samples_a, 0)))
