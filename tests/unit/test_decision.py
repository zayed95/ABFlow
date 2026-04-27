"""Unit tests for core.sequential.decision."""
import pytest
from core.sequential.bayesian import BetaBinomialPosterior
from core.sequential.decision import evaluate_decision, Decision, expected_loss


def test_evaluate_decision_b_wins():
    """Test that heavily favoring B leads to STOP_WINNER."""
    a = BetaBinomialPosterior()
    a.update(n_new_conversions=20, n_new_trials=100)
    
    b = BetaBinomialPosterior()
    b.update(n_new_conversions=50, n_new_trials=100)
    
    result = evaluate_decision(a, b, threshold_win=0.95, threshold_null=0.05, min_samples=100)
    
    assert result.decision == Decision.STOP_WINNER
    assert result.prob_b_beats_a > 0.95
    assert result.n_total == 200


def test_evaluate_decision_burn_in():
    """Test that n < min_samples returns CONTINUE regardless of probability."""
    # Probability would be high, but samples are few
    a = BetaBinomialPosterior()
    a.update(n_new_conversions=1, n_new_trials=2)
    
    b = BetaBinomialPosterior()
    b.update(n_new_conversions=2, n_new_trials=2)
    
    # n_total = 4, min_samples = 10
    result = evaluate_decision(a, b, threshold_win=0.90, threshold_null=0.10, min_samples=10)
    
    assert result.decision == Decision.CONTINUE
    assert result.n_total == 4


def test_evaluate_decision_a_wins():
    """Test that heavily favoring A leads to STOP_NULL."""
    a = BetaBinomialPosterior()
    a.update(n_new_conversions=50, n_new_trials=100)
    
    b = BetaBinomialPosterior()
    b.update(n_new_conversions=20, n_new_trials=100)
    
    result = evaluate_decision(a, b, threshold_win=0.95, threshold_null=0.05, min_samples=100)
    
    assert result.decision == Decision.STOP_NULL
    assert result.prob_b_beats_a < 0.05


def test_evaluate_decision_continue():
    """Test that ambiguous results return CONTINUE."""
    a = BetaBinomialPosterior()
    a.update(n_new_conversions=45, n_new_trials=100)
    
    b = BetaBinomialPosterior()
    b.update(n_new_conversions=55, n_new_trials=100)
    
    # Prob will be around 0.90ish, not 0.95
    result = evaluate_decision(a, b, threshold_win=0.99, threshold_null=0.01, min_samples=100)
    
    assert result.decision == Decision.CONTINUE


def test_expected_loss_symmetry():
    """Test basic properties of expected loss."""
    a = BetaBinomialPosterior()
    a.update(10, 100)
    b = BetaBinomialPosterior()
    b.update(15, 100)
    
    loss_ab = expected_loss(a, b)
    
    # Swapping should give similar loss if leader is swapped
    loss_ba = expected_loss(b, a)
    
    # Since leader is B in first and A in second, loss calculation logic is same
    assert abs(loss_ab - loss_ba) < 0.001
    assert loss_ab > 0
