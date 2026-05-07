"""Unit tests for core.sequential.decision."""
import pytest
import numpy as np
from core.sequential.bayesian import BetaBinomialPosterior, prob_b_beats_a
from core.sequential.decision import evaluate_decision, Decision, expected_loss


class TestDecisionOutcomes:
    def test_stop_winner(self):
        """Test that B winning by a large margin triggers STOP_WINNER."""
        a = BetaBinomialPosterior()
        a.update(n_new_conversions=10, n_new_trials=100) # 10%
        b = BetaBinomialPosterior()
        b.update(n_new_conversions=30, n_new_trials=100) # 30%
        
        result = evaluate_decision(a, b, min_samples=100)
        assert result.decision == Decision.STOP_WINNER
        assert result.prob_b_beats_a > 0.99

    def test_stop_null(self):
        """Test that A winning by a large margin triggers STOP_NULL."""
        a = BetaBinomialPosterior()
        a.update(n_new_conversions=30, n_new_trials=100) # 30%
        b = BetaBinomialPosterior()
        b.update(n_new_conversions=10, n_new_trials=100) # 10%
        
        result = evaluate_decision(a, b, min_samples=100)
        assert result.decision == Decision.STOP_NULL
        assert result.prob_b_beats_a < 0.01

    def test_continue(self):
        """Test that close results trigger CONTINUE."""
        a = BetaBinomialPosterior()
        a.update(n_new_conversions=20, n_new_trials=100) # 20%
        b = BetaBinomialPosterior()
        b.update(n_new_conversions=22, n_new_trials=100) # 22%
        
        result = evaluate_decision(a, b, min_samples=100)
        assert result.decision == Decision.CONTINUE


class TestDecisionGuards:
    def test_min_samples_guard(self):
        """Test that min_samples prevents stopping even with high probability."""
        a = BetaBinomialPosterior()
        a.update(n_new_conversions=0, n_new_trials=2)
        b = BetaBinomialPosterior()
        b.update(n_new_conversions=2, n_new_trials=2)
        
        # Prob B > A will be very high (approx 0.94 for Beta(1,1) priors)
        # But n_total = 4 < 100
        result = evaluate_decision(a, b, min_samples=100)
        assert result.decision == Decision.CONTINUE
        assert result.n_total == 4

    def test_invalid_parameters(self):
        a, b = BetaBinomialPosterior(), BetaBinomialPosterior()
        with pytest.raises(ValueError, match="Thresholds must satisfy"):
            evaluate_decision(a, b, threshold_win=0.5, threshold_null=0.6)
        with pytest.raises(ValueError, match="min_samples must be a positive integer"):
            evaluate_decision(a, b, min_samples=0)


class TestStatisticalProperties:
    def test_prob_b_beats_a_identical(self):
        """Test that P(B > A) is 0.5 for identical posteriors."""
        np.random.seed(42)
        a = BetaBinomialPosterior()
        a.update(10, 100)
        b = BetaBinomialPosterior()
        b.update(10, 100)
        
        p = prob_b_beats_a(a, b, n_samples=50_000)
        assert p == pytest.approx(0.5, abs=0.01)

    def test_expected_loss_a_leader(self):
        """Test expected loss calculation when A is the leader."""
        np.random.seed(42)
        a = BetaBinomialPosterior()
        a.update(20, 100) # Leader
        b = BetaBinomialPosterior()
        b.update(10, 100)
        
        loss = expected_loss(a, b)
        assert loss > 0
        # Since A is leader, it should use the "else" branch in expected_loss
        assert a.mean() > b.mean()

    def test_expected_loss_identical_peaked(self):
        """Test that expected loss is near zero for identical, highly peaked posteriors."""
        np.random.seed(42)
        a = BetaBinomialPosterior()
        a.update(1000, 10000) # 10% CR, very high confidence
        b = BetaBinomialPosterior()
        b.update(1000, 10000) # Identical
        
        loss = expected_loss(a, b, n_samples=50_000)
        # For peaked distributions, the overlap/tail is very small.
        # Expected loss should be very small.
        assert loss < 0.005
        assert loss >= 0
