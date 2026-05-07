"""Unit tests for core.sequential.frequentist."""
import pytest
from core.sequential.frequentist import OBrienFlemingBoundary


class TestOBrienFlemingBoundary:
    def test_initialization(self):
        """Test basic initialization and validation."""
        obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)
        assert obf.alpha == 0.05
        assert obf.n_planned_looks == 5
        assert len(obf.get_all_boundaries()) == 5

    def test_boundaries_decrease(self):
        """
        Verify the O'Brien-Fleming property: boundaries (critical values) 
        must decrease as the look number increases.
        """
        obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)
        boundaries = obf.get_all_boundaries()
        
        # Check that each subsequent boundary is smaller than the previous one
        for i in range(len(boundaries) - 1):
            assert boundaries[i] > boundaries[i+1], f"Boundary at look {i+1} should be > look {i+2}"
        
        # Last boundary should be close to the standard normal critical value (approx 1.96 for alpha=0.05)
        # For OBF, the last boundary is exactly z_alpha/2 if k=K.
        assert boundaries[-1] == pytest.approx(1.95996, abs=1e-4)

    def test_large_z_stat_rejects(self):
        """Test that a very large z-statistic triggers reject=True."""
        obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)
        
        # Look 1 boundary for alpha=0.05, K=5 is 1.96 * sqrt(5) approx 4.38
        # A Z-score of 10 should definitely reject.
        res = obf.test_at_look(n_control=100, conversions_control=10, 
                               n_treatment=100, conversions_treatment=90, 
                               look_number=1)
        
        assert res["reject"] is True
        assert res["z_stat"] > res["boundary"]
        assert res["p_value"] < 0.0001

    def test_small_z_stat_continues(self):
        """Test that a small z-statistic does not trigger reject."""
        obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)
        
        # No difference
        res = obf.test_at_look(n_control=100, conversions_control=10, 
                               n_treatment=100, conversions_treatment=10, 
                               look_number=1)
        
        assert res["reject"] is False
        assert res["z_stat"] == 0.0

    def test_invalid_look_number(self):
        obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)
        with pytest.raises(ValueError):
            obf.get_boundary(0)
        with pytest.raises(ValueError):
            obf.get_boundary(6)

    def test_invalid_initialization(self):
        with pytest.raises(ValueError, match="Alpha must be"):
            OBrienFlemingBoundary(alpha=1.5)
        with pytest.raises(ValueError, match="n_planned_looks must be"):
            OBrienFlemingBoundary(n_planned_looks=0)

    def test_evaluate_method(self):
        obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)
        # Boundary for look 5 is approx 1.96
        assert obf.evaluate(z_score=2.0, look_number=5)
        assert not obf.evaluate(z_score=1.0, look_number=5)

    def test_empty_data_cases(self):
        obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)
        # Zero trials
        res = obf.test_at_look(0, 0, 0, 0, 1)
        assert res["reject"] is False
        assert res["z_stat"] == 0.0
        
        # Zero conversions (p_pool = 0)
        res = obf.test_at_look(100, 0, 100, 0, 1)
        assert res["reject"] is False
        assert res["z_stat"] == 0.0

    def test_repr(self):
        obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)
        assert "OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)" in repr(obf)
