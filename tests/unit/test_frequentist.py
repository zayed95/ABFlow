"""Unit tests for core.sequential.frequentist."""
import pytest
import numpy as np
from core.sequential.frequentist import OBrienFlemingBoundary

def test_obrien_fleming_boundaries():
    """Verify that boundaries are conservative early and lenient late."""
    n_looks = 5
    obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=n_looks)
    
    boundaries = obf.get_all_boundaries()
    assert len(boundaries) == n_looks
    
    # Boundary 1 should be significantly larger than Boundary 5
    assert boundaries[0] > boundaries[-1]
    
    # Final boundary should be approximately 1.96 for alpha=0.05
    assert pytest.approx(boundaries[-1], rel=1e-3) == 1.960
    
    # Check monotonicity
    for i in range(len(boundaries) - 1):
        assert boundaries[i] > boundaries[i+1]

def test_test_at_look_early_stop():
    """Test significance at different looks with a fixed effect."""
    obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=5)
    
    # Case: Strong effect that might not be significant at Look 1 but is at Look 5
    # p1=0.1, p2=0.15 (delta=0.05)
    # n=500 per variant
    n_c, c_c = 500, 50
    n_t, c_t = 500, 75
    
    # Look 1: Boundary is ~4.38
    res1 = obf.test_at_look(n_c, c_c, n_t, c_t, look_number=1)
    # Z-stat will be ~2.4ish
    assert not res1["reject"]
    assert res1["boundary"] > 4.0
    
    # Look 5: Boundary is ~1.96
    res5 = obf.test_at_look(n_c, c_c, n_t, c_t, look_number=5)
    assert res5["reject"]
    assert pytest.approx(res5["boundary"], rel=1e-3) == 1.960

def test_z_score_calculation():
    """Verify the Z-score calculation logic."""
    obf = OBrienFlemingBoundary(alpha=0.05, n_planned_looks=1)
    
    # p1=0.1, p2=0.2, n=100
    # p_pool = 30/200 = 0.15
    # SE = sqrt(0.15 * 0.85 * (1/100 + 1/100)) = sqrt(0.1275 * 0.02) = sqrt(0.00255) = 0.050497
    # Z = (0.2 - 0.1) / 0.050497 = 1.9803
    res = obf.test_at_look(100, 10, 100, 20, look_number=1)
    
    assert pytest.approx(res["z_stat"], rel=1e-3) == 1.9803
    assert res["reject"] is True # 1.98 > 1.96
