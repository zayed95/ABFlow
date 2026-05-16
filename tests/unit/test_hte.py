import pytest
from core.segmentation.hte import run_segment_test, SegmentMetric

def test_run_segment_test_success():
    """
    Verify run_segment_test with known inputs.
    Control: 100 users, 10 conversions (10%)
    Treatment: 100 users, 20 conversions (20%)
    """
    control = SegmentMetric(
        segment_id=1, variant="control", n_users=100, 
        n_conversions=10, conversion_rate=0.1, is_testable=True
    )
    treatment = SegmentMetric(
        segment_id=1, variant="treatment", n_users=100, 
        n_conversions=20, conversion_rate=0.2, is_testable=True
    )
    
    result = run_segment_test(control, treatment)
    
    assert result.segment_id == 1
    assert result.cr_control == 0.1
    assert result.cr_treatment == 0.2
    assert result.delta == pytest.approx(0.1)
    assert result.relative_lift == pytest.approx(1.0) # (0.2 - 0.1) / 0.1 = 1.0
    
    # Check CI
    # SE = sqrt(0.1*0.9/100 + 0.2*0.8/100) = sqrt(0.0009 + 0.0016) = sqrt(0.0025) = 0.05
    # CI = 0.1 +/- 1.96 * 0.05 = 0.1 +/- 0.098 = [0.002, 0.198]
    assert result.ci_lower == pytest.approx(0.002)
    assert result.ci_upper == pytest.approx(0.198)
    
    assert result.raw_p_value < 0.05
    assert result.z_stat > 0
    assert result.significant is True

def test_run_segment_test_large_sample():
    """
    Verify run_segment_test with n=1000 per variant.
    Control: CR=0.10 (100 conv)
    Treatment: CR=0.15 (150 conv)
    Expected: delta=0.05, p < 0.01, CI does not contain 0.
    """
    # Using n=1000 per variant to ensure p-value < 0.01 for the 5% delta
    control = SegmentMetric(
        segment_id=1, variant="control", n_users=1000, 
        n_conversions=100, conversion_rate=0.1, is_testable=True
    )
    treatment = SegmentMetric(
        segment_id=1, variant="treatment", n_users=1000, 
        n_conversions=150, conversion_rate=0.15, is_testable=True
    )
    
    result = run_segment_test(control, treatment)
    
    assert result.delta == pytest.approx(0.05)
    assert result.raw_p_value < 0.01
    assert result.ci_lower > 0
    assert result.ci_upper > result.ci_lower
    assert result.significant is True

def test_run_segment_test_mismatch():
    control = SegmentMetric(segment_id=1, variant="control", n_users=100, n_conversions=10, conversion_rate=0.1, is_testable=True)
    treatment = SegmentMetric(segment_id=2, variant="treatment", n_users=100, n_conversions=20, conversion_rate=0.2, is_testable=True)
    
    with pytest.raises(ValueError, match="Segment ID mismatch"):
        run_segment_test(control, treatment)
