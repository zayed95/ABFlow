import pytest
from core.segmentation.hte import run_segment_test, apply_bh_correction, SegmentMetric, SegmentTestResult

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

def test_apply_bh_correction():
    """
    Verify Benjamini-Hochberg correction with m=3 tests.
    p-values: 0.01, 0.04, 0.10. alpha=0.05.
    Thresholds: 
      Rank 1: (1/3)*0.05 = 0.0166. p=0.01 <= 0.0166 (Significant)
      Rank 2: (2/3)*0.05 = 0.0333. p=0.04 > 0.0333 (Not Significant)
      Rank 3: (3/3)*0.05 = 0.05. p=0.10 > 0.05 (Not Significant)
    Max i = 1. Only Rank 1 is significant.
    """
    res1 = SegmentTestResult(
        segment_id=1, n_control=100, n_treatment=100, cr_control=0.1, cr_treatment=0.2, 
        delta=0.1, relative_lift=1.0, ci_lower=0.0, ci_upper=0.2, z_stat=1.0, 
        raw_p_value=0.01, corrected_p_value=None, significant=True
    )
    res2 = SegmentTestResult(
        segment_id=2, n_control=100, n_treatment=100, cr_control=0.1, cr_treatment=0.2, 
        delta=0.1, relative_lift=1.0, ci_lower=0.0, ci_upper=0.2, z_stat=1.0, 
        raw_p_value=0.04, corrected_p_value=None, significant=True
    )
    res3 = SegmentTestResult(
        segment_id=3, n_control=100, n_treatment=100, cr_control=0.1, cr_treatment=0.2, 
        delta=0.1, relative_lift=1.0, ci_lower=0.0, ci_upper=0.2, z_stat=1.0, 
        raw_p_value=0.10, corrected_p_value=None, significant=True
    )
    
    results = [res1, res2, res3]
    corrected = apply_bh_correction(results, alpha=0.05)
    
    # Original order should be preserved
    assert corrected[0].segment_id == 1
    assert corrected[1].segment_id == 2
    assert corrected[2].segment_id == 3
    
    # Significance
    assert corrected[0].significant is True
    assert corrected[1].significant is False
    assert corrected[2].significant is False
    
    # Corrected p-values (min(p*m/rank, 1.0))
    assert corrected[0].corrected_p_value == pytest.approx(0.03) # 0.01 * 3 / 1
    assert corrected[1].corrected_p_value == pytest.approx(0.06) # 0.04 * 3 / 2
    assert corrected[2].corrected_p_value == pytest.approx(0.10) # 0.10 * 3 / 3

import random

def test_bh_simulation():
    """
    Simulate 10 segments: 3 true effects, 7 null effects.
    Verify BH properties.
    """
    random.seed(42)
    results = []
    
    # 3 True Effects (p < 0.01)
    for i in range(3):
        results.append(SegmentTestResult(
            segment_id=i, n_control=100, n_treatment=100, cr_control=0.1, cr_treatment=0.2, 
            delta=0.1, relative_lift=1.0, ci_lower=0.0, ci_upper=0.2, z_stat=2.0, 
            raw_p_value=random.uniform(0.001, 0.009), corrected_p_value=None, significant=False
        ))
        
    # 7 Null Effects (p ~ Uniform(0.05, 0.99))
    # Note: To show uncorrected false positive reduction, we'd need some p between 0.01 and 0.05.
    # We'll add one borderline null p=0.04 to demonstrate.
    results.append(SegmentTestResult(
        segment_id=3, n_control=100, n_treatment=100, cr_control=0.1, cr_treatment=0.1, 
        delta=0.0, relative_lift=0.0, ci_lower=-0.1, ci_upper=0.1, z_stat=0.0, 
        raw_p_value=0.04, corrected_p_value=None, significant=False
    ))
    
    for i in range(4, 10):
        results.append(SegmentTestResult(
            segment_id=i, n_control=100, n_treatment=100, cr_control=0.1, cr_treatment=0.1, 
            delta=0.0, relative_lift=0.0, ci_lower=-0.1, ci_upper=0.1, z_stat=0.0, 
            raw_p_value=random.uniform(0.05, 0.99), corrected_p_value=None, significant=False
        ))
        
    # Uncorrected significance count
    uncorrected_significant = [r for r in results if r.raw_p_value < 0.05]
    assert len(uncorrected_significant) == 4 # 3 true + 1 borderline null
    
    # Apply BH
    corrected = apply_bh_correction(results, alpha=0.05)
    
    # Verify at least 2 of 3 true effects survive
    true_significant = [r for r in corrected if r.segment_id < 3 and r.significant]
    assert len(true_significant) >= 2
    
    # Verify false positive (segment 3) is filtered out
    fp = next(r for r in corrected if r.segment_id == 3)
    assert fp.significant is False
    
    # Verify uncorrected FP count (4) > corrected significant count
    corrected_significant = [r for r in corrected if r.significant]
    assert len(uncorrected_significant) > len(corrected_significant)
