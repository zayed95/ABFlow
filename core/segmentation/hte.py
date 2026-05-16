import math
from dataclasses import dataclass
from typing import Optional
from statsmodels.stats.proportion import proportions_ztest
from db.repositories.event_repo import SegmentMetric

@dataclass
class SegmentTestResult:
    """
    Result of a statistical test comparing variants within a segment.
    """
    segment_id: Optional[int]
    n_control: int
    n_treatment: int
    cr_control: float
    cr_treatment: float
    delta: float
    relative_lift: float
    ci_lower: float
    ci_upper: float
    z_stat: float
    raw_p_value: float
    corrected_p_value: Optional[float]
    significant: bool

def run_segment_test(control: SegmentMetric, treatment: SegmentMetric) -> SegmentTestResult:
    """
    Performs a two-proportion z-test comparing conversion rates between variants.
    Computes delta, relative lift, and 95% confidence intervals.
    """
    if control.segment_id != treatment.segment_id:
        raise ValueError(
            f"Segment ID mismatch: {control.segment_id} vs {treatment.segment_id}"
        )

    # 1. Basic Stats
    n_c, n_t = control.n_users, treatment.n_users
    p_c, p_t = control.conversion_rate, treatment.conversion_rate
    
    delta = p_t - p_c
    relative_lift = delta / p_c if p_c > 0 else 0.0

    # 2. Z-Test
    count = [treatment.n_conversions, control.n_conversions]
    nobs = [n_t, n_c]
    z_stat, p_value = proportions_ztest(count=count, nobs=nobs, alternative='two-sided')

    # 3. 95% Confidence Interval for Delta (Normal Approximation)
    # SE = sqrt( p_c*(1-p_c)/n_c + p_t*(1-p_t)/n_t )
    se = math.sqrt((p_c * (1 - p_c) / n_c) + (p_t * (1 - p_t) / n_t))
    margin_of_error = 1.96 * se
    ci_lower = delta - margin_of_error
    ci_upper = delta + margin_of_error

    return SegmentTestResult(
        segment_id=control.segment_id,
        n_control=n_c,
        n_treatment=n_t,
        cr_control=p_c,
        cr_treatment=p_t,
        delta=float(delta),
        relative_lift=float(relative_lift),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        z_stat=float(z_stat),
        raw_p_value=float(p_value),
        corrected_p_value=None,  # To be filled in subsequent steps
        significant=bool(p_value < 0.05)
    )
