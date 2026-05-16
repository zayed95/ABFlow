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

def apply_bh_correction(results: list[SegmentTestResult], alpha=0.05) -> list[SegmentTestResult]:
    """
    Applies the Benjamini-Hochberg (BH) procedure to control the False Discovery Rate (FDR).
    
    Why BH instead of Bonferroni?
    -----------------------------
    Bonferroni correction controls the Family-Wise Error Rate (FWER), which is the 
    probability of making AT LEAST ONE false positive. This is extremely conservative 
    when the number of tests (k) is large, often leading to high false negative rates.
    
    Benjamini-Hochberg controls the False Discovery Rate (FDR), which is the expected 
    proportion of "discoveries" (significant results) that are actually false positives. 
    BH is significantly more powerful than Bonferroni, especially when k > 3, as it 
    allows for some false positives in exchange for a much higher discovery rate.
    
    1. Sort results by raw_p_value ascending.
    2. Identify the largest rank i where p_value <= (i/m) * alpha.
    3. Flag all tests with rank <= i as significant.
    4. Compute corrected p-values.
    
    :param results: List of SegmentTestResult objects
    :param alpha: Target FDR level (default 0.05)
    :return: List of results with updated corrected_p_value and significant flag.
    """
    m = len(results)
    if m == 0:
        return []

    # Sort results by raw p-value to determine ranks
    # We use a temporary list with original indices to restore order later
    indexed_results = sorted(enumerate(results), key=lambda x: x[1].raw_p_value)
    
    # Find the largest rank i where p_i <= (i/m) * alpha
    max_i = -1
    for i, (_, res) in enumerate(indexed_results, start=1):
        threshold = (i / m) * alpha
        if res.raw_p_value <= threshold:
            max_i = i

    # Update corrected p-values and significance
    # Formula for corrected p-value: min(raw_p_value * m / rank, 1.0)
    for i, (_, res) in enumerate(indexed_results, start=1):
        res.corrected_p_value = min(res.raw_p_value * m / i, 1.0)
        res.significant = (i <= max_i) if max_i != -1 else False

    # Restore original order
    sorted_by_index = sorted(indexed_results, key=lambda x: x[0])
    return [res for _, res in sorted_by_index]
