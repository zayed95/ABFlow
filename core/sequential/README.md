# Sequential Analysis Module

This module provides the core statistical logic for sequential A/B testing in ABFlow, supporting both Bayesian and Frequentist (O'Brien-Fleming) approaches.

## Bayesian Sequential Analysis

The primary engine of ABFlow uses a **Beta-Binomial** model to perform interim analyses as data arrives. Unlike fixed-horizon tests, sequential analysis allows for early stopping if a variant is significantly better or if the test is unlikely to reach a conclusion (futility).

### Prior Options
We support three standard prior configurations to represent initial belief before data is collected:

- **Uniform (`uniform`)**: $Beta(1, 1)$. Represents complete ignorance, assigning equal probability density to all conversion rates in $[0, 1]$.
- **Jeffreys (`jeffreys`)**: $Beta(0.5, 0.5)$. A non-informative prior invariant under reparameterization; preferred for objective Bayesian analysis.
- **Informed (`informed`)**: $Beta(2, 1/CR - 2)$. Used when a historical baseline conversion rate (CR) is known. It centers the belief around the baseline while maintaining enough flexibility to adapt to new data.

### Decision Thresholds
Decisions are based on the posterior probability $P(B > A)$:

- **STOP_WINNER**: If $P(B > A) > 0.95$ (default), the treatment is declared a winner.
- **STOP_NULL**: If $P(B > A) < 0.05$ (default), the control is declared superior or no lift is found.
- **CONTINUE**: Otherwise, the experiment continues collecting data.

A **Min Samples** guard (default: 100) is implemented to prevent premature stopping on very small sample sizes where the posterior is highly sensitive to the first few observations.

---

## Frequentist Alternative: O'Brien-Fleming

For projects requiring traditional frequentist p-values, we provide **O'Brien-Fleming group sequential boundaries**. 

This method preserves the overall Type I error ($\alpha$) by using extremely conservative critical values in the early stages of the experiment and gradually relaxing them as more data (looks) are collected.

- **Look 1**: Requires a very high Z-score (e.g., $> 4.0$).
- **Final Look**: Converges to the standard Z-score (e.g., $1.96$ for $\alpha=0.05$).

---

## Quick Start Example

```python
from core.sequential.bayesian import BetaBinomialPosterior
from core.sequential.decision import evaluate_decision

# 1. Initialize Posteriors
# Variant A (Control) with a Uniform prior
post_a = BetaBinomialPosterior(prior_type='uniform')

# Variant B (Treatment) with an Informed prior (Historical CR = 10%)
post_b = BetaBinomialPosterior(prior_type='informed', historical_cr=0.10)

# 2. Update with new data
post_a.update(n_new_conversions=15, n_new_trials=200)
post_b.update(n_new_conversions=25, n_new_trials=200)

# 3. Evaluate Decision
result = evaluate_decision(post_a, post_b, min_samples=100)

if result.decision == "STOP_WINNER":
    print(f"B is the winner! P(B > A) = {result.prob_b_beats_a:.2%}")
elif result.decision == "CONTINUE":
    print(f"Keep testing. Current P(B > A) = {result.prob_b_beats_a:.2%}")
```

## Directory Structure
- `bayesian.py`: Beta-Binomial posterior updates and Monte Carlo probability estimates.
- `decision.py`: Sequential stopping rules and expected loss calculations.
- `frequentist.py`: O'Brien-Fleming boundary calculations and Z-tests.
- `models.py`: Shared data structures for posterior states.
