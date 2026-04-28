import sys
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from core.sequential.bayesian import BetaBinomialPosterior
from core.sequential.decision import evaluate_decision, Decision, expected_loss

def run_experiment(p_a, p_b, max_users, batch_size, rule='default'):
    """
    Simulates a single A/B experiment with a given stopping rule.
    """
    n_a_total = max_users // 2
    n_b_total = max_users - n_a_total
    
    # Pre-generate assignments and outcomes
    assignments = np.array([0] * n_a_total + [1] * n_b_total)
    np.random.shuffle(assignments)
    
    # Outcomes depend on the variant
    outcomes = np.zeros(max_users, dtype=int)
    a_mask = (assignments == 0)
    b_mask = (assignments == 1)
    outcomes[a_mask] = np.random.binomial(n=1, p=p_a, size=np.sum(a_mask))
    outcomes[b_mask] = np.random.binomial(n=1, p=p_b, size=np.sum(b_mask))
    
    post_a = BetaBinomialPosterior()
    post_b = BetaBinomialPosterior()
    
    for i in range(0, max_users, batch_size):
        batch_assignments = assignments[i : i + batch_size]
        batch_outcomes = outcomes[i : i + batch_size]
        
        a_indices = (batch_assignments == 0)
        b_indices = (batch_assignments == 1)
        
        n_new_a = int(np.sum(a_indices))
        s_new_a = int(np.sum(batch_outcomes[a_indices]))
        n_new_b = int(np.sum(b_indices))
        s_new_b = int(np.sum(batch_outcomes[b_indices]))
        
        if n_new_a > 0: post_a.update(s_new_a, n_new_a)
        if n_new_b > 0: post_b.update(s_new_b, n_new_b)
        
        n_total = int((post_a.alpha_posterior - post_a.alpha_prior) + 
                      (post_a.beta_posterior - post_a.beta_prior) + 
                      (post_b.alpha_posterior - post_b.alpha_prior) + 
                      (post_b.beta_posterior - post_b.beta_prior))
        
        # Burn-in: ignore decision evaluator if n_total < 100 (matching evaluate_decision default)
        if n_total < 100:
            continue
            
        if rule == 'default':
            result = evaluate_decision(post_a, post_b)
            if result.decision != Decision.CONTINUE:
                return result.decision, n_total
        elif rule == 'expected_loss':
            loss = expected_loss(post_a, post_b)
            if loss < 0.001:
                # If we stop, we pick the winner based on posterior mean
                decision = Decision.STOP_WINNER if post_b.mean() > post_a.mean() else Decision.STOP_NULL
                return decision, n_total
                
    return Decision.CONTINUE, max_users

def compute_stats(results, n_experiments):
    decisions = [r[0] for r in results]
    sample_sizes = [r[1] for r in results]
    
    tp = decisions.count(Decision.STOP_WINNER)
    fn = decisions.count(Decision.STOP_NULL) + decisions.count(Decision.CONTINUE)
    
    power = tp / n_experiments
    
    # Median sample size for CORRECT stops (STOP_WINNER)
    correct_stops = [s for d, s in results if d == Decision.STOP_WINNER]
    median_n = float(np.median(correct_stops)) if correct_stops else 0.0
    
    return {
        "power": power,
        "true_positives": tp,
        "false_negatives": fn,
        "median_sample_size": median_n,
        "stop_null": decisions.count(Decision.STOP_NULL),
        "never_stopped": decisions.count(Decision.CONTINUE)
    }

def main():
    N_EXPERIMENTS = 1000
    MAX_USERS = 5000  # High enough to capture 80% power
    P_A = 0.10
    P_B = 0.13
    BATCH_SIZE = 50
    
    # Fixed-horizon calculation
    alpha = 0.05
    beta = 0.20
    p_avg = (P_A + P_B) / 2
    delta = abs(P_B - P_A)
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(1 - beta)
    n_fixed_per_variant = int(np.ceil(2 * (z_alpha + z_beta)**2 * p_avg * (1 - p_avg) / (delta**2)))
    n_fixed_total = 2 * n_fixed_per_variant
    
    print(f"Running A/B simulation: {N_EXPERIMENTS} experiments...")
    print(f"Fixed-horizon total N (for 80% power): {n_fixed_total}")
    
    # 1. Default Stopping Rule (P(B>A) > 0.95)
    print("\nSimulating default rule (prob > 0.95)...")
    results_default = []
    for _ in tqdm(range(N_EXPERIMENTS)):
        results_default.append(run_experiment(P_A, P_B, MAX_USERS, BATCH_SIZE, 'default'))
    stats_default = compute_stats(results_default, N_EXPERIMENTS)
    
    # 2. Alternative Rule (Expected Loss < 0.001)
    print("\nSimulating alternative rule (expected_loss < 0.001)...")
    results_loss = []
    for _ in tqdm(range(N_EXPERIMENTS)):
        results_loss.append(run_experiment(P_A, P_B, MAX_USERS, BATCH_SIZE, 'expected_loss'))
    stats_loss = compute_stats(results_loss, N_EXPERIMENTS)
    
    final_results = {
        "config": {
            "p_a": P_A,
            "p_b": P_B,
            "n_experiments": N_EXPERIMENTS,
            "max_users": MAX_USERS,
            "batch_size": BATCH_SIZE
        },
        "fixed_horizon": {
            "total_n": n_fixed_total
        },
        "default_rule": stats_default,
        "loss_rule": stats_loss
    }
    
    # Save to file
    output_path = Path(__file__).parent / "ab_results.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\nResults saved to {output_path}")
    print("\n--- Summary ---")
    print(f"Default Power: {stats_default['power']:.2%} (Median N: {stats_default['median_sample_size']:.0f})")
    print(f"Loss-based Power: {stats_loss['power']:.2%} (Median N: {stats_loss['median_sample_size']:.0f})")
    print(f"Fixed-Horizon N: {n_fixed_total}")
    
    # Pytest assertion
    print(f"\nChecking power > 0.78: {stats_default['power'] > 0.78}")
    assert stats_default['power'] > 0.78, f"Power {stats_default['power']} is below target 0.78"

if __name__ == "__main__":
    main()
