import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from core.sequential.bayesian import BetaBinomialPosterior
from core.sequential.decision import evaluate_decision, Decision

def run_aa_with_prior(n_users, p_true, batch_size, alpha_prior, beta_prior):
    """
    Simulates a single A/A experiment with specific priors.
    """
    n_a = n_users // 2
    n_b = n_users - n_a
    assignments = np.array([0] * n_a + [1] * n_b)
    np.random.shuffle(assignments)
    outcomes = np.random.binomial(n=1, p=p_true, size=n_users)
    
    # Initialize with custom priors
    post_a = BetaBinomialPosterior(alpha_prior=alpha_prior, beta_prior=beta_prior)
    post_b = BetaBinomialPosterior(alpha_prior=alpha_prior, beta_prior=beta_prior)
    
    for i in range(0, n_users, batch_size):
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
            
        result = evaluate_decision(post_a, post_b)
        
        if result.decision != Decision.CONTINUE:
            return result.decision, result.n_total
            
    return Decision.CONTINUE, n_users

def main():
    N_EXPERIMENTS = 1000
    USERS_PER_EXPERIMENT = 2000
    P_TRUE = 0.10
    BATCH_SIZE = 50
    
    priors = [
        {"name": "Uniform Beta(1,1)", "alpha": 1.0, "beta": 1.0},
        {"name": "Informed Beta(2,8)", "alpha": 2.0, "beta": 8.0}, # Note: User specified Beta(2,8) for 10% CR, but 2/10 = 20%.
        {"name": "Jeffreys Beta(0.5,0.5)", "alpha": 0.5, "beta": 0.5},
    ]
    
    summary_results = []
    
    for prior in priors:
        print(f"\nRunning sensitivity test for {prior['name']}...")
        results = []
        for _ in tqdm(range(N_EXPERIMENTS)):
            results.append(run_aa_with_prior(
                USERS_PER_EXPERIMENT, P_TRUE, BATCH_SIZE, 
                prior['alpha'], prior['beta']
            ))
            
        decisions = [r[0] for r in results]
        sample_sizes = [r[1] for r in results]
        
        fpr = decisions.count(Decision.STOP_WINNER) / N_EXPERIMENTS
        avg_n = np.mean(sample_sizes)
        
        summary_results.append({
            "Prior": prior['name'],
            "FPR": f"{fpr:.2%}",
            "Avg N": f"{avg_n:.1f}"
        })
        
    # Tabulate results
    print("\n" + "="*50)
    print(f"{'Prior':<25} | {'FPR':<10} | {'Avg N':<10}")
    print("-" * 50)
    for res in summary_results:
        print(f"{res['Prior']:<25} | {res['FPR']:<10} | {res['Avg N']:<10}")
    print("="*50)

if __name__ == "__main__":
    main()
