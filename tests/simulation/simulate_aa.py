import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to sys.path to allow imports from core
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from core.sequential.bayesian import BetaBinomialPosterior
from core.sequential.decision import evaluate_decision, Decision

def run_single_experiment(n_users, p_true, batch_size):
    """
    Simulates a single A/A experiment.
    """
    # 1. Prepare users: 50/50 split
    n_a = n_users // 2
    n_b = n_users - n_a
    
    # Assignments: 0 for A, 1 for B
    assignments = np.array([0] * n_a + [1] * n_b)
    np.random.shuffle(assignments)
    
    # Outcomes: 1 for conversion, 0 for not
    outcomes = np.random.binomial(n=1, p=p_true, size=n_users)
    
    # Posteriors
    post_a = BetaBinomialPosterior()
    post_b = BetaBinomialPosterior()
    
    # Process in batches
    for i in range(0, n_users, batch_size):
        batch_assignments = assignments[i : i + batch_size]
        batch_outcomes = outcomes[i : i + batch_size]
        
        # Count successes and trials for each variant in this batch
        a_indices = (batch_assignments == 0)
        b_indices = (batch_assignments == 1)
        
        n_new_a = int(np.sum(a_indices))
        s_new_a = int(np.sum(batch_outcomes[a_indices]))
        
        n_new_b = int(np.sum(b_indices))
        s_new_b = int(np.sum(batch_outcomes[b_indices]))
        
        # Update posteriors
        if n_new_a > 0:
            post_a.update(s_new_a, n_new_a)
        if n_new_b > 0:
            post_b.update(s_new_b, n_new_b)
            
        # Evaluate decision
        result = evaluate_decision(post_a, post_b)
        
        if result.decision != Decision.CONTINUE:
            return result.decision, result.n_total
            
    return Decision.CONTINUE, n_users

def main():
    N_EXPERIMENTS = 1000
    USERS_PER_EXPERIMENT = 2000
    P_TRUE = 0.10
    BATCH_SIZE = 50
    
    print(f"Starting A/A simulation: {N_EXPERIMENTS} experiments...")
    print(f"Users per experiment: {USERS_PER_EXPERIMENT}, Split: 50/50")
    print(f"True conversion rate: {P_TRUE}, Batch size: {BATCH_SIZE}")
    
    results = []
    
    # Using tqdm for progress bar
    for _ in tqdm(range(N_EXPERIMENTS)):
        decision, n_total = run_single_experiment(USERS_PER_EXPERIMENT, P_TRUE, BATCH_SIZE)
        results.append((decision, n_total))
        
    # Analyze results
    decisions = [r[0] for r in results]
    sample_sizes = [r[1] for r in results]
    
    winners = decisions.count(Decision.STOP_WINNER)
    nulls = decisions.count(Decision.STOP_NULL)
    continues = decisions.count(Decision.CONTINUE)
    
    false_positive_rate = winners / N_EXPERIMENTS
    total_stop_rate = (winners + nulls) / N_EXPERIMENTS
    avg_sample_size = np.mean(sample_sizes)
    
    print("\n--- Simulation Results ---")
    print(f"Total Experiments:  {N_EXPERIMENTS}")
    print(f"Stop Winner (FP):   {winners} ({false_positive_rate:.2%})")
    print(f"Stop Null:          {nulls} ({nulls/N_EXPERIMENTS:.2%})")
    print(f"Reached End:        {continues} ({continues/N_EXPERIMENTS:.2%})")
    print(f"Avg Sample Size:    {avg_sample_size:.1f}")
    
    if false_positive_rate > 0.05:
        print("\nWARNING: False Positive Rate is higher than 5% (nominal alpha).")
    else:
        print("\nFalse Positive Rate is within expected range.")

if __name__ == "__main__":
    main()
