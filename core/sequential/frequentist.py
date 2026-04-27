import numpy as np
from scipy import stats
from typing import List, Optional

class OBrienFlemingBoundary:
    """
    Implements O'Brien-Fleming group sequential boundaries for frequentist testing.
    
    This design is conservative early in the experiment (requiring very high Z-scores)
    to preserve the overall alpha level across multiple interim looks.
    
    The boundary for the Z-score at analysis k (out of K) is approximately:
        Z_k = C(alpha, K) / sqrt(k/K)
    """

    def __init__(self, alpha: float = 0.05, n_planned_looks: int = 5):
        """
        Initialize the boundary calculator using the Lan-DeMets approximation.
        
        Args:
            alpha: Overall Type I error rate (default 0.05).
            n_planned_looks: Total planned number of analyses.
        """
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1.")
        if n_planned_looks < 1:
            raise ValueError("n_planned_looks must be at least 1.")

        self.alpha = alpha
        self.n_planned_looks = n_planned_looks
        
        # Calculate z_alpha/2
        self.z_alpha_2 = stats.norm.ppf(1 - alpha / 2)
        
        # Precompute boundaries for each look k (1 to K)
        # Formula: z_k = z_alpha/2 * sqrt(K / k)
        self.boundaries = [
            self.z_alpha_2 * np.sqrt(self.n_planned_looks / k)
            for k in range(1, self.n_planned_looks + 1)
        ]

    def get_boundary(self, look_number: int) -> float:
        """
        Get the Z-score boundary for a specific look.
        
        Args:
            look_number: The number of the current look (1 to n_planned_looks).
            
        Returns:
            The critical Z-score value.
        """
        if not (1 <= look_number <= self.n_planned_looks):
            raise ValueError(f"look_number must be between 1 and {self.n_planned_looks}")
        
        return self.boundaries[look_number - 1]

    def get_all_boundaries(self) -> List[float]:
        """Returns the Z-score boundaries for all planned looks."""
        return self.boundaries

    def evaluate(self, z_score: float, look_number: int) -> bool:
        """
        Evaluate if the observed Z-score exceeds the boundary.
        
        Args:
            z_score: The calculated Z-score for the test.
            look_number: The current look number.
            
        Returns:
            True if significant (exceeds boundary), False otherwise.
        """
        return abs(z_score) >= self.get_boundary(look_number)

    def test_at_look(
        self, 
        n_control: int, 
        conversions_control: int, 
        n_treatment: int, 
        conversions_treatment: int, 
        look_number: int
    ) -> dict:
        """
        Perform a two-proportion Z-test at a specific look and compare against O'Brien-Fleming boundary.
        
        Args:
            n_control: Total trials for control variant.
            conversions_control: Conversions for control variant.
            n_treatment: Total trials for treatment variant.
            conversions_treatment: Conversions for treatment variant.
            look_number: Current look number (1 to n_planned_looks).
            
        Returns:
            Dictionary containing z_stat, boundary, reject flag, and p_value.
        """
        if n_control == 0 or n_treatment == 0:
            return {
                "z_stat": 0.0,
                "boundary": self.get_boundary(look_number),
                "reject": False,
                "p_value": 1.0
            }

        p1 = conversions_control / n_control
        p2 = conversions_treatment / n_treatment
        p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)
        
        # Avoid division by zero if there are no conversions at all
        if p_pool == 0 or p_pool == 1:
            z_stat = 0.0
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
            z_stat = (p2 - p1) / se
            
        boundary = self.get_boundary(look_number)
        reject = abs(z_stat) >= boundary
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            "z_stat": float(z_stat),
            "boundary": float(boundary),
            "reject": bool(reject),
            "p_value": float(p_value)
        }

    def __repr__(self):
        return f"OBrienFlemingBoundary(alpha={self.alpha}, n_planned_looks={self.n_planned_looks})"
