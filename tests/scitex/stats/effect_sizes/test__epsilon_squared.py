#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for epsilon-squared effect size.

Tests cover:
- Non-parametric effect size for Kruskal-Wallis
- Range validation (0 <= ε² <= 1)
- Comparison with eta-squared
- Non-normal distributions
- Known values validation
- Mathematical properties
"""

import numpy as np
import pandas as pd
import pytest
from scitex.stats.effect_sizes import epsilon_squared, interpret_epsilon_squared


class TestBasicComputation:
    """Tests for basic epsilon-squared computations."""

    def test_two_groups_basic(self):
        """Test basic two-group comparison."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        eps2 = epsilon_squared([group1, group2])

        assert isinstance(eps2, float)
        assert 0 <= eps2 <= 1

    def test_three_groups_basic(self):
        """Test three-group comparison."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        group3 = np.array([5, 6, 7, 8, 9])
        eps2 = epsilon_squared([group1, group2, group3])

        assert isinstance(eps2, float)
        assert 0 <= eps2 <= 1
        # Should be high due to clear separation
        assert eps2 > 0.5

    def test_skewed_distributions(self):
        """Test with non-normal (skewed) distributions."""
        np.random.seed(42)
        group1 = np.random.exponential(1, 30)
        group2 = np.random.exponential(2, 30)
        group3 = np.random.exponential(3, 30)

        eps2 = epsilon_squared([group1, group2, group3])

        assert isinstance(eps2, float)
        assert 0 <= eps2 <= 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_effect_identical_groups(self):
        """Test that identical groups give ε² close to 0."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([1, 2, 3, 4, 5])
        eps2 = epsilon_squared([group1, group2])

        # Should be very small (near 0)
        # Note: Due to ties, might not be exactly 0
        assert eps2 < 0.1

    def test_perfect_separation(self):
        """Test that perfect separation gives ε² close to 1."""
        group1 = np.array([1, 1, 1, 1, 1])
        group2 = np.array([10, 10, 10, 10, 10])
        eps2 = epsilon_squared([group1, group2])

        assert eps2 > 0.8  # Should be high

    def test_nan_handling(self):
        """Test that NaN values are properly removed."""
        group1 = np.array([1, 2, np.nan, 4, 5])
        group2 = np.array([3, np.nan, 5, 6, 7])
        eps2 = epsilon_squared([group1, group2])

        assert isinstance(eps2, float)
        assert not np.isnan(eps2)

    def test_ties_handling(self):
        """Test handling of tied values."""
        group1 = np.array([1, 1, 1, 2, 2])
        group2 = np.array([2, 2, 3, 3, 3])
        eps2 = epsilon_squared([group1, group2])

        assert 0 <= eps2 <= 1


class TestKnownValues:
    """Tests with manually calculated or known values."""

    def test_complete_separation(self):
        """Test with completely separated groups."""
        group1 = np.array([1, 2, 3])
        group2 = np.array([10, 11, 12])
        eps2 = epsilon_squared([group1, group2])

        # Should be high (substantial effect)
        assert eps2 > 0.7  # Reasonable threshold for complete separation

    def test_overlapping_groups(self):
        """Test with overlapping groups."""
        group1 = np.array([1, 2, 3, 4, 5, 6, 7])
        group2 = np.array([4, 5, 6, 7, 8, 9, 10])
        eps2 = epsilon_squared([group1, group2])

        # Should show moderate effect
        assert 0.1 < eps2 < 0.8


class TestMathematicalProperties:
    """Tests for mathematical properties of epsilon-squared."""

    def test_range_constraint(self):
        """Test that ε² is always between 0 and 1."""
        np.random.seed(42)
        for _ in range(10):
            n_groups = np.random.randint(2, 6)
            groups = [np.random.exponential(i + 1, 20) for i in range(n_groups)]
            eps2 = epsilon_squared(groups)

            assert 0 <= eps2 <= 1

    def test_monotone_transformation_invariance(self):
        """Test that ε² is invariant to monotone transformations (rank-based)."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])

        eps2_original = epsilon_squared([group1, group2])

        # Apply monotone transformation (square)
        eps2_transformed = epsilon_squared([group1 ** 2, group2 ** 2])

        # Should be identical (rank-based, so monotone invariant)
        assert abs(eps2_original - eps2_transformed) < 0.01

    def test_increases_with_separation(self):
        """Test that ε² increases with group separation."""
        np.random.seed(42)

        eps2_values = []
        for scale in [1.0, 1.5, 2.0, 3.0]:
            g1 = np.random.exponential(1, 30)
            g2 = np.random.exponential(scale, 30)
            eps2_values.append(epsilon_squared([g1, g2]))

        # Should generally increase
        assert eps2_values[-1] > eps2_values[0]


class TestComparisonWithEtaSquared:
    """Tests comparing epsilon-squared with eta-squared."""

    def test_similar_for_normal_data(self):
        """Test that ε² and η² are similar for normal data."""
        from scitex.stats.effect_sizes import eta_squared

        np.random.seed(42)
        group1 = np.random.normal(0, 1, 40)
        group2 = np.random.normal(0.8, 1, 40)
        group3 = np.random.normal(1.5, 1, 40)

        eps2 = epsilon_squared([group1, group2, group3])
        eta2 = eta_squared([group1, group2, group3])

        # Should be reasonably similar for normal data
        assert abs(eps2 - eta2) < 0.3

    def test_robust_to_outliers(self):
        """Test that ε² is more robust to outliers than η²."""
        from scitex.stats.effect_sizes import eta_squared

        # Normal groups
        group1_normal = np.array([1, 2, 3, 4, 5])
        group2_normal = np.array([6, 7, 8, 9, 10])

        eps2_normal = epsilon_squared([group1_normal, group2_normal])
        eta2_normal = eta_squared([group1_normal, group2_normal])

        # With outlier
        group1_outlier = np.array([1, 2, 3, 4, 5])
        group2_outlier = np.array([6, 7, 8, 9, 100])  # Extreme outlier

        eps2_outlier = epsilon_squared([group1_outlier, group2_outlier])
        eta2_outlier = eta_squared([group1_outlier, group2_outlier])

        # ε² should be more stable (smaller change)
        eps_change = abs(eps2_normal - eps2_outlier)
        eta_change = abs(eta2_normal - eta2_outlier)

        # Epsilon should be less affected
        assert eps_change < eta_change

    def test_preferred_for_skewed_data(self):
        """Test that ε² works well with skewed distributions."""
        np.random.seed(42)
        # Highly skewed data
        group1 = np.random.exponential(1, 40)
        group2 = np.random.exponential(2, 40)
        group3 = np.random.exponential(3, 40)

        eps2 = epsilon_squared([group1, group2, group3])

        # Should detect the difference
        assert eps2 > 0.1


class TestInterpretation:
    """Tests for effect size interpretation."""

    def test_interpret_negligible(self):
        """Test negligible effect interpretation."""
        assert interpret_epsilon_squared(0.005) == 'negligible'
        assert interpret_epsilon_squared(0.009) == 'negligible'

    def test_interpret_small(self):
        """Test small effect interpretation."""
        assert interpret_epsilon_squared(0.03) == 'small'
        assert interpret_epsilon_squared(0.05) == 'small'

    def test_interpret_medium(self):
        """Test medium effect interpretation."""
        assert interpret_epsilon_squared(0.10) == 'medium'
        assert interpret_epsilon_squared(0.12) == 'medium'

    def test_interpret_large(self):
        """Test large effect interpretation."""
        assert interpret_epsilon_squared(0.20) == 'large'
        assert interpret_epsilon_squared(0.50) == 'large'

    def test_interpret_boundaries(self):
        """Test interpretation at boundaries."""
        # Boundaries: 0.01 (small), 0.06 (medium), 0.14 (large)
        assert interpret_epsilon_squared(0.01) == 'small'
        assert interpret_epsilon_squared(0.06) == 'medium'
        assert interpret_epsilon_squared(0.14) == 'large'


class TestSpecialCases:
    """Tests for special scenarios."""

    def test_ordinal_data(self):
        """Test with ordinal data (Likert scales)."""
        # Simulated Likert scale responses (1-5)
        group1 = np.array([1, 1, 2, 2, 2, 3, 3])
        group2 = np.array([3, 3, 4, 4, 4, 5, 5])
        eps2 = epsilon_squared([group1, group2])

        assert 0 <= eps2 <= 1
        assert eps2 > 0.3  # Should show effect

    def test_unbalanced_groups(self):
        """Test with very unbalanced group sizes."""
        group1 = np.array([1, 2, 3])
        group2 = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        eps2 = epsilon_squared([group1, group2])

        assert 0 <= eps2 <= 1
        assert eps2 > 0.4  # Should show substantial separation

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_epsilon_squared.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 21:05:00 (ywatanabe)"
# # File: ./src/scitex/stats/effect_sizes/_epsilon_squared.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/stats/effect_sizes/_epsilon_squared.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Functionalities:
#   - Compute epsilon-squared (ε²) effect size for Kruskal-Wallis test
#   - Non-parametric analog of eta-squared
#   - Measure variance in ranks explained by groups
#   - Provide interpretation guidelines
# 
# Dependencies:
#   - packages: numpy, pandas, scipy
# 
# IO:
#   - input: List of sample arrays (one per group)
#   - output: Effect size value (float, ranges from 0 to 1)
# """
# 
# """Imports"""
# import argparse
# from typing import List
# 
# import numpy as np
# import pandas as pd
# import scitex as stx
# from scipy import stats
# from scitex.logging import getLogger
# 
# logger = getLogger(__name__)
# 
# """Functions"""
# 
# 
# def epsilon_squared(groups: List[np.ndarray]) -> float:
#     """
#     Compute epsilon-squared (ε²) effect size for Kruskal-Wallis test.
# 
#     Parameters
#     ----------
#     groups : list of arrays
#         List of sample arrays for each group
# 
#     Returns
#     -------
#     float
#         Epsilon-squared value (0 to 1)
# 
#     Notes
#     -----
#     Epsilon-squared (ε²) is the non-parametric analog of eta-squared (η²)
#     for the Kruskal-Wallis test. It measures the proportion of variance in
#     ranks explained by group membership.
# 
#     .. math::
#         \\epsilon^2 = \\frac{H}{(n^2 - 1) / (n + 1)}
# 
#     Where:
#     - H: Kruskal-Wallis H statistic
#     - n: Total sample size
# 
#     Alternative formula (based on ranks):
# 
#     .. math::
#         \\epsilon^2 = \\frac{H - k + 1}{n - k}
# 
#     Where:
#     - H: Kruskal-Wallis H statistic
#     - k: Number of groups
#     - n: Total sample size
# 
#     Interpretation (similar to η²):
#     - ε² < 0.01:  negligible
#     - ε² < 0.06:  small
#     - ε² < 0.14:  medium
#     - ε² ≥ 0.14:  large
# 
#     References
#     ----------
#     .. [1] Tomczak, M., & Tomczak, E. (2014). "The need to report effect size
#            estimates revisited. An overview of some recommended measures of
#            effect size". Trends in Sport Sciences, 21(1), 19-25.
#     .. [2] Kerby, D. S. (2014). "The simple difference formula: An approach to
#            teaching nonparametric correlation". Comprehensive Psychology, 3, 11.
# 
#     Examples
#     --------
#     >>> group1 = np.array([1, 2, 3, 4, 5])
#     >>> group2 = np.array([3, 4, 5, 6, 7])
#     >>> group3 = np.array([5, 6, 7, 8, 9])
#     >>> epsilon_squared([group1, group2, group3])
#     0.857...
# 
#     >>> # No effect
#     >>> group1 = np.array([1, 2, 3, 4, 5])
#     >>> group2 = np.array([1, 2, 3, 4, 5])
#     >>> epsilon_squared([group1, group2])
#     0.0
#     """
#     # Convert all groups to numpy arrays and remove NaN
#     groups = [np.asarray(g) for g in groups]
#     groups = [g[~np.isnan(g)] for g in groups]
# 
#     # Get group sizes
#     k = len(groups)
#     n = sum(len(g) for g in groups)
# 
#     # Perform Kruskal-Wallis test to get H statistic
#     h_stat, _ = stats.kruskal(*groups)
# 
#     # Compute epsilon-squared using H statistic
#     # Formula: ε² = (H - k + 1) / (n - k)
#     if n == k:
#         return 0.0
# 
#     epsilon2 = (h_stat - k + 1) / (n - k)
# 
#     # Ensure value is in valid range [0, 1]
#     epsilon2 = max(0.0, min(1.0, epsilon2))
# 
#     return float(epsilon2)
# 
# 
# def interpret_epsilon_squared(epsilon2: float) -> str:
#     """
#     Interpret epsilon-squared effect size.
# 
#     Parameters
#     ----------
#     epsilon2 : float
#         Epsilon-squared value
# 
#     Returns
#     -------
#     str
#         Interpretation string
# 
#     Examples
#     --------
#     >>> interpret_epsilon_squared(0.005)
#     'negligible'
#     >>> interpret_epsilon_squared(0.03)
#     'small'
#     >>> interpret_epsilon_squared(0.10)
#     'medium'
#     >>> interpret_epsilon_squared(0.20)
#     'large'
#     """
#     if epsilon2 < 0.01:
#         return "negligible"
#     elif epsilon2 < 0.06:
#         return "small"
#     elif epsilon2 < 0.14:
#         return "medium"
#     else:
#         return "large"
# 
# 
# """Main function"""
# 
# 
# def main(args):
#     """Demonstrate epsilon-squared computation."""
#     logger.info("Demonstrating epsilon-squared effect size for Kruskal-Wallis")
# 
#     # Set random seed
#     np.random.seed(42)
# 
#     # Example 1: Three groups with non-normal distributions
#     logger.info("\n=== Example 1: Three skewed groups ===")
# 
#     group1 = np.random.exponential(1, 30)
#     group2 = np.random.exponential(1.5, 30)
#     group3 = np.random.exponential(2, 30)
# 
#     eps2 = epsilon_squared([group1, group2, group3])
#     interpretation = interpret_epsilon_squared(eps2)
# 
#     logger.info(f"Epsilon-squared = {eps2:.3f} ({interpretation})")
#     logger.info(f"{eps2:.1%} of rank variance explained by group membership")
# 
#     # Example 2: Comparison with eta-squared
#     logger.info("\n=== Example 2: Epsilon-squared vs Eta-squared ===")
# 
#     from ._eta_squared import eta_squared
# 
#     # Normal data
#     norm1 = np.random.normal(0, 1, 40)
#     norm2 = np.random.normal(0.8, 1, 40)
#     norm3 = np.random.normal(1.5, 1, 40)
# 
#     eps2_norm = epsilon_squared([norm1, norm2, norm3])
#     eta2_norm = eta_squared([norm1, norm2, norm3])
# 
#     logger.info(f"Normal data:   ε² = {eps2_norm:.3f}, η² = {eta2_norm:.3f}")
# 
#     # Skewed data
#     skew1 = np.random.exponential(1, 40)
#     skew2 = np.random.exponential(2, 40)
#     skew3 = np.random.exponential(3, 40)
# 
#     eps2_skew = epsilon_squared([skew1, skew2, skew3])
#     eta2_skew = eta_squared([skew1, skew2, skew3])
# 
#     logger.info(f"Skewed data:   ε² = {eps2_skew:.3f}, η² = {eta2_skew:.3f}")
#     logger.info("Epsilon-squared is more appropriate for non-normal data")
# 
#     # Example 3: Different effect sizes
#     logger.info("\n=== Example 3: Different effect sizes ===")
# 
#     for scale in [1.0, 1.5, 2.0, 3.0]:
#         g1 = np.random.exponential(1, 30)
#         g2 = np.random.exponential(scale, 30)
# 
#         eps2 = epsilon_squared([g1, g2])
#         interpretation = interpret_epsilon_squared(eps2)
# 
#         logger.info(f"Scale = {scale:.1f}: ε² = {eps2:.3f} ({interpretation})")
# 
#     # Visualization
#     logger.info("\n=== Creating visualization ===")
# 
#     fig, axes = stx.plt.subplots(1, 2, figsize=(12, 5))
# 
#     # Plot 1: Distribution comparison
#     ax = axes[0]
#     ax.hist(group1, bins=20, alpha=0.5, label="Group 1", density=True)
#     ax.hist(group2, bins=20, alpha=0.5, label="Group 2", density=True)
#     ax.hist(group3, bins=20, alpha=0.5, label="Group 3", density=True)
# 
#     ax.set_xlabel("Value")
#     ax.set_ylabel("Density")
#     ax.set_title(f"Skewed Distributions (ε² = {eps2:.2f})")
#     ax.legend()
# 
#     # Plot 2: Comparison of methods
#     ax = axes[1]
#     scales = np.linspace(1, 3, 10)
#     eps2_values = []
#     eta2_values = []
# 
#     for scale in scales:
#         g1 = np.random.exponential(1, 50)
#         g2 = np.random.exponential(scale, 50)
# 
#         eps2_values.append(epsilon_squared([g1, g2]))
#         eta2_values.append(eta_squared([g1, g2]))
# 
#     ax.plot(scales, eps2_values, "o-", label="Epsilon-squared (ε²)", linewidth=2)
#     ax.plot(scales, eta2_values, "s-", label="Eta-squared (η²)", linewidth=2)
# 
#     ax.set_xlabel("Distribution Scale Difference")
#     ax.set_ylabel("Effect Size")
#     ax.set_title("Non-parametric vs Parametric Effect Size")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
# 
#     stx.plt.tight_layout()
#     stx.io.save(fig, "./epsilon_squared_demo.jpg")
#     logger.info("Visualization saved")
# 
#     return 0
# 
# 
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Demonstrate epsilon-squared effect size calculation"
#     )
#     parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
#     return parser.parse_args()
# 
# 
# def run_main():
#     """Initialize SciTeX framework and run main."""
#     global CONFIG, sys, plt, rng
# 
#     import sys
#     import matplotlib.pyplot as plt
# 
#     args = parse_args()
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(
#         sys,
#         plt,
#         args=args,
#         file=__file__,
#         verbose=args.verbose,
#         agg=True,
#     )
# 
#     exit_status = main(args)
# 
#     stx.session.close(
#         CONFIG,
#         verbose=args.verbose,
#         exit_status=exit_status,
#     )
# 
# 
# if __name__ == "__main__":
#     run_main()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_epsilon_squared.py
# --------------------------------------------------------------------------------
