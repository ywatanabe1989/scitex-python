#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for eta-squared effect size.

Tests cover:
- Basic ANOVA computations
- Multiple groups
- Range validation (0 <= η² <= 1)
- Perfect separation (η²=1)
- No effect (η²=0)
- Known values validation
- Mathematical properties
"""

import numpy as np
import pandas as pd
import pytest
from scitex.stats.effect_sizes import eta_squared, interpret_eta_squared


class TestBasicComputation:
    """Tests for basic eta-squared computations."""

    def test_two_groups_basic(self):
        """Test basic two-group comparison."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        eta2 = eta_squared([group1, group2])

        assert isinstance(eta2, float)
        assert 0 <= eta2 <= 1

    def test_three_groups_basic(self):
        """Test three-group ANOVA."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        group3 = np.array([5, 6, 7, 8, 9])
        eta2 = eta_squared([group1, group2, group3])

        assert isinstance(eta2, float)
        assert 0 <= eta2 <= 1
        # Should be high due to clear separation
        assert eta2 > 0.5

    def test_pandas_series_input(self):
        """Test that pandas Series work as input."""
        group1 = pd.Series([1, 2, 3, 4, 5])
        group2 = pd.Series([3, 4, 5, 6, 7])
        eta2 = eta_squared([group1, group2])

        assert isinstance(eta2, float)
        assert not np.isnan(eta2)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_effect_identical_groups(self):
        """Test that identical groups give η² = 0."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([1, 2, 3, 4, 5])
        eta2 = eta_squared([group1, group2])

        assert abs(eta2) < 0.01  # Should be very close to 0

    def test_perfect_separation(self):
        """Test that perfect separation gives η² close to 1."""
        group1 = np.array([1, 1, 1, 1, 1])
        group2 = np.array([10, 10, 10, 10, 10])
        eta2 = eta_squared([group1, group2])

        assert eta2 > 0.95  # Should be very close to 1

    def test_nan_handling(self):
        """Test that NaN values are properly removed."""
        group1 = np.array([1, 2, np.nan, 4, 5])
        group2 = np.array([3, np.nan, 5, 6, 7])
        eta2 = eta_squared([group1, group2])

        assert isinstance(eta2, float)
        assert not np.isnan(eta2)

    def test_all_same_values(self):
        """Test when all values across groups are identical."""
        group1 = np.array([5, 5, 5])
        group2 = np.array([5, 5, 5])
        group3 = np.array([5, 5, 5])
        eta2 = eta_squared([group1, group2, group3])

        # Total variance is 0, should return 0
        assert eta2 == 0.0

    def test_single_observation_per_group(self):
        """Test with single observation per group."""
        group1 = np.array([1])
        group2 = np.array([5])
        group3 = np.array([10])
        eta2 = eta_squared([group1, group2, group3])

        assert 0 <= eta2 <= 1


class TestKnownValues:
    """Tests with manually calculated known values."""

    def test_known_value_simple(self):
        """Test with simple manually calculated example."""
        # Groups: [0, 0], [5, 5]
        # Grand mean = 2.5
        # SS_total = (0-2.5)² + (0-2.5)² + (5-2.5)² + (5-2.5)² = 25
        # SS_between = 2*(0-2.5)² + 2*(5-2.5)² = 25
        # η² = 25/25 = 1.0
        group1 = np.array([0, 0])
        group2 = np.array([5, 5])
        eta2 = eta_squared([group1, group2])

        assert abs(eta2 - 1.0) < 0.01

    def test_known_value_medium_effect(self):
        """Test with known medium effect."""
        # Create groups with controlled variance
        np.random.seed(42)
        n = 50
        group1 = np.random.normal(0, 1, n)
        group2 = np.random.normal(0.5, 1, n)
        group3 = np.random.normal(1.0, 1, n)

        eta2 = eta_squared([group1, group2, group3])

        # Should show substantial effect
        assert 0.1 < eta2 < 0.5


class TestMathematicalProperties:
    """Tests for mathematical properties of eta-squared."""

    def test_range_constraint(self):
        """Test that η² is always between 0 and 1."""
        np.random.seed(42)
        for _ in range(10):
            n_groups = np.random.randint(2, 6)
            groups = [np.random.normal(i, 1, 20) for i in range(n_groups)]
            eta2 = eta_squared(groups)

            assert 0 <= eta2 <= 1

    def test_increases_with_group_separation(self):
        """Test that η² increases as groups become more separated."""
        np.random.seed(42)
        base = np.random.normal(0, 1, 30)

        # Increasing separation
        eta2_values = []
        for separation in [0.0, 0.5, 1.0, 2.0]:
            g1 = np.random.normal(0, 1, 30)
            g2 = np.random.normal(separation, 1, 30)
            eta2_values.append(eta_squared([g1, g2]))

        # Should generally increase
        # (with some tolerance for random variation)
        assert eta2_values[-1] > eta2_values[0]

    def test_invariant_to_linear_transformation(self):
        """Test that η² is invariant to linear transformations."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        group3 = np.array([5, 6, 7, 8, 9])

        eta2_original = eta_squared([group1, group2, group3])

        # Scale and shift all groups equally
        scale = 10
        shift = 100
        eta2_transformed = eta_squared([
            group1 * scale + shift,
            group2 * scale + shift,
            group3 * scale + shift
        ])

        # Should be identical (relative variance is preserved)
        assert abs(eta2_original - eta2_transformed) < 0.001

    def test_more_groups_can_increase_eta2(self):
        """Test that adding well-separated groups increases η²."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 30)
        group2 = np.random.normal(2, 1, 30)

        eta2_two = eta_squared([group1, group2])

        # Add a third group well separated
        group3 = np.random.normal(4, 1, 30)
        eta2_three = eta_squared([group1, group2, group3])

        # Usually increases with more separated groups
        assert eta2_three > eta2_two * 0.8  # Allow some variation


class TestInterpretation:
    """Tests for effect size interpretation."""

    def test_interpret_negligible(self):
        """Test negligible effect interpretation."""
        assert interpret_eta_squared(0.005) == 'negligible'
        assert interpret_eta_squared(0.009) == 'negligible'

    def test_interpret_small(self):
        """Test small effect interpretation."""
        assert interpret_eta_squared(0.03) == 'small'
        assert interpret_eta_squared(0.05) == 'small'

    def test_interpret_medium(self):
        """Test medium effect interpretation."""
        assert interpret_eta_squared(0.10) == 'medium'
        assert interpret_eta_squared(0.12) == 'medium'

    def test_interpret_large(self):
        """Test large effect interpretation."""
        assert interpret_eta_squared(0.20) == 'large'
        assert interpret_eta_squared(0.50) == 'large'

    def test_interpret_boundaries(self):
        """Test interpretation at boundaries."""
        # Boundaries: 0.01 (small), 0.06 (medium), 0.14 (large)
        assert interpret_eta_squared(0.01) == 'small'
        assert interpret_eta_squared(0.06) == 'medium'
        assert interpret_eta_squared(0.14) == 'large'


class TestComparison:
    """Tests comparing eta-squared across scenarios."""

    def test_within_vs_between_variance(self):
        """Test effect of within-group vs between-group variance."""
        np.random.seed(42)

        # High within-group variance
        g1_high_var = np.random.normal(0, 2, 30)
        g2_high_var = np.random.normal(1, 2, 30)
        eta2_high_var = eta_squared([g1_high_var, g2_high_var])

        # Low within-group variance
        g1_low_var = np.random.normal(0, 0.5, 30)
        g2_low_var = np.random.normal(1, 0.5, 30)
        eta2_low_var = eta_squared([g1_low_var, g2_low_var])

        # Lower within-group variance should give higher η²
        assert eta2_low_var > eta2_high_var

    def test_balanced_vs_unbalanced_groups(self):
        """Test that η² works with unbalanced group sizes."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        eta2 = eta_squared([group1, group2])

        assert 0 <= eta2 <= 1
        assert eta2 > 0.5  # Should show strong effect

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_eta_squared.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 21:05:00 (ywatanabe)"
# # File: ./src/scitex/stats/effect_sizes/_eta_squared.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/stats/effect_sizes/_eta_squared.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Functionalities:
#   - Compute eta-squared (η²) effect size for ANOVA
#   - Measure proportion of variance explained by group membership
#   - Provide interpretation guidelines
#   - Support multiple groups
# 
# Dependencies:
#   - packages: numpy, pandas
# 
# IO:
#   - input: List of sample arrays (one per group)
#   - output: Effect size value (float, ranges from 0 to 1)
# """
# 
# """Imports"""
# import argparse
# from typing import List, Union
# 
# import numpy as np
# import pandas as pd
# import scitex as stx
# from scitex.logging import getLogger
# 
# logger = getLogger(__name__)
# 
# """Functions"""
# 
# 
# def eta_squared(groups: List[Union[np.ndarray, pd.Series]], ddof: int = 1) -> float:
#     """
#     Compute eta-squared (η²) effect size for ANOVA.
# 
#     Parameters
#     ----------
#     groups : list of arrays or Series
#         List of samples, one per group
#     ddof : int, default 1
#         Degrees of freedom correction for variance
# 
#     Returns
#     -------
#     float
#         Eta-squared value (ranges from 0 to 1)
# 
#     Notes
#     -----
#     Eta-squared (η²) measures the proportion of total variance explained
#     by group membership in ANOVA designs.
# 
#     .. math::
#         \\eta^2 = \\frac{SS_{between}}{SS_{total}}
# 
#     Where:
#     - SS_between: Sum of squares between groups
#     - SS_total: Total sum of squares
# 
#     Interpretation (Cohen, 1988):
#     - η² < 0.01:  negligible
#     - η² < 0.06:  small
#     - η² < 0.14:  medium
#     - η² ≥ 0.14:  large
# 
#     **Variants:**
#     - η²: Biased, overestimates population effect
#     - ω² (omega-squared): Less biased estimate
#     - partial η²: Used in factorial designs
# 
#     Relationship to F-statistic:
# 
#     .. math::
#         \\eta^2 = \\frac{F \\cdot df_{between}}{F \\cdot df_{between} + df_{within}}
# 
#     References
#     ----------
#     .. [1] Cohen, J. (1988). Statistical Power Analysis for the Behavioral
#            Sciences (2nd ed.). Routledge.
#     .. [2] Richardson, J. T. E. (2011). "Eta squared and partial eta squared
#            as measures of effect size in educational research". Educational
#            Research Review, 6(2), 135-147.
# 
#     Examples
#     --------
#     >>> group1 = np.array([1, 2, 3, 4, 5])
#     >>> group2 = np.array([3, 4, 5, 6, 7])
#     >>> group3 = np.array([5, 6, 7, 8, 9])
#     >>> eta_squared([group1, group2, group3])
#     0.857...
# 
#     >>> # No effect
#     >>> group1 = np.array([1, 2, 3, 4, 5])
#     >>> group2 = np.array([1, 2, 3, 4, 5])
#     >>> eta_squared([group1, group2])
#     0.0
#     """
#     # Convert all groups to numpy arrays and remove NaN
#     groups = [np.asarray(g) for g in groups]
#     groups = [g[~np.isnan(g)] for g in groups]
# 
#     # Compute grand mean
#     all_data = np.concatenate(groups)
#     grand_mean = np.mean(all_data)
# 
#     # Compute total sum of squares
#     ss_total = np.sum((all_data - grand_mean) ** 2)
# 
#     # Compute between-group sum of squares
#     ss_between = 0
#     for group in groups:
#         group_mean = np.mean(group)
#         n_group = len(group)
#         ss_between += n_group * (group_mean - grand_mean) ** 2
# 
#     # Compute eta-squared
#     if ss_total == 0:
#         return 0.0
# 
#     eta2 = ss_between / ss_total
# 
#     return float(eta2)
# 
# 
# def interpret_eta_squared(eta2: float) -> str:
#     """
#     Interpret eta-squared effect size.
# 
#     Parameters
#     ----------
#     eta2 : float
#         Eta-squared value
# 
#     Returns
#     -------
#     str
#         Interpretation string
# 
#     Examples
#     --------
#     >>> interpret_eta_squared(0.005)
#     'negligible'
#     >>> interpret_eta_squared(0.03)
#     'small'
#     >>> interpret_eta_squared(0.10)
#     'medium'
#     >>> interpret_eta_squared(0.20)
#     'large'
#     """
#     if eta2 < 0.01:
#         return "negligible"
#     elif eta2 < 0.06:
#         return "small"
#     elif eta2 < 0.14:
#         return "medium"
#     else:
#         return "large"
# 
# 
# """Main function"""
# 
# 
# def main(args):
#     """Demonstrate eta-squared computation."""
#     logger.info("Demonstrating eta-squared effect size for ANOVA")
# 
#     # Set random seed
#     np.random.seed(42)
# 
#     # Example 1: Three groups with clear differences
#     logger.info("\n=== Example 1: Three groups with differences ===")
# 
#     group1 = np.random.normal(0, 1, 30)
#     group2 = np.random.normal(0.5, 1, 30)
#     group3 = np.random.normal(1.0, 1, 30)
# 
#     eta2 = eta_squared([group1, group2, group3])
#     interpretation = interpret_eta_squared(eta2)
# 
#     logger.info(f"Eta-squared = {eta2:.3f} ({interpretation})")
#     logger.info(f"{eta2:.1%} of variance explained by group membership")
# 
#     # Example 2: Different effect sizes
#     logger.info("\n=== Example 2: Different effect sizes ===")
# 
#     control = np.random.normal(0, 1, 40)
# 
#     for shift in [0.0, 0.3, 0.6, 1.0]:
#         treatment1 = np.random.normal(shift, 1, 40)
#         treatment2 = np.random.normal(shift * 1.5, 1, 40)
# 
#         eta2 = eta_squared([control, treatment1, treatment2])
#         interpretation = interpret_eta_squared(eta2)
# 
#         logger.info(f"Shift = {shift:.1f}: η² = {eta2:.3f} ({interpretation})")
# 
#     # Example 3: Many groups
#     logger.info("\n=== Example 3: Five groups ===")
# 
#     groups = [np.random.normal(i * 0.3, 1, 25) for i in range(5)]
#     eta2_many = eta_squared(groups)
# 
#     logger.info(
#         f"Five groups: η² = {eta2_many:.3f} ({interpret_eta_squared(eta2_many)})"
#     )
# 
#     # Visualization
#     logger.info("\n=== Creating visualization ===")
# 
#     fig, axes = stx.plt.subplots(1, 2, figsize=(12, 5))
# 
#     # Plot 1: Group distributions
#     ax = axes[0]
#     positions = [0, 1, 2]
#     data_viz = [group1, group2, group3]
# 
#     bp = ax.boxplot(data_viz, positions=positions, widths=0.6, patch_artist=True)
#     for patch in bp["boxes"]:
#         patch.set_facecolor("lightblue")
# 
#     ax.set_xlabel("Group")
#     ax.set_ylabel("Value")
#     ax.set_title(f"Three Groups (η² = {eta2:.2f})")
#     ax.set_xticklabels(["Group 1", "Group 2", "Group 3"])
#     ax.grid(True, alpha=0.3, axis="y")
# 
#     # Plot 2: Effect size vs number of groups
#     ax = axes[1]
#     n_groups_list = [2, 3, 4, 5, 6]
#     eta2_values = []
# 
#     for n_groups in n_groups_list:
#         groups_test = [np.random.normal(i * 0.4, 1, 30) for i in range(n_groups)]
#         eta2_values.append(eta_squared(groups_test))
# 
#     ax.plot(n_groups_list, eta2_values, "o-", linewidth=2, markersize=8)
#     ax.set_xlabel("Number of Groups")
#     ax.set_ylabel("Eta-squared")
#     ax.set_title("Effect Size vs Number of Groups")
#     ax.grid(True, alpha=0.3)
# 
#     stx.plt.tight_layout()
#     stx.io.save(fig, "./eta_squared_demo.jpg")
#     logger.info("Visualization saved")
# 
#     return 0
# 
# 
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Demonstrate eta-squared effect size calculation"
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_eta_squared.py
# --------------------------------------------------------------------------------
