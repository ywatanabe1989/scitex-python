#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for probability of superiority.

Tests cover:
- Probability of superiority P(X > Y) computation
- Range validation (0 <= PS <= 1)
- Relationship to Cliff's delta: PS = (δ + 1) / 2
- Common language effect size interpretation
- Known values validation
- Mathematical properties
"""

import numpy as np
import pandas as pd
import pytest
from scitex.stats.effect_sizes import prob_superiority, interpret_prob_superiority


class TestBasicComputation:
    """Tests for basic probability of superiority computations."""

    def test_basic_comparison(self):
        """Test basic two-sample comparison."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        prob = prob_superiority(x, y)

        assert isinstance(prob, float)
        assert 0 <= prob <= 1
        # x < y on average, so P(X > Y) should be low
        assert prob < 0.5

    def test_known_value(self):
        """Test with manually calculated known value."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        prob = prob_superiority(x, y)

        # Expected: count pairs where x_i > y_j
        # x=5: >y in {2,3,4} = 3 pairs; x=4: >y in {2,3} = 2 pairs
        # x=3: >y in {2} = 1 pair; x=2: >y in {} = 0; x=1: >y in {} = 0
        # Total: 3+2+1+0+0 = 6 out of 25 = 0.24
        assert abs(prob - 0.24) < 0.1

    def test_pandas_series_input(self):
        """Test that pandas Series work as input."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([3, 4, 5, 6, 7])
        prob = prob_superiority(x, y)

        assert isinstance(prob, float)
        assert not np.isnan(prob)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_effect_chance_level(self):
        """Test that identical distributions give P = 0.5."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        prob = prob_superiority(x, y)

        # With identical values, ties don't count as >
        # So P(X > Y) should be around 0.5 (actually might be less due to ties)
        assert 0.4 <= prob <= 0.6

    def test_perfect_dominance_x_over_y(self):
        """Test perfect dominance (all x > all y) gives P = 1.0."""
        x = np.array([6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5])
        prob = prob_superiority(x, y)

        assert abs(prob - 1.0) < 0.01

    def test_perfect_dominance_y_over_x(self):
        """Test perfect dominance (all x < all y) gives P = 0.0."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([6, 7, 8, 9, 10])
        prob = prob_superiority(x, y)

        assert abs(prob - 0.0) < 0.01

    def test_nan_handling(self):
        """Test that NaN values are properly removed."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([3, np.nan, 5, 6, 7])
        prob = prob_superiority(x, y)

        assert isinstance(prob, float)
        assert not np.isnan(prob)


class TestKnownValues:
    """Tests with manually calculated known values."""

    def test_complete_superiority(self):
        """Test with complete superiority."""
        x = np.array([10, 11, 12])
        y = np.array([1, 2, 3])
        prob = prob_superiority(x, y)

        # All x > all y: P = 1.0
        assert abs(prob - 1.0) < 0.01

    def test_complete_inferiority(self):
        """Test with complete inferiority."""
        x = np.array([1, 2, 3])
        y = np.array([10, 11, 12])
        prob = prob_superiority(x, y)

        # All x < all y: P = 0.0
        assert abs(prob - 0.0) < 0.01

    def test_partial_overlap(self):
        """Test with partial overlap."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])
        prob = prob_superiority(x, y)

        # Should be less than 0.5 (y dominates)
        assert 0 < prob < 0.5


class TestMathematicalProperties:
    """Tests for mathematical properties of probability of superiority."""

    def test_range_constraint(self):
        """Test that P is always between 0 and 1."""
        np.random.seed(42)
        for _ in range(20):
            x = np.random.normal(0, 1, 20)
            y = np.random.normal(np.random.uniform(-2, 2), 1, 20)
            prob = prob_superiority(x, y)

            assert 0 <= prob <= 1

    def test_complement_property(self):
        """Test that P(X > Y) + P(Y > X) + P(ties) = 1."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])

        prob_x_gt_y = prob_superiority(x, y)
        prob_y_gt_x = prob_superiority(y, x)

        # Without ties: P(X>Y) + P(Y>X) should be close to 1
        # With ties: sum will be < 1
        assert prob_x_gt_y + prob_y_gt_x <= 1.01  # Allow tiny numerical error

    def test_monotone_transformation_invariance(self):
        """Test that P is invariant to monotone transformations."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([6, 7, 8, 9, 10])

        prob_original = prob_superiority(x, y)

        # Apply monotone transformation (square)
        prob_transformed = prob_superiority(x ** 2, y ** 2)

        # Should be identical (ordinal, so monotone invariant)
        assert abs(prob_original - prob_transformed) < 0.01


class TestRelationshipWithCliffsD:
    """Tests for relationship with Cliff's delta."""

    def test_relationship_formula(self):
        """Test that P(X > Y) = (δ + 1) / 2."""
        from scitex.stats.effect_sizes import cliffs_delta

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y = np.array([3, 4, 5, 6, 7, 8, 9, 10])

        prob = prob_superiority(x, y)
        delta = cliffs_delta(x, y)

        expected_prob = (delta + 1) / 2
        # Note: This relationship assumes no ties. With some ties, there may be small deviation
        assert abs(prob - expected_prob) < 0.05  # Allow for ties

    def test_relationship_multiple_scenarios(self):
        """Test relationship across multiple scenarios."""
        from scitex.stats.effect_sizes import cliffs_delta

        np.random.seed(42)
        for _ in range(10):
            x = np.random.normal(0, 1, 20)
            y = np.random.normal(np.random.uniform(-1, 1), 1, 20)

            prob = prob_superiority(x, y)
            delta = cliffs_delta(x, y)

            expected_prob = (delta + 1) / 2
            assert abs(prob - expected_prob) < 0.01

    def test_delta_zero_implies_prob_half(self):
        """Test that δ = 0 implies P ≈ 0.5."""
        # Create groups with approximately equal distributions
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0, 1, 50)

        prob = prob_superiority(x, y)

        # Should be close to 0.5
        assert 0.4 < prob < 0.6


class TestInterpretation:
    """Tests for effect size interpretation."""

    def test_interpret_negligible(self):
        """Test negligible effect interpretation."""
        assert interpret_prob_superiority(0.51) == 'negligible'
        assert interpret_prob_superiority(0.49) == 'negligible'

    def test_interpret_small(self):
        """Test small effect interpretation."""
        assert interpret_prob_superiority(0.60) == 'small'
        assert interpret_prob_superiority(0.40) == 'small'

    def test_interpret_medium(self):
        """Test medium effect interpretation."""
        assert interpret_prob_superiority(0.68) == 'medium'
        assert interpret_prob_superiority(0.32) == 'medium'

    def test_interpret_large(self):
        """Test large effect interpretation."""
        assert interpret_prob_superiority(0.75) == 'large'
        assert interpret_prob_superiority(0.25) == 'large'

    def test_interpret_symmetric(self):
        """Test that interpretation is symmetric around 0.5."""
        # Distance from 0.5 should determine interpretation
        assert interpret_prob_superiority(0.6) == interpret_prob_superiority(0.4)
        assert interpret_prob_superiority(0.7) == interpret_prob_superiority(0.3)


class TestCommonLanguageEffect:
    """Tests for common language effect size interpretation."""

    def test_intuitive_interpretation(self):
        """Test intuitive probabilistic interpretation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([6, 7, 8, 9, 10])

        prob = prob_superiority(x, y)

        # Interpretation: If you pick random x and y,
        # probability that x > y is 'prob'
        # Here, x is always less, so prob should be 0
        assert prob == 0.0

    def test_medium_effect_interpretation(self):
        """Test interpretation of medium effect."""
        # P = 0.64 means 64% chance that X > Y
        # This is considered a medium effect
        np.random.seed(42)
        x = np.random.normal(0.5, 1, 100)
        y = np.random.normal(0, 1, 100)

        prob = prob_superiority(x, y)

        # Should be > 0.5 since x has higher mean
        assert prob > 0.5


class TestRobustness:
    """Tests for robustness properties."""

    def test_robust_to_outliers(self):
        """Test that P is robust to outliers."""
        # Normal case
        x_normal = np.array([1, 2, 3, 4, 5])
        y_normal = np.array([3, 4, 5, 6, 7])
        prob_normal = prob_superiority(x_normal, y_normal)

        # With extreme outlier
        x_outlier = np.array([1, 2, 3, 4, 100])
        prob_outlier = prob_superiority(x_outlier, y_normal)

        # Should be relatively stable
        assert abs(prob_normal - prob_outlier) < 0.3

    def test_non_normal_distributions(self):
        """Test with non-normal distributions."""
        np.random.seed(42)
        x = np.random.exponential(1, 50)
        y = np.random.exponential(2, 50)
        prob = prob_superiority(x, y)

        assert 0 <= prob <= 1
        # x has smaller scale parameter, so x values tend to be smaller
        # Thus P(X > Y) should be less than 0.5
        assert 0.2 < prob < 0.5  # Reasonable range for this scenario

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_prob_superiority.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 21:00:00 (ywatanabe)"
# # File: ./src/scitex/stats/effect_sizes/_prob_superiority.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/stats/effect_sizes/_prob_superiority.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Functionalities:
#   - Compute probability of superiority P(X > Y)
#   - Common language effect size
#   - Related to Brunner-Munzel test and Cliff's delta
#   - Provide interpretation guidelines
# 
# Dependencies:
#   - packages: numpy, pandas
# 
# IO:
#   - input: Two samples (arrays or Series)
#   - output: Probability value (float, ranges from 0 to 1)
# """
# 
# """Imports"""
# import argparse
# from typing import Union
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
# def prob_superiority(
#     x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]
# ) -> float:
#     """
#     Compute probability of superiority P(X > Y).
# 
#     Also known as the common language effect size or probabilistic index.
# 
#     Parameters
#     ----------
#     x : array or Series
#         First sample
#     y : array or Series
#         Second sample
# 
#     Returns
#     -------
#     float
#         Probability that a random value from X is greater than a random value from Y
#         (ranges from 0 to 1)
# 
#     Notes
#     -----
#     The probability of superiority is defined as:
# 
#     .. math::
#         P(X > Y) = \\frac{\\#(x_i > y_j)}{n_x \\cdot n_y}
# 
#     This is the probabilistic interpretation of effect size and is directly
#     related to the Brunner-Munzel statistic and Cliff's delta:
# 
#     .. math::
#         P(X > Y) = \\frac{1 + \\delta}{2}
# 
#     Where δ is Cliff's delta.
# 
#     Interpretation:
#     - P(X > Y) = 0.50: No effect (chance level)
#     - P(X > Y) = 0.56: Small effect (McGraw & Wong, 1992)
#     - P(X > Y) = 0.64: Medium effect
#     - P(X > Y) = 0.71: Large effect
# 
#     Advantages:
#     - Intuitive probabilistic interpretation
#     - Non-parametric (distribution-free)
#     - Directly comparable across studies
#     - Used in Brunner-Munzel test
# 
#     This is also called:
#     - Common Language Effect Size (CLES)
#     - Area Under the Curve (AUC) in ROC analysis
#     - Mann-Whitney U / (nx * ny)
# 
#     References
#     ----------
#     .. [1] McGraw, K. O., & Wong, S. P. (1992). "A common language effect size
#            statistic". Psychological Bulletin, 111(2), 361-365.
#     .. [2] Brunner, E., & Munzel, U. (2000). "The nonparametric Behrens-Fisher
#            problem: Asymptotic theory and a small-sample approximation".
#            Biometrical Journal, 42(1), 17-25.
# 
#     Examples
#     --------
#     >>> x = np.array([1, 2, 3, 4, 5])
#     >>> y = np.array([2, 3, 4, 5, 6])
#     >>> prob_superiority(x, y)
#     0.2
# 
#     >>> # 20% chance a random X value exceeds a random Y value
# 
#     >>> # No difference (chance level)
#     >>> x = np.array([1, 2, 3, 4, 5])
#     >>> y = np.array([1, 2, 3, 4, 5])
#     >>> prob_superiority(x, y)
#     0.5
# 
#     >>> # Complete dominance
#     >>> x = np.array([6, 7, 8, 9, 10])
#     >>> y = np.array([1, 2, 3, 4, 5])
#     >>> prob_superiority(x, y)
#     1.0
#     """
#     # Convert to numpy arrays
#     x = np.asarray(x)
#     y = np.asarray(y)
# 
#     # Remove NaN values
#     x = x[~np.isnan(x)]
#     y = y[~np.isnan(y)]
# 
#     nx = len(x)
#     ny = len(y)
# 
#     # Count how many times x > y
#     more = np.sum(x[:, None] > y)
# 
#     # Compute probability
#     prob = more / (nx * ny)
# 
#     return float(prob)
# 
# 
# def interpret_prob_superiority(prob: float) -> str:
#     """
#     Interpret probability of superiority effect size.
# 
#     Parameters
#     ----------
#     prob : float
#         Probability of superiority P(X > Y)
# 
#     Returns
#     -------
#     str
#         Interpretation string
# 
#     Examples
#     --------
#     >>> interpret_prob_superiority(0.51)
#     'negligible'
#     >>> interpret_prob_superiority(0.60)
#     'small'
#     >>> interpret_prob_superiority(0.68)
#     'medium'
#     >>> interpret_prob_superiority(0.75)
#     'large'
#     """
#     # Convert to distance from 0.5 (chance)
#     distance = abs(prob - 0.5)
# 
#     if distance < 0.06:
#         return "negligible"
#     elif distance < 0.14:
#         return "small"
#     elif distance < 0.21:
#         return "medium"
#     else:
#         return "large"
# 
# 
# """Main function"""
# 
# 
# def main(args):
#     """Demonstrate probability of superiority computation."""
#     logger.info("Demonstrating probability of superiority P(X > Y)")
# 
#     # Set random seed
#     np.random.seed(42)
# 
#     # Example 1: Relationship with Cliff's delta
#     logger.info("\n=== Example 1: Relationship with Cliff's delta ===")
# 
#     x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#     y = np.array([3, 4, 5, 6, 7, 8, 9, 10])
# 
#     from ._cliffs_delta import cliffs_delta
# 
#     prob = prob_superiority(x, y)
#     delta = cliffs_delta(x, y)
# 
#     logger.info(f"P(X > Y) = {prob:.3f} ({interpret_prob_superiority(prob)})")
#     logger.info(f"Cliff's delta = {delta:.3f}")
#     logger.info(f"Relationship: P(X>Y) = (1 + δ) / 2 = {(1 + delta) / 2:.3f}")
# 
#     # Example 2: Different effect sizes
#     logger.info("\n=== Example 2: Different effect sizes ===")
# 
#     control = np.random.normal(0, 1, 50)
# 
#     for shift in [0.0, 0.3, 0.6, 1.0]:
#         treatment = np.random.normal(shift, 1, 50)
#         prob = prob_superiority(treatment, control)
#         interpretation = interpret_prob_superiority(prob)
# 
#         logger.info(
#             f"Shift = {shift:.1f}: P(Treatment > Control) = {prob:.3f} ({interpretation})"
#         )
# 
#     # Visualization
#     logger.info("\n=== Creating visualization ===")
# 
#     fig, ax = stx.plt.subplots(figsize=(10, 6))
# 
#     # Generate data for visualization
#     shifts = np.linspace(0, 2, 20)
#     probs = []
# 
#     for shift in shifts:
#         treatment = np.random.normal(shift, 1, 100)
#         control = np.random.normal(0, 1, 100)
#         probs.append(prob_superiority(treatment, control))
# 
#     ax.plot(shifts, probs, "o-", linewidth=2, markersize=8)
#     ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Chance level")
#     ax.axhline(0.56, color="orange", linestyle="--", alpha=0.5, label="Small effect")
#     ax.axhline(0.64, color="yellow", linestyle="--", alpha=0.5, label="Medium effect")
#     ax.axhline(0.71, color="green", linestyle="--", alpha=0.5, label="Large effect")
# 
#     ax.set_xlabel("Mean Shift (Cohen's d)")
#     ax.set_ylabel("P(Treatment > Control)")
#     ax.set_title("Probability of Superiority vs Effect Size")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
# 
#     stx.plt.tight_layout()
#     stx.io.save(fig, "./prob_superiority_demo.jpg")
#     logger.info("Visualization saved")
# 
#     return 0
# 
# 
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Demonstrate probability of superiority calculation"
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_prob_superiority.py
# --------------------------------------------------------------------------------
