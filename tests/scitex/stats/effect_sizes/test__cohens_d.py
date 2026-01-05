#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for Cohen's d effect size.

Tests cover:
- Basic computations (independent samples)
- One-sample tests (against zero)
- Paired samples
- Different corrections (None, 'hedges', 'glass')
- Edge cases (equal means, NaN handling, empty arrays)
- Known values validation
- Interpretation levels
"""

import numpy as np
import pandas as pd
import pytest
from scitex.stats.effect_sizes import cohens_d, interpret_cohens_d


class TestBasicComputation:
    """Tests for basic Cohen's d computations."""

    def test_independent_samples_basic(self):
        """Test basic independent samples."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        d = cohens_d(x, y)

        # Mean diff = -1, pooled std = 1.58..., d should be ~ -0.632
        assert isinstance(d, float)
        assert d < 0  # y has higher mean
        assert -1.5 < d < 0

    def test_independent_samples_known_value(self):
        """Test with manually calculated known value."""
        # Two groups with mean diff = 1, both std = 1
        x = np.array([0, 1, 2])  # mean=1, std=1
        y = np.array([1, 2, 3])  # mean=2, std=1
        d = cohens_d(x, y)

        # Expected: d = (1-2) / pooled_std(1) = -1.0
        assert abs(d - (-1.0)) < 0.01

    def test_one_sample_against_zero(self):
        """Test one-sample Cohen's d against zero."""
        x = np.array([2, 4, 6, 8, 10])  # mean=6, std=sqrt(10)
        d = cohens_d(x, y=None)

        # d = mean / std
        expected_d = np.mean(x) / np.std(x, ddof=1)
        assert abs(d - expected_d) < 0.001

    def test_pandas_series_input(self):
        """Test that pandas Series work as input."""
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 3, 4, 5, 6])
        d = cohens_d(x, y)

        assert isinstance(d, float)
        assert not np.isnan(d)


class TestPairedSamples:
    """Tests for paired sample Cohen's d."""

    def test_paired_samples_basic(self):
        """Test paired samples computation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        d = cohens_d(x, y, paired=True)

        # diff = x - y = [-1, -1, -1, -1, -1]
        # mean_diff = -1, std_diff = 0
        # Since std is 0, this will give inf or very large value
        assert isinstance(d, float)

    def test_paired_different_lengths_raises_error(self):
        """Test that paired samples with different lengths raise error."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4])

        with pytest.raises(ValueError, match="same length"):
            cohens_d(x, y, paired=True)

    def test_paired_vs_independent(self):
        """Test that paired gives different result than independent."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 20)
        followup = baseline + 0.5 + np.random.normal(0, 0.3, 20)

        d_independent = cohens_d(baseline, followup, paired=False)
        d_paired = cohens_d(baseline, followup, paired=True)

        # They should be different
        assert abs(d_independent - d_paired) > 0.1


class TestCorrections:
    """Tests for different correction methods."""

    def test_no_correction(self):
        """Test standard Cohen's d without correction."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])
        d = cohens_d(x, y, correction=None)

        assert isinstance(d, float)

    def test_hedges_correction(self):
        """Test Hedges' g correction."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])

        d_standard = cohens_d(x, y, correction=None)
        d_hedges = cohens_d(x, y, correction='hedges')

        # Hedges' g should be slightly smaller (correction factor < 1)
        assert abs(d_hedges) < abs(d_standard)
        assert abs(d_hedges / d_standard) > 0.9  # But close

    def test_glass_correction(self):
        """Test Glass's delta (uses control group SD only)."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])

        d_glass = cohens_d(x, y, correction='glass')

        # Glass's delta = mean_diff / std(y)
        mean_diff = np.mean(x) - np.mean(y)
        expected = mean_diff / np.std(y, ddof=1)
        assert abs(d_glass - expected) < 0.001

    def test_hedges_small_sample(self):
        """Test that Hedges' correction matters for small samples."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        d_standard = cohens_d(x, y, correction=None)
        d_hedges = cohens_d(x, y, correction='hedges')

        # For small samples, difference should be more noticeable
        assert abs(d_standard - d_hedges) > 0.01


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_equal_means_zero_effect(self):
        """Test that equal means give d close to zero."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        d = cohens_d(x, y)

        assert abs(d) < 0.01

    def test_nan_handling(self):
        """Test that NaN values are properly removed."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, np.nan, 4, 5, 6])
        d = cohens_d(x, y)

        # Should compute without error
        assert isinstance(d, float)
        assert not np.isnan(d)

    def test_all_nan_array(self):
        """Test behavior with all NaN values."""
        x = np.array([np.nan, np.nan])
        y = np.array([1, 2, 3])

        # This should either handle gracefully or raise appropriate error
        # Depending on implementation
        try:
            d = cohens_d(x, y)
            # If it doesn't raise, check result
            assert np.isnan(d) or np.isinf(d)
        except (ValueError, ZeroDivisionError):
            # Also acceptable
            pass

    def test_single_value_arrays(self):
        """Test with single-value arrays (std = 0)."""
        x = np.array([5])
        y = np.array([10])

        # Will have division by zero
        d = cohens_d(x, y)
        assert np.isinf(d) or np.isnan(d)


class TestKnownValues:
    """Tests with manually calculated known values."""

    def test_textbook_example_1(self):
        """Test classic example: mean diff = 0.5, pooled std = 1."""
        # Create data with controlled properties
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = np.random.normal(0.5, 1, n)

        d = cohens_d(x, y)

        # Should be approximately -0.5
        assert abs(d - (-0.5)) < 0.2  # Allow some random variation

    def test_textbook_example_2(self):
        """Test large effect size."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        y = np.random.normal(1.0, 1, n)

        d = cohens_d(x, y)

        # Should be approximately -1.0 (large effect)
        assert abs(d - (-1.0)) < 0.3  # Allow more variation for random data


class TestMathematicalProperties:
    """Tests for mathematical properties of Cohen's d."""

    def test_symmetry(self):
        """Test that swapping x and y flips the sign."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])

        d_xy = cohens_d(x, y)
        d_yx = cohens_d(y, x)

        # Should be negatives of each other
        assert abs(d_xy + d_yx) < 0.001

    def test_scale_invariance(self):
        """Test that d is invariant to scaling both groups equally."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])

        d1 = cohens_d(x, y)
        d2 = cohens_d(x * 10, y * 10)

        # Should be approximately equal (scale invariant)
        assert abs(d1 - d2) < 0.001

    def test_location_invariance(self):
        """Test that d is invariant to shifting both groups equally."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([3, 4, 5, 6, 7])

        d1 = cohens_d(x, y)
        d2 = cohens_d(x + 100, y + 100)

        # Should be approximately equal (location invariant)
        assert abs(d1 - d2) < 0.001


class TestInterpretation:
    """Tests for effect size interpretation."""

    def test_interpret_negligible(self):
        """Test negligible effect interpretation."""
        assert interpret_cohens_d(0.1) == 'negligible'
        assert interpret_cohens_d(-0.15) == 'negligible'

    def test_interpret_small(self):
        """Test small effect interpretation."""
        assert interpret_cohens_d(0.3) == 'small'
        assert interpret_cohens_d(-0.4) == 'small'

    def test_interpret_medium(self):
        """Test medium effect interpretation."""
        assert interpret_cohens_d(0.6) == 'medium'
        assert interpret_cohens_d(-0.7) == 'medium'

    def test_interpret_large(self):
        """Test large effect interpretation."""
        assert interpret_cohens_d(0.9) == 'large'
        assert interpret_cohens_d(-1.2) == 'large'

    def test_interpret_boundaries(self):
        """Test interpretation at boundaries."""
        # Boundaries: 0.2 (small), 0.5 (medium), 0.8 (large)
        assert interpret_cohens_d(0.2) == 'small'
        assert interpret_cohens_d(0.5) == 'medium'
        assert interpret_cohens_d(0.8) == 'large'

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_cohens_d.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 21:00:00 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/effect_sizes/_cohens_d.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Functionalities:
#   - Compute Cohen's d effect size for t-tests
#   - Support both independent and paired samples
#   - Handle various pooling methods (standard, Hedges' g, Glass's delta)
#   - Provide interpretation guidelines
# 
# Dependencies:
#   - packages: numpy, pandas
# 
# IO:
#   - input: Two samples (arrays or Series)
#   - output: Effect size value (float)
# """
# 
# """Imports"""
# import argparse
# from typing import Literal, Optional, Union
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
# def cohens_d(
#     x: Union[np.ndarray, pd.Series],
#     y: Optional[Union[np.ndarray, pd.Series]] = None,
#     paired: bool = False,
#     correction: Literal["hedges", "glass", None] = None,
# ) -> float:
#     """
#     Compute Cohen's d effect size.
# 
#     Parameters
#     ----------
#     x : array or Series
#         First sample
#     y : array or Series, optional
#         Second sample. If None, computes one-sample effect size against zero.
#     paired : bool, default False
#         Whether samples are paired
#     correction : {'hedges', 'glass', None}, default None
#         Correction method:
#         - None: Standard Cohen's d
#         - 'hedges': Hedges' g (corrected for small samples)
#         - 'glass': Glass's delta (uses only control group SD)
# 
#     Returns
#     -------
#     float
#         Effect size value
# 
#     Notes
#     -----
#     Cohen's d is calculated as:
# 
#     .. math::
#         d = \\frac{\\bar{x}_1 - \\bar{x}_2}{s_{pooled}}
# 
#     where :math:`s_{pooled}` is the pooled standard deviation:
# 
#     .. math::
#         s_{pooled} = \\sqrt{\\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
# 
#     Interpretation guidelines (Cohen, 1988):
#     - Small effect: d = 0.2
#     - Medium effect: d = 0.5
#     - Large effect: d = 0.8
# 
#     For paired samples, d is computed as:
# 
#     .. math::
#         d = \\frac{\\bar{d}}{s_d}
# 
#     where :math:`\\bar{d}` is the mean difference and :math:`s_d` is the
#     standard deviation of differences.
# 
#     References
#     ----------
#     .. [1] Cohen, J. (1988). Statistical Power Analysis for the Behavioral
#            Sciences (2nd ed.). Routledge.
# 
#     Examples
#     --------
#     >>> x = np.array([1, 2, 3, 4, 5])
#     >>> y = np.array([2, 3, 4, 5, 6])
#     >>> cohens_d(x, y)
#     -1.0
# 
#     >>> # Paired samples
#     >>> cohens_d(x, y, paired=True)
#     -1.58...
# 
#     >>> # With Hedges' correction
#     >>> cohens_d(x, y, correction='hedges')
#     -0.95...
#     """
#     # Convert to numpy arrays
#     x = np.asarray(x)
#     if y is not None:
#         y = np.asarray(y)
# 
#     # Remove NaN values
#     x = x[~np.isnan(x)]
#     if y is not None:
#         y = y[~np.isnan(y)]
# 
#     # Compute effect size
#     if y is None:
#         # One-sample: compare to zero
#         d = np.mean(x) / np.std(x, ddof=1)
#     elif paired:
#         # Paired samples
#         if len(x) != len(y):
#             raise ValueError("Paired samples must have same length")
#         diff = x - y
#         d = np.mean(diff) / np.std(diff, ddof=1)
#     else:
#         # Independent samples
#         n1, n2 = len(x), len(y)
#         mean_diff = np.mean(x) - np.mean(y)
# 
#         if correction == "glass":
#             # Glass's delta: use only control group (y) SD
#             d = mean_diff / np.std(y, ddof=1)
#         else:
#             # Pooled standard deviation
#             var1 = np.var(x, ddof=1)
#             var2 = np.var(y, ddof=1)
#             pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
#             d = mean_diff / pooled_std
# 
#         # Apply Hedges' correction for small samples
#         if correction == "hedges":
#             # Hedges' g correction factor
#             correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
#             d = d * correction_factor
# 
#     return float(d)
# 
# 
# def interpret_cohens_d(d: float) -> str:
#     """
#     Interpret Cohen's d effect size.
# 
#     Parameters
#     ----------
#     d : float
#         Cohen's d value
# 
#     Returns
#     -------
#     str
#         Interpretation string
# 
#     Examples
#     --------
#     >>> interpret_cohens_d(0.3)
#     'small'
#     >>> interpret_cohens_d(0.6)
#     'medium'
#     >>> interpret_cohens_d(0.9)
#     'large'
#     """
#     d_abs = abs(d)
# 
#     if d_abs < 0.2:
#         return "negligible"
#     elif d_abs < 0.5:
#         return "small"
#     elif d_abs < 0.8:
#         return "medium"
#     else:
#         return "large"
# 
# 
# """Main function"""
# 
# 
# def main(args):
#     """Demonstrate Cohen's d computation."""
#     logger.info("Demonstrating Cohen's d effect size")
# 
#     # Set random seed
#     np.random.seed(42)
# 
#     # Example 1: Different effect sizes
#     logger.info("\n=== Example 1: Different effect sizes ===")
# 
#     n = 50
#     control = np.random.normal(0, 1, n)
# 
#     effect_sizes = [0.0, 0.2, 0.5, 0.8, 1.2]
#     results = []
# 
#     for true_d in effect_sizes:
#         treatment = np.random.normal(true_d, 1, n)
#         computed_d = cohens_d(control, treatment)
#         interpretation = interpret_cohens_d(computed_d)
# 
#         logger.info(
#             f"True d = {true_d:.1f}, "
#             f"Computed d = {computed_d:.3f}, "
#             f"Interpretation: {interpretation}"
#         )
# 
#         results.append(
#             {
#                 "true_d": true_d,
#                 "computed_d": computed_d,
#                 "interpretation": interpretation,
#             }
#         )
# 
#     # Example 2: Paired vs independent
#     logger.info("\n=== Example 2: Paired vs Independent ===")
# 
#     n_pairs = 30
#     baseline = np.random.normal(0, 1, n_pairs)
#     noise = np.random.normal(0, 0.3, n_pairs)
#     followup = baseline + 0.5 + noise
# 
#     d_independent = cohens_d(baseline, followup, paired=False)
#     d_paired = cohens_d(baseline, followup, paired=True)
# 
#     logger.info(f"Independent samples d = {d_independent:.3f}")
#     logger.info(f"Paired samples d = {d_paired:.3f}")
#     logger.info(f"Paired is more sensitive due to correlation")
# 
#     # Example 3: Correction methods
#     logger.info("\n=== Example 3: Correction methods ===")
# 
#     small_n = 10
#     x_small = np.random.normal(0, 1, small_n)
#     y_small = np.random.normal(0.5, 1, small_n)
# 
#     d_standard = cohens_d(x_small, y_small)
#     d_hedges = cohens_d(x_small, y_small, correction="hedges")
#     d_glass = cohens_d(x_small, y_small, correction="glass")
# 
#     logger.info(f"Standard Cohen's d = {d_standard:.3f}")
#     logger.info(f"Hedges' g = {d_hedges:.3f} (corrected for small n)")
#     logger.info(f"Glass's delta = {d_glass:.3f} (uses control SD only)")
# 
#     # Visualization
#     logger.info("\n=== Creating visualization ===")
# 
#     fig, axes = stx.plt.subplots(1, 2, figsize=(12, 5))
# 
#     # Plot 1: Distribution visualization
#     ax = axes[0]
#     control_demo = np.random.normal(0, 1, 1000)
#     treatment_demo = np.random.normal(0.8, 1, 1000)
# 
#     ax.hist(control_demo, bins=30, alpha=0.5, label="Control", density=True)
#     ax.hist(treatment_demo, bins=30, alpha=0.5, label="Treatment", density=True)
# 
#     d_demo = cohens_d(control_demo, treatment_demo)
#     ax.axvline(np.mean(control_demo), color="blue", linestyle="--", alpha=0.7)
#     ax.axvline(np.mean(treatment_demo), color="orange", linestyle="--", alpha=0.7)
# 
#     ax.set_xlabel("Value")
#     ax.set_ylabel("Density")
#     ax.set_title(f"Distributions (Cohen's d = {d_demo:.2f})")
#     ax.legend()
# 
#     # Plot 2: Effect size comparison
#     ax = axes[1]
#     df_results = pd.DataFrame(results)
#     ax.scatter(df_results["true_d"], df_results["computed_d"], s=100)
#     ax.plot([-0.5, 1.5], [-0.5, 1.5], "k--", alpha=0.5, label="Perfect agreement")
#     ax.set_xlabel("True Effect Size")
#     ax.set_ylabel("Computed Cohen's d")
#     ax.set_title("True vs Computed Effect Sizes")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
# 
#     stx.plt.tight_layout()
#     stx.io.save(fig, "./cohens_d_demo.jpg")
#     logger.info("Visualization saved")
# 
#     return 0
# 
# 
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Demonstrate Cohen's d effect size calculation"
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/effect_sizes/_cohens_d.py
# --------------------------------------------------------------------------------
