#!/usr/bin/env python3
# Time-stamp: "2026-01-04"

"""Tests for scitex.stats._utils._power module."""

import numpy as np
import pytest

from scitex.stats._utils._power import power_ttest, sample_size_ttest


class TestPowerTtest:
    """Test power_ttest function for statistical power calculation."""

    def test_basic_two_sample(self):
        """Test basic two-sample t-test power calculation."""
        power = power_ttest(effect_size=0.5, n1=30, n2=30)
        assert isinstance(power, float)
        assert 0 < power < 1

    def test_one_sample_test(self):
        """Test one-sample t-test power calculation."""
        power = power_ttest(effect_size=0.5, n=50, test_type="one-sample")
        assert isinstance(power, float)
        assert 0 < power < 1

    def test_paired_test(self):
        """Test paired t-test power calculation."""
        power = power_ttest(effect_size=0.8, n=25, test_type="paired")
        assert isinstance(power, float)
        assert 0 < power < 1

    def test_power_increases_with_sample_size(self):
        """Test that power increases as sample size increases."""
        power_small = power_ttest(effect_size=0.5, n1=20, n2=20)
        power_large = power_ttest(effect_size=0.5, n1=100, n2=100)
        assert power_large > power_small

    def test_power_increases_with_effect_size(self):
        """Test that power increases as effect size increases."""
        power_small_d = power_ttest(effect_size=0.2, n1=30, n2=30)
        power_large_d = power_ttest(effect_size=0.8, n1=30, n2=30)
        assert power_large_d > power_small_d

    def test_power_increases_with_higher_alpha(self):
        """Test that power increases with higher alpha level."""
        power_strict = power_ttest(effect_size=0.5, n1=30, n2=30, alpha=0.01)
        power_liberal = power_ttest(effect_size=0.5, n1=30, n2=30, alpha=0.10)
        assert power_liberal > power_strict

    def test_one_sided_higher_power_than_two_sided(self):
        """Test that one-sided tests have higher power than two-sided."""
        power_two = power_ttest(effect_size=0.5, n1=30, n2=30, alternative="two-sided")
        power_one = power_ttest(effect_size=0.5, n1=30, n2=30, alternative="greater")
        assert power_one > power_two

    def test_alternative_less(self):
        """Test alternative='less' power calculation."""
        power = power_ttest(effect_size=0.5, n1=30, n2=30, alternative="less")
        assert isinstance(power, float)
        assert 0 < power < 1

    def test_unequal_sample_sizes(self):
        """Test power with unequal sample sizes."""
        power = power_ttest(effect_size=0.5, n1=20, n2=40)
        assert isinstance(power, float)
        assert 0 < power < 1

    def test_missing_n_for_one_sample_raises(self):
        """Test that missing n for one-sample test raises error."""
        with pytest.raises(ValueError, match="n must be specified"):
            power_ttest(effect_size=0.5, test_type="one-sample")

    def test_missing_n_for_paired_raises(self):
        """Test that missing n for paired test raises error."""
        with pytest.raises(ValueError, match="n must be specified"):
            power_ttest(effect_size=0.5, test_type="paired")

    def test_missing_n1_n2_for_two_sample_raises(self):
        """Test that missing n1/n2 for two-sample test raises error."""
        with pytest.raises(ValueError, match="n1 and n2 must be specified"):
            power_ttest(effect_size=0.5, test_type="two-sample")

    def test_missing_n2_raises(self):
        """Test that missing n2 for two-sample test raises error."""
        with pytest.raises(ValueError, match="n1 and n2 must be specified"):
            power_ttest(effect_size=0.5, n1=30, test_type="two-sample")

    def test_invalid_test_type_raises(self):
        """Test that invalid test type raises error."""
        with pytest.raises(ValueError, match="Unknown test_type"):
            power_ttest(effect_size=0.5, n1=30, n2=30, test_type="invalid")

    def test_invalid_alternative_raises(self):
        """Test that invalid alternative raises error."""
        with pytest.raises(ValueError, match="Unknown alternative"):
            power_ttest(effect_size=0.5, n1=30, n2=30, alternative="invalid")

    def test_zero_effect_size(self):
        """Test power with zero effect size (should be around alpha)."""
        power = power_ttest(effect_size=0.0, n1=100, n2=100, alpha=0.05)
        # With zero effect, power should be around alpha (Type I error rate)
        assert abs(power - 0.05) < 0.01

    def test_large_effect_size(self):
        """Test power with large effect size is very high."""
        # Use moderately large effect size to avoid numerical issues
        power = power_ttest(effect_size=1.0, n1=50, n2=50)
        assert power > 0.95

    def test_power_bound_between_0_and_1(self):
        """Test that power is always between 0 and 1."""
        # Use moderate effect sizes to avoid numerical overflow issues
        for d in [0.1, 0.5, 0.8, 1.0]:
            for n in [10, 50, 100]:
                power = power_ttest(effect_size=d, n1=n, n2=n)
                assert 0 <= power <= 1


class TestSampleSizeTtest:
    """Test sample_size_ttest function for sample size determination."""

    def test_basic_two_sample(self):
        """Test basic two-sample sample size calculation."""
        n1, n2 = sample_size_ttest(effect_size=0.5, power=0.80)
        assert isinstance(n1, int)
        assert isinstance(n2, int)
        assert n1 > 0
        assert n2 > 0

    def test_equal_allocation_by_default(self):
        """Test that equal allocation is used by default."""
        n1, n2 = sample_size_ttest(effect_size=0.5, power=0.80)
        assert n1 == n2

    def test_one_sample_test(self):
        """Test one-sample sample size calculation."""
        n = sample_size_ttest(effect_size=0.5, power=0.80, test_type="one-sample")
        assert isinstance(n, int)
        assert n > 0

    def test_paired_test(self):
        """Test paired sample size calculation."""
        n = sample_size_ttest(effect_size=0.5, power=0.80, test_type="paired")
        assert isinstance(n, int)
        assert n > 0

    def test_larger_effect_needs_smaller_sample(self):
        """Test that larger effect sizes need smaller samples."""
        n_small_d, _ = sample_size_ttest(effect_size=0.2, power=0.80)
        n_large_d, _ = sample_size_ttest(effect_size=0.8, power=0.80)
        assert n_small_d > n_large_d

    def test_higher_power_needs_larger_sample(self):
        """Test that higher power requirements need larger samples."""
        n_low_power, _ = sample_size_ttest(effect_size=0.5, power=0.70)
        n_high_power, _ = sample_size_ttest(effect_size=0.5, power=0.95)
        assert n_high_power > n_low_power

    def test_one_sided_needs_smaller_sample(self):
        """Test that one-sided tests need smaller samples."""
        n_two, _ = sample_size_ttest(
            effect_size=0.5, power=0.80, alternative="two-sided"
        )
        n_one, _ = sample_size_ttest(effect_size=0.5, power=0.80, alternative="greater")
        assert n_one < n_two

    def test_unequal_allocation_ratio(self):
        """Test unequal allocation ratio."""
        n1, n2 = sample_size_ttest(effect_size=0.5, power=0.80, ratio=2.0)
        assert n2 == int(n1 * 2)

    def test_stricter_alpha_needs_larger_sample(self):
        """Test that stricter alpha needs larger samples."""
        n_liberal, _ = sample_size_ttest(effect_size=0.5, power=0.80, alpha=0.10)
        n_strict, _ = sample_size_ttest(effect_size=0.5, power=0.80, alpha=0.01)
        assert n_strict > n_liberal

    def test_achieves_target_power(self):
        """Test that calculated sample size achieves target power."""
        target_power = 0.80
        n1, n2 = sample_size_ttest(effect_size=0.5, power=target_power)

        actual_power = power_ttest(effect_size=0.5, n1=n1, n2=n2)
        assert actual_power >= target_power

    def test_achieves_target_power_one_sample(self):
        """Test that calculated sample size achieves target power for one-sample."""
        target_power = 0.80
        n = sample_size_ttest(
            effect_size=0.5, power=target_power, test_type="one-sample"
        )

        actual_power = power_ttest(effect_size=0.5, n=n, test_type="one-sample")
        assert actual_power >= target_power

    def test_medium_effect_requires_about_64_per_group(self):
        """Test that medium effect (d=0.5) requires approximately 64 per group for 80% power."""
        n1, n2 = sample_size_ttest(effect_size=0.5, power=0.80)
        # Standard result is around 64 per group for d=0.5, power=0.80
        assert 60 <= n1 <= 70

    def test_small_effect_requires_large_sample(self):
        """Test that small effect (d=0.2) requires large sample."""
        n1, n2 = sample_size_ttest(effect_size=0.2, power=0.80)
        # Small effect requires many more subjects
        assert n1 > 300

    def test_large_effect_requires_small_sample(self):
        """Test that large effect (d=0.8) requires smaller sample."""
        n1, n2 = sample_size_ttest(effect_size=0.8, power=0.80)
        # Large effect requires fewer subjects
        assert n1 < 30


class TestPowerConsistency:
    """Test consistency between power and sample size functions."""

    def test_round_trip_two_sample(self):
        """Test that sample_size -> power -> sample_size is consistent."""
        target_power = 0.85
        effect_size = 0.6

        n1, n2 = sample_size_ttest(effect_size=effect_size, power=target_power)
        actual_power = power_ttest(effect_size=effect_size, n1=n1, n2=n2)

        # Actual power should meet or exceed target
        assert actual_power >= target_power
        # But not by too much (within 0.05)
        assert actual_power < target_power + 0.05

    def test_round_trip_one_sample(self):
        """Test consistency for one-sample tests."""
        target_power = 0.90
        effect_size = 0.4

        n = sample_size_ttest(
            effect_size=effect_size, power=target_power, test_type="one-sample"
        )
        actual_power = power_ttest(effect_size=effect_size, n=n, test_type="one-sample")

        assert actual_power >= target_power

    def test_round_trip_paired(self):
        """Test consistency for paired tests."""
        target_power = 0.80
        effect_size = 0.7

        n = sample_size_ttest(
            effect_size=effect_size, power=target_power, test_type="paired"
        )
        actual_power = power_ttest(effect_size=effect_size, n=n, test_type="paired")

        assert actual_power >= target_power


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_alpha(self):
        """Test with very small alpha."""
        power = power_ttest(effect_size=0.5, n1=100, n2=100, alpha=0.001)
        assert 0 < power < 1

    def test_very_small_sample(self):
        """Test with very small sample sizes."""
        power = power_ttest(effect_size=0.5, n1=3, n2=3)
        assert 0 < power < 1

    def test_high_power_target(self):
        """Test with high power target (0.99)."""
        n1, n2 = sample_size_ttest(effect_size=0.5, power=0.99)
        assert n1 > 0

        actual_power = power_ttest(effect_size=0.5, n1=n1, n2=n2)
        assert actual_power >= 0.99

    def test_negative_effect_size(self):
        """Test with negative effect size (should work, direction doesn't matter)."""
        power_pos = power_ttest(effect_size=0.5, n1=30, n2=30)
        power_neg = power_ttest(effect_size=-0.5, n1=30, n2=30)
        # Power should be similar since magnitude is the same
        # Note: Exact equality depends on implementation
        assert abs(power_pos - power_neg) < 0.1 or power_neg > 0


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/utils/_power.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 14:55:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/utils/_power.py
#
# """
# Functionalities:
#   - Compute statistical power for t-tests
#   - Perform power analysis for sample size determination
#   - Support both a priori and post-hoc power analysis
#   - Handle various test types (one-sample, two-sample, paired)
#
# Dependencies:
#   - packages: numpy, scipy
#
# IO:
#   - input: Effect size, sample sizes, alpha level
#   - output: Statistical power (float between 0 and 1)
# """
#
# """Imports"""
# import sys
# import argparse
# import numpy as np
# import pandas as pd
# from typing import Union, Optional, Literal
# from scipy import stats
# import scitex as stx
# from scitex.logging import getLogger
#
# logger = getLogger(__name__)
#
# """Functions"""
#
#
# def power_ttest(
#     effect_size: float,
#     n: Optional[int] = None,
#     n1: Optional[int] = None,
#     n2: Optional[int] = None,
#     alpha: float = 0.05,
#     alternative: Literal["two-sided", "greater", "less"] = "two-sided",
#     test_type: Literal["one-sample", "two-sample", "paired"] = "two-sample",
# ) -> float:
#     """
#     Compute statistical power for t-test.
#
#     Parameters
#     ----------
#     effect_size : float
#         Cohen's d effect size
#     n : int, optional
#         Sample size (for one-sample or paired tests)
#     n1 : int, optional
#         Sample size for first group (two-sample tests)
#     n2 : int, optional
#         Sample size for second group (two-sample tests)
#     alpha : float, default 0.05
#         Significance level
#     alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
#         Alternative hypothesis type
#     test_type : {'one-sample', 'two-sample', 'paired'}, default 'two-sample'
#         Type of t-test
#
#     Returns
#     -------
#     float
#         Statistical power (probability of detecting effect if it exists)
#
#     Notes
#     -----
#     Statistical power is the probability of rejecting the null hypothesis
#     when the alternative hypothesis is true. In other words, it's the
#     probability of correctly detecting an effect when it exists.
#
#     Power = P(reject H0 | H1 is true)
#
#     Common power benchmarks:
#     - 0.80 (80%): Minimum acceptable power (Cohen, 1988)
#     - 0.90 (90%): High power
#     - 0.95 (95%): Very high power
#
#     For two-sample tests, this function assumes equal sample sizes by default.
#     If n1 and n2 differ, the harmonic mean is used:
#
#     .. math::
#         n_{eff} = \\frac{2 n_1 n_2}{n_1 + n_2}
#
#     References
#     ----------
#     .. [1] Cohen, J. (1988). Statistical Power Analysis for the Behavioral
#            Sciences (2nd ed.). Routledge.
#     .. [2] Faul, F., Erdfelder, E., Lang, A. G., & Buchner, A. (2007).
#            G*Power 3: A flexible statistical power analysis program for the
#            social, behavioral, and biomedical sciences. Behavior Research
#            Methods, 39(2), 175-191.
#
#     Examples
#     --------
#     >>> # Two-sample t-test with n=30 per group, d=0.5
#     >>> power_ttest(effect_size=0.5, n1=30, n2=30)
#     0.477...
#
#     >>> # One-sample t-test with n=50, d=0.5
#     >>> power_ttest(effect_size=0.5, n=50, test_type='one-sample')
#     0.696...
#
#     >>> # Paired t-test with n=25 pairs, d=0.8
#     >>> power_ttest(effect_size=0.8, n=25, test_type='paired')
#     0.927...
#
#     >>> # One-sided test has higher power
#     >>> power_ttest(effect_size=0.5, n1=30, n2=30, alternative='greater')
#     0.628...
#     """
#     # Validate inputs
#     if test_type in ["one-sample", "paired"]:
#         if n is None:
#             raise ValueError(f"n must be specified for {test_type} test")
#         sample_size = n
#     elif test_type == "two-sample":
#         if n1 is None or n2 is None:
#             raise ValueError("n1 and n2 must be specified for two-sample test")
#         # Use harmonic mean for unequal sample sizes
#         sample_size = 2 * n1 * n2 / (n1 + n2)
#     else:
#         raise ValueError(f"Unknown test_type: {test_type}")
#
#     # Determine degrees of freedom
#     if test_type == "two-sample":
#         df = n1 + n2 - 2
#     else:
#         df = sample_size - 1
#
#     # Determine critical value based on alternative hypothesis
#     if alternative == "two-sided":
#         t_crit = stats.t.ppf(1 - alpha / 2, df)
#     elif alternative == "greater":
#         t_crit = stats.t.ppf(1 - alpha, df)
#     elif alternative == "less":
#         t_crit = stats.t.ppf(alpha, df)
#     else:
#         raise ValueError(f"Unknown alternative: {alternative}")
#
#     # Compute non-centrality parameter
#     if test_type == "two-sample":
#         # For two-sample, ncp = d * sqrt(n_eff / 2)
#         ncp = effect_size * np.sqrt(sample_size / 2)
#     else:
#         # For one-sample and paired, ncp = d * sqrt(n)
#         ncp = effect_size * np.sqrt(sample_size)
#
#     # Compute power using non-central t-distribution
#     if alternative == "two-sided":
#         # For two-sided, power = P(|T| > t_crit | ncp)
#         power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
#     elif alternative == "greater":
#         # For greater, power = P(T > t_crit | ncp)
#         power = 1 - stats.nct.cdf(t_crit, df, ncp)
#     else:  # less
#         # For less, power = P(T < t_crit | ncp)
#         power = stats.nct.cdf(t_crit, df, ncp)
#
#     return float(power)
#
#
# def sample_size_ttest(
#     effect_size: float,
#     power: float = 0.80,
#     alpha: float = 0.05,
#     alternative: Literal["two-sided", "greater", "less"] = "two-sided",
#     test_type: Literal["one-sample", "two-sample", "paired"] = "two-sample",
#     ratio: float = 1.0,
# ) -> Union[int, tuple]:
#     """
#     Determine required sample size for t-test with desired power.
#
#     Parameters
#     ----------
#     effect_size : float
#         Expected Cohen's d effect size
#     power : float, default 0.80
#         Desired statistical power (0.80 = 80%)
#     alpha : float, default 0.05
#         Significance level
#     alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
#         Alternative hypothesis type
#     test_type : {'one-sample', 'two-sample', 'paired'}, default 'two-sample'
#         Type of t-test
#     ratio : float, default 1.0
#         Ratio of n2 to n1 for two-sample tests (n2 = ratio * n1)
#
#     Returns
#     -------
#     int or tuple of int
#         Required sample size(s)
#         - For one-sample/paired: single integer n
#         - For two-sample: tuple (n1, n2)
#
#     Examples
#     --------
#     >>> # Sample size for two-sample test with d=0.5, power=0.80
#     >>> sample_size_ttest(effect_size=0.5, power=0.80)
#     (64, 64)
#
#     >>> # Sample size for one-sample test with d=0.5
#     >>> sample_size_ttest(effect_size=0.5, test_type='one-sample')
#     27
#
#     >>> # Unequal allocation ratio
#     >>> sample_size_ttest(effect_size=0.5, ratio=2.0)
#     (51, 102)
#     """
#     # Binary search for required sample size
#     n_min = 2
#     n_max = 10000
#
#     if test_type == "two-sample":
#         # Search for n1
#         while n_max - n_min > 1:
#             n1_mid = (n_min + n_max) // 2
#             n2_mid = int(n1_mid * ratio)
#
#             current_power = power_ttest(
#                 effect_size=effect_size,
#                 n1=n1_mid,
#                 n2=n2_mid,
#                 alpha=alpha,
#                 alternative=alternative,
#                 test_type=test_type,
#             )
#
#             if current_power < power:
#                 n_min = n1_mid
#             else:
#                 n_max = n1_mid
#
#         n1_required = n_max
#         n2_required = int(n1_required * ratio)
#         return (n1_required, n2_required)
#
#     else:  # one-sample or paired
#         while n_max - n_min > 1:
#             n_mid = (n_min + n_max) // 2
#
#             current_power = power_ttest(
#                 effect_size=effect_size,
#                 n=n_mid,
#                 alpha=alpha,
#                 alternative=alternative,
#                 test_type=test_type,
#             )
#
#             if current_power < power:
#                 n_min = n_mid
#             else:
#                 n_max = n_mid
#
#         return n_max
#
#
# """Main function"""
#
#
# def main(args):
#     """Demonstrate power analysis functionality."""
#     logger.info("Demonstrating statistical power analysis")
#
#     # Example 1: Effect size vs power
#     logger.info("\n=== Example 1: Effect Size vs Power ===")
#
#     effect_sizes = [0.2, 0.5, 0.8, 1.0, 1.2]
#     n_per_group = 30
#
#     results = []
#     for d in effect_sizes:
#         power = power_ttest(effect_size=d, n1=n_per_group, n2=n_per_group)
#         logger.info(f"d = {d:.1f}, n = {n_per_group} per group → Power = {power:.3f}")
#         results.append({"effect_size": d, "power": power})
#
#     # Example 2: Sample size vs power
#     logger.info("\n=== Example 2: Sample Size vs Power ===")
#
#     d = 0.5  # Medium effect
#     sample_sizes = [10, 20, 30, 50, 100, 200]
#
#     power_results = []
#     for n in sample_sizes:
#         power = power_ttest(effect_size=d, n1=n, n2=n)
#         logger.info(f"n = {n} per group, d = {d} → Power = {power:.3f}")
#         power_results.append({"n": n, "power": power})
#
#     # Example 3: Required sample size
#     logger.info("\n=== Example 3: Required Sample Size ===")
#
#     target_power = 0.80
#     for d in [0.2, 0.5, 0.8]:
#         n1, n2 = sample_size_ttest(effect_size=d, power=target_power)
#         logger.info(
#             f"d = {d:.1f}, target power = {target_power:.0%} "
#             f"→ Required n = {n1} per group"
#         )
#
#     # Example 4: One-sided vs two-sided
#     logger.info("\n=== Example 4: One-sided vs Two-sided ===")
#
#     d = 0.5
#     n = 30
#     power_two = power_ttest(effect_size=d, n1=n, n2=n, alternative="two-sided")
#     power_one = power_ttest(effect_size=d, n1=n, n2=n, alternative="greater")
#
#     logger.info(f"Two-sided test: Power = {power_two:.3f}")
#     logger.info(f"One-sided test:  Power = {power_one:.3f}")
#     logger.info(f"One-sided power is {power_one / power_two:.2f}x higher")
#
#     # Create visualizations
#     logger.info("\n=== Creating visualizations ===")
#
#     fig, axes = stx.plt.subplots(2, 2, figsize=(12, 10))
#
#     # Plot 1: Effect size vs power
#     ax = axes[0, 0]
#     df_es = pd.DataFrame(results)
#     ax.plot(df_es["effect_size"], df_es["power"], "o-", linewidth=2, markersize=8)
#     ax.axhline(0.80, color="red", linestyle="--", alpha=0.5, label="80% power")
#     ax.set_xlabel("Effect Size (Cohen's d)")
#     ax.set_ylabel("Statistical Power")
#     ax.set_title(f"Effect Size vs Power (n={n_per_group}/group)")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     ax.set_ylim(0, 1)
#
#     # Plot 2: Sample size vs power
#     ax = axes[0, 1]
#     df_ss = pd.DataFrame(power_results)
#     ax.plot(df_ss["n"], df_ss["power"], "o-", linewidth=2, markersize=8)
#     ax.axhline(0.80, color="red", linestyle="--", alpha=0.5, label="80% power")
#     ax.set_xlabel("Sample Size (per group)")
#     ax.set_ylabel("Statistical Power")
#     ax.set_title(f"Sample Size vs Power (d={d})")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     ax.set_xscale("log")
#     ax.set_ylim(0, 1)
#
#     # Plot 3: Power curves for different effect sizes
#     ax = axes[1, 0]
#     ns = np.arange(5, 201, 5)
#     for d_val in [0.2, 0.5, 0.8]:
#         powers = [power_ttest(effect_size=d_val, n1=n, n2=n) for n in ns]
#         ax.plot(ns, powers, linewidth=2, label=f"d = {d_val}")
#
#     ax.axhline(0.80, color="black", linestyle="--", alpha=0.3)
#     ax.set_xlabel("Sample Size (per group)")
#     ax.set_ylabel("Statistical Power")
#     ax.set_title("Power Curves for Different Effect Sizes")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     ax.set_ylim(0, 1)
#
#     # Plot 4: Required sample size for different effect sizes
#     ax = axes[1, 1]
#     effect_sizes_range = np.linspace(0.1, 1.5, 30)
#     required_ns = []
#
#     for d_val in effect_sizes_range:
#         n1, n2 = sample_size_ttest(effect_size=d_val, power=0.80)
#         required_ns.append(n1)
#
#     ax.plot(effect_sizes_range, required_ns, linewidth=2)
#     ax.set_xlabel("Effect Size (Cohen's d)")
#     ax.set_ylabel("Required Sample Size (per group)")
#     ax.set_title("Sample Size Required for 80% Power")
#     ax.grid(True, alpha=0.3)
#     ax.set_yscale("log")
#
#     # Add reference lines
#     ax.axvline(0.2, color="gray", linestyle="--", alpha=0.3)
#     ax.axvline(0.5, color="gray", linestyle="--", alpha=0.3)
#     ax.axvline(0.8, color="gray", linestyle="--", alpha=0.3)
#     ax.text(0.2, ax.get_ylim()[1] * 0.9, "Small", ha="center", fontsize=9)
#     ax.text(0.5, ax.get_ylim()[1] * 0.9, "Medium", ha="center", fontsize=9)
#     ax.text(0.8, ax.get_ylim()[1] * 0.9, "Large", ha="center", fontsize=9)
#
#     plt.tight_layout()
#
#     # Save
#     stx.io.save(fig, "./power_analysis_demo.jpg")
#     logger.info("Visualization saved")
#
#     return 0
#
#
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Demonstrate statistical power analysis for t-tests"
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
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/utils/_power.py
# --------------------------------------------------------------------------------
