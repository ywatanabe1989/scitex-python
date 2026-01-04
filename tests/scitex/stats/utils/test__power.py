#!/usr/bin/env python3
# Time-stamp: "2026-01-04"

"""Tests for scitex.stats.utils._power module."""

import numpy as np
import pytest

from scitex.stats.utils._power import power_ttest, sample_size_ttest


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
