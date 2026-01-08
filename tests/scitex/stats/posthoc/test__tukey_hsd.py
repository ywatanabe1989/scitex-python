#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test file for scitex.stats.posthoc._tukey_hsd

import pytest
import numpy as np
import pandas as pd
from scitex.stats.posthoc import posthoc_tukey


class TestBasicComputations:
    """Test basic Tukey HSD computations."""

    def test_three_groups_basic(self):
        """Test basic three-group comparison."""
        np.random.seed(42)
        group1 = np.random.normal(10, 2, 20)
        group2 = np.random.normal(12, 2, 20)
        group3 = np.random.normal(14, 2, 20)

        results = posthoc_tukey([group1, group2, group3])

        # Should return DataFrame by default
        assert isinstance(results, pd.DataFrame)

        # Should have 3 comparisons (3 choose 2)
        assert len(results) == 3

        # Check required columns
        required_cols = ['group_i', 'group_j', 'mean_diff', 'pvalue',
                        'significant', 'q_statistic', 'ci_lower', 'ci_upper']
        for col in required_cols:
            assert col in results.columns

    def test_four_groups_all_pairs(self):
        """Test that all pairwise comparisons are performed."""
        np.random.seed(123)
        groups = [np.random.normal(i*2, 1, 15) for i in range(4)]

        results = posthoc_tukey(groups)

        # 4 groups = 6 pairwise comparisons (4 choose 2)
        assert len(results) == 6

        # Verify all pairs are present
        pairs = set()
        for _, row in results.iterrows():
            pairs.add((row['group_i'], row['group_j']))

        expected_pairs = {
            ('Group 1', 'Group 2'), ('Group 1', 'Group 3'), ('Group 1', 'Group 4'),
            ('Group 2', 'Group 3'), ('Group 2', 'Group 4'), ('Group 3', 'Group 4')
        }
        assert pairs == expected_pairs

    def test_known_values_equal_means(self):
        """Test with known values where groups have equal means."""
        # Three groups with identical means
        group1 = np.array([10, 10, 10, 10, 10])
        group2 = np.array([10, 10, 10, 10, 10])
        group3 = np.array([10, 10, 10, 10, 10])

        results = posthoc_tukey([group1, group2, group3])

        # All comparisons should be non-significant
        assert all(~results['significant'])

        # All mean differences should be 0
        assert all(abs(results['mean_diff']) < 1e-10)


class TestInputFormats:
    """Test different input formats."""

    def test_list_of_arrays(self):
        """Test with list of numpy arrays."""
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        results = posthoc_tukey(groups)
        assert len(results) == 3

    def test_pandas_series(self):
        """Test with pandas Series."""
        groups = [
            pd.Series([1, 2, 3]),
            pd.Series([4, 5, 6]),
            pd.Series([7, 8, 9])
        ]
        results = posthoc_tukey(groups)
        assert len(results) == 3

    def test_mixed_types(self):
        """Test with mixed array types."""
        groups = [
            np.array([1, 2, 3]),
            pd.Series([4, 5, 6]),
            [7, 8, 9]
        ]
        results = posthoc_tukey(groups)
        assert len(results) == 3

    def test_custom_group_names(self):
        """Test with custom group names."""
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        group_names = ['Control', 'Treatment']

        results = posthoc_tukey(groups, group_names=group_names)

        assert results.iloc[0]['group_i'] == 'Control'
        assert results.iloc[0]['group_j'] == 'Treatment'

    def test_return_dict_format(self):
        """Test return_as='dict' option."""
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        results = posthoc_tukey(groups, return_as='dict')

        assert isinstance(results, list)
        assert isinstance(results[0], dict)
        assert 'mean_diff' in results[0]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_two_groups_minimum(self):
        """Test with minimum of two groups."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])

        results = posthoc_tukey([group1, group2])

        # Should have exactly 1 comparison
        assert len(results) == 1

    def test_single_group_raises_error(self):
        """Test that single group raises ValueError."""
        group = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Need at least 2 groups"):
            posthoc_tukey([group])

    def test_identical_groups(self):
        """Test with completely identical groups."""
        identical_data = np.array([5, 5, 5, 5, 5])
        groups = [identical_data.copy() for _ in range(3)]

        results = posthoc_tukey(groups)

        # All comparisons should be non-significant
        assert all(~results['significant'])
        # All p-values should be NaN or close to 1 (zero variance case)
        # When variance is zero, p-values may be NaN
        assert all(results['pvalue'].isna() | (results['pvalue'] > 0.9))

    def test_unequal_sample_sizes(self):
        """Test Tukey-Kramer with unequal sample sizes."""
        group1 = np.random.normal(10, 2, 10)
        group2 = np.random.normal(12, 2, 20)
        group3 = np.random.normal(14, 2, 30)

        results = posthoc_tukey([group1, group2, group3])

        # Should handle unequal sizes without error
        assert len(results) == 3
        # Check sample sizes are recorded
        assert results.iloc[0]['n_i'] == 10
        assert results.iloc[0]['n_j'] == 20

    def test_very_large_differences(self):
        """Test with very large mean differences."""
        group1 = np.array([0, 0, 0, 0, 0])
        group2 = np.array([100, 100, 100, 100, 100])

        results = posthoc_tukey([group1, group2])

        # Should be highly significant
        assert results.iloc[0]['significant']
        assert results.iloc[0]['pvalue'] < 0.001

    def test_mismatched_group_names_length(self):
        """Test error when group_names length doesn't match."""
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        group_names = ['Group1']  # Only 1 name for 2 groups

        with pytest.raises(ValueError, match="Expected 2 group names"):
            posthoc_tukey(groups, group_names=group_names)


class TestStatisticalProperties:
    """Test statistical properties of results."""

    def test_alpha_level_respected(self):
        """Test that alpha level affects significance."""
        np.random.seed(42)
        group1 = np.random.normal(10, 2, 20)
        group2 = np.random.normal(11, 2, 20)

        results_005 = posthoc_tukey([group1, group2], alpha=0.05)
        results_001 = posthoc_tukey([group1, group2], alpha=0.01)

        # More conservative alpha should have higher critical value
        assert results_001.iloc[0]['q_critical'] >= results_005.iloc[0]['q_critical']

    def test_confidence_intervals(self):
        """Test confidence interval properties."""
        np.random.seed(42)
        groups = [np.random.normal(10 + i, 2, 20) for i in range(3)]

        results = posthoc_tukey(groups)

        for _, row in results.iterrows():
            # CI should contain the mean difference
            assert row['ci_lower'] <= row['mean_diff'] <= row['ci_upper']

            # CI width should be positive
            assert row['ci_upper'] > row['ci_lower']

    def test_q_statistic_calculation(self):
        """Test q-statistic calculation."""
        np.random.seed(42)
        groups = [np.random.normal(10, 2, 20), np.random.normal(15, 2, 20)]

        results = posthoc_tukey(groups)

        # q_statistic should be positive
        assert results.iloc[0]['q_statistic'] > 0

        # Significance should match q_stat vs q_critical
        row = results.iloc[0]
        assert row['significant'] == (row['q_statistic'] > row['q_critical'])

    def test_symmetric_comparisons(self):
        """Test that mean_diff has correct sign."""
        np.random.seed(42)
        group1 = np.random.normal(10, 2, 20)
        group2 = np.random.normal(15, 2, 20)

        results = posthoc_tukey([group1, group2], group_names=['A', 'B'])

        # mean_diff should be A - B (negative since B > A)
        assert results.iloc[0]['mean_diff'] < 0
        assert abs(results.iloc[0]['mean_diff']) == pytest.approx(5, abs=1)


class TestOutputStructure:
    """Test output structure and completeness."""

    def test_dataframe_output_structure(self):
        """Test DataFrame output has all required fields."""
        groups = [np.random.normal(i, 1, 10) for i in range(3)]
        results = posthoc_tukey(groups)

        required_fields = [
            'group_i', 'group_j', 'n_i', 'n_j',
            'mean_i', 'mean_j', 'mean_diff', 'std_error',
            'q_statistic', 'q_critical', 'pvalue', 'significant',
            'pstars', 'ci_lower', 'ci_upper', 'alpha'
        ]

        for field in required_fields:
            assert field in results.columns, f"Missing field: {field}"

    def test_dict_output_structure(self):
        """Test dict output has all required fields."""
        groups = [np.random.normal(i, 1, 10) for i in range(3)]
        results = posthoc_tukey(groups, return_as='dict')

        required_fields = [
            'group_i', 'group_j', 'mean_diff', 'pvalue',
            'significant', 'q_statistic', 'ci_lower', 'ci_upper'
        ]

        for result in results:
            for field in required_fields:
                assert field in result, f"Missing field: {field}"

    def test_pstars_format(self):
        """Test p-value stars format."""
        # Create groups with very different means for highly significant result
        group1 = np.array([0, 0, 0, 0, 0])
        group2 = np.array([10, 10, 10, 10, 10])

        results = posthoc_tukey([group1, group2])

        # Should have pstars field
        assert 'pstars' in results.columns
        # Should be a string
        assert isinstance(results.iloc[0]['pstars'], str)


class TestRobustness:
    """Test robustness to various data conditions."""

    def test_small_sample_sizes(self):
        """Test with very small sample sizes."""
        # Minimum viable sample size
        group1 = np.array([1, 2])
        group2 = np.array([3, 4])
        group3 = np.array([5, 6])

        results = posthoc_tukey([group1, group2, group3])

        # Should run without error
        assert len(results) == 3

    def test_large_number_of_groups(self):
        """Test with many groups."""
        np.random.seed(42)
        groups = [np.random.normal(i, 1, 10) for i in range(10)]

        results = posthoc_tukey(groups)

        # 10 groups = 45 pairwise comparisons
        assert len(results) == 45

    def test_high_variance_data(self):
        """Test with high variance data."""
        np.random.seed(42)
        group1 = np.random.normal(10, 100, 20)  # Very high variance
        group2 = np.random.normal(12, 100, 20)

        results = posthoc_tukey([group1, group2])

        # Should handle without error
        assert len(results) == 1
        # High variance should lead to non-significance
        assert results.iloc[0]['std_error'] > 10


class TestComparison:
    """Test comparisons and relationships."""

    def test_multiple_alpha_levels(self):
        """Test behavior across different alpha levels."""
        np.random.seed(42)
        groups = [np.random.normal(10 + i*0.5, 2, 20) for i in range(3)]

        alphas = [0.01, 0.05, 0.10]
        critical_values = []

        for alpha in alphas:
            results = posthoc_tukey(groups, alpha=alpha)
            critical_values.append(results.iloc[0]['q_critical'])

        # More conservative alpha should have higher critical values
        assert critical_values[0] > critical_values[1] > critical_values[2]

    def test_consistency_with_anova(self):
        """Test that Tukey identifies differences when ANOVA would."""
        # Create groups with clear differences
        np.random.seed(42)
        group1 = np.random.normal(10, 1, 30)
        group2 = np.random.normal(15, 1, 30)
        group3 = np.random.normal(20, 1, 30)

        results = posthoc_tukey([group1, group2, group3])

        # All pairwise comparisons should be significant
        assert all(results['significant'])

    def test_no_false_positives_null(self):
        """Test that identical groups don't show false positives."""
        np.random.seed(42)
        # All groups from same distribution
        groups = [np.random.normal(10, 2, 50) for _ in range(5)]

        results = posthoc_tukey(groups, alpha=0.05)

        # With alpha=0.05, we expect ~5% false positives
        # With 10 comparisons, likely 0-1 false positives
        n_significant = sum(results['significant'])
        assert n_significant <= 2  # Allow some random chance

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/posthoc/_tukey_hsd.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 19:30:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/posthoc/_tukey_hsd.py
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
#   - Perform Tukey HSD (Honestly Significant Difference) post-hoc test
#   - All pairwise comparisons after ANOVA
#   - Control family-wise error rate
#   - Assumes equal variances and balanced/unbalanced designs
# 
# Dependencies:
#   - packages: numpy, pandas, scipy
# 
# IO:
#   - input: Multiple groups data
#   - output: Pairwise comparison results (DataFrame)
# """
# 
# """Imports"""
# import numpy as np
# import pandas as pd
# from typing import Union, List, Optional
# from scipy import stats
# from scitex.stats.utils._formatters import p2stars
# from scitex.stats.utils._normalizers import convert_results
# 
# 
# def studentized_range_critical(k: int, df: int, alpha: float = 0.05) -> float:
#     """
#     Get critical value from studentized range distribution.
# 
#     Parameters
#     ----------
#     k : int
#         Number of groups
#     df : int
#         Degrees of freedom for error
#     alpha : float
#         Significance level
# 
#     Returns
#     -------
#     q_crit : float
#         Critical value
# 
#     Notes
#     -----
#     Uses approximation since scipy doesn't have exact studentized range.
#     For exact values, we approximate using the relationship between
#     studentized range and normal distribution.
#     """
#     # Approximate critical value using normal distribution
#     # This is less accurate than exact tables but serviceable
#     # For production, consider using statsmodels or R integration
# 
#     # Use Bonferroni-adjusted critical value as conservative approximation
#     # Actual studentized range is less conservative
#     alpha_adj = alpha / (k * (k - 1) / 2)  # Number of pairwise comparisons
#     t_crit = stats.t.ppf(1 - alpha_adj / 2, df)
# 
#     # Studentized range is approximately sqrt(2) * t for equal sample sizes
#     q_crit = np.sqrt(2) * t_crit
# 
#     return float(q_crit)
# 
# 
# def posthoc_tukey(
#     groups: List[Union[np.ndarray, pd.Series]],
#     group_names: Optional[List[str]] = None,
#     alpha: float = 0.05,
#     return_as: str = "dataframe",
# ) -> Union[pd.DataFrame, List[dict]]:
#     """
#     Perform Tukey HSD post-hoc test for pairwise comparisons.
# 
#     Conducts all pairwise comparisons between groups after ANOVA,
#     controlling the family-wise error rate.
# 
#     Parameters
#     ----------
#     groups : list of arrays
#         List of sample arrays for each group
#     group_names : list of str, optional
#         Names for each group. If None, uses 'Group 1', 'Group 2', etc.
#     alpha : float, default 0.05
#         Family-wise error rate
#     return_as : {'dataframe', 'dict'}, default 'dataframe'
#         Output format
# 
#     Returns
#     -------
#     results : DataFrame or list of dict
#         Pairwise comparison results including:
#         - group_i: First group name
#         - group_j: Second group name
#         - mean_i: Mean of group i
#         - mean_j: Mean of group j
#         - mean_diff: Difference in means (i - j)
#         - std_error: Standard error of difference
#         - q_statistic: Studentized range statistic
#         - q_critical: Critical value
#         - pvalue: Approximate p-value
#         - significant: Whether difference is significant
#         - ci_lower: Lower bound of 95% CI
#         - ci_upper: Upper bound of 95% CI
# 
#     Notes
#     -----
#     Tukey's Honestly Significant Difference (HSD) test is used for all
#     pairwise comparisons between group means after a significant ANOVA.
# 
#     **Test Statistic (Studentized Range)**:
# 
#     .. math::
#         q = \\frac{|\\bar{x}_i - \\bar{x}_j|}{\\sqrt{MS_{error}/n}}
# 
#     Where:
#     - MS_error: Mean square error from ANOVA
#     - n: Harmonic mean of sample sizes (for unbalanced designs)
# 
#     **Assumptions**:
#     1. Independence of observations
#     2. Normality within each group
#     3. Homogeneity of variance across groups (same as ANOVA)
# 
#     **Advantages**:
#     - Controls family-wise error rate exactly at α
#     - More powerful than Bonferroni correction
#     - Provides confidence intervals for differences
# 
#     **Disadvantages**:
#     - Assumes equal variances (use Games-Howell if violated)
#     - Less powerful than Bonferroni for small number of comparisons
#     - Requires significant ANOVA first (recommended practice)
# 
#     **When to use**:
#     - After significant one-way ANOVA
#     - When variances are approximately equal
#     - For all pairwise comparisons (not subset)
# 
#     Examples
#     --------
#     >>> import numpy as np
#     >>> from scitex.stats.posthoc import posthoc_tukey
#     >>>
#     >>> # Example: Compare 4 treatment groups
#     >>> np.random.seed(42)
#     >>> control = np.random.normal(10, 2, 20)
#     >>> treatment1 = np.random.normal(12, 2, 20)
#     >>> treatment2 = np.random.normal(11, 2, 20)
#     >>> treatment3 = np.random.normal(13, 2, 20)
#     >>>
#     >>> results = posthoc_tukey(
#     ...     [control, treatment1, treatment2, treatment3],
#     ...     group_names=['Control', 'Treat1', 'Treat2', 'Treat3']
#     ... )
#     >>>
#     >>> print(results[['group_i', 'group_j', 'mean_diff', 'pvalue', 'significant']])
# 
#     References
#     ----------
#     .. [1] Tukey, J. W. (1949). "Comparing individual means in the analysis
#            of variance". Biometrics, 5(2), 99-114.
#     .. [2] Kramer, C. Y. (1956). "Extension of multiple range tests to group
#            means with unequal numbers of replications". Biometrics, 12(3),
#            307-310.
# 
#     See Also
#     --------
#     posthoc_games_howell : For unequal variances
#     posthoc_dunnett : For comparisons vs control
#     correct_bonferroni : Simple but conservative alternative
#     """
#     # Convert to list of arrays
#     groups = [np.asarray(g) for g in groups]
# 
#     k = len(groups)
# 
#     if k < 2:
#         raise ValueError("Need at least 2 groups for pairwise comparisons")
# 
#     # Group names
#     if group_names is None:
#         group_names = [f"Group {i + 1}" for i in range(k)]
# 
#     if len(group_names) != k:
#         raise ValueError(f"Expected {k} group names, got {len(group_names)}")
# 
#     # Compute group statistics
#     n_groups = [len(g) for g in groups]
#     means = [np.mean(g) for g in groups]
# 
#     # Total sample size and degrees of freedom
#     N = sum(n_groups)
#     df_error = N - k
# 
#     # Pooled variance (MS_error from ANOVA)
#     ss_error = 0
#     for g in groups:
#         ss_error += np.sum((g - np.mean(g)) ** 2)
# 
#     ms_error = ss_error / df_error
# 
#     # Get critical value
#     q_crit = studentized_range_critical(k, df_error, alpha)
# 
#     # Perform all pairwise comparisons
#     results = []
# 
#     for i in range(k):
#         for j in range(i + 1, k):
#             n_i = n_groups[i]
#             n_j = n_groups[j]
#             mean_i = means[i]
#             mean_j = means[j]
# 
#             # Mean difference
#             mean_diff = mean_i - mean_j
# 
#             # Standard error for unequal sample sizes (Tukey-Kramer)
#             se = np.sqrt(ms_error * (1 / n_i + 1 / n_j) / 2)
# 
#             # Studentized range statistic
#             q_stat = abs(mean_diff) / se
# 
#             # Approximate p-value
#             # Using conservative approximation
#             # Exact p-value requires studentized range distribution
#             t_equiv = q_stat / np.sqrt(2)
#             pvalue = 2 * (1 - stats.t.cdf(t_equiv, df_error))
# 
#             # Determine significance
#             significant = q_stat > q_crit
# 
#             # Confidence interval
#             margin = q_crit * se
#             ci_lower = mean_diff - margin
#             ci_upper = mean_diff + margin
# 
#             results.append(
#                 {
#                     "group_i": group_names[i],
#                     "group_j": group_names[j],
#                     "n_i": n_i,
#                     "n_j": n_j,
#                     "mean_i": round(float(mean_i), 3),
#                     "mean_j": round(float(mean_j), 3),
#                     "mean_diff": round(float(mean_diff), 3),
#                     "std_error": round(float(se), 3),
#                     "q_statistic": round(float(q_stat), 3),
#                     "q_critical": round(float(q_crit), 3),
#                     "pvalue": round(float(pvalue), 4),
#                     "significant": bool(significant),
#                     "pstars": p2stars(pvalue),
#                     "ci_lower": round(float(ci_lower), 3),
#                     "ci_upper": round(float(ci_upper), 3),
#                     "alpha": alpha,
#                 }
#             )
# 
#     # Return format
#     if return_as == "dataframe":
#         return pd.DataFrame(results)
#     else:
#         return results
# 
# 
# if __name__ == "__main__":
#     import sys
#     import argparse
#     import scitex as stx
# 
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args([])
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys=sys,
#         plt=None,
#         args=args,
#         file=__FILE__,
#         verbose=True,
#         agg=True,
#     )
# 
#     logger = stx.logging.getLogger(__name__)
# 
#     logger.info("=" * 70)
#     logger.info("Tukey HSD Post-hoc Test Examples")
#     logger.info("=" * 70)
# 
#     # Example 1: Basic usage after ANOVA
#     logger.info("\n[Example 1] Basic Tukey HSD after significant ANOVA")
#     logger.info("-" * 70)
# 
#     np.random.seed(42)
#     control = np.random.normal(10, 2, 20)
#     treatment1 = np.random.normal(12, 2, 20)
#     treatment2 = np.random.normal(11, 2, 20)
#     treatment3 = np.random.normal(13, 2, 20)
# 
#     # First run ANOVA
#     from ..tests.parametric import test_anova
# 
#     anova_result = test_anova(
#         [control, treatment1, treatment2, treatment3],
#         var_names=["Control", "Treat1", "Treat2", "Treat3"],
#     )
# 
#     logger.info(
#         f"ANOVA: F = {anova_result['statistic']:.3f}, p = {anova_result['pvalue']:.4f}"
#     )
# 
#     if anova_result["significant"]:
#         logger.info("\nANOVA significant, conducting Tukey HSD...")
# 
#         results = posthoc_tukey(
#             [control, treatment1, treatment2, treatment3],
#             group_names=["Control", "Treat1", "Treat2", "Treat3"],
#         )
# 
#         logger.info(
#             f"\n{results[['group_i', 'group_j', 'mean_diff', 'pvalue', 'significant']].to_string()}"
#         )
# 
#     # Example 2: Unbalanced design (Tukey-Kramer)
#     logger.info("\n[Example 2] Unbalanced design (different sample sizes)")
#     logger.info("-" * 70)
# 
#     group_a = np.random.normal(50, 10, 15)
#     group_b = np.random.normal(60, 10, 25)
#     group_c = np.random.normal(55, 10, 20)
# 
#     results_unbalanced = posthoc_tukey(
#         [group_a, group_b, group_c], group_names=["A", "B", "C"]
#     )
# 
#     logger.info(f"Sample sizes: A={len(group_a)}, B={len(group_b)}, C={len(group_c)}")
#     logger.info(f"\n{results_unbalanced.to_string()}")
# 
#     # Example 3: With confidence intervals
#     logger.info("\n[Example 3] Confidence intervals for differences")
#     logger.info("-" * 70)
# 
#     for _, row in results.iterrows():
#         if row["significant"]:
#             logger.info(
#                 f"{row['group_i']} vs {row['group_j']}: "
#                 f"Diff = {row['mean_diff']:.2f}, "
#                 f"95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}] {row['pstars']}"
#             )
# 
#     # Example 4: Export results
#     logger.info("\n[Example 4] Export results")
#     logger.info("-" * 70)
# 
#     convert_results(results, return_as="excel", path="./tukey_hsd_results.xlsx")
#     logger.info("Saved to: ./tukey_hsd_results.xlsx")
# 
#     stx.session.close(
#         CONFIG,
#         verbose=False,
#         notify=False,
#         exit_status=0,
#     )
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/posthoc/_tukey_hsd.py
# --------------------------------------------------------------------------------
