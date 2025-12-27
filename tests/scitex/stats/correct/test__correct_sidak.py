#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Šidák correction method."""

import pytest
import numpy as np
import pandas as pd
from scitex.stats.correct import correct_sidak
from scitex.stats.correct import correct_bonferroni


class TestSidakBasic:
    """Basic functionality tests for Šidák correction."""

    def test_basic_correction_list(self):
        """Test basic Šidák correction with list of dicts."""
        results = [
            {'test_name': 'test1', 'pvalue': 0.01},
            {'test_name': 'test2', 'pvalue': 0.02},
            {'test_name': 'test3', 'pvalue': 0.03},
            {'test_name': 'test4', 'pvalue': 0.04},
            {'test_name': 'test5', 'pvalue': 0.05}
        ]
        corrected = correct_sidak(results, verbose=False)

        assert isinstance(corrected, list)
        assert len(corrected) == len(results)
        assert all('pvalue_adjusted' in r for r in corrected)

        # Šidák uses formula: p_adj = 1 - (1-p)^m
        m = len(results)
        for i, r in enumerate(corrected):
            expected = 1.0 - (1.0 - results[i]['pvalue']) ** m
            np.testing.assert_almost_equal(r['pvalue_adjusted'], expected, decimal=5)

    def test_single_pvalue(self):
        """Test with single p-value."""
        result = correct_sidak({'pvalue': 0.01}, verbose=False)

        assert isinstance(result, dict)
        np.testing.assert_almost_equal(result['pvalue_adjusted'], 0.01, decimal=5)  # Single test, no adjustment
        assert 'rejected' in result

    def test_significance_threshold(self):
        """Test significance determination with different alpha."""
        results = [
            {'pvalue': 0.001},
            {'pvalue': 0.01},
            {'pvalue': 0.02},
            {'pvalue': 0.05},
            {'pvalue': 0.1}
        ]

        result_005 = correct_sidak(results, alpha=0.05, verbose=False)
        result_001 = correct_sidak(results, alpha=0.01, verbose=False)

        # Check that significance changes with alpha
        assert all('rejected' in r for r in result_005)
        assert all('rejected' in r for r in result_001)

        n_rejected_005 = sum(r['rejected'] for r in result_005)
        n_rejected_001 = sum(r['rejected'] for r in result_001)
        assert n_rejected_005 >= n_rejected_001

    def test_alpha_adjustment_formula(self):
        """Test that Šidák alpha adjustment uses correct formula."""
        results = [{'pvalue': 0.01 * i} for i in range(1, 6)]
        m = len(results)
        alpha = 0.05

        corrected = correct_sidak(results, alpha=alpha, verbose=False)

        # Šidák adjusted alpha: α_adj = 1 - (1 - α)^(1/m)
        expected_alpha_adj = 1.0 - (1.0 - alpha) ** (1.0 / m)

        for r in corrected:
            np.testing.assert_almost_equal(r['alpha_adjusted'], expected_alpha_adj, decimal=6)


class TestSidakInputFormats:
    """Test different input formats."""

    def test_single_dict_input(self):
        """Test with single dict input."""
        result = correct_sidak({'pvalue': 0.01}, verbose=False)
        assert isinstance(result, dict)
        assert 'pvalue_adjusted' in result

    def test_list_of_dicts_input(self):
        """Test with list of dicts input containing p-values."""
        test_results = [
            {'test_name': 'test1', 'pvalue': 0.01},
            {'test_name': 'test2', 'pvalue': 0.02},
            {'test_name': 'test3', 'pvalue': 0.03}
        ]
        result = correct_sidak(test_results, verbose=False)
        assert isinstance(result, list)
        assert all('pvalue_adjusted' in r for r in result)
        assert all('test_name' in r for r in result)  # Original fields preserved

    def test_dataframe_input(self):
        """Test with DataFrame input."""
        df = pd.DataFrame({
            'test': ['t1', 't2', 't3'],
            'pvalue': [0.01, 0.02, 0.03]
        })
        result = correct_sidak(df, verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert 'pvalue_adjusted' in result.columns
        assert 'alpha_adjusted' in result.columns
        assert 'rejected' in result.columns
        assert 'pstars' in result.columns


class TestSidakEdgeCases:
    """Test edge cases and error handling."""

    def test_pvalue_clipping(self):
        """Test that corrected p-values are clipped at 1.0."""
        results = [
            {'pvalue': 0.5},
            {'pvalue': 0.6},
            {'pvalue': 0.7}
        ]
        corrected = correct_sidak(results, verbose=False)

        # All corrected values should be <= 1.0
        assert all(r['pvalue_adjusted'] <= 1.0 for r in corrected)

    def test_zero_pvalues(self):
        """Test handling of zero p-values."""
        results = [
            {'pvalue': 0.0},
            {'pvalue': 0.01},
            {'pvalue': 0.02}
        ]
        corrected = correct_sidak(results, verbose=False)

        # 1 - (1-0)^m = 1 - 1^m = 0
        assert corrected[0]['pvalue_adjusted'] == 0.0

    def test_one_pvalue(self):
        """Test handling of p-value = 1.0."""
        results = [
            {'pvalue': 0.01},
            {'pvalue': 0.5},
            {'pvalue': 1.0}
        ]
        corrected = correct_sidak(results, verbose=False)

        # 1 - (1-1)^m = 1 - 0^m = 1
        assert corrected[2]['pvalue_adjusted'] == 1.0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            'pvalue': [0.01, np.nan, 0.03]
        })

        # NaN handling may raise error or propagate NaN
        # This test checks the behavior doesn't crash
        try:
            result = correct_sidak(df, verbose=False)
            # If it succeeds, check NaN is handled
            if len(result) == 3:
                # NaN may remain NaN or be handled
                pass
        except (ValueError, TypeError, KeyError):
            # If it raises an error on NaN, that's also acceptable
            pass


class TestSidakComparison:
    """Test Šidák correction compared to Bonferroni."""

    def test_sidak_more_powerful_than_bonferroni(self):
        """Test that Šidák is more powerful than Bonferroni under independence."""
        p_values = [0.001, 0.01, 0.02, 0.03, 0.04]
        results = [{'pvalue': p} for p in p_values]

        sidak_corrected = correct_sidak(results, alpha=0.05, verbose=False)
        bonf_corrected = correct_bonferroni(results, alpha=0.05, verbose=False)

        # Šidák alpha should be >= Bonferroni alpha (less conservative)
        assert sidak_corrected[0]['alpha_adjusted'] >= bonf_corrected[0]['alpha_adjusted']

        # Number of rejections: Šidák should reject >= Bonferroni
        n_rejected_sidak = sum(r['rejected'] for r in sidak_corrected)
        n_rejected_bonf = sum(r['rejected'] for r in bonf_corrected)
        assert n_rejected_sidak >= n_rejected_bonf

    def test_known_values(self):
        """Test against manually calculated Šidák values."""
        # With m=3, alpha=0.05:
        # - p-values: 0.01, 0.02, 0.03
        # - Šidák p_adj: 1-(1-p)^3
        results = [
            {'pvalue': 0.01},
            {'pvalue': 0.02},
            {'pvalue': 0.03}
        ]

        corrected = correct_sidak(results, verbose=False)
        m = 3

        # First: 1 - (1-0.01)^3 = 1 - 0.99^3 = 1 - 0.970299 = 0.029701
        expected_1 = 1.0 - (1.0 - 0.01) ** m
        np.testing.assert_almost_equal(corrected[0]['pvalue_adjusted'], expected_1, decimal=5)

        # Second: 1 - (1-0.02)^3 = 1 - 0.98^3 = 1 - 0.941192 = 0.058808
        expected_2 = 1.0 - (1.0 - 0.02) ** m
        np.testing.assert_almost_equal(corrected[1]['pvalue_adjusted'], expected_2, decimal=5)

        # Third: 1 - (1-0.03)^3 = 1 - 0.97^3 = 1 - 0.912673 = 0.087327
        expected_3 = 1.0 - (1.0 - 0.03) ** m
        np.testing.assert_almost_equal(corrected[2]['pvalue_adjusted'], expected_3, decimal=5)

    def test_alpha_comparison_with_bonferroni(self):
        """Test that Šidák adjusted alpha is always >= Bonferroni adjusted alpha."""
        for m in [2, 5, 10, 20, 50]:
            results = [{'pvalue': 0.01 * i} for i in range(1, m + 1)]
            alpha = 0.05

            sidak = correct_sidak(results, alpha=alpha, verbose=False)
            bonf = correct_bonferroni(results, alpha=alpha, verbose=False)

            # Šidák: α_adj = 1 - (1 - α)^(1/m)
            # Bonferroni: α_adj = α/m
            # Šidák should always be >= Bonferroni (more powerful)
            assert sidak[0]['alpha_adjusted'] >= bonf[0]['alpha_adjusted']

            # For large m, the difference becomes more significant
            ratio = sidak[0]['alpha_adjusted'] / bonf[0]['alpha_adjusted']
            assert ratio >= 1.0


class TestSidakOutput:
    """Test output structure and format."""

    def test_dict_output_keys(self):
        """Test that single dict input returns expected keys."""
        result = correct_sidak({'pvalue': 0.01}, verbose=False)

        assert 'pvalue_adjusted' in result
        assert 'alpha_adjusted' in result
        assert 'rejected' in result
        assert 'pstars' in result

    def test_list_output_keys(self):
        """Test that list input returns list with expected keys."""
        results = [
            {'pvalue': 0.01},
            {'pvalue': 0.02},
            {'pvalue': 0.03}
        ]
        corrected = correct_sidak(results, verbose=False)

        assert isinstance(corrected, list)
        assert all('pvalue_adjusted' in r for r in corrected)
        assert all('alpha_adjusted' in r for r in corrected)
        assert all('rejected' in r for r in corrected)
        assert all('pstars' in r for r in corrected)

    def test_stars_annotation(self):
        """Test significance stars are added."""
        test_results = [
            {'test_name': 'test1', 'pvalue': 0.001},
            {'test_name': 'test2', 'pvalue': 0.01},
            {'test_name': 'test3', 'pvalue': 0.05}
        ]
        result = correct_sidak(test_results, verbose=False)

        # Check stars are added (field is 'pstars')
        assert all('pstars' in r for r in result)
        # Very small adjusted p-values should have stars
        # Note: stars are based on adjusted p-values
        for r in result:
            assert isinstance(r['pstars'], str)


class TestSidakMathematicalProperties:
    """Test mathematical properties of Šidák correction."""

    def test_exponential_formula(self):
        """Test that Šidák uses exponential formula correctly."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        m = len(p_values)
        results = [{'pvalue': p} for p in p_values]

        corrected = correct_sidak(results, verbose=False)

        for i, r in enumerate(corrected):
            # p_adj = 1 - (1 - p)^m
            expected = 1.0 - (1.0 - p_values[i]) ** m
            np.testing.assert_almost_equal(r['pvalue_adjusted'], expected, decimal=6)

    def test_independence_assumption(self):
        """Test properties that rely on independence assumption."""
        # Under independence, Šidák controls FWER exactly at alpha
        # With m tests and Šidák adjustment:
        # P(at least one false positive) = 1 - P(no false positives)
        # = 1 - (1 - α_adj)^m = 1 - (1 - (1 - (1-α)^(1/m)))^m
        # which simplifies to approximately α

        results = [{'pvalue': 0.01 * i} for i in range(1, 11)]
        m = len(results)
        alpha = 0.05

        corrected = correct_sidak(results, alpha=alpha, verbose=False)

        alpha_adj = corrected[0]['alpha_adjusted']

        # Verify: (1 - alpha_adj)^m ≈ (1 - alpha)
        left_side = (1.0 - alpha_adj) ** m
        right_side = 1.0 - alpha

        np.testing.assert_almost_equal(left_side, right_side, decimal=6)

    def test_monotonic_increase_with_m(self):
        """Test that adjusted p-values increase with more tests."""
        p = 0.01

        for m in [1, 2, 5, 10, 20]:
            results = [{'pvalue': p}] * m
            corrected = correct_sidak(results, verbose=False)

            # p_adj = 1 - (1-p)^m increases with m
            p_adj = corrected[0]['pvalue_adjusted']
            expected = 1.0 - (1.0 - p) ** m
            np.testing.assert_almost_equal(p_adj, expected, decimal=6)

        # Check monotonicity
        p_adj_values = []
        for m in [1, 2, 5, 10, 20]:
            results = [{'pvalue': p}] * m
            corrected = correct_sidak(results, verbose=False)
            p_adj_values.append(corrected[0]['pvalue_adjusted'])

        # Should be increasing
        for i in range(len(p_adj_values) - 1):
            assert p_adj_values[i] <= p_adj_values[i + 1]

    def test_limiting_behavior(self):
        """Test limiting behavior as m approaches infinity."""
        # For small p and large m: 1 - (1-p)^m ≈ 1 - e^(-mp) ≈ mp (for small mp)
        # This is close to Bonferroni for large m
        p = 0.001
        m = 100

        results = [{'pvalue': p}] * m
        corrected = correct_sidak(results, verbose=False)

        sidak_adj = corrected[0]['pvalue_adjusted']
        bonferroni_adj = min(p * m, 1.0)

        # For large m and small p, Šidák should approach Bonferroni
        # The difference should be relatively small
        difference = abs(sidak_adj - bonferroni_adj)
        assert difference < 0.01  # Allow small difference


class TestSidakSpecialCases:
    """Test special cases specific to Šidák."""

    def test_two_tests(self):
        """Test the simple case of two tests."""
        results = [
            {'pvalue': 0.01},
            {'pvalue': 0.02}
        ]
        alpha = 0.05
        corrected = correct_sidak(results, alpha=alpha, verbose=False)

        # For m=2:
        # α_adj = 1 - (1-0.05)^(1/2) = 1 - 0.95^0.5 ≈ 0.0253
        expected_alpha = 1.0 - (1.0 - alpha) ** 0.5
        np.testing.assert_almost_equal(corrected[0]['alpha_adjusted'], expected_alpha, decimal=4)

        # p_adj = 1 - (1-p)^2
        expected_p1 = 1.0 - (1.0 - 0.01) ** 2
        expected_p2 = 1.0 - (1.0 - 0.02) ** 2
        np.testing.assert_almost_equal(corrected[0]['pvalue_adjusted'], expected_p1, decimal=5)
        np.testing.assert_almost_equal(corrected[1]['pvalue_adjusted'], expected_p2, decimal=5)

    def test_very_small_pvalues(self):
        """Test behavior with very small p-values."""
        results = [
            {'pvalue': 1e-10},
            {'pvalue': 1e-9},
            {'pvalue': 1e-8}
        ]
        corrected = correct_sidak(results, verbose=False)

        # Even with small p-values, adjustment should work correctly
        m = 3
        for i, r in enumerate(corrected):
            expected = 1.0 - (1.0 - results[i]['pvalue']) ** m
            np.testing.assert_almost_equal(r['pvalue_adjusted'], expected, decimal=10)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/correct/_correct_sidak.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 20:29:45 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/correct/_correct_sidak.py
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
#   - Apply Šidák correction for multiple comparisons
#   - Adjust p-values and significance thresholds assuming independence
#   - Support both dict and DataFrame inputs
#   - More powerful than Bonferroni under independence assumption
# 
# Dependencies:
#   - packages: numpy, pandas
# 
# IO:
#   - input: Test results with p-values (dict, list of dicts, or DataFrame)
#   - output: Results with adjusted p-values and significance (same format as input)
# """
# 
# """Imports"""
# from typing import Any, Dict, List, Optional, Union
# 
# import matplotlib
# import matplotlib.axes
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
# def correct_sidak(
#     results: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
#     alpha: float = 0.05,
#     return_as: str = None,
#     verbose: bool = True,
#     plot: bool = False,
#     ax: Optional[matplotlib.axes.Axes] = None,
# ) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
#     """
#     Apply Šidák correction for multiple comparisons.
# 
#     Parameters
#     ----------
#     results : dict, list of dict, or DataFrame
#         Test results containing 'pvalue' field(s)
#         - Single dict: one test result
#         - List of dicts: multiple test results
#         - DataFrame: multiple test results (one per row)
#     alpha : float, default 0.05
#         Family-wise error rate (FWER) to control
#     return_as : {'dict', 'dataframe', None}, optional
#         Force specific return format. If None, matches input format.
#     verbose : bool, default True
#         Whether to log progress information
#     plot : bool, default False
#         Whether to generate visualization
#     ax : matplotlib.axes.Axes, optional
#         Axes object to plot on. If None and plot=True, creates new figure.
#         If provided, automatically enables plotting.
# 
#     Returns
#     -------
#     results : dict, list of dict, or DataFrame
#         Results with added fields:
#         - pvalue_adjusted: Šidák-adjusted p-value
#         - alpha_adjusted: Šidák-adjusted alpha threshold
#         - rejected: Whether null hypothesis is rejected (using adjusted values)
#         - pstars: Significance stars (using adjusted p-value)
# 
#     Notes
#     -----
#     The Šidák correction is less conservative than Bonferroni and assumes
#     independence between tests. It controls the family-wise error rate (FWER).
# 
#     For m tests with family-wise error rate α:
# 
#     .. math::
#         \\alpha_{adj} = 1 - (1 - \\alpha)^{1/m}
# 
#     .. math::
#         p_{adj,i} = 1 - (1 - p_i)^m
# 
#     The method guarantees (under independence):
# 
#     .. math::
#         P(\\text{at least one false positive}) \\leq \\alpha
# 
#     **Advantages:**
#     - More powerful than Bonferroni under independence
#     - Still controls FWER exactly
#     - Simple interpretation
# 
#     **Disadvantages:**
#     - Assumes independence between tests
#     - Less conservative than Bonferroni if dependence exists
#     - Can be overly liberal if tests are positively correlated
# 
#     **When to use:**
#     - When tests are truly independent
#     - When Bonferroni is too conservative
#     - When FWER control is required but more power is needed
# 
#     Examples
#     --------
#     >>> from scitex.stats.tests import test_ttest_ind
#     >>> from scitex.stats.correct import correct_sidak
#     >>> import numpy as np
#     >>>
#     >>> # Multiple independent t-tests
#     >>> results = []
#     >>> for i in range(5):
#     ...     x = np.random.normal(0, 1, 30)
#     ...     y = np.random.normal(0.5, 1, 30)
#     ...     r = test_ttest_ind(x, y)
#     ...     results.append(r)
#     >>>
#     >>> # Apply Šidák correction
#     >>> corrected = correct_sidak(results, alpha=0.05)
#     >>>
#     >>> # Compare with Bonferroni
#     >>> from scitex.stats.correct import correct_bonferroni
#     >>> bonf = correct_bonferroni(results, alpha=0.05)
#     >>>
#     >>> print(f"Šidák alpha: {corrected[0]['alpha_adjusted']:.4f}")
#     >>> print(f"Bonferroni alpha: {bonf[0]['alpha_adjusted']:.4f}")
# 
#     References
#     ----------
#     .. [1] Šidák, Z. (1967). "Rectangular Confidence Regions for the Means of
#            Multivariate Normal Distributions". Journal of the American Statistical
#            Association, 62(318), 626-633.
#     .. [2] Abdi, H. (2007). "Bonferroni and Šidák corrections for multiple
#            comparisons". Encyclopedia of Measurement and Statistics, 3, 103-107.
# 
#     See Also
#     --------
#     correct_bonferroni : More conservative alternative
#     correct_holm : Sequential Bonferroni method
#     correct_fdr : FDR control (less conservative)
#     """
#     from scitex.stats.utils._formatters import p2stars
# 
#     if verbose:
#         logger.info("Applying Šidák correction")
# 
#     # Determine input format
#     single_result = False
#     if isinstance(results, dict):
#         results = [results]
#         single_result = True
#     elif isinstance(results, pd.DataFrame):
#         results_list = results.to_dict("records")
#         input_was_df = True
#     else:
#         results_list = results
#         input_was_df = False
# 
#     if isinstance(results, list):
#         results_list = results
#         input_was_df = False
# 
#     # Get number of tests
#     m = len(results_list)
# 
#     if verbose:
#         logger.info(f"Number of tests: {m}, alpha: {alpha}")
# 
#     # Calculate Šidák-adjusted alpha
#     # α_adj = 1 - (1 - α)^(1/m)
#     alpha_adj = 1.0 - (1.0 - alpha) ** (1.0 / m)
# 
#     # Apply correction to each result
#     corrected_results = []
#     for r in results_list:
#         r_copy = r.copy()
#         pval = r["pvalue"]
# 
#         # Calculate adjusted p-value
#         # p_adj = 1 - (1 - p)^m
#         # Handle edge cases
#         if pval >= 1.0:
#             pval_adj = 1.0
#         else:
#             pval_adj = 1.0 - (1.0 - pval) ** m
#             pval_adj = min(pval_adj, 1.0)  # Cap at 1.0
# 
#         # Update result
#         r_copy["pvalue_adjusted"] = pval_adj
#         r_copy["alpha_adjusted"] = alpha_adj
#         r_copy["rejected"] = pval_adj < alpha
#         r_copy["significant"] = r_copy["rejected"]
#         r_copy["pstars"] = p2stars(pval_adj)
# 
#         corrected_results.append(r_copy)
# 
#     # Log results summary
#     if verbose:
#         rejections = sum(r["rejected"] for r in corrected_results)
#         logger.info(f"Šidák correction complete: {rejections}/{m} hypotheses rejected")
#         logger.info(f"Adjusted alpha threshold: {alpha_adj:.6f}")
# 
#         # Log detailed results if not too many tests
#         if m <= 10:
#             logger.info("\nDetailed results:")
#             for r in corrected_results:
#                 comparison = ""
#                 if "var_x" in r and "var_y" in r:
#                     comparison = f"{r['var_x']} vs {r['var_y']}: "
#                 elif "test_method" in r:
#                     comparison = f"{r['test_method']}: "
#                 elif "comparison" in r:
#                     comparison = f"{r['comparison']}: "
# 
#                 logger.info(
#                     f"  {comparison}"
#                     f"p = {r['pvalue']:.4f} → p_adj = {r['pvalue_adjusted']:.4f} "
#                     f"{r['pstars']}, rejected = {r['rejected']}"
#                 )
# 
#     # Auto-enable plotting if ax is provided
#     if ax is not None:
#         plot = True
# 
#     # Generate plot if requested
#     if plot:
#         if ax is None:
#             import matplotlib.pyplot as plt
# 
#             fig, ax = plt.subplots(figsize=(10, 6))
#         _plot_sidak(corrected_results, alpha, alpha_adj, m, ax)
# 
#     # Format output
#     if single_result:
#         return corrected_results[0]
# 
#     if return_as == "dataframe" or (return_as is None and input_was_df):
#         return pd.DataFrame(corrected_results)
# 
#     return corrected_results
# 
# 
# def _plot_sidak(corrected_results, alpha, alpha_adj, m, ax):
#     """Create visualization for Šidák correction on given axes."""
#     x = np.arange(m)
#     pvalues = [r["pvalue"] for r in corrected_results]
#     pvalues_adj = [r["pvalue_adjusted"] for r in corrected_results]
# 
#     # Plot original and adjusted p-values
#     ax.scatter(x, pvalues, label="Original p-values", alpha=0.7, s=100, color="C0")
#     ax.scatter(
#         x,
#         pvalues_adj,
#         label="Adjusted p-values",
#         alpha=0.7,
#         s=100,
#         color="C1",
#         marker="s",
#     )
# 
#     # Connect original to adjusted with lines
#     for i in range(m):
#         ax.plot([i, i], [pvalues[i], pvalues_adj[i]], "k-", alpha=0.3, linewidth=0.5)
# 
#     # Add significance thresholds
#     ax.axhline(
#         alpha, color="red", linestyle="--", linewidth=2, alpha=0.5, label=f"α = {alpha}"
#     )
#     ax.axhline(
#         alpha_adj,
#         color="orange",
#         linestyle="--",
#         linewidth=2,
#         alpha=0.5,
#         label=f"α_adj = {alpha_adj:.4f}",
#     )
# 
#     # Formatting
#     ax.set_xlabel("Test Index")
#     ax.set_ylabel("P-value")
#     rejections = sum(r["rejected"] for r in corrected_results)
#     ax.set_title(
#         f"Šidák Correction (m={m} tests)\n{rejections}/{m} hypotheses rejected"
#     )
#     ax.set_yscale("log")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
# 
#     # Set x-axis labels if there are comparison names
#     if m <= 20:
#         labels = []
#         for r in corrected_results:
#             if "var_x" in r and "var_y" in r:
#                 labels.append(f"{r['var_x']}\nvs\n{r['var_y']}")
#             elif "test_method" in r:
#                 labels.append(r["test_method"])
#             elif "comparison" in r:
#                 labels.append(r["comparison"])
#             else:
#                 labels.append(f"Test {len(labels) + 1}")
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
#     else:
#         ax.set_xlabel(f"Test Index (1-{m})")
# 
# 
# def main():
#     """Comprehensive examples of Šidák correction."""
#     import argparse
#     import sys
# 
#     import matplotlib.pyplot as plt
#     import numpy as np
# 
#     from ..tests.parametric import test_ttest_ind
#     from . import correct_bonferroni, correct_holm
# 
#     # Parse empty args for session
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--verbose", action="store_true", default=True, help="Enable verbose output"
#     )
#     args = parser.parse_args([])
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(
#         sys=sys,
#         plt=plt,
#         args=args,
#         file=__FILE__,
#         verbose=True,
#         agg=True,
#     )
# 
#     logger.info("=" * 70)
#     logger.info("Šidák Correction Examples")
#     logger.info("=" * 70)
# 
#     # Example 1: Basic usage with 5 independent tests
#     logger.info("\n[Example 1] Basic Šidák correction with 5 independent t-tests")
#     logger.info("-" * 70)
# 
#     np.random.seed(42)
#     results = []
#     for i in range(5):
#         x = np.random.normal(0, 1, 30)
#         y = np.random.normal(0.3, 1, 30)
#         r = test_ttest_ind(x, y, var_x=f"Group_{i}_A", var_y=f"Group_{i}_B")
#         results.append(r)
# 
#     corrected = correct_sidak(results, alpha=0.05, verbose=args.verbose)
# 
#     # Example 2: Comparison with Bonferroni
#     logger.info("\n[Example 2] Šidák vs Bonferroni comparison")
#     logger.info("-" * 70)
# 
#     bonf = correct_bonferroni(results, alpha=0.05, verbose=False)
# 
#     logger.info(f"Number of tests: {len(results)}")
#     logger.info(f"Šidák alpha: {corrected[0]['alpha_adjusted']:.6f}")
#     logger.info(f"Bonferroni alpha: {bonf[0]['alpha_adjusted']:.6f}")
#     logger.info(
#         f"Difference: {corrected[0]['alpha_adjusted'] - bonf[0]['alpha_adjusted']:.6f}"
#     )
# 
#     n_rejected_sidak = sum(r["rejected"] for r in corrected)
#     n_rejected_bonf = sum(r["rejected"] for r in bonf)
# 
#     logger.info(f"\nŠidák rejections: {n_rejected_sidak}/{len(results)}")
#     logger.info(f"Bonferroni rejections: {n_rejected_bonf}/{len(results)}")
# 
#     # Example 3: Large number of tests
#     logger.info("\n[Example 3] Large number of tests (m=20)")
#     logger.info("-" * 70)
# 
#     np.random.seed(123)
#     results_20 = []
#     for i in range(20):
#         x = np.random.normal(0, 1, 50)
#         y = np.random.normal(0.2, 1, 50)
#         r = test_ttest_ind(x, y)
#         results_20.append(r)
# 
#     corrected_20 = correct_sidak(results_20, alpha=0.05, verbose=False)
#     bonf_20 = correct_bonferroni(results_20, alpha=0.05, verbose=False)
# 
#     logger.info(f"With 20 tests:")
#     logger.info(f"  Šidák alpha: {corrected_20[0]['alpha_adjusted']:.6f}")
#     logger.info(f"  Bonferroni alpha: {bonf_20[0]['alpha_adjusted']:.6f}")
#     logger.info(
#         f"  Power gain: {(corrected_20[0]['alpha_adjusted'] / bonf_20[0]['alpha_adjusted'] - 1) * 100:.2f}%"
#     )
# 
#     # Example 4: Comparison with Holm
#     logger.info("\n[Example 4] Šidák vs Holm (sequential Bonferroni)")
#     logger.info("-" * 70)
# 
#     holm = correct_holm(results, alpha=0.05, verbose=False)
# 
#     n_rejected_holm = sum(r["rejected"] for r in holm)
# 
#     logger.info(f"Number of tests: {len(results)}")
#     logger.info(f"Šidák rejections: {n_rejected_sidak}")
#     logger.info(f"Holm rejections: {n_rejected_holm}")
#     logger.info(f"Bonferroni rejections: {n_rejected_bonf}")
#     logger.info(
#         "\nNote: Holm is typically more powerful than both Šidák and Bonferroni"
#     )
# 
#     # Example 5: DataFrame input/output
#     logger.info("\n[Example 5] DataFrame input and output")
#     logger.info("-" * 70)
# 
#     df_input = pd.DataFrame(results)
#     df_corrected = correct_sidak(df_input, alpha=0.05, verbose=args.verbose)
# 
#     if args.verbose:
#         logger.info(f"Input type: {type(df_input)}")
#         logger.info(f"Output type: {type(df_corrected)}")
#         logger.info(f"\nCorrected DataFrame (first 3 rows):")
#         logger.info(
#             df_corrected[["var_x", "var_y", "pvalue", "pvalue_adjusted", "rejected"]]
#             .head(3)
#             .to_string()
#         )
# 
#     # Example 6: Single test
#     logger.info("\n[Example 6] Single test (returns dict)")
#     logger.info("-" * 70)
# 
#     single = correct_sidak(results[0], alpha=0.05, verbose=False)
#     logger.info(f"Input: single dict")
#     logger.info(f"Output: {type(single)}")
#     logger.info(f"Original p-value: {results[0]['pvalue']:.4f}")
#     logger.info(f"Adjusted p-value: {single['pvalue_adjusted']:.4f}")
# 
#     # Example 7: Edge cases
#     logger.info("\n[Example 7] Edge cases")
#     logger.info("-" * 70)
# 
#     # Very small p-value
#     edge_results = [{"pvalue": 0.001}, {"pvalue": 0.05}, {"pvalue": 0.99}]
#     edge_corrected = correct_sidak(edge_results, alpha=0.05, verbose=False)
# 
#     for i, (orig, corr) in enumerate(zip(edge_results, edge_corrected)):
#         logger.info(
#             f"Test {i + 1}: p = {orig['pvalue']:.4f} -> "
#             f"p_adj = {corr['pvalue_adjusted']:.4f}"
#         )
# 
#     # Example 8: Different alpha levels
#     logger.info("\n[Example 8] Different alpha levels")
#     logger.info("-" * 70)
# 
#     for alpha_val in [0.01, 0.05, 0.10]:
#         corr = correct_sidak(results, alpha=alpha_val, verbose=False)
#         n_rej = sum(r["rejected"] for r in corr)
#         logger.info(
#             f"Alpha = {alpha_val:.2f}: alpha_adj = {corr[0]['alpha_adjusted']:.4f}, "
#             f"rejections = {n_rej}/{len(results)}"
#         )
# 
#     # Example 9: Export to Excel
#     logger.info("\n[Example 9] Export corrected results")
#     logger.info("-" * 70)
# 
#     # Save as Excel
#     stx.io.save(df_corrected, "./sidak_corrected.xlsx")
# 
#     # Save as CSV
#     stx.io.save(df_corrected, "./sidak_corrected.csv")
# 
#     # Example 10: Mathematical properties
#     logger.info("\n[Example 10] Mathematical properties demonstration")
#     logger.info("-" * 70)
# 
#     m_values = [2, 5, 10, 20, 50, 100]
#     alpha = 0.05
# 
#     logger.info(f"For alpha = {alpha}:")
#     logger.info(f"{'m':<5} {'Bonferroni':<12} {'Šidák':<12} {'Ratio':<8}")
#     logger.info("-" * 40)
# 
#     for m in m_values:
#         bonf_alpha = alpha / m
#         sidak_alpha = 1.0 - (1.0 - alpha) ** (1.0 / m)
#         ratio = sidak_alpha / bonf_alpha
#         logger.info(f"{m:<5} {bonf_alpha:.6f}     {sidak_alpha:.6f}     {ratio:.4f}")
# 
#     logger.info("\nNote: Šidák is always ≥ Bonferroni (more powerful)")
#     logger.info("Difference increases with larger m")
# 
#     stx.session.close(
#         CONFIG,
#         verbose=False,
#         notify=False,
#         exit_status=0,
#     )
# 
# 
# if __name__ == "__main__":
#     main()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/correct/_correct_sidak.py
# --------------------------------------------------------------------------------
