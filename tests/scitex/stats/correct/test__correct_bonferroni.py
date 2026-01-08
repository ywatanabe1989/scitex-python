#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Bonferroni correction method."""

import pytest
import numpy as np
import pandas as pd
from scitex.stats.correct import correct_bonferroni


class TestBonferroniBasic:
    """Basic functionality tests for Bonferroni correction."""

    def test_basic_correction_list(self):
        """Test basic Bonferroni correction with list of dicts."""
        results = [
            {'test_name': 'test1', 'pvalue': 0.01},
            {'test_name': 'test2', 'pvalue': 0.02},
            {'test_name': 'test3', 'pvalue': 0.03},
            {'test_name': 'test4', 'pvalue': 0.04},
            {'test_name': 'test5', 'pvalue': 0.05}
        ]
        corrected = correct_bonferroni(results, verbose=False)

        assert isinstance(corrected, list)
        assert len(corrected) == len(results)
        assert all('pvalue_adjusted' in r for r in corrected)

        # Bonferroni multiplies by number of tests
        for i, r in enumerate(corrected):
            expected = min(results[i]['pvalue'] * len(results), 1.0)
            np.testing.assert_almost_equal(r['pvalue_adjusted'], expected)

    def test_single_pvalue(self):
        """Test with single p-value."""
        result = correct_bonferroni({'pvalue': 0.01}, verbose=False)

        assert isinstance(result, dict)
        assert result['pvalue_adjusted'] == 0.01  # Single test, no adjustment
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

        result_005 = correct_bonferroni(results, alpha=0.05, verbose=False)
        result_001 = correct_bonferroni(results, alpha=0.01, verbose=False)

        # Check that significance changes with alpha
        assert all('rejected' in r for r in result_005)
        assert all('rejected' in r for r in result_001)

        n_rejected_005 = sum(r['rejected'] for r in result_005)
        n_rejected_001 = sum(r['rejected'] for r in result_001)
        assert n_rejected_005 >= n_rejected_001


class TestBonferroniInputFormats:
    """Test different input formats."""

    def test_single_dict_input(self):
        """Test with single dict input."""
        result = correct_bonferroni({'pvalue': 0.01}, verbose=False)
        assert isinstance(result, dict)
        assert 'pvalue_adjusted' in result

    def test_list_of_dicts_input(self):
        """Test with list of dicts input containing p-values."""
        test_results = [
            {'test_name': 'test1', 'pvalue': 0.01},
            {'test_name': 'test2', 'pvalue': 0.02},
            {'test_name': 'test3', 'pvalue': 0.03}
        ]
        result = correct_bonferroni(test_results, verbose=False)
        assert isinstance(result, list)
        assert all('pvalue_adjusted' in r for r in result)
        assert all('test_name' in r for r in result)  # Original fields preserved

    def test_dataframe_input(self):
        """Test with DataFrame input."""
        df = pd.DataFrame({
            'test': ['t1', 't2', 't3'],
            'pvalue': [0.01, 0.02, 0.03]
        })
        result = correct_bonferroni(df, verbose=False)
        assert isinstance(result, pd.DataFrame)
        assert 'pvalue_adjusted' in result.columns
        assert 'alpha_adjusted' in result.columns
        assert 'rejected' in result.columns
        assert 'pstars' in result.columns


class TestBonferroniEdgeCases:
    """Test edge cases and error handling."""

    def test_pvalue_clipping(self):
        """Test that corrected p-values are clipped at 1.0."""
        results = [
            {'pvalue': 0.5},
            {'pvalue': 0.6},
            {'pvalue': 0.7}
        ]
        corrected = correct_bonferroni(results, verbose=False)

        # All corrected values should be <= 1.0
        assert all(r['pvalue_adjusted'] <= 1.0 for r in corrected)
        # With 3 tests, 0.5*3=1.5 should be clipped to 1.0
        assert corrected[0]['pvalue_adjusted'] == 1.0
        assert corrected[1]['pvalue_adjusted'] == 1.0
        assert corrected[2]['pvalue_adjusted'] == 1.0

    def test_zero_pvalues(self):
        """Test handling of zero p-values."""
        results = [
            {'pvalue': 0.0},
            {'pvalue': 0.01},
            {'pvalue': 0.02}
        ]
        corrected = correct_bonferroni(results, verbose=False)

        assert corrected[0]['pvalue_adjusted'] == 0.0

    def test_one_pvalue(self):
        """Test handling of p-value = 1.0."""
        results = [
            {'pvalue': 0.01},
            {'pvalue': 0.5},
            {'pvalue': 1.0}
        ]
        corrected = correct_bonferroni(results, verbose=False)

        assert corrected[2]['pvalue_adjusted'] == 1.0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            'pvalue': [0.01, np.nan, 0.03]
        })
        result = correct_bonferroni(df, verbose=False)

        # NaN should remain NaN
        assert pd.isna(result['pvalue_adjusted'].iloc[1])


class TestBonferroniComparison:
    """Test Bonferroni correction against known values."""

    def test_known_values(self):
        """Test against manually calculated values."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        results = [{'pvalue': p} for p in p_values]

        expected = [min(p * 5, 1.0) for p in p_values]  # Bonferroni multiplies by n

        corrected = correct_bonferroni(results, verbose=False)

        for i, r in enumerate(corrected):
            np.testing.assert_almost_equal(r['pvalue_adjusted'], expected[i])

    def test_rejection_count(self):
        """Test number of rejections at different alpha levels."""
        results = [
            {'pvalue': 0.001},
            {'pvalue': 0.005},
            {'pvalue': 0.01},
            {'pvalue': 0.02},
            {'pvalue': 0.05}
        ]

        corrected = correct_bonferroni(results, alpha=0.05, verbose=False)
        n_rejected = sum(r['rejected'] for r in corrected)

        # At alpha=0.05, adjusted alpha = 0.05/5 = 0.01
        # Adjusted p-values: 0.005, 0.025, 0.05, 0.1, 0.25
        # Should reject p_adjusted < alpha_adjusted = 0.01
        # So only 0.005 should be rejected
        assert n_rejected == 1


class TestBonferroniOutput:
    """Test output structure and format."""

    def test_dict_output_keys(self):
        """Test that single dict input returns expected keys."""
        result = correct_bonferroni({'pvalue': 0.01}, verbose=False)

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
        corrected = correct_bonferroni(results, verbose=False)

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
        result = correct_bonferroni(test_results, verbose=False)

        # Check stars are added (field is 'pstars')
        assert all('pstars' in r for r in result)
        # Very small p-values should have stars
        assert result[0]['pstars'] in ['***', '**', '*']

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/correct/_correct_bonferroni.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-01 20:47:38 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/correct/_correct_bonferroni.py
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
#   - Apply Bonferroni correction for multiple comparisons
#   - Adjust p-values and significance thresholds
#   - Support both dict and DataFrame inputs
#   - Maintain full result information with adjusted values
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
# import argparse
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
# def correct_bonferroni(
#     results: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
#     alpha: float = 0.05,
#     return_as: str = None,
#     verbose: bool = True,
#     plot: bool = False,
#     ax: Optional[matplotlib.axes.Axes] = None,
# ) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
#     """
#     Apply Bonferroni correction for multiple comparisons.
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
#         - pvalue_adjusted: Bonferroni-adjusted p-value
#         - alpha_adjusted: Bonferroni-adjusted alpha threshold
#         - rejected: Whether null hypothesis is rejected (using adjusted values)
#         - pstars: Significance stars (using adjusted p-value)
# 
#     Notes
#     -----
#     The Bonferroni correction is the most conservative method for controlling
#     the family-wise error rate (FWER) in multiple comparisons.
# 
#     For m tests with family-wise error rate α:
# 
#     .. math::
#         p_{adj,i} = \\min(m \\cdot p_i, 1.0)
# 
#     .. math::
#         \\alpha_{adj} = \\frac{\\alpha}{m}
# 
#     The method guarantees:
# 
#     .. math::
#         P(\\text{at least one false positive}) \\leq \\alpha
# 
#     **Advantages:**
#     - Simple and intuitive
#     - Strong FWER control
#     - No assumptions about test dependencies
# 
#     **Disadvantages:**
#     - Very conservative (low statistical power)
#     - Power decreases linearly with number of tests
#     - May miss true effects (Type II errors)
# 
#     **When to use:**
#     - Small number of planned comparisons (m < 10)
#     - Need strict control of false positives
#     - Tests are independent or positively correlated
# 
#     References
#     ----------
#     .. [1] Bonferroni, C. E. (1936). "Teoria statistica delle classi e calcolo
#            delle probabilità". Pubblicazioni del R Istituto Superiore di
#            Scienze Economiche e Commerciali di Firenze, 8, 3-62.
#     .. [2] Dunn, O. J. (1961). "Multiple Comparisons Among Means". Journal of
#            the American Statistical Association, 56(293), 52-64.
# 
#     Examples
#     --------
#     >>> # Single test (no correction needed)
#     >>> result = {'pvalue': 0.04, 'var_x': 'A', 'var_y': 'B'}
#     >>> corrected = correct_bonferroni(result)
#     >>> corrected['pvalue_adjusted']
#     0.04
# 
#     >>> # Multiple tests
#     >>> results = [
#     ...     {'pvalue': 0.01, 'var_x': 'A', 'var_y': 'B'},
#     ...     {'pvalue': 0.03, 'var_x': 'A', 'var_y': 'C'},
#     ...     {'pvalue': 0.05, 'var_x': 'B', 'var_y': 'C'}
#     ... ]
#     >>> corrected = correct_bonferroni(results)
#     >>> [r['pvalue_adjusted'] for r in corrected]
#     [0.03, 0.09, 0.15]
# 
#     >>> # DataFrame input
#     >>> df = pd.DataFrame({'pvalue': [0.01, 0.03, 0.05]})
#     >>> df_corrected = correct_bonferroni(df)
#     >>> df_corrected['alpha_adjusted'].iloc[0]
#     0.0166...
#     """
#     from scitex.stats.utils._formatters import p2stars
#     from scitex.stats.utils._normalizers import force_dataframe, to_dict
# 
#     if verbose:
#         logger.info("Applying Bonferroni correction")
# 
#     # Store original input type
#     input_type = type(results)
#     is_single_dict = isinstance(results, dict)
# 
#     # Convert to DataFrame for processing
#     if isinstance(results, dict):
#         results_list = [results]
#     elif isinstance(results, list):
#         results_list = results
#     else:  # DataFrame
#         results_list = None
# 
#     if results_list is not None:
#         df = force_dataframe(results_list, fill_na=False, enforce_types=False)
#     else:
#         df = results.copy()
# 
#     # Number of tests
#     m = len(df)
#     if verbose:
#         logger.info(f"Number of tests: {m}, alpha: {alpha}")
# 
#     # Compute adjusted p-values (Bonferroni)
#     df["pvalue_adjusted"] = np.minimum(df["pvalue"] * m, 1.0)
# 
#     # Compute adjusted alpha threshold
#     if "alpha" in df.columns:
#         alpha_values = df["alpha"].fillna(alpha)
#     else:
#         alpha_values = alpha
# 
#     df["alpha_adjusted"] = alpha_values / m
# 
#     # Update rejection decisions based on adjusted values
#     df["rejected"] = df["pvalue_adjusted"] < df["alpha_adjusted"]
# 
#     # Update significance stars based on adjusted p-values
#     df["pstars"] = df["pvalue_adjusted"].apply(p2stars)
# 
#     # Log results summary
#     if verbose:
#         rejections = df["rejected"].sum()
#         logger.info(
#             f"Bonferroni correction complete: {rejections}/{m} hypotheses rejected"
#         )
#         logger.info(f"Adjusted alpha threshold: {alpha / m:.6f}")
# 
#         # Log detailed results if not too many tests
#         if m <= 10:
#             logger.info("\nDetailed results:")
#             for idx, row in df.iterrows():
#                 comparison = ""
#                 if "var_x" in row and "var_y" in row:
#                     comparison = f"{row['var_x']} vs {row['var_y']}: "
#                 elif "test_method" in row:
#                     comparison = f"{row['test_method']}: "
#                 elif "comparison" in row:
#                     comparison = f"{row['comparison']}: "
# 
#                 logger.info(
#                     f"  {comparison}"
#                     f"p = {row['pvalue']:.4f} → p_adj = {row['pvalue_adjusted']:.4f} "
#                     f"{row['pstars']}, rejected = {row['rejected']}"
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
#         _plot_bonferroni(df, alpha, ax)
# 
#     # Determine return format
#     if return_as == "dataframe":
#         return df
#     elif return_as == "dict":
#         if is_single_dict:
#             return to_dict(df, row=0)
#         else:
#             return df.to_dict("records")
#     else:
#         # Match input format
#         if input_type == dict:
#             return to_dict(df, row=0)
#         elif input_type == list:
#             return df.to_dict("records")
#         else:  # DataFrame
#             return df
# 
# 
# def _plot_bonferroni(df, alpha, ax):
#     """Create visualization for Bonferroni correction on given axes."""
#     m = len(df)
#     x = np.arange(m)
# 
#     # Plot original and adjusted p-values
#     ax.scatter(x, df["pvalue"], label="Original p-values", alpha=0.7, s=100, color="C0")
#     ax.scatter(
#         x,
#         df["pvalue_adjusted"],
#         label="Adjusted p-values",
#         alpha=0.7,
#         s=100,
#         color="C1",
#         marker="s",
#     )
# 
#     # Connect original to adjusted with lines
#     for i in range(m):
#         ax.plot(
#             [i, i],
#             [df["pvalue"].iloc[i], df["pvalue_adjusted"].iloc[i]],
#             "k-",
#             alpha=0.3,
#             linewidth=0.5,
#         )
# 
#     # Add significance thresholds
#     ax.axhline(
#         alpha, color="red", linestyle="--", linewidth=2, alpha=0.5, label=f"α = {alpha}"
#     )
#     ax.axhline(
#         alpha / m,
#         color="orange",
#         linestyle="--",
#         linewidth=2,
#         alpha=0.5,
#         label=f"α_adj = {alpha / m:.4f}",
#     )
# 
#     # Formatting
#     ax.set_xlabel("Test Index")
#     ax.set_ylabel("P-value")
#     ax.set_title(
#         f"Bonferroni Correction (m={m} tests)\n"
#         f"{df['rejected'].sum()}/{m} hypotheses rejected"
#     )
#     ax.set_yscale("log")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
# 
#     # Set x-axis labels if there are comparison names
#     if m <= 20:  # Only show labels for reasonable number of tests
#         labels = []
#         for idx, row in df.iterrows():
#             if "var_x" in row and "var_y" in row:
#                 labels.append(f"{row['var_x']}\nvs\n{row['var_y']}")
#             elif "test_method" in row:
#                 labels.append(row["test_method"])
#             elif "comparison" in row:
#                 labels.append(row["comparison"])
#             else:
#                 labels.append(f"Test {idx + 1}")
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
#     else:
#         ax.set_xlabel(f"Test Index (1-{m})")
# 
# 
# """Main function"""
# 
# 
# def main(args):
#     """Demonstrate Bonferroni correction."""
#     logger.info("Demonstrating Bonferroni correction")
# 
#     # Example 1: Single test (no correction needed)
#     logger.info("\n=== Example 1: Single test ===")
# 
#     single_result = {
#         "var_x": "Control",
#         "var_y": "Treatment",
#         "pvalue": 0.04,
#         "alpha": 0.05,
#     }
# 
#     corrected_single = correct_bonferroni(single_result, verbose=args.verbose)
# 
#     # Example 2: Multiple tests
#     logger.info("\n=== Example 2: Three pairwise comparisons ===")
# 
#     multiple_results = [
#         {"var_x": "A", "var_y": "B", "pvalue": 0.01},
#         {"var_x": "A", "var_y": "C", "pvalue": 0.03},
#         {"var_x": "B", "var_y": "C", "pvalue": 0.05},
#     ]
# 
#     corrected_multiple = correct_bonferroni(
#         multiple_results, alpha=0.05, verbose=args.verbose
#     )
# 
#     # Example 3: Many tests (demonstrate conservativeness)
#     logger.info("\n=== Example 3: Many tests (m=20) ===")
# 
#     np.random.seed(42)
#     many_results = []
# 
#     for i in range(20):
#         # Mix of significant and non-significant
#         p = np.random.uniform(0.001, 0.1)
#         many_results.append({"var_x": f"Var_{i}", "var_y": "Control", "pvalue": p})
# 
#     corrected_many = correct_bonferroni(many_results, verbose=args.verbose)
# 
#     # Count rejections
#     n_rejected_before = sum(r["pvalue"] < 0.05 for r in many_results)
#     n_rejected_after = sum(r["rejected"] for r in corrected_many)
# 
#     logger.info(f"Tests with p < 0.05 before correction: {n_rejected_before}")
#     logger.info(f"Tests rejected after correction:       {n_rejected_after}")
# 
#     # Example 4: DataFrame input/output
#     logger.info("\n=== Example 4: DataFrame workflow ===")
# 
#     df_input = pd.DataFrame(
#         {
#             "var_x": ["A", "A", "B"],
#             "var_y": ["B", "C", "C"],
#             "pvalue": [0.002, 0.025, 0.048],
#             "effect_size": [0.8, 0.5, 0.3],
#         }
#     )
# 
#     if args.verbose:
#         logger.info("\nBefore correction:")
#         logger.info(df_input[["var_x", "var_y", "pvalue"]])
# 
#     df_corrected = correct_bonferroni(df_input, verbose=args.verbose)
# 
#     if args.verbose:
#         logger.info("\nAfter correction:")
#         logger.info(
#             df_corrected[
#                 [
#                     "var_x",
#                     "var_y",
#                     "pvalue",
#                     "pvalue_adjusted",
#                     "alpha_adjusted",
#                     "pstars",
#                     "rejected",
#                 ]
#             ]
#         )
# 
#     # Create visualization
#     logger.info("\n=== Creating visualization ===")
# 
#     fig, axes = stx.plt.subplots(2, 2, figsize=(12, 10))
# 
#     # Plot 1: Adjusted vs original p-values
#     ax = axes[0, 0]
# 
#     m_vals = [3, 5, 10, 20]
#     p_orig = 0.03
# 
#     for m in m_vals:
#         p_adj = min(p_orig * m, 1.0)
#         ax.scatter(m, p_adj, s=100, label=f"m = {m}")
# 
#     ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="α = 0.05")
#     ax.axhline(p_orig, color="blue", linestyle="--", alpha=0.5, label="Original p")
#     ax.set_xlabel("Number of Tests (m)")
#     ax.set_ylabel("Adjusted P-value")
#     ax.set_title(f"Bonferroni Adjustment (p_original = {p_orig})")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
# 
#     # Plot 2: Power loss with increasing tests
#     ax = axes[0, 1]
# 
#     m_range = np.arange(1, 51)
#     alpha = 0.05
#     alpha_adj = alpha / m_range
# 
#     ax.plot(m_range, alpha_adj, linewidth=2)
#     ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, label="Original α")
#     ax.set_xlabel("Number of Tests (m)")
#     ax.set_ylabel("Adjusted α Threshold")
#     ax.set_title("Bonferroni: Threshold Decreases Linearly")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     ax.set_yscale("log")
# 
#     # Plot 3: Before/after comparison
#     ax = axes[1, 0]
# 
#     # Generate test results
#     np.random.seed(42)
#     n_tests = 15
#     p_values = np.random.beta(2, 20, n_tests)  # Mix of small p-values
#     p_adjusted = np.minimum(p_values * n_tests, 1.0)
# 
#     x_pos = np.arange(n_tests)
#     width = 0.35
# 
#     bars1 = ax.bar(x_pos - width / 2, p_values, width, label="Original", alpha=0.7)
#     bars2 = ax.bar(x_pos + width / 2, p_adjusted, width, label="Adjusted", alpha=0.7)
# 
#     # Color bars by significance
#     for i, (p_orig, p_adj) in enumerate(zip(p_values, p_adjusted)):
#         if p_orig < 0.05:
#             bars1[i].set_color("green")
#         else:
#             bars1[i].set_color("gray")
# 
#         if p_adj < 0.05:
#             bars2[i].set_color("green")
#         else:
#             bars2[i].set_color("gray")
# 
#     ax.axhline(0.05, color="red", linestyle="--", alpha=0.5, linewidth=2)
#     ax.set_xlabel("Test Index")
#     ax.set_ylabel("P-value")
#     ax.set_title("Before vs After Bonferroni Correction")
#     ax.legend()
#     ax.grid(True, alpha=0.3, axis="y")
# 
#     # Plot 4: Comparison table
#     ax = axes[1, 1]
#     ax.axis("off")
# 
#     # Create comparison data
#     methods_data = [
#         ["Method", "Adjusted α\n(m=10)", "Power", "FWER Control"],
#         ["None", "0.050", "High", "No"],
#         ["Bonferroni", "0.005", "Low", "Strong"],
#         ["Holm", "0.005-0.05", "Medium", "Strong"],
#         ["FDR", "~0.05", "High", "Weak (FDR)"],
#     ]
# 
#     table = ax.table(
#         cellText=methods_data,
#         cellLoc="center",
#         loc="center",
#         bbox=[0, 0, 1, 1],
#     )
# 
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
# 
#     # Header styling
#     for i in range(4):
#         table[(0, i)].set_facecolor("#40466e")
#         table[(0, i)].set_text_props(weight="bold", color="white")
# 
#     # Row styling
#     for i in range(1, 5):
#         for j in range(4):
#             if i % 2 == 0:
#                 table[(i, j)].set_facecolor("#f0f0f0")
# 
#     ax.set_title("Multiple Comparison Methods Comparison", pad=20, fontweight="bold")
# 
#     plt.tight_layout()
# 
#     # Save
#     stx.io.save(fig, "./bonferroni_demo.jpg")
#     logger.info("Visualization saved")
# 
#     return 0
# 
# 
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description="Demonstrate Bonferroni correction")
#     parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
#     return parser.parse_args()
# 
# 
# def run_main():
#     """Initialize SciTeX framework and run main."""
#     global CONFIG, sys, plt, rng
# 
#     import sys
# 
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/correct/_correct_bonferroni.py
# --------------------------------------------------------------------------------
