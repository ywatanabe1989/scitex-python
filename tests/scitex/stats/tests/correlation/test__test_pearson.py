# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/tests/correlation/_test_pearson.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-11-29 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/tests/correlation/_test_pearson.py
# 
# """
# Pearson correlation test with publication-ready output.
# 
# Functionalities:
#   - Compute Pearson correlation coefficient and p-value
#   - Calculate confidence intervals using Fisher's z-transformation
#   - Optional scatter plot with regression line and statistics
#   - Return standardized result dictionary
# 
# Dependencies:
#   - packages: numpy, pandas, scipy
# 
# IO:
#   - input: Two continuous variables (arrays or Series)
#   - output: Correlation result dictionary with r, p-value, CI
# """
# 
# import argparse
# from typing import Optional, Union, Tuple
# 
# import numpy as np
# import pandas as pd
# from scipy import stats as scipy_stats
# 
# import scitex as stx
# from scitex.logging import getLogger
# 
# logger = getLogger(__name__)
# 
# 
# def test_pearson(
#     x: Union[np.ndarray, pd.Series],
#     y: Union[np.ndarray, pd.Series],
#     var_x: Optional[str] = None,
#     var_y: Optional[str] = None,
#     alpha: float = 0.05,
#     plot: bool = False,
#     **plot_kwargs,
# ) -> Union["StatResult", Tuple["StatResult", "matplotlib.figure.Figure"]]:
#     """
#     Pearson correlation test for linear relationship between two continuous variables.
# 
#     Parameters
#     ----------
#     x : array or Series
#         First variable (continuous)
#     y : array or Series
#         Second variable (continuous)
#     var_x : str, optional
#         Name of first variable for display
#     var_y : str, optional
#         Name of second variable for display
#     alpha : float, default 0.05
#         Significance level for confidence intervals
#     plot : bool, default False
#         Whether to create scatter plot with regression line
#     **plot_kwargs
#         Additional arguments passed to plotting function
# 
#     Returns
#     -------
#     result : StatResult
#         StatResult instance containing:
#         - statistic: Pearson's r coefficient
#         - p_value: Two-tailed p-value
#         - stars: Significance stars (*, **, ***, ns)
#         - ci_95: Confidence interval [lower, upper]
#         - effect_size: R² and interpretation
#         - samples: Sample information
#         Access as attributes or use .to_dict() for dictionary
#     fig : matplotlib.figure.Figure, optional
#         Figure object if plot=True
# 
#     Notes
#     -----
#     Pearson correlation coefficient r is calculated as:
# 
#     .. math::
#         r = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^{n}(x_i - \\bar{x})^2 \\sum_{i=1}^{n}(y_i - \\bar{y})^2}}
# 
#     Confidence intervals use Fisher's z-transformation:
# 
#     .. math::
#         z = 0.5 \\ln\\left(\\frac{1+r}{1-r}\\right)
# 
#     Interpretation guidelines (Cohen, 1988):
#     - |r| < 0.1: Negligible
#     - |r| < 0.3: Small
#     - |r| < 0.5: Medium
#     - |r| ≥ 0.5: Large
# 
#     Examples
#     --------
#     >>> import numpy as np
#     >>> from scitex.stats.tests.correlation import test_pearson
#     >>>
#     >>> # Generate correlated data
#     >>> np.random.seed(42)
#     >>> x = np.random.randn(100)
#     >>> y = 2 * x + np.random.randn(100) * 0.5
#     >>>
#     >>> # Basic usage
#     >>> result = test_pearson(x, y)
#     >>> print(f"r = {result.statistic['value']:.3f}, p = {result.p_value:.3e}")
#     >>>
#     >>> # With variable names and plot
#     >>> result, fig = test_pearson(
#     ...     x, y,
#     ...     var_x='Height',
#     ...     var_y='Weight',
#     ...     plot=True
#     ... )
#     >>>
#     >>> # Check significance
#     >>> if result.p_value < 0.05:
#     ...     print(f"Significant correlation: r = {result.statistic['value']:.3f}{result.stars}")
#     >>>
#     >>> # Use as dictionary (backward compatibility)
#     >>> result_dict = result.to_dict()
#     >>> print(result_dict['statistic']['value'])
#     """
#     from scitex.stats.utils._formatters import p2stars
#     from scitex.stats._schema import StatResult
# 
#     # Convert to numpy arrays and remove NaN
#     x = np.asarray(x)
#     y = np.asarray(y)
# 
#     # Check for matching lengths
#     if len(x) != len(y):
#         raise ValueError(f"x and y must have same length (got {len(x)} and {len(y)})")
# 
#     # Remove NaN pairs
#     mask = ~(np.isnan(x) | np.isnan(y))
#     x_clean = x[mask]
#     y_clean = y[mask]
#     n = len(x_clean)
# 
#     if n < 3:
#         raise ValueError(f"Need at least 3 valid pairs (got {n})")
# 
#     # Compute Pearson correlation
#     r, p_value = scipy_stats.pearsonr(x_clean, y_clean)
# 
#     # Calculate confidence interval using Fisher's z-transformation
#     z = np.arctanh(r)  # Fisher's z = 0.5 * ln((1+r)/(1-r))
#     se = 1 / np.sqrt(n - 3)  # Standard error of z
#     z_crit = scipy_stats.norm.ppf(1 - alpha / 2)  # Critical value for two-tailed test
# 
#     ci_lower_z = z - z_crit * se
#     ci_upper_z = z + z_crit * se
# 
#     # Transform back to r scale
#     ci_lower = np.tanh(ci_lower_z)
#     ci_upper = np.tanh(ci_upper_z)
# 
#     # Convert p-value to stars
#     stars = p2stars(p_value, ns_symbol=False)
#     if not stars:
#         stars = "ns"
# 
#     # Calculate R²
#     r_squared = r**2
# 
#     # Interpret effect size
#     r_abs = abs(r)
#     if r_abs < 0.1:
#         interpretation = "negligible"
#     elif r_abs < 0.3:
#         interpretation = "small"
#     elif r_abs < 0.5:
#         interpretation = "medium"
#     else:
#         interpretation = "large"
# 
#     # Set variable names
#     if var_x is None:
#         var_x = "X"
#     if var_y is None:
#         var_y = "Y"
# 
#     # Build StatResult instance
#     result = StatResult(
#         test_type="pearson",
#         test_category="correlation",
#         statistic={"name": "r", "value": r},
#         p_value=p_value,
#         stars=stars,
#         effect_size={
#             "name": "r_squared",
#             "value": r_squared,
#             "interpretation": interpretation,
#             "ci_95": [ci_lower, ci_upper],
#         },
#         samples={"n_total": n, "n_valid": n, "var_x": var_x, "var_y": var_y},
#         ci_95=[ci_lower, ci_upper],
#         extra={
#             "test": "Pearson correlation",
#             "ci_level": 1 - alpha,
#             "method": "pearson",
#             "alpha": alpha,
#         },
#         software_version=stx.__version__,
#     )
# 
#     # Create plot if requested
#     if plot:
#         fig = _plot_pearson(x_clean, y_clean, result, **plot_kwargs)
#         return result, fig
# 
#     return result
# 
# 
# def _plot_pearson(
#     x: np.ndarray, y: np.ndarray, result: "StatResult", **kwargs
# ) -> "matplotlib.figure.Figure":
#     """
#     Create scatter plot with regression line for Pearson correlation.
# 
#     Internal function called by test_pearson when plot=True.
#     """
#     # Create figure
#     fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
# 
#     # Scatter plot
#     scatter = ax.scatter(x, y, alpha=0.6, c="steelblue", label="Data")
#     stx.plt.ax.style_scatter(scatter, size_mm=0.8)
# 
#     # Add fitted line with statistics
#     stx.plt.ax.add_fitted_line(
#         ax, x, y, color="black", linestyle="--", label="Fit", show_stats=True
#     )
# 
#     # Labels
#     var_x = result.samples.get("var_x", "X")
#     var_y = result.samples.get("var_y", "Y")
#     ax.set_xlabel(stx.plt.ax.format_label(var_x, ""))
#     ax.set_ylabel(stx.plt.ax.format_label(var_y, ""))
#     ax.set_title(
#         f"Pearson Correlation (r = {result.statistic['value']:.3f}{result.stars})"
#     )
#     ax.legend(frameon=False, fontsize=6)
# 
#     return fig
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/tests/correlation/_test_pearson.py
# --------------------------------------------------------------------------------
