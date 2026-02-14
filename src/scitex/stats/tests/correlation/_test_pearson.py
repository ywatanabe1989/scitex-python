#!/usr/bin/env python3
# Timestamp: "2025-10-01 21:47:27 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/correlation/_test_pearson.py
# ----------------------------------------
from __future__ import annotations

"""
Functionalities:
  - Perform Pearson correlation test
  - Compute correlation coefficient with confidence intervals
  - Test significance of correlation
  - Generate scatter plots with regression lines
  - Support flexible output formats (dict or DataFrame)

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: Two continuous variables (arrays or Series)
  - output: Test results (dict or DataFrame) and optional figure
"""

import os
from typing import Literal, Optional, Union

import matplotlib.axes
import numpy as np
import pandas as pd
from scipy import stats

import scitex as stx
from scitex.logging import getLogger
from scitex.stats._utils._formatters import fmt_stat, fmt_sym

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)

"""Functions"""


def test_pearson(  # noqa: C901
    x: Union[np.ndarray, pd.Series, str],
    y: Union[np.ndarray, pd.Series, str],
    var_x: str = "x",
    var_y: str = "y",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    data: Union[pd.DataFrame, str, None] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    decimals: int = 3,
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    r"""
    Perform Pearson correlation test.

    Parameters
    ----------
    x, y : arrays or Series
        Two continuous variables
    var_x, var_y : str
        Labels for variables
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Alternative hypothesis
    alpha : float, default 0.05
        Significance level for confidence interval
    plot : bool, default False
        Whether to generate scatter plot
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None and plot=True, creates new figure.
        If provided, automatically enables plotting.
    data : DataFrame, str, or None, optional
        DataFrame or CSV path. When provided, string values for x/y
        are resolved as column names (seaborn-style).
    return_as : {'dict', 'dataframe'}, default 'dict'
        Output format
    decimals : int, default 3
        Number of decimal places for rounding
    verbose : bool, default False
        Whether to print test results

    Returns
    -------
    results : dict or DataFrame
        Test results including:
        - test_method: 'Pearson correlation'
        - statistic: Pearson correlation coefficient
        - pvalue: p-value
        - stars: Significance stars
        - significant: Whether null hypothesis is rejected
        - ci_lower, ci_upper: Confidence interval bounds
        - r_squared: Coefficient of determination
        - effect_size: Correlation coefficient (same as statistic)
        - effect_size_metric: 'Pearson r'
        - effect_size_interpretation: Interpretation
        - n: Sample size (after removing NaN pairs)
        - var_x, var_y: Variable labels
        - H0: Null hypothesis description

    Notes
    -----
    Pearson correlation coefficient measures the linear relationship between
    two continuous variables.

    **Null Hypothesis (H0)**: No linear correlation (ρ = 0)

    **Pearson's r**:

    .. math::
        r = \frac{\\sum(x_i - \bar{x})(y_i - \bar{y})}{\\sqrt{\\sum(x_i - \bar{x})^2 \\sum(y_i - \bar{y})^2}}

    Range: -1 ≤ r ≤ 1
    - r = 1: Perfect positive linear relationship
    - r = 0: No linear relationship
    - r = -1: Perfect negative linear relationship

    **Coefficient of determination (R²)**:

    .. math::
        R^2 = r^2

    R² represents the proportion of variance in y explained by x.

    **Interpretation (Cohen, 1988)**:
    - |r| < 0.1:  negligible
    - |r| < 0.3:  small
    - |r| < 0.5:  medium
    - |r| ≥ 0.5:  large

    **Assumptions**:
    1. **Linearity**: Relationship between variables is linear
    2. **Normality**: Both variables are normally distributed (for hypothesis testing)
    3. **Homoscedasticity**: Variance is constant across the range
    4. **Independence**: Observations are independent

    **When to use**:
    - Assessing linear relationship between two continuous variables
    - Both variables approximately normally distributed
    - No major outliers present
    - Relationship appears linear on scatter plot

    **When NOT to use**:
    - Non-linear relationships (consider transformation or Spearman)
    - Ordinal data (use Spearman)
    - Severe outliers present (use Spearman)
    - Non-normal distributions (use Spearman)

    **Confidence Interval**:
    Computed using Fisher's z-transformation:

    .. math::
        z = 0.5 \\ln\\left(\frac{1+r}{1-r}\right)

    References
    ----------
    .. [1] Pearson, K. (1896). "Mathematical contributions to the theory of
           evolution. III. Regression, heredity, and panmixia". Philosophical
           Transactions of the Royal Society of London. Series A, 187, 253-318.
    .. [2] Cohen, J. (1988). Statistical Power Analysis for the Behavioral
           Sciences (2nd ed.). Routledge.

    Examples
    --------
    >>> # Strong positive correlation
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 5, 7, 8])
    >>> result = test_pearson(x, y)
    >>> result['statistic']
    0.98...

    >>> # With visualization
    >>> result, fig = test_pearson(x, y, plot=True)
    """
    # Resolve column names from DataFrame (seaborn-style data= parameter)
    if data is not None:
        from scitex.stats._utils._csv_support import resolve_columns

        resolved = resolve_columns(data, x=x, y=y)
        x, y = resolved["x"], resolved["y"]

    from scitex.stats._utils._formatters import p2stars
    from scitex.stats._utils._normalizers import convert_results, force_dataframe

    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN pairs (pairwise deletion)
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    n = len(x)

    if n < 3:
        raise ValueError("Pearson correlation requires at least 3 valid pairs")

    # Perform Pearson correlation test
    r, pvalue = stats.pearsonr(x, y)
    r = float(r)
    pvalue = float(pvalue)

    # Adjust p-value for alternative hypothesis
    if alternative == "less":
        if r > 0:
            pvalue = 1 - pvalue / 2
        else:
            pvalue = pvalue / 2
    elif alternative == "greater":
        if r < 0:
            pvalue = 1 - pvalue / 2
        else:
            pvalue = pvalue / 2

    # Determine rejection
    rejected = pvalue < alpha

    # Compute confidence interval using Fisher's z-transformation
    z = np.arctanh(r)  # Fisher's z
    se = 1 / np.sqrt(n - 3)  # Standard error of z
    z_crit = stats.norm.ppf(1 - alpha / 2)

    z_lower = z - z_crit * se
    z_upper = z + z_crit * se

    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)

    # Compute R-squared
    r_squared = r**2

    # Interpret effect size
    r_abs = abs(r)
    if r_abs < 0.1:
        effect_interp = "negligible"
    elif r_abs < 0.3:
        effect_interp = "small"
    elif r_abs < 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    # Compile results
    result = {
        "test_method": "Pearson correlation",
        "statistic": round(r, decimals),
        "stat_symbol": "r",
        "n": n,
        "var_x": var_x,
        "var_y": var_y,
        "pvalue": round(pvalue, decimals),
        "stars": p2stars(pvalue),
        "alpha": alpha,
        "significant": rejected,
        "ci_lower": round(ci_lower, decimals),
        "ci_upper": round(ci_upper, decimals),
        "r_squared": round(r_squared, decimals),
        "effect_size": round(r, decimals),
        "effect_size_metric": "Pearson r",
        "effect_size_interpretation": effect_interp,
        "H0": f"No linear correlation between {var_x} and {var_y}",
    }

    # Add interpretation
    if rejected:
        direction = "positive" if r > 0 else "negative"
        result["interpretation"] = (
            f"Significant {direction} correlation detected "
            f"(r = {r:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}])"
        )
    else:
        result["interpretation"] = f"No significant correlation detected (r = {r:.3f})"

    # Log results if verbose
    if verbose:
        logger.info(f"Pearson: r = {r:.3f}, p = {pvalue:.4f} {p2stars(pvalue)}")
        logger.info(
            f"R² = {r_squared:.3f} ({effect_interp}), 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]"
        )

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            fig, ax = stx.plt.subplots()
        _plot_pearson(x, y, var_x, var_y, result, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)
    elif return_as not in ["dict", "dataframe"]:
        return convert_results(result, return_as=return_as)

    return result


def _plot_pearson(x, y, var_x, var_y, result, ax) -> None:
    """Create scatter plot with regression line on given axes."""
    from scitex.stats._plot_helpers import scatter_regression, stats_text_box

    scatter_regression(ax, x, y)
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.set_title("Pearson Correlation")

    stats_text_box(
        ax,
        [
            fmt_stat("r", result["statistic"]),
            fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
            f"95% CI [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]",
            fmt_stat("R2", result["r_squared"]),
            f"{fmt_sym('n')} = {result['n']}",
        ],
    )


# EOF
