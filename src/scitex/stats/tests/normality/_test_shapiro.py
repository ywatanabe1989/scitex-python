#!/usr/bin/env python3
# Timestamp: "2025-10-01 15:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/normality/_test_shapiro.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------


"""
Functionalities:
  - Perform Shapiro-Wilk test for normality
  - Generate Q-Q plots for visual assessment
  - Provide interpretation and recommendations
  - Support flexible output formats (dict or DataFrame)

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: One sample (array or Series)
  - output: Test results (dict or DataFrame) and optional figure
"""

"""Imports"""
from typing import Literal, Optional, Union  # noqa: E402

import matplotlib.axes  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import scitex as stx  # noqa: E402
from scitex.logging import getLogger  # noqa: E402
from scitex.stats._utils._formatters import fmt_stat, fmt_sym  # noqa: E402

logger = getLogger(__name__)

"""Functions"""


def test_shapiro(  # noqa: C901
    x: Union[np.ndarray, pd.Series, str],
    var_x: str = "x",
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    data: Union[pd.DataFrame, str, None] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    """
    Perform Shapiro-Wilk test for normality.

    Parameters
    ----------
    x : array or Series
        Sample to test
    var_x : str, default 'x'
        Label for sample
    alpha : float, default 0.05
        Significance level
    plot : bool, default False
        Whether to generate Q-Q plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If provided, plot is set to True
    data : DataFrame, str, or None, optional
        DataFrame or CSV path. When provided, string value for x
        is resolved as a column name (seaborn-style).
    return_as : {'dict', 'dataframe'}, default 'dict'
        Output format
    verbose : bool, default False
        If True, print test results to logger

    Returns
    -------
    results : dict or DataFrame
        Test results including:
        - test_method: 'Shapiro-Wilk test'
        - statistic: W-statistic value (0 to 1, closer to 1 = more normal)
        - pvalue: p-value
        - stars: Significance stars
        - significant: Whether null hypothesis is rejected (True = not normal)
        - normal: Whether data appears normal (True = normal)
        - recommendation: Suggested statistical approach
        - n: Sample size
        - var_x: Variable label

    Notes
    -----
    The Shapiro-Wilk test tests the null hypothesis that data come from a
    normal distribution.

    **Null Hypothesis (H0)**: Data are normally distributed

    **Test Statistic W**: Ranges from 0 to 1
    - W close to 1: Data appear normal
    - W much less than 1: Data deviate from normality

    **p-value interpretation**:
    - p > α (typically 0.05): Fail to reject H0, data appear normal
    - p ≤ α: Reject H0, data significantly deviate from normality

    **Important considerations**:
    - Sensitive to sample size: with n > 50, may detect trivial deviations
    - Works best for 3 ≤ n ≤ 5000
    - Should be combined with visual inspection (Q-Q plots)
    - Large samples: focus on Q-Q plots over p-values
    - Small samples: test may lack power to detect non-normality

    **Recommendations based on results**:
    - Normal (p > 0.05): Use parametric tests (t-test, ANOVA, Pearson)
    - Non-normal (p ≤ 0.05): Use non-parametric tests (Brunner-Munzel, Wilcoxon, Spearman)
    - Borderline: Check Q-Q plot and consider robustness

    References
    ----------
    .. [1] Shapiro, S. S., & Wilk, M. B. (1965). "An analysis of variance test
           for normality (complete samples)". Biometrika, 52(3-4), 591-611.
    .. [2] Razali, N. M., & Wah, Y. B. (2011). "Power comparisons of
           Shapiro-Wilk, Kolmogorov-Smirnov, Lilliefors and Anderson-Darling
           tests". Journal of Statistical Modeling and Analytics, 2(1), 21-33.

    Examples
    --------
    >>> # Normal data
    >>> x = np.random.normal(0, 1, 100)
    >>> result = test_shapiro(x)
    >>> result['normal']
    True

    >>> # Non-normal data
    >>> x = np.random.exponential(2, 100)
    >>> result = test_shapiro(x)
    >>> result['normal']
    False

    >>> # With Q-Q plot
    >>> result, fig = test_shapiro(x, plot=True)
    """
    from scitex.stats._utils._formatters import p2stars
    from scitex.stats._utils._normalizers import force_dataframe

    # Resolve column names from DataFrame (seaborn-style data= parameter)
    if data is not None:
        from scitex.stats._utils._csv_support import resolve_columns

        resolved = resolve_columns(data, x=x)
        x = resolved["x"]

    # Convert to numpy array and remove NaN
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    n_x = len(x)

    # Check sample size
    if n_x < 3:
        raise ValueError("Shapiro-Wilk test requires at least 3 observations")
    if n_x > 5000:
        logger.warning(
            f"Sample size n={n_x} is large. "
            "Shapiro-Wilk may detect trivial deviations. "
            "Consider visual inspection (Q-Q plot) instead."
        )

    # Perform Shapiro-Wilk test
    sw_result = stats.shapiro(x)
    w_stat = float(sw_result.statistic)
    pvalue = float(sw_result.pvalue)

    # Determine if data appear normal
    normal = pvalue > alpha
    rejected = not normal

    # Generate recommendation
    if normal:
        recommendation = "Data appear normal. Parametric tests (t-test, ANOVA, Pearson) are appropriate."
    else:
        recommendation = "Data deviate from normality. Consider non-parametric tests (Brunner-Munzel, Wilcoxon, Spearman)."

    # Add sample size consideration
    if n_x > 100:
        recommendation += (
            " Note: Large sample size - inspect Q-Q plot for practical significance."
        )
    elif n_x < 20:
        recommendation += " Note: Small sample size - test may have low power."

    # Compile results
    result = {
        "test_method": "Shapiro-Wilk test",
        "statistic": w_stat,
        "stat_symbol": "W",
        "n": n_x,
        "var_x": var_x,
        "pvalue": pvalue,
        "stars": p2stars(pvalue),
        "alpha": alpha,
        "significant": rejected,
        "normal": normal,
        "recommendation": recommendation,
    }

    # Log results if verbose
    if verbose:
        logger.info(
            f"Shapiro-Wilk: W = {w_stat:.4f}, p = {pvalue:.4f} {p2stars(pvalue)}"
        )
        logger.info(f"Normal: {normal}")

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            fig, axes = stx.plt.subplots(1, 2, figsize=(12, 5))
            _plot_qq_full(x, var_x, result, axes)
        else:
            _plot_qq_simple(x, var_x, result, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)
    elif return_as not in ["dict", "dataframe"]:
        from scitex.stats._utils._normalizers import convert_results

        return convert_results(result, return_as=return_as)

    return result


def _plot_qq_full(x, var_x, result, axes):
    """Create 2-panel Q-Q plot with histogram."""
    from scitex.stats._plot_helpers import stats_text_box

    # Plot 1: Q-Q plot
    ax = axes[0]

    # Compute theoretical quantiles
    (osm, osr), (slope, intercept, _r) = stats.probplot(x, dist="norm")

    # Plot
    ax.scatter(osm, osr)
    ax.plot(osm, slope * osm + intercept, label="Expected (normal)")

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Shapiro-Wilk Test")
    ax.legend()

    # Add text with results
    stats_text_box(
        ax,
        [
            fmt_stat("W", result["statistic"], fmt=".4f"),
            fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
            f"Normal: {result['normal']}",
            f"{fmt_sym('n')} = {len(x)}",
        ],
    )

    # Plot 2: Histogram with normal curve overlay
    ax = axes[1]

    # Histogram
    _n, _bins, _patches = ax.hist(x, bins="auto", density=True)

    # Fit normal distribution
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = stats.norm.pdf(x_fit, mu, sigma)

    ax.plot(x_fit, y_fit, label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram")
    ax.legend()


def _plot_qq_simple(x, var_x, result, ax):
    """Create single Q-Q plot on provided axes."""
    from scitex.stats._plot_helpers import stats_text_box

    # Compute theoretical quantiles
    (osm, osr), (slope, intercept, _r) = stats.probplot(x, dist="norm")

    # Plot
    ax.scatter(osm, osr)
    ax.plot(osm, slope * osm + intercept, label="Expected (normal)")

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("Shapiro-Wilk Test")
    ax.legend()

    # Add text with results
    stats_text_box(
        ax,
        [
            fmt_stat("W", result["statistic"], fmt=".4f"),
            fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
            f"Normal: {result['normal']}",
            f"{fmt_sym('n')} = {len(x)}",
        ],
    )


def test_normality(
    *samples,
    var_names: Optional[list] = None,
    alpha: float = 0.05,
    warn: bool = True,
) -> dict:
    """
    Check normality for multiple samples using Shapiro-Wilk test.

    Parameters
    ----------
    *samples : arrays
        Samples to check
    var_names : list of str, optional
        Names for each sample
    alpha : float, default 0.05
        Significance level
    warn : bool, default True
        Whether to log warnings for non-normal data

    Returns
    -------
    dict
        Dictionary with results for each sample:
        - 'all_normal': bool, True if all samples are normal
        - 'results': list of individual test results
        - 'recommendation': str, overall recommendation

    Examples
    --------
    >>> x = np.random.normal(0, 1, 50)
    >>> y = np.random.exponential(2, 50)
    >>> check = check_normality(x, y, var_names=['Normal', 'Exponential'])
    >>> check['all_normal']
    False
    >>> check['recommendation']
    'Some samples deviate from normality. Consider non-parametric tests.'
    """
    if var_names is None:
        var_names = [f"sample_{i}" for i in range(len(samples))]

    if len(var_names) != len(samples):
        raise ValueError("Number of var_names must match number of samples")

    results = []
    for sample, var_name in zip(samples, var_names):
        result = test_shapiro(sample, var_x=var_name, alpha=alpha, return_as="dict")
        results.append(result)

        if warn and not result["normal"]:
            logger.warning(
                f"{var_name}: Data deviate from normality "
                f"(W={result['statistic']:.4f}, p={result['pvalue']:.4f})"
            )

    all_normal = all(r["normal"] for r in results)

    if all_normal:
        recommendation = "All samples appear normal. Parametric tests are appropriate."
    else:
        non_normal = [r["var_x"] for r in results if not r["normal"]]
        recommendation = (
            f"Samples {', '.join(non_normal)} deviate from normality. "
            "Consider non-parametric tests (Brunner-Munzel, Wilcoxon, Kruskal-Wallis)."
        )

    return {
        "all_normal": all_normal,
        "results": results,
        "recommendation": recommendation,
    }


# EOF
