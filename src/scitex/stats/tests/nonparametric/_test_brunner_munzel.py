#!/usr/bin/env python3
# Timestamp: "2025-10-01 22:40:43 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/nonparametric/_test_brunner_munzel.py

r"""Brunner-Munzel test (non-parametric alternative to t-test).

Functionalities:
  - Perform Brunner-Munzel test (non-parametric alternative to t-test)
  - Compute both P(X>Y) and Cliff's delta effect sizes
  - Generate visualizations with significance indicators
  - Support flexible output formats (dict or DataFrame)

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: Two samples (arrays or Series)
  - output: Test results (dict or DataFrame) and optional figure
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Union

import matplotlib
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


def test_brunner_munzel(
    x: Union[np.ndarray, pd.Series, str],
    y: Union[np.ndarray, pd.Series, str],
    var_x: str = "x",
    var_y: str = "y",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    data: Union[pd.DataFrame, str, None] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    r"""
    Perform Brunner-Munzel test (non-parametric).

    Parameters
    ----------
    x : array or Series
        First sample
    y : array or Series
        Second sample
    var_x : str, default 'x'
        Label for first sample
    var_y : str, default 'y'
        Label for second sample
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Alternative hypothesis:
        - 'two-sided': distributions differ
        - 'greater': x tends to be greater than y
        - 'less': x tends to be less than y
    alpha : float, default 0.05
        Significance level
    plot : bool, default False
        Whether to generate visualization
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None and plot=True, creates new figure.
        If provided, automatically enables plotting.
    data : DataFrame, str, or None, optional
        DataFrame or CSV path. When provided, string values for x/y
        are resolved as column names (seaborn-style).
    return_as : {'dict', 'dataframe'}, default 'dict'
        Output format
    verbose : bool, default False
        Whether to print test results

    Returns
    -------
    results : dict or DataFrame
        Test results including:
        - test_method: 'Brunner-Munzel test'
        - statistic_name: 'W'
        - statistic: W-statistic value
        - pvalue: p-value
        - stars: Significance stars
        - rejected: Whether null hypothesis is rejected
        - effect_size: P(X > Y) (primary effect size)
        - effect_size_metric: 'P(X>Y)'
        - effect_size_interpretation: Interpretation of P(X>Y)
        - effect_size_secondary: Cliff's delta (secondary effect size)
        - effect_size_secondary_metric: "Cliff's delta"
        - effect_size_secondary_interpretation: Interpretation of delta
        - n_x, n_y: Sample sizes
        - var_x, var_y: Variable labels
        - H0: Null hypothesis description

    Notes
    -----
    The Brunner-Munzel test is a non-parametric test for comparing two independent
    samples. It is more robust than the t-test when:
    - Distributions are non-normal
    - Variances are unequal
    - Sample sizes differ
    - Data contain outliers

    Unlike Mann-Whitney U test, Brunner-Munzel does not assume equal variances
    and provides better control of Type I error rate.

    The test statistic W is approximately t-distributed:

    .. math::
        W = \frac{\hat{p} - 0.5}{\sqrt{\hat{\sigma}^2}}

    where :math:`\hat{p}` is an estimate of P(X > Y).

    **Effect Sizes:**

    1. **P(X > Y)**: Probability that a random value from X exceeds a random
       value from Y. Interpretation:
       - 0.50: No effect (chance)
       - 0.56: Small effect
       - 0.64: Medium effect
       - 0.71: Large effect

    2. **Cliff's delta (δ)**: Ranges from -1 to 1, related to P(X>Y) by:
       δ = 2×P(X>Y) - 1. Interpretation:
       - |δ| < 0.147: Negligible
       - |δ| < 0.33: Small
       - |δ| < 0.474: Medium
       - |δ| ≥ 0.474: Large

    References
    ----------
    .. [1] Brunner, E., & Munzel, U. (2000). "The nonparametric Behrens-Fisher
           problem: Asymptotic theory and a small-sample approximation".
           Biometrical Journal, 42(1), 17-25.
    .. [2] Neubert, K., & Brunner, E. (2007). "A studentized permutation test
           for the non-parametric Behrens-Fisher problem". Computational
           Statistics & Data Analysis, 51(10), 5192-5204.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 3, 4, 5, 6])
    >>> result = test_brunner_munzel(x, y)
    >>> result['pvalue']
    0.109...
    >>> result['effect_size']  # P(X > Y)
    0.2
    >>> result['effect_size_secondary']  # Cliff's delta
    -0.6

    >>> # With auto-created figure
    >>> result = test_brunner_munzel(x, y, plot=True)

    >>> # Plot on existing axes
    >>> fig, ax = plt.subplots()
    >>> result = test_brunner_munzel(x, y, ax=ax)

    >>> # As DataFrame
    >>> df = test_brunner_munzel(x, y, return_as='dataframe')
    """
    # Resolve column names from DataFrame (seaborn-style data= parameter)
    if data is not None:
        from scitex.stats._utils._csv_support import resolve_columns

        resolved = resolve_columns(data, x=x, y=y)
        x, y = resolved["x"], resolved["y"]

    from scitex.stats._utils._effect_size import (
        cliffs_delta,
        interpret_cliffs_delta,
        interpret_prob_superiority,
        prob_superiority,
    )
    from scitex.stats._utils._formatters import p2stars
    from scitex.stats._utils._normalizers import force_dataframe

    # Convert to numpy arrays and remove NaN
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n_x = len(x)
    n_y = len(y)

    # Perform Brunner-Munzel test
    bm_result = stats.brunnermunzel(x, y, alternative=alternative)
    w_stat = float(bm_result.statistic)
    pvalue = float(bm_result.pvalue)

    # Compute effect sizes
    prob_xy = prob_superiority(x, y)
    delta = cliffs_delta(x, y)

    # Interpretations
    prob_interp = interpret_prob_superiority(prob_xy)
    delta_interp = interpret_cliffs_delta(delta)

    # Create null hypothesis description
    if alternative == "two-sided":
        H0 = f"P({var_x} > {var_y}) = 0.5"
    elif alternative == "greater":
        H0 = f"P({var_x} > {var_y}) ≤ 0.5"
    else:  # less
        H0 = f"P({var_x} > {var_y}) ≥ 0.5"

    # Compile results
    result = {
        "test_method": "Brunner-Munzel test",
        "statistic": w_stat,
        "stat_symbol": "BM",
        "alternative": alternative,
        "n_x": n_x,
        "n_y": n_y,
        "var_x": var_x,
        "var_y": var_y,
        "pvalue": pvalue,
        "stars": p2stars(pvalue),
        "alpha": alpha,
        "significant": pvalue < alpha,
        "effect_size": prob_xy,
        "effect_size_metric": "P(X>Y)",
        "effect_size_interpretation": prob_interp,
        "effect_size_secondary": delta,
        "effect_size_secondary_metric": "Cliff's delta",
        "effect_size_secondary_interpretation": delta_interp,
        "H0": H0,
    }

    # Log results if verbose
    if verbose:
        logger.info(
            f"Brunner-Munzel: W = {w_stat:.3f}, p = {pvalue:.4f} {p2stars(pvalue)}"
        )
        logger.info(
            f"P(X>Y) = {prob_xy:.3f} ({prob_interp}), Cliff's δ = {delta:.3f} ({delta_interp})"
        )

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            _fig, ax = stx.plt.subplots()
        _plot_brunner_munzel(x, y, var_x, var_y, result, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)

    return result


def _plot_brunner_munzel(x, y, var_x, var_y, result, ax):
    """Create violin+swarm visualization for Brunner-Munzel test on given axes."""
    from scitex.stats._plot_helpers import (
        significance_bracket,
        stats_text_box,
        violin_swarm,
    )

    violin_swarm(ax, [x, y], [0, 1], [var_x, var_y])
    significance_bracket(ax, 0, 1, result["stars"], [x, y])

    stats_text_box(
        ax,
        [
            fmt_stat("W", result["statistic"]),
            fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
            f"P({fmt_sym('X')}>{fmt_sym('Y')}) = {result['effect_size']:.3f}",
            fmt_stat("delta", result["effect_size_secondary"]),
        ],
    )

    ax.set_title("Brunner-Munzel Test")


# EOF
