#!/usr/bin/env python3
# Timestamp: "2025-10-01 16:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/categorical/_test_mcnemar.py
# ----------------------------------------

r"""McNemar's test for paired categorical data.

Functionalities:
  - McNemar's test for paired categorical data
  - Test for marginal homogeneity in 2×2 contingency tables
  - Compute odds ratio for matched pairs
  - Generate visualizations for paired outcomes

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: 2×2 contingency table (array or DataFrame)
  - output: Test results (dict or DataFrame) and optional figure
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Union

import matplotlib.axes
import numpy as np
import pandas as pd
from scipy import stats

import scitex as stx
from scitex.logging import getLogger
from scitex.stats._utils._formatters import p2stars
from scitex.stats._utils._normalizers import convert_results, force_dataframe

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)


def mcnemar_odds_ratio(b: int, c: int) -> float:
    """
    Compute odds ratio for McNemar's test from discordant pairs.

    Parameters
    ----------
    b : int
        Count in cell [0,1] (success before, failure after)
    c : int
        Count in cell [1,0] (failure before, success after)

    Returns
    -------
    or_val : float
        Odds ratio = b / c

    Notes
    -----
    For McNemar's test, OR = b/c where b and c are the discordant pairs.
    OR > 1 indicates more b than c (positive change)
    OR < 1 indicates more c than b (negative change)
    OR = 1 indicates no change
    """
    if c == 0:
        if b == 0:
            return 1.0  # No discordant pairs
        return float("inf")  # Only b discordant pairs
    return float(b / c)


def interpret_mcnemar_or(or_val: float) -> str:
    """
    Interpret McNemar's odds ratio.

    Parameters
    ----------
    or_val : float
        Odds ratio

    Returns
    -------
    interpretation : str
        Interpretation
    """
    if or_val == 1.0:
        return "no change"
    elif or_val > 1.0:
        if or_val < 2.0:
            return "small increase"
        elif or_val < 4.0:
            return "medium increase"
        else:
            return "large increase"
    else:  # or_val < 1.0
        if or_val > 0.5:
            return "small decrease"
        elif or_val > 0.25:
            return "medium decrease"
        else:
            return "large decrease"


def test_mcnemar(  # noqa: C901
    observed: Union[np.ndarray, pd.DataFrame, list],
    var_before: Optional[str] = None,
    var_after: Optional[str] = None,
    correction: bool = True,
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    decimals: int = 3,
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    r"""
    Perform McNemar's test for paired categorical data.

    Tests whether there is a significant change in proportions for paired binary data.
    Appropriate for before-after studies with binary outcomes.

    Parameters
    ----------
    observed : array-like, shape (2, 2)
        2×2 contingency table:
        [[a, b],
         [c, d]]
        where:
        - a: both conditions negative (0,0)
        - b: before negative, after positive (0,1)
        - c: before positive, after negative (1,0)
        - d: both conditions positive (1,1)
    var_before : str, optional
        Name for before condition
    var_after : str, optional
        Name for after condition
    correction : bool, default True
        Whether to apply continuity correction (recommended for small samples)
    alpha : float, default 0.05
        Significance level
    plot : bool, default False
        Whether to generate visualization
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If provided, plot is set to True
    return_as : {'dict', 'dataframe'}, default 'dict'
        Output format
    decimals : int, default 3
        Number of decimal places for rounding
    verbose : bool, default False
        If True, print test results to logger

    Returns
    -------
    result : dict or DataFrame
        Test results including:
        - test_method: Name of test
        - statistic: χ² statistic
        - pvalue: p-value
        - df: degrees of freedom (always 1)
        - b: count of (before=0, after=1)
        - c: count of (before=1, after=0)
        - odds_ratio: b / c
        - effect_size: odds ratio
        - effect_size_interpretation: interpretation
        - significant: whether to reject null hypothesis
        - stars: significance stars

    Notes
    -----
    McNemar's test is used for paired nominal data, testing whether row and column
    marginal frequencies are equal (marginal homogeneity).

    The test statistic is based on the discordant pairs (b and c):

    .. math::
        \\chi^2 = \\frac{(b - c)^2}{b + c}  \\quad \\text{(without correction)}

    .. math::
        \\chi^2 = \\frac{(|b - c| - 1)^2}{b + c}  \\quad \\text{(with correction)}

    **Null hypothesis:** The marginal proportions are equal (no change)
    **Alternative:** The marginal proportions differ (significant change)

    **Assumptions:**
    - Paired data (matched observations)
    - Binary outcomes for both conditions
    - Large enough sample (b + c ≥ 10 recommended for chi-square approximation)

    **Effect size (Odds Ratio):**
    OR = b / c
    - OR = 1: no change
    - OR > 1: increase (more transitions from 0→1 than 1→0)
    - OR < 1: decrease (more transitions from 1→0 than 0→1)

    Examples
    --------
    >>> import numpy as np
    >>> from scitex.stats.tests.categorical import test_mcnemar
    >>>
    >>> # Example: Treatment effectiveness (before/after)
    >>> # Rows: before, Columns: after
    >>> # [[no→no, no→yes],
    >>> #  [yes→no, yes→yes]]
    >>> observed = [[59, 6],   # 59 stayed negative, 6 improved
    ...             [16, 19]]  # 16 relapsed, 19 stayed positive
    >>>
    >>> result = test_mcnemar(observed, var_before='Before Treatment',
    ...                       var_after='After Treatment', plot=True)
    >>> print(f"χ² = {result['statistic']:.2f}, p = {result['pvalue']:.4f}")
    >>> print(f"Odds Ratio = {result['odds_ratio']:.2f}")

    References
    ----------
    .. [1] McNemar, Q. (1947). "Note on the sampling error of the difference between
           correlated proportions or percentages". Psychometrika, 12(2), 153-157.
    .. [2] Edwards, A. L. (1948). "Note on the correction for continuity in testing
           the significance of the difference between correlated proportions".
           Psychometrika, 13(3), 185-187.

    See Also
    --------
    test_chi2 : For independent (unpaired) categorical data
    test_fisher : For 2×2 tables with small expected frequencies
    """
    # Convert to numpy array
    if isinstance(observed, pd.DataFrame):
        observed_array = observed.values
    elif isinstance(observed, list):
        observed_array = np.array(observed)
    else:
        observed_array = np.asarray(observed)

    # Validate shape
    if observed_array.shape != (2, 2):
        raise ValueError(
            f"McNemar's test requires a 2×2 table, got shape {observed_array.shape}"
        )

    # Extract cells
    a, b = observed_array[0]
    c, d = observed_array[1]

    # Validate data types
    if not all(
        isinstance(x, (int, np.integer))
        or (isinstance(x, (float, np.floating)) and x == int(x))  # type: ignore[unreachable]
        for x in [a, b, c, d]
    ):
        raise ValueError("All values must be non-negative integers (counts)")

    if any(x < 0 for x in [a, b, c, d]):
        raise ValueError("All counts must be non-negative")

    a, b, c, d = int(a), int(b), int(c), int(d)

    # Perform McNemar's test
    # scipy.stats.contingency.mcnemar is not available in older scipy
    # We'll compute manually
    n_discordant = b + c

    if n_discordant == 0:
        # No discordant pairs - perfect agreement
        statistic = 0.0
        pvalue = 1.0
    else:
        if correction:
            # With continuity correction
            statistic = (abs(b - c) - 1.0) ** 2 / n_discordant
        else:
            # Without continuity correction
            statistic = (b - c) ** 2 / n_discordant

        # p-value from chi-square distribution with df=1
        pvalue = 1.0 - stats.chi2.cdf(statistic, df=1)

    # Compute odds ratio
    odds_ratio = mcnemar_odds_ratio(b, c)
    or_interpretation = interpret_mcnemar_or(odds_ratio)

    # Variable names
    var_before = var_before or "Before"
    var_after = var_after or "After"

    # Build result dictionary
    result = {
        "test_method": "McNemar's test",
        "var_before": var_before,
        "var_after": var_after,
        "statistic": round(float(statistic), decimals),
        "pvalue": round(float(pvalue), decimals + 1),
        "df": 1,
        "b": int(b),  # Changed (0→1)
        "c": int(c),  # Changed (1→0)
        "n_discordant": int(n_discordant),
        "odds_ratio": (
            round(float(odds_ratio), decimals)
            if np.isfinite(odds_ratio)
            else odds_ratio
        ),
        "effect_size": (
            round(float(odds_ratio), decimals)
            if np.isfinite(odds_ratio)
            else odds_ratio
        ),
        "effect_size_metric": "Odds ratio",
        "effect_size_interpretation": or_interpretation,
        "correction": correction,
        "alpha": alpha,
        "significant": pvalue < alpha,
        "stars": p2stars(pvalue),
    }

    # Log results if verbose
    if verbose:
        logger.info(
            f"McNemar: χ² = {statistic:.3f}, p = {pvalue:.4f} {p2stars(pvalue)}"
        )
        logger.info(f"OR = {odds_ratio:.3f}, {or_interpretation} (b={b}, c={c})")

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            fig, axes = stx.plt.subplots(1, 3, figsize=(12, 4))
            _plot_mcnemar_full(observed_array, result, var_before, var_after, axes)
        else:
            _plot_mcnemar_simple(observed_array, result, var_before, var_after, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)
    elif return_as not in ["dict", "dataframe"]:
        return convert_results(result, return_as=return_as)

    return result


def _plot_mcnemar_full(observed, result, var_before, var_after, axes):
    """Create 3-panel visualization for McNemar's test."""
    a, b = observed[0]
    c, d = observed[1]

    # Panel 1: Contingency table heatmap
    ax = axes[0]
    im = ax.imshow(observed, cmap="Blues", aspect="auto")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(observed[i, j]),
                ha="center",
                va="center",
                color="black",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    ax.set_xlabel(var_after)
    ax.set_ylabel(var_before)
    ax.set_title("Contingency Table")
    stx.plt.colorbar(im, ax=ax)

    # Panel 2: Discordant pairs comparison
    ax = axes[1]
    categories = ["0→1\n(b)", "1→0\n(c)"]
    counts = [b, c]

    bars = ax.bar(categories, counts)

    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Count")
    ax.set_title("Discordant Pairs")
    ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)

    # Panel 3: McNemar's Test stats
    ax = axes[2]
    ax.axis("off")
    ax.set_title("McNemar's Test")

    # Add stats text box
    stars_text = result["stars"].replace("ns", "$n$s")
    if np.isfinite(result["odds_ratio"]):
        text_str = (
            f"$\\chi^2$ = {result['statistic']:.3f}\n"
            f"$p$ = {result['pvalue']:.4f} {stars_text}\n"
            f"$b$ = {result['b']}, $c$ = {result['c']}\n"
            f"$OR$ = {result['odds_ratio']:.3f}"
        )
    else:
        text_str = (
            f"$\\chi^2$ = {result['statistic']:.3f}\n"
            f"$p$ = {result['pvalue']:.4f} {stars_text}\n"
            f"$b$ = {result['b']}, $c$ = {result['c']}"
        )
    ax.text(
        0.5,
        0.5,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        color="black",
    )


def _plot_mcnemar_simple(observed, result, var_before, var_after, ax):
    """Create simplified single-panel discordant pairs plot on given axes."""
    a, b = observed[0]
    c, d = observed[1]

    # Discordant pairs comparison
    categories = ["0→1\n(b)", "1→0\n(c)"]
    counts = [b, c]

    bars = ax.bar(categories, counts)

    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Count")
    ax.set_title("McNemar's Test")
    ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)

    # Add stats text box
    stars_text = result["stars"].replace("ns", "$n$s")
    if np.isfinite(result["odds_ratio"]):
        text_str = (
            f"$\\chi^2$ = {result['statistic']:.3f}\n"
            f"$p$ = {result['pvalue']:.4f} {stars_text}\n"
            f"$b$ = {result['b']}, $c$ = {result['c']}\n"
            f"$OR$ = {result['odds_ratio']:.3f}"
        )
    else:
        text_str = (
            f"$\\chi^2$ = {result['statistic']:.3f}\n"
            f"$p$ = {result['pvalue']:.4f} {stars_text}\n"
            f"$b$ = {result['b']}, $c$ = {result['c']}"
        )
    ax.text(
        0.02,
        0.98,
        text_str,
        transform=ax.transAxes,
        verticalalignment="top",
        color="black",
        fontsize=6,
    )


# Demo: python -m scitex.stats.tests.categorical._demo_mcnemar

# EOF
