#!/usr/bin/env python3
# Time-stamp: "2025-01-15 00:00:00 (ywatanabe)"
# File: ./scitex_repo/src/scitex/stats/tests/categorical/_test_fisher.py
# ----------------------------------------

r"""
Fisher's exact test for 2×2 contingency tables.

Tests association between two binary categorical variables with small sample sizes.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Tuple, Union

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


def odds_ratio_ci(
    a: int, b: int, c: int, d: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute confidence interval for odds ratio using log transformation.

    Parameters
    ----------
    a, b, c, d : int
        2×2 table: [[a, b], [c, d]]
    alpha : float
        Significance level

    Returns
    -------
    ci_lower, ci_upper : float
        Confidence interval bounds

    Notes
    -----
    Uses log transformation for asymptotic normality.
    Formula: log(OR) ± z * SE(log(OR))
    where SE(log(OR)) = sqrt(1/a + 1/b + 1/c + 1/d)
    """
    # Handle zero cells (add 0.5 continuity correction)
    if a == 0 or b == 0 or c == 0 or d == 0:
        a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_adj, b_adj, c_adj, d_adj = a, b, c, d

    or_val = (a_adj * d_adj) / (b_adj * c_adj)
    log_or = np.log(or_val)

    # Standard error of log(OR)
    se_log_or = np.sqrt(1 / a_adj + 1 / b_adj + 1 / c_adj + 1 / d_adj)

    # Z critical value
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # CI on log scale
    log_ci_lower = log_or - z_crit * se_log_or
    log_ci_upper = log_or + z_crit * se_log_or

    # Transform back to OR scale
    ci_lower = np.exp(log_ci_lower)
    ci_upper = np.exp(log_ci_upper)

    return float(ci_lower), float(ci_upper)


def interpret_odds_ratio(or_val: float) -> str:
    """
    Interpret odds ratio effect size.

    Parameters
    ----------
    or_val : float
        Odds ratio

    Returns
    -------
    interpretation : str
        Interpretation of effect size
    """
    if or_val == 1.0:
        return "no association"
    elif or_val > 1.0:
        if or_val < 1.5:
            return "weak positive association"
        elif or_val < 3.0:
            return "moderate positive association"
        elif or_val < 9.0:
            return "strong positive association"
        else:
            return "very strong positive association"
    else:  # or_val < 1.0
        inv_or = 1.0 / or_val
        if inv_or < 1.5:
            return "weak negative association"
        elif inv_or < 3.0:
            return "moderate negative association"
        elif inv_or < 9.0:
            return "strong negative association"
        else:
            return "very strong negative association"


def test_fisher(  # noqa: C901
    observed: Union[np.ndarray, pd.DataFrame, list],
    var_row: Optional[str] = None,
    var_col: Optional[str] = None,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    decimals: int = 3,
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    """
    Fisher's exact test for 2×2 contingency tables.

    Tests association between two binary categorical variables.
    Exact test (no large-sample approximation required).

    Parameters
    ----------
    observed : array-like or DataFrame
        2×2 contingency table as [[a, b], [c, d]]
        If DataFrame, row/column names used as variable names
    var_row : str, optional
        Name of row variable (default: 'row_variable')
    var_col : str, optional
        Name of column variable (default: 'col_variable')
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Alternative hypothesis:
        - 'two-sided': odds ratio ≠ 1
        - 'less': odds ratio < 1
        - 'greater': odds ratio > 1
    alpha : float, default 0.05
        Significance level for confidence interval
    plot : bool, default False
        If True, create visualization
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If provided, plot is set to True
    return_as : {'dict', 'dataframe'}, default 'dict'
        Return format
    decimals : int, default 3
        Number of decimal places for rounding
    verbose : bool, default False
        If True, print test results to logger

    Returns
    -------
    result : dict or DataFrame
        Test results with:
        - test_method: Name of test
        - statistic: Odds ratio
        - pvalue: Exact p-value
        - alternative: Alternative hypothesis
        - alpha: Significance level
        - significant: Whether result is significant
        - stars: Significance stars
        - effect_size: Odds ratio
        - effect_size_metric: 'Odds ratio'
        - effect_size_interpretation: Interpretation
        - ci_lower: Lower CI bound for odds ratio
        - ci_upper: Upper CI bound for odds ratio
        - n: Total sample size
        - var_row: Row variable name
        - var_col: Column variable name

    Notes
    -----
    Fisher's exact test computes exact probability of observed table
    (and more extreme tables) under independence assumption.

    H₀: Two binary variables are independent (OR = 1)
    H₁: Variables are associated (OR ≠ 1)

    **Odds Ratio (OR)**:
    For table [[a, b], [c, d]]:
    OR = (a × d) / (b × c)

    Interpretation:
    - OR = 1: No association
    - OR > 1: Positive association
    - OR < 1: Negative association

    **When to use**:
    - 2×2 contingency tables
    - Small sample sizes (any cell < 5)
    - Need exact p-value (not approximation)

    **Advantages over chi-square**:
    - Exact test (valid for any sample size)
    - No minimum expected frequency requirement
    - More powerful for small samples

    References
    ----------
    Fisher, R. A. (1922). On the interpretation of χ² from contingency
    tables, and the calculation of P. Journal of the Royal Statistical
    Society, 85(1), 87-94.

    Examples
    --------
    >>> import numpy as np
    >>> from scitex.stats.tests.categorical import test_fisher

    # Example 1: Small 2×2 table (treatment × outcome)
    >>> observed = [[8, 2], [1, 5]]
    >>> result = test_fisher(observed, var_row='Treatment', var_col='Response', plot=True)
    >>> print(result)

    # Example 2: Case-control study
    >>> exposed_cases = 12
    >>> unexposed_cases = 5
    >>> exposed_controls = 8
    >>> unexposed_controls = 20
    >>> observed = [[exposed_cases, unexposed_cases],
    ...             [exposed_controls, unexposed_controls]]
    >>> result = test_fisher(observed, var_row='Exposure', var_col='Disease')
    >>> print(f"OR = {result['statistic']:.2f}, 95% CI [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
    >>> print(f"p = {result['pvalue']:.4f}")

    # Example 3: One-tailed test (expect positive association)
    >>> observed = [[10, 2], [3, 8]]
    >>> result = test_fisher(observed, alternative='greater')
    >>> print(f"One-tailed p = {result['pvalue']:.4f}")

    # Example 4: Using pandas DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame([[15, 5], [3, 10]],
    ...                    index=['Group A', 'Group B'],
    ...                    columns=['Success', 'Failure'])
    >>> result = test_fisher(df, plot=True)

    # Example 5: Compare with chi-square
    >>> from scitex.stats.tests.categorical import test_chi2
    >>> observed = [[5, 10], [10, 5]]
    >>> fisher_result = test_fisher(observed)
    >>> chi2_result = test_chi2(observed)
    >>> print(f"Fisher's exact p = {fisher_result['pvalue']:.4f}")
    >>> print(f"Chi-square p = {chi2_result['pvalue']:.4f}")
    """
    # Convert to numpy array
    if isinstance(observed, pd.DataFrame):
        if var_row is None:
            var_row = observed.index.name or "row_variable"
        if var_col is None:
            var_col = observed.columns.name or "col_variable"
        observed = observed.values
    else:
        observed = np.asarray(observed)
        if var_row is None:
            var_row = "row_variable"
        if var_col is None:
            var_col = "col_variable"

    # Check dimensions
    if observed.shape != (2, 2):
        raise ValueError(
            f"Fisher's exact test requires 2×2 table (got {observed.shape})"
        )

    # Extract table values
    a, b = int(observed[0, 0]), int(observed[0, 1])
    c, d = int(observed[1, 0]), int(observed[1, 1])

    # Total sample size
    n = a + b + c + d

    if n == 0:
        raise ValueError("Contingency table is empty (sum = 0)")

    # Perform Fisher's exact test
    or_val, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative=alternative)
    or_val = float(or_val)
    pvalue = float(pvalue)

    # Compute confidence interval for odds ratio
    ci_lower, ci_upper = odds_ratio_ci(a, b, c, d, alpha)

    # Interpret effect size
    interpretation = interpret_odds_ratio(or_val)

    # Check significance
    significant = pvalue < alpha
    stars = p2stars(pvalue)

    # Build result
    result = {
        "test_method": "Fisher's exact test",
        "statistic": round(or_val, decimals),
        "pvalue": round(pvalue, decimals),
        "alternative": alternative,
        "alpha": alpha,
        "significant": significant,
        "stars": stars,
        "effect_size": round(or_val, decimals),
        "effect_size_metric": "Odds ratio",
        "effect_size_interpretation": interpretation,
        "ci_lower": round(ci_lower, decimals),
        "ci_upper": round(ci_upper, decimals),
        "n": n,
        "var_row": var_row,
        "var_col": var_col,
    }

    # Log results if verbose
    if verbose:
        logger.info(f"Fisher: OR = {or_val:.3f}, p = {pvalue:.4f} {p2stars(pvalue)}")
        logger.info(f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}], {interpretation}")

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            fig, axes = stx.plt.subplots(1, 2, figsize=(12, 5))
            _plot_fisher_full(
                [[a, b], [c, d]],
                or_val,
                pvalue,
                ci_lower,
                ci_upper,
                var_row,
                var_col,
                axes,
            )
        else:
            _plot_fisher_simple(
                [[a, b], [c, d]],
                or_val,
                pvalue,
                ci_lower,
                ci_upper,
                var_row,
                var_col,
                ax,
            )

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)
    elif return_as not in ["dict", "dataframe"]:
        return convert_results(result, return_as=return_as)

    return result


def _plot_fisher_full(
    observed, or_val, pvalue, ci_lower, ci_upper, var_row, var_col, axes
):
    """Create 2-panel visualization for Fisher's exact test."""
    observed = np.array(observed)

    # Panel 1: 2×2 table heatmap
    ax = axes[0]
    im = ax.imshow(observed, cmap="Blues", aspect="auto")
    ax.set_title("Observed")
    ax.set_xlabel(var_col)
    ax.set_ylabel(var_row)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["C1", "C2"])
    ax.set_yticklabels(["R1", "R2"])

    # Add values
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{observed[i, j]:.0f}",
                ha="center",
                va="center",
                color="white",
            )

    stx.plt.colorbar(im, ax=ax)

    # Panel 2: Odds ratio with confidence interval
    ax = axes[1]

    # Plot OR point estimate
    ax.plot([or_val], [0], "o", zorder=3)

    # Plot CI
    ax.plot([ci_lower, ci_upper], [0, 0], "-", zorder=2)
    ax.plot([ci_lower, ci_lower], [-0.1, 0.1], "-", zorder=2)
    ax.plot([ci_upper, ci_upper], [-0.1, 0.1], "-", zorder=2)

    # Add reference line at OR = 1
    ax.axvline(1, linestyle="--", alpha=0.5, label="OR = 1 (null)")

    # Set x-axis (log scale for OR)
    if or_val > 0:
        x_min = min(0.1, ci_lower * 0.5)
        x_max = max(10, ci_upper * 2)
        ax.set_xlim(x_min, x_max)
        ax.set_xscale("log")

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Odds Ratio (log scale)")

    ax.set_title("Fisher's Exact Test")
    ax.legend(loc="upper right")

    # Add stats text box
    stars_text = p2stars(pvalue).replace("ns", "$n$s")
    text_str = (
        f"$OR$ = {or_val:.3f} {stars_text}\n"
        f"$p$ = {pvalue:.4f}\n"
        f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]"
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


def _plot_fisher_simple(
    observed, or_val, pvalue, ci_lower, ci_upper, var_row, var_col, ax
):
    """Create simplified single-panel OR plot on given axes."""
    # Plot OR point estimate
    ax.plot([or_val], [0], "o", zorder=3)

    # Plot CI
    ax.plot([ci_lower, ci_upper], [0, 0], "-", zorder=2)
    ax.plot([ci_lower, ci_lower], [-0.1, 0.1], "-", zorder=2)
    ax.plot([ci_upper, ci_upper], [-0.1, 0.1], "-", zorder=2)

    # Add reference line at OR = 1
    ax.axvline(1, linestyle="--", alpha=0.5, label="OR = 1")

    # Set x-axis (log scale for OR)
    if or_val > 0:
        x_min = min(0.1, ci_lower * 0.5)
        x_max = max(10, ci_upper * 2)
        ax.set_xlim(x_min, x_max)
        ax.set_xscale("log")

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Odds Ratio (log scale)")

    ax.set_title("Fisher's Exact Test")
    ax.legend()

    # Add stats text box
    stars_text = p2stars(pvalue).replace("ns", "$n$s")
    text_str = (
        f"$OR$ = {or_val:.3f} {stars_text}\n"
        f"$p$ = {pvalue:.4f}\n"
        f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]"
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


# Demo: python -m scitex.stats.tests.categorical._demo_fisher

# EOF
