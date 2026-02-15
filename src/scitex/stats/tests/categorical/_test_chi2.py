#!/usr/bin/env python3
# Timestamp: "2025-10-01 18:53:25 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/categorical/_test_chi2.py
# ----------------------------------------

r"""
Chi-square test of independence for categorical data.

Tests association between two categorical variables in a contingency table.
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


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """
    Compute Cramér's V effect size for chi-square test.

    Parameters
    ----------
    chi2 : float
        Chi-square statistic
    n : int
        Total sample size
    r : int
        Number of rows
    c : int
        Number of columns

    Returns
    -------
    v : float
        Cramér's V (0 to 1)

    Notes
    -----
    Formula: V = sqrt(χ² / (n × (min(r,c) - 1)))  # noqa: D301

    Interpretation (Cramér, 1946):
    For df* = min(r-1, c-1):
    - df*=1: small=0.10, medium=0.30, large=0.50
    - df*=2: small=0.07, medium=0.21, large=0.35
    - df*=3: small=0.06, medium=0.17, large=0.29
    """
    if n == 0:
        return 0.0

    min_dim = min(r, c)
    if min_dim <= 1:
        return 0.0

    v = np.sqrt(chi2 / (n * (min_dim - 1)))
    return float(v)


def interpret_cramers_v(v: float, df_star: int) -> str:
    """
    Interpret Cramér's V effect size.

    Parameters
    ----------
    v : float
        Cramér's V
    df_star : int
        min(rows-1, cols-1)

    Returns
    -------
    interpretation : str
        'negligible', 'small', 'medium', or 'large'
    """
    if df_star == 1:
        thresholds = (0.10, 0.30, 0.50)
    elif df_star == 2:
        thresholds = (0.07, 0.21, 0.35)
    else:  # df_star >= 3
        thresholds = (0.06, 0.17, 0.29)

    if v < thresholds[0]:
        return "negligible"
    elif v < thresholds[1]:
        return "small"
    elif v < thresholds[2]:
        return "medium"
    else:
        return "large"


def test_chi2(  # noqa: C901
    observed: Union[np.ndarray, pd.DataFrame],
    var_row: Optional[str] = None,
    var_col: Optional[str] = None,
    alpha: float = 0.05,
    correction: bool = True,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    decimals: int = 3,
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    """
    Chi-square test of independence for contingency tables.

    Tests whether two categorical variables are independent.

    Parameters
    ----------
    observed : array-like or DataFrame
        Observed frequencies as contingency table (rows × columns)
        If DataFrame, row/column names used as variable names
    var_row : str, optional
        Name of row variable (default: 'row_variable')
    var_col : str, optional
        Name of column variable (default: 'col_variable')
    alpha : float, default 0.05
        Significance level
    correction : bool, default True
        Apply Yates' continuity correction for 2×2 tables
    plot : bool, default False
        If True, create mosaic plot visualization
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
        - statistic: Chi-square statistic (χ²)
        - pvalue: p-value
        - df: Degrees of freedom
        - alpha: Significance level
        - significant: Whether result is significant
        - stars: Significance stars
        - effect_size: Cramér's V
        - effect_size_metric: "Cramér's V"
        - effect_size_interpretation: Interpretation
        - n: Total sample size
        - expected_min: Minimum expected frequency
        - var_row: Row variable name
        - var_col: Column variable name

    Notes
    -----
    Chi-square test of independence tests:
    H₀: Two categorical variables are independent
    H₁: Two categorical variables are associated

    Test statistic:
    χ² = Σ[(O - E)² / E]
    where O = observed frequencies, E = expected frequencies

    Assumptions:
    1. Independence of observations
    2. Expected frequencies ≥ 5 in at least 80% of cells
    3. No expected frequencies < 1

    For 2×2 tables with small expected frequencies, use Fisher's exact test instead.

    Cramér's V measures strength of association (0 to 1):
    - 0 = no association
    - 1 = perfect association

    References
    ----------
    Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.

    Examples
    --------
    >>> import numpy as np
    >>> from scitex.stats.tests.categorical import test_chi2

    # Example 1: 2×2 contingency table (treatment × outcome)
    >>> observed = np.array([[30, 10], [20, 40]])
    >>> result = test_chi2(observed, var_row='Treatment', var_col='Outcome', plot=True)
    >>> print(result)

    # Example 2: Using DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame([[12, 8, 5], [15, 20, 10]],
    ...                    index=['Group A', 'Group B'],
    ...                    columns=['Low', 'Med', 'High'])
    >>> result = test_chi2(df, plot=True)

    # Example 3: Test gender × preference association
    >>> observed = np.array([
    ...     [20, 30, 15],  # Male: product A, B, C
    ...     [25, 20, 40]   # Female: product A, B, C
    ... ])
    >>> result = test_chi2(observed, var_row='Gender', var_col='Product', plot=True)
    >>> print(f"χ² = {result['statistic']:.2f}, p = {result['pvalue']:.4f}")
    >>> print(f"Cramér's V = {result['effect_size']:.3f} ({result['effect_size_interpretation']})")

    # Example 4: Small expected frequencies warning
    >>> observed = np.array([[2, 8], [3, 7]])  # Small counts
    >>> result = test_chi2(observed)

    # Example 5: Export to various formats
    >>> result = test_chi2(observed, return_as='dataframe')
    >>> convert_results(result, return_as='latex', path='chi2_test.tex')
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
    if observed.ndim != 2:
        raise ValueError(f"Contingency table must be 2D (got {observed.ndim}D)")

    rows, cols = observed.shape
    if rows < 2 or cols < 2:
        raise ValueError(f"Need at least 2×2 table (got {rows}×{cols})")

    # Total sample size
    n = int(np.sum(observed))

    if n == 0:
        raise ValueError("Contingency table is empty (sum = 0)")

    # Perform chi-square test
    # For 2×2 tables, apply Yates' correction if requested
    if rows == 2 and cols == 2 and correction:
        chi2_result = stats.chi2_contingency(observed, correction=True)
    else:
        chi2_result = stats.chi2_contingency(observed, correction=False)

    chi2_stat, pvalue, dof, expected = chi2_result
    chi2_stat = float(chi2_stat)
    pvalue = float(pvalue)
    dof = int(dof)

    # Compute Cramér's V effect size
    v = cramers_v(chi2_stat, n, rows, cols)
    df_star = min(rows - 1, cols - 1)
    interpretation = interpret_cramers_v(v, df_star)

    # Check assumptions
    expected_min = float(np.min(expected))
    expected_lt5 = np.sum(expected < 5)
    expected_lt1 = np.sum(expected < 1)

    warnings = []
    assumptions_met = True

    if expected_lt1 > 0:
        warnings.append(f"{expected_lt1} cells have expected frequency < 1")
        assumptions_met = False

    if expected_lt5 > 0.2 * expected.size:
        pct = 100 * expected_lt5 / expected.size
        warnings.append(
            f"{expected_lt5}/{expected.size} cells ({pct:.1f}%) have expected frequency < 5"
        )
        assumptions_met = False

    if not assumptions_met:
        if rows == 2 and cols == 2:
            warnings.append(
                "Consider using Fisher's exact test for 2×2 table with small counts"
            )

    # Check significance
    significant = pvalue < alpha
    stars = p2stars(pvalue)

    # Build result
    result = {
        "test_method": "Chi-square test of independence",
        "statistic": round(chi2_stat, decimals),
        "stat_symbol": "χ²",
        "pvalue": round(pvalue, decimals),
        "df": dof,
        "alpha": alpha,
        "significant": significant,
        "stars": stars,
        "effect_size": round(v, decimals),
        "effect_size_metric": "Cramér's V",
        "effect_size_interpretation": interpretation,
        "n": n,
        "n_rows": rows,
        "n_cols": cols,
        "expected_min": round(expected_min, decimals),
        "assumptions_met": assumptions_met,
        "var_row": var_row,
        "var_col": var_col,
    }

    if warnings:
        result["warnings"] = "; ".join(warnings)

    # Log results if verbose
    if verbose:
        logger.info(
            f"Chi-square: χ² = {chi2_stat:.3f}, df = {dof}, p = {pvalue:.4f} {p2stars(pvalue)}"
        )
        logger.info(f"Cramér's V = {v:.3f} ({interpretation})")
        if warnings:
            logger.warning(f"⚠ {'; '.join(warnings)}")

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            # For chi2, we need 3 panels, so create a figure with subplots
            fig, axes = stx.plt.subplots(1, 3, figsize=(15, 5))
            _plot_chi2_full(
                observed,
                expected,
                chi2_stat,
                pvalue,
                v,
                var_row,
                var_col,
                axes,
            )
        else:
            # If single ax provided, create simplified single-panel plot
            _plot_chi2_simple(
                observed, expected, chi2_stat, pvalue, v, var_row, var_col, ax
            )

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)
    elif return_as not in ["dict", "dataframe"]:
        return convert_results(result, return_as=return_as)

    return result


def _plot_chi2_full(observed, expected, chi2_stat, pvalue, v, var_row, var_col, axes):
    """Create 3-panel visualization for chi-square test."""
    rows, cols = observed.shape

    # Panel 1: Observed frequencies heatmap
    ax = axes[0]
    im1 = ax.imshow(observed, cmap="Blues", aspect="auto")
    ax.set_title("Observed")
    ax.set_xlabel(var_col)
    ax.set_ylabel(var_row)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f"C{i + 1}" for i in range(cols)])
    ax.set_yticklabels([f"R{i + 1}" for i in range(rows)])

    # Add values
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, f"{observed[i, j]:.0f}", ha="center", va="center")

    stx.plt.colorbar(im1, ax=ax)

    # Panel 2: Expected frequencies heatmap
    ax = axes[1]
    im2 = ax.imshow(expected, cmap="Oranges", aspect="auto")
    ax.set_title("Expected")
    ax.set_xlabel(var_col)
    ax.set_ylabel(var_row)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f"C{i + 1}" for i in range(cols)])
    ax.set_yticklabels([f"R{i + 1}" for i in range(rows)])

    # Add values
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, f"{expected[i, j]:.1f}", ha="center", va="center")

    stx.plt.colorbar(im2, ax=ax)

    # Panel 3: Residuals (standardized)
    ax = axes[2]
    residuals = (observed - expected) / np.sqrt(expected)
    vmax = max(abs(residuals.min()), abs(residuals.max()))
    im3 = ax.imshow(residuals, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_title("Chi-Square Test")
    ax.set_xlabel(var_col)
    ax.set_ylabel(var_row)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f"C{i + 1}" for i in range(cols)])
    ax.set_yticklabels([f"R{i + 1}" for i in range(rows)])

    # Add values
    for i in range(rows):
        for j in range(cols):
            color = "white" if abs(residuals[i, j]) > vmax / 2 else "black"
            ax.text(
                j,
                i,
                f"{residuals[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
            )

    stx.plt.colorbar(im3, ax=ax)

    # Add stats text box
    stars_text = p2stars(pvalue).replace("ns", "$n$s")
    text_str = (
        f"$\\chi^2$ = {chi2_stat:.3f}\n$p$ = {pvalue:.4f} {stars_text}\n$V$ = {v:.3f}"
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


def _plot_chi2_simple(observed, expected, chi2_stat, pvalue, v, var_row, var_col, ax):
    """Create simplified single-panel residuals plot on given axes."""
    rows, cols = observed.shape

    # Show standardized residuals
    residuals = (observed - expected) / np.sqrt(expected)
    vmax = max(abs(residuals.min()), abs(residuals.max()))
    im = ax.imshow(residuals, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_title("Chi-Square Test")
    ax.set_xlabel(var_col)
    ax.set_ylabel(var_row)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f"C{i + 1}" for i in range(cols)])
    ax.set_yticklabels([f"R{i + 1}" for i in range(rows)])

    # Add values
    for i in range(rows):
        for j in range(cols):
            color = "white" if abs(residuals[i, j]) > vmax / 2 else "black"
            ax.text(
                j,
                i,
                f"{residuals[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
            )

    stx.plt.colorbar(im, ax=ax)

    # Add stats text box
    stars_text = p2stars(pvalue).replace("ns", "$n$s")
    text_str = (
        f"$\\chi^2$ = {chi2_stat:.3f}\n$p$ = {pvalue:.4f} {stars_text}\n$V$ = {v:.3f}"
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


# Demo: python -m scitex.stats.tests.categorical._demo_chi2

# EOF
