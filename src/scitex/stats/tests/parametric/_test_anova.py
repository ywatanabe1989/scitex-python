#!/usr/bin/env python3
# Timestamp: "2025-10-01 16:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/parametric/_test_anova.py
# ----------------------------------------
from __future__ import annotations

r"""
Functionalities:
  - Perform one-way ANOVA for independent samples
  - Compute eta-squared effect size
  - Generate box plots and distribution visualizations
  - Support flexible output formats (dict or DataFrame)
  - Automatic normality and homogeneity checking

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: Multiple independent samples (arrays or Series)
  - output: Test results (dict or DataFrame) and optional figure
"""

"""Imports"""
import os  # noqa: E402
from typing import List, Literal, Optional, Union  # noqa: E402

import matplotlib.axes  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import scitex as stx  # noqa: E402
from scitex.logging import getLogger  # noqa: E402
from scitex.stats._utils._formatters import fmt_stat, fmt_sym  # noqa: E402

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)

"""Functions"""


def test_anova(  # noqa: C901
    groups: Optional[List[Union[np.ndarray, pd.Series]]] = None,
    var_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    check_assumptions: bool = True,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    data: Union[pd.DataFrame, str, None] = None,
    value_col: Optional[str] = None,
    group_col: Optional[str] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    decimals: int = 3,
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    r"""
    Perform one-way ANOVA for independent samples.

    Parameters
    ----------
    groups : list of arrays
        List of sample arrays for each group (minimum 2 groups)
    var_names : list of str, optional
        Names for each group. If None, uses 'Group 1', 'Group 2', etc.
    alpha : float, default 0.05
        Significance level
    check_assumptions : bool, default True
        Whether to check normality and homogeneity assumptions
    plot : bool, default False
        Whether to generate visualization
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None and plot=True, creates new figure.
        If provided, automatically enables plotting.
    data : DataFrame, str, or None, optional
        DataFrame or CSV path. When provided with value_col and group_col,
        groups are extracted automatically (seaborn-style).
    value_col : str, optional
        Column containing measurement values (used with data=).
    group_col : str, optional
        Column containing group labels (used with data=).
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
        - test_method: 'One-way ANOVA'
        - statistic: F-statistic value
        - pvalue: p-value
        - stars: Significance stars
        - significant: Whether null hypothesis is rejected
        - effect_size: Eta-squared (η²)
        - effect_size_metric: 'eta-squared'
        - effect_size_interpretation: Interpretation of eta-squared
        - n_groups: Number of groups
        - n_samples: Sample sizes for each group
        - df_between: Degrees of freedom between groups
        - df_within: Degrees of freedom within groups
        - var_names: Group labels
        - assumptions_met: Whether assumptions are satisfied
        - H0: Null hypothesis description

    Notes
    -----
    One-way ANOVA (Analysis of Variance) tests whether samples from different
    groups have the same population mean.

    **Null Hypothesis (H0)**: All groups have equal population means

    **Alternative Hypothesis (H1)**: At least one group mean differs

    **Assumptions**:
    1. **Independence**: Observations within and between groups are independent
    2. **Normality**: Data in each group are normally distributed
       - Can be checked with test_shapiro()
       - Robust to moderate violations with large samples (n > 30 per group)
    3. **Homogeneity of variance**: Groups have equal population variances
       - Can be checked with Levene's test
       - If violated, consider Welch's ANOVA or non-parametric alternative

    **When assumptions are violated**:
    - Non-normality: Use test_kruskal() (Kruskal-Wallis test)
    - Unequal variances: Use Welch's ANOVA (not yet implemented)
    - Outliers present: Use test_kruskal() or remove outliers

    **F-Statistic**:

    .. math::
        F = \\frac{MS_{between}}{MS_{within}} = \\frac{SS_{between}/(k-1)}{SS_{within}/(N-k)}

    Where:
    - k: Number of groups
    - N: Total sample size
    - SS: Sum of squares
    - MS: Mean square

    **Effect Size (Eta-squared)**:

    .. math::
        \\eta^2 = \\frac{SS_{between}}{SS_{total}}

    Interpretation:
    - η² < 0.01:  negligible
    - η² < 0.06:  small
    - η² < 0.14:  medium
    - η² ≥ 0.14:  large

    **Post-hoc tests**:
    If significant, perform pairwise comparisons with correction:
    - test_ttest_ind() for all pairs (if assumptions met)
    - test_brunner_munzel() for all pairs (robust alternative)
    - correct_bonferroni() or correct_fdr() for multiple comparisons

    References
    ----------
    .. [1] Fisher, R. A. (1925). Statistical Methods for Research Workers.
           Oliver and Boyd.
    .. [2] Cohen, J. (1988). Statistical Power Analysis for the Behavioral
           Sciences (2nd ed.). Routledge.
    .. [3] Maxwell, S. E., & Delaney, H. D. (2004). Designing Experiments
           and Analyzing Data: A Model Comparison Perspective (2nd ed.).
           Psychology Press.

    Examples
    --------
    >>> # Three groups with different means
    >>> group1 = np.array([1, 2, 3, 4, 5])
    >>> group2 = np.array([3, 4, 5, 6, 7])
    >>> group3 = np.array([5, 6, 7, 8, 9])
    >>> result = test_anova([group1, group2, group3])
    >>> result['rejected']
    True

    >>> # With auto-created figure
    >>> result = test_anova(
    ...     [group1, group2, group3],
    ...     var_names=['Control', 'Treatment 1', 'Treatment 2'],
    ...     plot=True
    ... )

    >>> # Plot on existing axes
    >>> fig, ax = plt.subplots()
    >>> result = test_anova([group1, group2, group3], ax=ax)

    >>> # Export results
    >>> from scitex.stats.utils._normalizers import convert_results
    >>> convert_results(result, return_as='excel', path='anova_results.xlsx')
    """
    from scitex.stats._utils._effect_size import eta_squared, interpret_eta_squared
    from scitex.stats._utils._formatters import p2stars
    from scitex.stats._utils._normalizers import convert_results, force_dataframe
    from scitex.stats.tests.normality._test_shapiro import test_normality

    # Resolve groups from DataFrame (seaborn-style data= parameter)
    if data is not None and value_col is not None and group_col is not None:
        from scitex.stats._utils._csv_support import resolve_groups

        groups, group_names = resolve_groups(data, value_col, group_col)
        if var_names is None:
            var_names = group_names

    # Validate input
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")

    # Convert to numpy arrays and remove NaN
    groups = [np.asarray(g) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]

    # Generate default names if not provided
    if var_names is None:
        var_names = [f"Group {i + 1}" for i in range(len(groups))]

    if len(var_names) != len(groups):
        raise ValueError("Number of var_names must match number of groups")

    # Get sample sizes
    n_samples = [len(g) for g in groups]
    n_groups = len(groups)
    n_total = sum(n_samples)

    # Check assumptions if requested
    assumptions_met = True
    assumption_warnings = []

    if check_assumptions:
        # Check normality for each group
        normality_check = test_normality(
            *groups, var_names=var_names, alpha=alpha, warn=False
        )

        if not normality_check["all_normal"]:
            assumptions_met = False
            non_normal = [
                r["var_x"] for r in normality_check["results"] if not r["normal"]
            ]
            warning_msg = (
                f"Normality assumption violated for: {', '.join(non_normal)}. "
                "Consider using test_kruskal() (Kruskal-Wallis test) instead."
            )
            assumption_warnings.append(warning_msg)
            logger.warning(warning_msg)

        # Check homogeneity of variance (Levene's test)
        _, levene_p = stats.levene(*groups)

        if levene_p < alpha:
            assumptions_met = False
            warning_msg = (
                f"Homogeneity of variance violated (Levene's test: p={levene_p:.4f}). "
                "Consider using Welch's ANOVA or test_kruskal()."
            )
            assumption_warnings.append(warning_msg)
            logger.warning(warning_msg)

    # Perform one-way ANOVA
    f_result = stats.f_oneway(*groups)
    f_stat = float(f_result.statistic)
    pvalue = float(f_result.pvalue)

    # Determine rejection
    rejected = pvalue < alpha

    # Compute effect size (eta-squared)
    effect_size = eta_squared(groups)
    effect_size_interp = interpret_eta_squared(effect_size)

    # Compute degrees of freedom
    df_between = n_groups - 1
    df_within = n_total - n_groups

    # Compile results
    result = {
        "test_method": "One-way ANOVA",
        "statistic": round(f_stat, decimals),
        "stat_symbol": "F",
        "n_groups": n_groups,
        "n_samples": n_samples,
        "df_between": df_between,
        "df_within": df_within,
        "var_names": var_names,
        "pvalue": round(pvalue, decimals),
        "stars": p2stars(pvalue),
        "alpha": alpha,
        "significant": rejected,
        "effect_size": round(effect_size, decimals),
        "effect_size_metric": "eta-squared",
        "effect_size_interpretation": effect_size_interp,
        "assumptions_met": assumptions_met,
        "H0": "All groups have equal population means",
    }

    # Add assumption warnings if any
    if assumption_warnings:
        result["assumption_warnings"] = assumption_warnings

    # Add post-hoc recommendation if significant
    if result["significant"]:
        if assumptions_met:
            result["recommendation"] = (
                "Significant difference detected. Perform post-hoc pairwise comparisons "
                "with test_ttest_ind() and apply correction (correct_bonferroni or correct_fdr)."
            )
        else:
            result["recommendation"] = (
                "Significant difference detected, but assumptions violated. "
                "Consider using test_kruskal() or performing pairwise test_brunner_munzel() with correction."
            )
    else:
        result["recommendation"] = "No significant difference between groups."

    # Log results if verbose
    if verbose:
        logger.info(
            f"One-way ANOVA: F({df_between}, {df_within}) = {f_stat:.3f}, p = {pvalue:.4f} {p2stars(pvalue)}"
        )
        logger.info(f"η² = {effect_size:.3f} ({effect_size_interp})")

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            fig, ax = stx.plt.subplots()
        _plot_anova(groups, var_names, result, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)
    elif return_as not in ["dict", "dataframe"]:
        # Use universal converter for other formats
        return convert_results(result, return_as=return_as)

    return result


def _plot_anova(groups, var_names, result, ax):
    """Create violin+swarm visualization for ANOVA results on given axes."""
    from scitex.stats._plot_helpers import (
        significance_bracket,
        stats_text_box,
        violin_swarm,
    )

    positions = list(np.arange(1, len(groups) + 1))

    # Violin + swarm
    violin_swarm(ax, groups, positions, var_names)

    ax.set_title("One-way ANOVA")

    # Stats text box
    n_total = sum(result["n_samples"])
    lines = [
        fmt_stat(
            "F",
            result["statistic"],
            df=f"{result['df_between']}, {result['df_within']}",
        ),
        fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
        f"{fmt_sym('eta^2')} = {result['effect_size']:.3f}",
        f"{fmt_sym('n')} = {n_total}",
    ]
    stats_text_box(ax, lines)

    # Add significance bracket
    if result["significant"]:
        significance_bracket(ax, positions[0], positions[-1], result["stars"], groups)


# EOF
