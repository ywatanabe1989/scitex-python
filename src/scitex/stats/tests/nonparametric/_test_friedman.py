#!/usr/bin/env python3
# Timestamp: "2025-10-01 22:43:58 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/nonparametric/_test_friedman.py

r"""Friedman test for repeated measures (non-parametric).

Functionalities:
  - Perform Friedman test for repeated measures (non-parametric)
  - Non-parametric alternative to repeated measures ANOVA
  - Test differences across 3+ related samples
  - Compute Kendall's W (coefficient of concordance)
  - Generate rank-based visualizations

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: Data in wide or long format (subjects × conditions)
  - output: Test results (dict or DataFrame) and optional figure
"""

from __future__ import annotations

import os
from typing import List, Literal, Optional, Union

import matplotlib.axes
import numpy as np
import pandas as pd
from scipy import stats

import scitex as stx
from scitex.logging import getLogger
from scitex.stats._utils._formatters import fmt_stat, p2stars

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)


def kendall_w(ranks: np.ndarray) -> float:
    """
    Compute Kendall's W (coefficient of concordance).

    Parameters
    ----------
    ranks : array, shape (n_subjects, n_conditions)
        Rank matrix

    Returns
    -------
    W : float
        Kendall's W (0 to 1)

    Notes
    -----
    W = 0: No agreement among subjects
    W = 1: Complete agreement among subjects
    """
    n, k = ranks.shape

    # Sum of ranks for each condition
    R = ranks.sum(axis=0)

    # Mean of rank sums
    R_mean = R.mean()

    # Sum of squared deviations
    S = np.sum((R - R_mean) ** 2)

    # Kendall's W
    W = (12 * S) / (n**2 * (k**3 - k))

    return float(W)


def interpret_kendall_w(W: float) -> str:
    """Interpret Kendall's W effect size."""
    if W < 0.1:
        return "negligible agreement"
    elif W < 0.3:
        return "weak agreement"
    elif W < 0.5:
        return "moderate agreement"
    elif W < 0.7:
        return "strong agreement"
    else:
        return "very strong agreement"


def test_friedman(  # noqa: C901
    data: Union[np.ndarray, pd.DataFrame],
    subject_col: Optional[str] = None,
    condition_col: Optional[str] = None,
    value_col: Optional[str] = None,
    condition_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    decimals: int = 3,
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    r"""
    Perform Friedman test for repeated measures (non-parametric).

    Non-parametric alternative to repeated measures ANOVA. Tests whether
    distributions differ across 3+ related samples using ranks.

    Parameters
    ----------
    data : array or DataFrame
        - If array: shape (n_subjects, n_conditions), wide format
        - If DataFrame with subject_col/condition_col: long format
        - If DataFrame without: wide format (rows=subjects, cols=conditions)
    subject_col : str, optional
        Column name for subject IDs (long format)
    condition_col : str, optional
        Column name for conditions (long format)
    value_col : str, optional
        Column name for values (long format)
    condition_names : list of str, optional
        Names for conditions (wide format)
    alpha : float, default 0.05
        Significance level
    plot : bool, default False
        Whether to generate visualization
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None and plot=True, creates new figure.
        If provided, automatically enables plotting.
    return_as : {'dict', 'dataframe'}, default 'dict'
        Output format
    decimals : int, default 3
        Number of decimal places for rounding
    verbose : bool, default False
        Whether to print test results

    Returns
    -------
    result : dict or DataFrame
        Test results including:
        - statistic: Chi-square statistic (Friedman's χ²)
        - pvalue: p-value
        - df: Degrees of freedom (k - 1)
        - kendall_w: Kendall's W (coefficient of concordance)
        - effect_size: Kendall's W
        - effect_size_interpretation: interpretation
        - n_subjects: Number of subjects
        - n_conditions: Number of conditions
        - mean_ranks: Mean rank for each condition
        - significant: Whether to reject null hypothesis

    Notes
    -----
    The Friedman test is the non-parametric alternative to repeated measures
    ANOVA. It is used when:
    - Normality assumption is violated
    - Data are ordinal (e.g., Likert scales)
    - Sample sizes are small

    **Null Hypothesis (H0)**: All conditions have the same distribution

    **Alternative Hypothesis (H1)**: At least one condition differs

    **Procedure**:
    1. Rank observations within each subject (across conditions)
    2. Compute sum of ranks for each condition
    3. Calculate test statistic based on rank sums

    **Test Statistic**:

    .. math::
        \chi^2_F = \frac{12}{nk(k+1)} \sum_{j=1}^{k} R_j^2 - 3n(k+1)

    Where:
    - n: Number of subjects
    - k: Number of conditions
    - R_j: Sum of ranks for condition j

    **Effect Size (Kendall's W)**:

    .. math::
        W = \frac{12 \sum_{j=1}^{k}(R_j - \bar{R})^2}{n^2(k^3 - k)}

    Interpretation:
    - W < 0.1: negligible agreement
    - W < 0.3: weak agreement
    - W < 0.5: moderate agreement
    - W < 0.7: strong agreement
    - W ≥ 0.7: very strong agreement

    **Assumptions**:
    - Paired/repeated observations (same subjects)
    - At least ordinal scale data
    - 3+ conditions (for 2 conditions, use Wilcoxon signed-rank test)

    **Post-hoc tests**:
    If significant:
    - Pairwise Wilcoxon signed-rank tests
    - Apply corrections: correct_bonferroni(), correct_holm()

    **Advantages**:
    - No normality assumption
    - Robust to outliers
    - Works with ordinal data
    - No sphericity assumption

    **Disadvantages**:
    - Less powerful than RM-ANOVA when assumptions are met
    - Requires at least ordinal data
    - Sensitive to ties

    Examples
    --------
    >>> import numpy as np
    >>> from scitex.stats.tests.nonparametric import test_friedman
    >>>
    >>> # Example: Pain ratings (ordinal) across 4 time points
    >>> data = np.array([
    ...     [7, 6, 5, 4],  # Subject 1
    ...     [8, 7, 6, 5],  # Subject 2
    ...     [6, 5, 4, 3],  # Subject 3
    ...     [9, 8, 7, 6],  # Subject 4
    ... ])
    >>>
    >>> result = test_friedman(
    ...     data,
    ...     condition_names=['Baseline', '1 week', '2 weeks', '3 weeks'],
    ...     plot=True
    ... )
    >>>
    >>> print(f"χ² = {result['statistic']:.2f}, p = {result['pvalue']:.4f}")
    >>> print(f"Kendall's W = {result['kendall_w']:.3f}")

    References
    ----------
    .. [1] Friedman, M. (1937). "The use of ranks to avoid the assumption of
           normality implicit in the analysis of variance". Journal of the
           American Statistical Association, 32(200), 675-701.
    .. [2] Kendall, M. G., & Babington Smith, B. (1939). "The problem of m
           rankings". The Annals of Mathematical Statistics, 10(3), 275-287.

    See Also
    --------
    test_anova_rm : Parametric alternative (repeated measures ANOVA)
    test_wilcoxon : For 2 related samples
    test_kruskal : For 3+ independent samples
    """
    # Convert data to wide format array
    if isinstance(data, pd.DataFrame):
        if (
            subject_col is not None
            and condition_col is not None
            and value_col is not None
        ):
            # Long format - pivot to wide
            data_wide = data.pivot(
                index=subject_col, columns=condition_col, values=value_col
            )
            data_array = data_wide.values
            if condition_names is None:
                condition_names = list(data_wide.columns)
        else:
            # Already wide format
            data_array = data.values
            if condition_names is None:
                condition_names = list(data.columns)
    else:
        data_array = np.asarray(data)
        if data_array.ndim != 2:
            raise ValueError("Data must be 2D (subjects × conditions)")

    n_subjects, n_conditions = data_array.shape

    if n_conditions < 3:
        raise ValueError(
            "Friedman test requires at least 3 conditions. Use test_wilcoxon for 2 conditions."
        )

    if n_subjects < 2:
        raise ValueError("Need at least 2 subjects")

    if condition_names is None:
        condition_names = [f"Condition {i + 1}" for i in range(n_conditions)]

    # Perform Friedman test
    statistic, pvalue = stats.friedmanchisquare(*data_array.T)

    # Compute ranks for each subject (across conditions)
    ranks = np.zeros_like(data_array)
    for i in range(n_subjects):
        ranks[i, :] = stats.rankdata(data_array[i, :])

    # Compute mean ranks for each condition
    mean_ranks = ranks.mean(axis=0)

    # Compute Kendall's W
    W = kendall_w(ranks)
    W_interpretation = interpret_kendall_w(W)

    # Degrees of freedom
    df = n_conditions - 1

    # Build result dictionary
    result = {
        "test": "Friedman test",
        "statistic": round(float(statistic), decimals),
        "pvalue": round(float(pvalue), decimals + 1),
        "df": int(df),
        "kendall_w": round(float(W), decimals),
        "effect_size": round(float(W), decimals),
        "effect_size_metric": "kendall_w",
        "effect_size_interpretation": W_interpretation,
        "n_subjects": int(n_subjects),
        "n_conditions": int(n_conditions),
        "condition_names": condition_names,
        "mean_ranks": [round(float(r), decimals) for r in mean_ranks],
        "alpha": alpha,
        "significant": pvalue < alpha,
        "stars": p2stars(pvalue),
    }

    # Log results if verbose
    if verbose:
        logger.info(
            f"Friedman: χ² = {statistic:.3f}, p = {pvalue:.4f} {p2stars(pvalue)}"
        )
        logger.info(f"Kendall's W = {W:.3f} ({W_interpretation})")

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            _fig, ax = stx.plt.subplots()
        _plot_friedman(data_array, ranks, result, condition_names, ax)

    # Return based on format
    if return_as == "dataframe":
        result_df = pd.DataFrame([result])
        return result_df
    else:
        return result


def _plot_friedman(data, ranks, result, condition_names, ax):
    """Create violin+swarm visualization on given axes."""
    from scitex.stats._plot_helpers import (
        significance_bracket,
        stats_text_box,
        violin_swarm,
    )

    n_subjects, n_conditions = data.shape
    positions = list(range(n_conditions))
    data_list = [data[:, i] for i in range(n_conditions)]

    violin_swarm(ax, data_list, positions, condition_names)

    if result["significant"]:
        significance_bracket(ax, 0, n_conditions - 1, result["stars"], data_list)

    stats_text_box(
        ax,
        [
            fmt_stat("chi2", result["statistic"]),
            fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
            fmt_stat("W", result["kendall_w"]),
        ],
    )

    ax.set_xticklabels(condition_names, rotation=45, ha="right")
    ax.set_title("Friedman Test")
    ax.grid(True, alpha=0.3, axis="y")


# EOF
