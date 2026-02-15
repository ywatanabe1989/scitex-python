#!/usr/bin/env python3
# Timestamp: "2025-10-01 17:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/parametric/_test_anova_rm.py
# ----------------------------------------
from __future__ import annotations

r"""
Functionalities:
  - Perform repeated measures ANOVA for within-subjects designs
  - Test sphericity assumption (Mauchly's test)
  - Apply Greenhouse-Geisser correction when sphericity violated
  - Compute partial eta-squared effect size
  - Generate profile plots and distribution visualizations

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib, pingouin

IO:
  - input: Data in wide or long format (subjects × conditions)
  - output: Test results (dict or DataFrame) and optional figure
"""

"""Imports"""
import os  # noqa: E402
from typing import List, Literal, Optional, Tuple, Union  # noqa: E402

import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from scitex.stats._utils._formatters import p2stars  # noqa: E402

from ._anova_helpers import (  # noqa: E402
    greenhouse_geisser_epsilon,
    interpret_eta_squared,
    mauchly_sphericity,
)
from ._anova_helpers import partial_eta_squared as partial_eta_squared_rm  # noqa: E402

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

HAS_PLT = True

# Try importing pingouin for sphericity test
try:
    import pingouin as pg  # noqa: E402

    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False


def test_anova_rm(  # noqa: C901
    data: Union[np.ndarray, pd.DataFrame],
    subject_col: Optional[str] = None,
    condition_col: Optional[str] = None,
    value_col: Optional[str] = None,
    condition_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    correction: Literal["auto", "none", "gg"] = "auto",
    check_sphericity: bool = True,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    decimals: int = 3,
    verbose: bool = False,
) -> Union[dict, pd.DataFrame, Tuple]:
    r"""
    Perform repeated measures ANOVA for within-subjects designs.

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
    correction : {'auto', 'none', 'gg'}, default 'auto'
        Correction method:
        - 'auto': Apply GG correction if sphericity violated
        - 'none': No correction
        - 'gg': Always apply Greenhouse-Geisser correction
    check_sphericity : bool, default True
        Whether to test sphericity assumption
    plot : bool, default False
        Whether to generate profile plot
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
        - statistic: F-statistic
        - pvalue: p-value (possibly corrected)
        - df_effect: Degrees of freedom for effect
        - df_error: Degrees of freedom for error
        - effect_size: Partial eta-squared
        - sphericity_W: Mauchly's W (if checked)
        - sphericity_pvalue: Sphericity test p-value
        - sphericity_met: Whether sphericity assumption met
        - epsilon_gg: Greenhouse-Geisser epsilon
        - correction_applied: Which correction was applied
        - significant: Whether to reject null hypothesis

    If plot=True, returns tuple of (result, figure)

    Notes
    -----
    Repeated measures ANOVA tests whether the means differ across multiple
    conditions measured on the same subjects (within-subjects factor).

    **Null Hypothesis (H0)**: All condition means are equal

    **Assumptions**:
    1. **Independence of subjects**: Different subjects are independent
    2. **Normality**: Differences between conditions are normally distributed
    3. **Sphericity**: Variances of differences between all pairs of conditions
       are equal (tested with Mauchly's test)

    **Sphericity**:
    The sphericity assumption is unique to repeated measures ANOVA. If violated:
    - Greenhouse-Geisser correction: More conservative, use when ε < 0.75
    - Huynh-Feldt correction: Less conservative (not implemented)
    - Multivariate approach: MANOVA (not implemented)

    **Greenhouse-Geisser Correction**:
    Adjusts degrees of freedom by multiplying by epsilon (ε):
    - df_effect_adj = ε × df_effect
    - df_error_adj = ε × df_error

    **Effect Size (Partial η²)**:

    .. math::
        \\eta_p^2 = \\frac{SS_{effect}}{SS_{effect} + SS_{error}}

    Interpretation same as regular eta-squared:
    - < 0.01: negligible
    - < 0.06: small
    - < 0.14: medium
    - ≥ 0.14: large

    **Post-hoc tests**:
    If significant, use pairwise t-tests with correction:
    - test_ttest_rel() for all pairs
    - correct_bonferroni() or correct_holm() for multiple comparisons

    Examples
    --------
    >>> import numpy as np
    >>> from scitex.stats.tests.parametric import test_anova_rm
    >>>
    >>> # Wide format: subjects × conditions
    >>> data = np.array([
    ...     [5.2, 6.1, 7.3, 6.8],  # Subject 1
    ...     [4.8, 5.9, 6.7, 6.2],  # Subject 2
    ...     [5.5, 6.4, 7.1, 7.0],  # Subject 3
    ...     [4.9, 5.7, 6.9, 6.5],  # Subject 4
    ... ])
    >>>
    >>> result = test_anova_rm(
    ...     data,
    ...     condition_names=['Baseline', 'Week 1', 'Week 2', 'Week 3'],
    ...     plot=True
    ... )
    >>>
    >>> print(f"F = {result['statistic']:.2f}, p = {result['pvalue']:.4f}")
    >>> print(f"Sphericity met: {result['sphericity_met']}")
    >>> print(f"Partial η² = {result['effect_size']:.3f}")

    References
    ----------
    .. [1] Greenhouse, S. W., & Geisser, S. (1959). "On methods in the analysis
           of profile data". Psychometrika, 24(2), 95-112.
    .. [2] Mauchly, J. W. (1940). "Significance test for sphericity of a normal
           n-variate distribution". The Annals of Mathematical Statistics,
           11(2), 204-209.

    See Also
    --------
    test_anova : One-way ANOVA for independent samples
    test_friedman : Non-parametric alternative (no sphericity assumption)
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

    if n_conditions < 2:
        raise ValueError("Need at least 2 conditions for repeated measures ANOVA")

    if condition_names is None:
        condition_names = [f"Condition {i + 1}" for i in range(n_conditions)]

    # Compute ANOVA
    grand_mean = data_array.mean()
    subject_means = data_array.mean(axis=1)
    condition_means = data_array.mean(axis=0)

    # Sum of squares
    ss_total = np.sum((data_array - grand_mean) ** 2)
    ss_subjects = n_conditions * np.sum((subject_means - grand_mean) ** 2)
    ss_conditions = n_subjects * np.sum((condition_means - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_conditions

    # Degrees of freedom
    df_conditions = n_conditions - 1
    df_subjects = n_subjects - 1
    df_error = df_conditions * df_subjects

    # Mean squares
    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error

    # F-statistic
    F_stat = ms_conditions / ms_error

    # Initial p-value (uncorrected)
    pvalue = 1 - stats.f.cdf(F_stat, df_conditions, df_error)

    # Test sphericity
    sphericity_met = True
    sphericity_W = None
    sphericity_chi2 = None
    sphericity_pvalue = None
    epsilon_gg = None
    correction_applied = "none"

    if check_sphericity and n_conditions > 2:
        try:
            if HAS_PINGOUIN:
                # Use pingouin for robust sphericity test
                spher = pg.sphericity(data_array, method="mauchly")
                sphericity_W = spher[0]
                sphericity_chi2 = spher[1]
                sphericity_pvalue = spher[2]
            else:
                # Use our implementation
                sphericity_W, sphericity_chi2, sphericity_pvalue = mauchly_sphericity(
                    data_array
                )

            sphericity_met = sphericity_pvalue >= alpha

            # Compute Greenhouse-Geisser epsilon
            if HAS_PINGOUIN:
                epsilon_gg = pg.epsilon(data_array, correction="gg")
            else:
                epsilon_gg = greenhouse_geisser_epsilon(data_array)

            # Apply correction if needed
            if correction == "gg" or (correction == "auto" and not sphericity_met):
                # Adjust degrees of freedom
                df_conditions_adj = df_conditions * epsilon_gg
                df_error_adj = df_error * epsilon_gg
                pvalue = 1 - stats.f.cdf(F_stat, df_conditions_adj, df_error_adj)
                correction_applied = "greenhouse-geisser"
                df_conditions = df_conditions_adj
                df_error = df_error_adj

        except Exception as e:
            # If sphericity test fails, continue without it
            import warnings

            warnings.warn(
                f"Sphericity test failed: {e}. Proceeding without correction."
            )
            sphericity_met = None

    # Compute effect size (partial eta-squared)
    partial_eta2 = partial_eta_squared_rm(ss_conditions, ss_error)
    eta2_interpretation = interpret_eta_squared(partial_eta2)

    # Build result dictionary
    result = {
        "test": "Repeated Measures ANOVA",
        "statistic": round(float(F_stat), decimals),
        "pvalue": round(float(pvalue), decimals + 1),
        "df_effect": round(float(df_conditions), decimals),
        "df_error": round(float(df_error), decimals),
        "n_subjects": int(n_subjects),
        "n_conditions": int(n_conditions),
        "condition_names": condition_names,
        "effect_size": round(float(partial_eta2), decimals),
        "effect_size_metric": "partial_eta_squared",
        "effect_size_interpretation": eta2_interpretation,
        "alpha": alpha,
        "significant": pvalue < alpha,
        "stars": p2stars(pvalue),
    }

    # Add sphericity results
    if sphericity_W is not None:
        result["sphericity_W"] = round(float(sphericity_W), decimals)
        result["sphericity_chi2"] = round(float(sphericity_chi2), decimals)
        result["sphericity_pvalue"] = round(float(sphericity_pvalue), decimals + 1)
        result["sphericity_met"] = sphericity_met
        result["epsilon_gg"] = round(float(epsilon_gg), decimals)
        result["correction_applied"] = correction_applied

    # Log results if verbose
    if verbose:
        from scitex.logging import getLogger

        logger = getLogger(__name__)
        logger.info(
            f"Repeated Measures ANOVA: F({result['df_effect']:.1f}, {result['df_error']:.1f}) = {result['statistic']:.3f}, p = {result['pvalue']:.4f} {result['stars']}"
        )
        logger.info(
            f"Partial η² = {result['effect_size']:.3f} ({result['effect_size_interpretation']})"
        )
        if "sphericity_met" in result:
            logger.info(f"Sphericity met: {result['sphericity_met']}")

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    fig = None
    if plot and HAS_PLT:
        if ax is None:
            fig = _plot_anova_rm(data_array, condition_names, result)
        else:
            # Use provided axes (not fully implemented for 1x3 layout)
            fig = _plot_anova_rm(data_array, condition_names, result)

    # Return based on format
    if return_as == "dataframe":
        result_df = pd.DataFrame([result])
        if plot and fig is not None:
            return result_df, fig
        return result_df
    else:
        if plot and fig is not None:
            return result, fig
        return result


def _plot_anova_rm(data, condition_names, result):
    """Create visualization for repeated measures ANOVA."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    n_subjects, n_conditions = data.shape
    conditions = np.arange(n_conditions)

    # Panel 1: Profile plot (individual subjects)
    ax = axes[0]
    for i in range(n_subjects):
        ax.plot(
            conditions,
            data[i, :],
            marker="o",
            alpha=0.3,
            color="gray",
            linewidth=0.5,
        )

    # Add mean profile
    means = data.mean(axis=0)
    sems = data.std(axis=0) / np.sqrt(n_subjects)
    ax.plot(
        conditions,
        means,
        marker="o",
        color="red",
        linewidth=2.5,
        markersize=8,
        label="Mean",
        zorder=10,
    )
    ax.fill_between(
        conditions,
        means - sems,
        means + sems,
        alpha=0.3,
        color="red",
        label="±SEM",
    )

    ax.set_xticks(conditions)
    ax.set_xticklabels(condition_names, rotation=45, ha="right")
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Repeated Measures ANOVA", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Stats text box - top-left corner
    stars_text = result["stars"].replace("ns", "$n$s")
    text_str = (
        f"$F$({result['df_effect']:.1f}, {result['df_error']:.1f}) = {result['statistic']:.3f} {stars_text}\n"
        f"$p$ = {result['pvalue']:.4f}\n"
        f"$\\eta_p^2$ = {result['effect_size']:.3f}\n"
        f"$n$ = {result['n_subjects']}"
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

    # Panel 2: Box plots
    ax = axes[1]
    positions = np.arange(1, n_conditions + 1)
    _ = ax.boxplot(
        [data[:, i] for i in range(n_conditions)],
        positions=positions,
        widths=0.6,
        patch_artist=True,
    )

    # Color boxes
    for patch in _["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(condition_names, rotation=45, ha="right")
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Distribution by Condition", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Results summary
    ax = axes[2]
    ax.axis("off")

    result_text = "Repeated Measures ANOVA\n"
    result_text += "=" * 35 + "\n\n"
    result_text += f"F({result['df_effect']:.1f}, {result['df_error']:.1f}) = {result['statistic']:.3f}\n"
    result_text += f"p-value = {result['pvalue']:.4f} {result['stars']}\n\n"
    result_text += f"Partial η² = {result['effect_size']:.3f}\n"
    result_text += f"Interpretation: {result['effect_size_interpretation']}\n\n"

    if "sphericity_W" in result:
        result_text += "Sphericity Test:\n"
        result_text += f"  Mauchly's W = {result['sphericity_W']:.3f}\n"
        result_text += f"  p = {result['sphericity_pvalue']:.4f}\n"
        result_text += f"  Met: {'Yes' if result['sphericity_met'] else 'No'}\n\n"
        if result["correction_applied"] != "none":
            result_text += f"  ε_GG = {result['epsilon_gg']:.3f}\n"
            result_text += f"  Correction: {result['correction_applied']}\n\n"

    result_text += f"Subjects: {result['n_subjects']}\n"
    result_text += f"Conditions: {result['n_conditions']}\n"
    result_text += f"Significant (α={result['alpha']}): "
    result_text += "Yes" if result["significant"] else "No"

    ax.text(
        0.1,
        0.5,
        result_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    return fig


# Demo: python -m scitex.stats.tests.parametric._demo_anova_rm

# EOF
