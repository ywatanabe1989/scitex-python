#!/usr/bin/env python3
# Timestamp: "2025-10-01 15:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/_test_ttest_ind.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------


"""
Functionalities:
  - Perform independent samples t-test
  - Compute effect size (Cohen's d) and statistical power
  - Generate visualizations with significance indicators
  - Support flexible output formats (dict or DataFrame)

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: Two samples (arrays or Series)
  - output: Test results (dict or DataFrame) and optional figure
"""

"""Imports"""
import argparse  # noqa: E402
from typing import Literal, Optional, Union  # noqa: E402

import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import scitex as stx  # noqa: E402
from scitex.logging import getLogger  # noqa: E402
from scitex.stats._utils._formatters import fmt_stat, fmt_sym  # noqa: E402

logger = getLogger(__name__)

"""Functions"""


def test_ttest_ind(
    x: Union[np.ndarray, pd.Series, str],
    y: Union[np.ndarray, pd.Series, str],
    var_x: str = "x",
    var_y: str = "y",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    equal_var: bool = True,
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    data: Union[pd.DataFrame, str, None] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    r"""
    Perform independent samples t-test.

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
        - 'two-sided': means are different
        - 'greater': mean of x is greater than y
        - 'less': mean of x is less than y
    equal_var : bool, default True
        Assume equal population variances (Student's t-test)
        If False, use Welch's t-test
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
        - test_method: Name of test performed
        - statistic: t-statistic value
        - pvalue: p-value
        - stars: Significance stars
        - significant: Whether null hypothesis is rejected
        - effect_size: Cohen's d
        - power: Statistical power
        - n_x, n_y: Sample sizes
        - var_x, var_y: Variable labels
        - H0: Null hypothesis description

    Notes
    -----
    The independent samples t-test compares means of two independent groups.

    Null hypothesis: μ_x = μ_y
    Alternative (two-sided): μ_x ≠ μ_y

    The t-statistic is computed as:

    .. math::
        t = \\frac{\\bar{x} - \\bar{y}}{s_p \\sqrt{\\frac{1}{n_x} + \\frac{1}{n_y}}}

    where :math:`s_p` is the pooled standard deviation.

    For Welch's t-test (unequal variances), the denominator uses separate
    variances and degrees of freedom are adjusted.

    References
    ----------
    .. [1] Student (1908). "The Probable Error of a Mean". Biometrika, 6(1), 1-25.
    .. [2] Welch, B. L. (1947). "The generalization of 'Student's' problem when
           several different population variances are involved". Biometrika, 34(1-2), 28-35.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 3, 4, 5, 6])
    >>> result = test_ttest_ind(x, y)
    >>> result['pvalue']
    0.109...

    >>> # With auto-created figure
    >>> result = test_ttest_ind(x, y, plot=True)

    >>> # Plot on existing axes
    >>> fig, ax = plt.subplots()
    >>> result = test_ttest_ind(x, y, ax=ax)

    >>> # As DataFrame
    >>> df = test_ttest_ind(x, y, return_as='dataframe')
    >>> df['stars'].iloc[0]
    'ns'
    """
    # Resolve column names from DataFrame (seaborn-style data= parameter)
    if data is not None:
        from scitex.stats._utils._csv_support import resolve_columns

        resolved = resolve_columns(data, x=x, y=y)
        x, y = resolved["x"], resolved["y"]

    from scitex.stats._utils._effect_size import cohens_d
    from scitex.stats._utils._formatters import p2stars
    from scitex.stats._utils._normalizers import force_dataframe
    from scitex.stats._utils._power import power_ttest

    # Convert to numpy arrays and remove NaN
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n_x = len(x)
    n_y = len(y)

    # Perform t-test
    t_result = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
    t_stat = float(t_result.statistic)
    pvalue = float(t_result.pvalue)

    # Compute effect size
    from scitex.stats._utils._effect_size import interpret_cohens_d

    effect_size = cohens_d(x, y, paired=False)
    effect_size_interpretation = interpret_cohens_d(effect_size)

    # Compute statistical power
    power = power_ttest(
        effect_size=abs(effect_size),
        n1=n_x,
        n2=n_y,
        alpha=alpha,
        alternative=alternative,
        test_type="two-sample",
    )

    # Determine test method name
    if equal_var:
        test_method = "Student's t-test (independent)"
    else:
        test_method = "Welch's t-test (independent)"

    # Create null hypothesis description
    if alternative == "two-sided":
        H0 = f"μ({var_x}) = μ({var_y})"
    elif alternative == "greater":
        H0 = f"μ({var_x}) ≤ μ({var_y})"
    else:  # less
        H0 = f"μ({var_x}) ≥ μ({var_y})"

    # Compile results
    result = {
        "test_method": test_method,
        "statistic": t_stat,
        "stat_symbol": "t",
        "alternative": alternative,
        "n_x": n_x,
        "n_y": n_y,
        "var_x": var_x,
        "var_y": var_y,
        "pvalue": pvalue,
        "stars": p2stars(pvalue),
        "alpha": alpha,
        "significant": pvalue < alpha,
        "effect_size": effect_size,
        "effect_size_metric": "Cohen's d",
        "effect_size_interpretation": effect_size_interpretation,
        "power": power,
        "H0": H0,
    }

    # Log results if verbose
    if verbose:
        logger.info(
            f"{test_method}: t = {t_stat:.3f}, p = {pvalue:.4f} {p2stars(pvalue)}"
        )
        logger.info(
            f"Cohen's d = {effect_size:.3f} ({effect_size_interpretation}), power = {power:.3f}"
        )

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            fig, ax = stx.plt.subplots()
        _plot_ttest_ind(x, y, var_x, var_y, result, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)

    return result


def _plot_ttest_ind(x, y, var_x, var_y, result, ax):
    """Create violin+swarm visualization for independent t-test on given axes."""
    from scitex.stats._plot_helpers import (
        significance_bracket,
        stats_text_box,
        violin_swarm,
    )

    positions = [0, 1]
    groups = [x, y]
    var_names = [var_x, var_y]

    # Violin + swarm
    violin_swarm(ax, groups, positions, var_names)

    # Significance bracket
    significance_bracket(ax, positions[0], positions[1], result["stars"], groups)

    ax.set_title(f"Student's {fmt_sym('t')}-test (independent)")

    # Stats text box
    lines = [
        fmt_stat("t", result["statistic"]),
        fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
        fmt_stat("d", result["effect_size"]),
        f"{fmt_sym('n_1')} = {result['n_x']}, {fmt_sym('n_2')} = {result['n_y']}",
    ]
    stats_text_box(ax, lines)


"""Main function"""


def main(args):
    """Demonstrate independent samples t-test functionality."""
    logger.info("Demonstrating independent samples t-test")

    # Set random seed
    np.random.seed(42)

    # Example 1: Significant difference
    logger.info("\n=== Example 1: Significant difference ===")

    x1 = np.random.normal(0, 1, 50)
    y1 = np.random.normal(0.8, 1, 50)  # Large effect

    test_ttest_ind(x1, y1, var_x="Control", var_y="Treatment", verbose=True)

    # Example 2: Non-significant difference
    logger.info("\n=== Example 2: Non-significant difference ===")

    x2 = np.random.normal(0, 1, 30)
    y2 = np.random.normal(0.2, 1, 30)  # Small effect

    test_ttest_ind(x2, y2, var_x="Group A", var_y="Group B", verbose=True)

    # Example 3: Welch's t-test (unequal variances)
    logger.info("\n=== Example 3: Welch's t-test ===")

    x3 = np.random.normal(0, 1, 40)
    y3 = np.random.normal(0.5, 2, 40)  # Different variance

    test_ttest_ind(
        x3,
        y3,
        var_x="Low Variance",
        var_y="High Variance",
        equal_var=False,
        verbose=True,
    )

    # Example 4: One-sided test
    logger.info("\n=== Example 4: One-sided test ===")

    x4 = np.random.normal(0, 1, 50)
    y4 = np.random.normal(0.6, 1, 50)

    test_ttest_ind(x4, y4, alternative="two-sided", verbose=True)
    test_ttest_ind(x4, y4, alternative="less", verbose=True)

    # Example 5: With visualization
    logger.info("\n=== Example 5: With visualization ===")

    x5 = np.random.normal(10, 2, 60)
    y5 = np.random.normal(12, 2, 60)

    test_ttest_ind(x5, y5, var_x="Baseline", var_y="Follow-up", plot=True, verbose=True)
    stx.io.save(plt.gcf(), "./.dev/ttest_ind_example5.jpg")
    plt.close()

    # Example 6: DataFrame output
    logger.info("\n=== Example 6: DataFrame output ===")

    df_result = test_ttest_ind(x1, y1, return_as="dataframe")
    logger.info(f"\n{df_result.T}")  # type: ignore[union-attr]

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate independent samples t-test"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Initialize SciTeX framework and run main."""
    import sys  # noqa: E402

    import matplotlib.pyplot as plt  # noqa: E402

    global CONFIG, sys, plt

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(  # type: ignore[name-defined]
        sys,  # type: ignore[name-defined]
        plt,
        args=args,
        file=__file__,
        verbose=args.verbose,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,  # type: ignore[name-defined]
        verbose=args.verbose,
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
