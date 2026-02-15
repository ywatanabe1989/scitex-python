#!/usr/bin/env python3
# Timestamp: "2025-10-01 15:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/_test_ttest_1samp.py
# ----------------------------------------
from __future__ import annotations

"""
Functionalities:
  - Perform one-sample t-test
  - Compute effect size (Cohen's d) and statistical power
  - Generate visualizations with reference line and confidence interval
  - Support flexible output formats (dict or DataFrame)

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: One sample (array or Series) and population mean
  - output: Test results (dict or DataFrame) and optional figure
"""

"""Imports"""
import argparse  # noqa: E402
import os  # noqa: E402
from typing import Literal, Optional, Union  # noqa: E402

import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
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


def test_ttest_1samp(
    x: Union[np.ndarray, pd.Series, str],
    popmean: float = 0,
    var_x: str = "sample",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    data: Union[pd.DataFrame, str, None] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
) -> Union[dict, pd.DataFrame]:
    r"""
    Perform one-sample t-test.

    Parameters
    ----------
    x : array or Series
        Sample data
    popmean : float, default 0
        Expected population mean (null hypothesis value)
    var_x : str, default 'sample'
        Label for sample
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Alternative hypothesis:
        - 'two-sided': mean ≠ popmean
        - 'greater': mean > popmean
        - 'less': mean < popmean
    alpha : float, default 0.05
        Significance level
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If provided, plots visualization on given axes.
    data : DataFrame, str, or None, optional
        DataFrame or CSV path. When provided, string value for x
        is resolved as a column name (seaborn-style).
    return_as : {'dict', 'dataframe'}, default 'dict'
        Output format

    Returns
    -------
    results : dict or DataFrame
        Test results

    Notes
    -----
    The one-sample t-test compares sample mean to a known population mean.

    **When to use:**
    - Test if sample mean differs from theoretical/known value
    - Compare observed data to standard/reference value
    - Test if mean differs from zero (common in difference scores)

    **Assumptions:**
    - Data are normally distributed
    - Observations are independent

    The test statistic is:

    .. math::
        t = \\frac{\\bar{x} - \\mu_0}{s / \\sqrt{n}}

    where :math:`\\mu_0` is the hypothesized population mean.

    **Effect size** (Cohen's d for one sample):

    .. math::
        d = \\frac{\\bar{x} - \\mu_0}{s}

    References
    ----------
    .. [1] Student (1908). "The Probable Error of a Mean". Biometrika, 6(1), 1-25.

    Examples
    --------
    >>> # Test if sample mean differs from 0
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> result = test_ttest_1samp(x, popmean=0)
    >>> result['pvalue']
    0.003...

    >>> # Test if sample mean differs from 100
    >>> scores = np.array([95, 98, 102, 105, 108])
    >>> result = test_ttest_1samp(scores, popmean=100)
    """
    from scitex.stats._utils._effect_size import cohens_d, interpret_cohens_d
    from scitex.stats._utils._formatters import p2stars
    from scitex.stats._utils._normalizers import force_dataframe
    from scitex.stats._utils._power import power_ttest

    # Resolve column names from DataFrame (seaborn-style data= parameter)
    if data is not None:
        from scitex.stats._utils._csv_support import resolve_columns

        resolved = resolve_columns(data, x=x)
        x = resolved["x"]

    # Convert to numpy array and remove NaN
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    n_x = len(x)

    # Perform one-sample t-test
    t_result = stats.ttest_1samp(x, popmean, alternative=alternative)
    t_stat = float(t_result.statistic)
    pvalue = float(t_result.pvalue)

    # Compute effect size (Cohen's d for one sample)
    effect_size = cohens_d(x, y=None, paired=False)  # One-sample version
    effect_size_interpretation = interpret_cohens_d(effect_size)

    # Compute statistical power
    power = power_ttest(
        effect_size=abs(effect_size),
        n=n_x,
        alpha=alpha,
        alternative=alternative,
        test_type="one-sample",
    )

    # Create null hypothesis description
    if alternative == "two-sided":
        H0 = f"μ({var_x}) = {popmean}"
    elif alternative == "greater":
        H0 = f"μ({var_x}) ≤ {popmean}"
    else:  # less
        H0 = f"μ({var_x}) ≥ {popmean}"

    # Compile results
    result = {
        "test_method": "One-sample t-test",
        "statistic": t_stat,
        "stat_symbol": "t",
        "alternative": alternative,
        "n_x": n_x,
        "var_x": var_x,
        "popmean": popmean,
        "sample_mean": float(np.mean(x)),
        "pvalue": pvalue,
        "stars": p2stars(pvalue),
        "alpha": alpha,
        "significant": pvalue < alpha,
        "effect_size": effect_size,
        "effect_size_metric": "Cohen's d (one-sample)",
        "effect_size_interpretation": effect_size_interpretation,
        "power": power,
        "H0": H0,
    }

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            _, ax = stx.plt.subplots()
        _plot_ttest_1samp(x, popmean, var_x, result, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)

    return result


def _plot_ttest_1samp(x, popmean, var_x, result, ax):
    """Create visualization for one-sample t-test on given axes."""
    from scitex.stats._plot_helpers import stats_text_box

    # Box plot - theme handles styling
    ax.boxplot([x], positions=[0], patch_artist=True, showfliers=True)

    # Add reference line for population mean
    ax.axhline(popmean, label=f"H0: μ = {popmean}")

    # Add confidence interval
    ci = stats.t.interval(
        1 - result["alpha"], len(x) - 1, loc=np.mean(x), scale=stats.sem(x)
    )
    ax.plot([0, 0], ci, label=f"{int((1 - result['alpha']) * 100)}% CI")

    ax.set_xticks([0])
    ax.set_xticklabels([var_x])
    ax.set_ylabel("Value")
    ax.set_title(f"Student's {fmt_sym('t')}-test (one-sample)")
    ax.legend()

    # Stats text box
    lines = [
        fmt_stat("t", result["statistic"]),
        fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
        fmt_stat("d", result["effect_size"]),
        f"{fmt_sym('n')} = {result['n_x']}",
    ]
    stats_text_box(ax, lines)


"""Main function"""


def main(args):
    """Demonstrate one-sample t-test functionality."""
    logger.info("Demonstrating one-sample t-test")

    # Set random seed
    np.random.seed(42)

    # Example 1: Test against zero
    logger.info("\n=== Example 1: Test against zero ===")

    x1 = np.random.normal(2, 1, 30)  # Mean around 2, should be significant vs 0

    test_ttest_1samp(x1, popmean=0, var_x="Differences")

    # Example 2: Test against non-zero value
    logger.info("\n=== Example 2: Test against reference value ===")

    scores = np.random.normal(100, 15, 50)

    test_ttest_1samp(scores, popmean=100, var_x="Test Scores")

    # Example 3: With visualization
    logger.info("\n=== Example 3: With visualization ===")

    test_ttest_1samp(x1, popmean=0, plot=True)
    stx.io.save(plt.gcf(), "./.dev/ttest_1samp_example3.jpg")
    plt.close()

    # Example 4: DataFrame output
    logger.info("\n=== Example 4: DataFrame output ===")

    df_result = test_ttest_1samp(x1, return_as="dataframe")
    logger.info(f"\n{df_result.T}")  # type: ignore[union-attr]

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demonstrate one-sample t-test")
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
