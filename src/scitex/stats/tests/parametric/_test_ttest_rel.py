#!/usr/bin/env python3
# Timestamp: "2025-10-01 15:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/_test_ttest_rel.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------


"""
Functionalities:
  - Perform paired samples t-test (related/dependent samples)
  - Compute effect size (Cohen's d) and statistical power
  - Generate visualizations with paired lines
  - Support flexible output formats (dict or DataFrame)

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: Two paired samples (arrays or Series)
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


def test_ttest_rel(
    x: Union[np.ndarray, pd.Series, str],
    y: Union[np.ndarray, pd.Series, str],
    var_x: str = "before",
    var_y: str = "after",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    data: Union[pd.DataFrame, str, None] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
) -> Union[dict, pd.DataFrame]:
    r"""
    Perform paired samples t-test (related/dependent samples).

    Parameters
    ----------
    x : array or Series
        First sample (e.g., pre-test, baseline)
    y : array or Series
        Second sample (e.g., post-test, follow-up)
        Must have same length as x
    var_x : str, default 'before'
        Label for first sample
    var_y : str, default 'after'
        Label for second sample
    alternative : {'two-sided', 'greater', 'less'}, default 'two-sided'
        Alternative hypothesis:
        - 'two-sided': means differ
        - 'greater': mean(x - y) > 0
        - 'less': mean(x - y) < 0
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

    Returns
    -------
    results : dict or DataFrame
        Test results (same structure as test_ttest_ind)

    Notes
    -----
    The paired t-test compares means of matched observations (within-subjects).

    **When to use:**
    - Before-after measurements on same subjects
    - Matched pairs (twins, siblings, matched controls)
    - Repeated measures at two time points

    **Assumptions:**
    - Differences (x - y) are normally distributed
    - Pairs are independent across subjects
    - No assumption about equality of variances

    The test statistic is:

    .. math::
        t = \\frac{\\bar{d}}{s_d / \\sqrt{n}}

    where :math:`\\bar{d}` is mean difference and :math:`s_d` is SD of differences.

    **Effect size** (Cohen's d for paired samples):

    .. math::
        d = \\frac{\\bar{d}}{s_d}

    This measures the standardized change from baseline.

    References
    ----------
    .. [1] Student (1908). "The Probable Error of a Mean". Biometrika, 6(1), 1-25.

    Examples
    --------
    >>> before = np.array([10, 12, 15, 18, 20])
    >>> after = np.array([12, 14, 17, 20, 22])
    >>> result = test_ttest_rel(before, after)
    >>> result['pvalue']
    0.001...

    >>> # With visualization
    >>> fig, ax = plt.subplots()
    >>> result = test_ttest_rel(before, after, ax=ax)
    >>> plt.show()
    """
    # Resolve column names from DataFrame (seaborn-style data= parameter)
    if data is not None:
        from scitex.stats._utils._csv_support import resolve_columns

        resolved = resolve_columns(data, x=x, y=y)
        x, y = resolved["x"], resolved["y"]

    from scitex.stats._utils._effect_size import cohens_d, interpret_cohens_d
    from scitex.stats._utils._formatters import p2stars
    from scitex.stats._utils._normalizers import force_dataframe
    from scitex.stats._utils._power import power_ttest

    # Convert to numpy arrays and remove NaN
    x = np.asarray(x)
    y = np.asarray(y)

    # Check for paired NaN removal
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) != len(y):
        raise ValueError(
            f"Paired samples must have same length after NaN removal: {len(x)} vs {len(y)}"
        )

    n_pairs = len(x)

    # Perform paired t-test
    t_result = stats.ttest_rel(x, y, alternative=alternative)
    t_stat = float(t_result.statistic)
    pvalue = float(t_result.pvalue)

    # Compute effect size (Cohen's d for paired samples)
    effect_size = cohens_d(x, y, paired=True)
    effect_size_interpretation = interpret_cohens_d(effect_size)

    # Compute statistical power
    power = power_ttest(
        effect_size=abs(effect_size),
        n=n_pairs,
        alpha=alpha,
        alternative=alternative,
        test_type="paired",
    )

    # Create null hypothesis description
    if alternative == "two-sided":
        H0 = f"μ({var_x} - {var_y}) = 0"
    elif alternative == "greater":
        H0 = f"μ({var_x} - {var_y}) ≤ 0"
    else:  # less
        H0 = f"μ({var_x} - {var_y}) ≥ 0"

    # Compile results
    result = {
        "test_method": "Paired t-test",
        "statistic": t_stat,
        "stat_symbol": "t",
        "alternative": alternative,
        "n_pairs": n_pairs,
        "var_x": var_x,
        "var_y": var_y,
        "pvalue": pvalue,
        "stars": p2stars(pvalue),
        "alpha": alpha,
        "significant": pvalue < alpha,
        "effect_size": effect_size,
        "effect_size_metric": "Cohen's d (paired)",
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
            fig, ax = stx.plt.subplots()
        _plot_ttest_rel(x, y, var_x, var_y, result, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)

    return result


def _plot_ttest_rel(x, y, var_x, var_y, result, ax):
    """Create visualization for paired t-test on given axes."""
    from scitex.stats._plot_helpers import stats_text_box

    # Plot paired lines with gray color
    for i in range(len(x)):
        ax.plot([0, 1], [x[i], y[i]], "o-", color="gray", alpha=0.3)

    # Plot means with error bars - theme handles styling
    ax.errorbar(
        [0],
        [np.mean(x)],
        yerr=[np.std(x, ddof=1)],
        fmt="o",
        label=var_x,
    )
    ax.errorbar(
        [1],
        [np.mean(y)],
        yerr=[np.std(y, ddof=1)],
        fmt="o",
        label=var_y,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([var_x, var_y])
    ax.set_ylabel("Value")
    ax.set_title(f"Student's {fmt_sym('t')}-test (paired)")
    ax.legend()

    # Stats text box
    lines = [
        fmt_stat("t", result["statistic"]),
        fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
        fmt_stat("d", result["effect_size"]),
        f"{fmt_sym('n')} = {result['n_pairs']}",
    ]
    stats_text_box(ax, lines)


"""Main function"""


def main(args):
    """Demonstrate paired samples t-test functionality."""
    logger.info("Demonstrating paired samples t-test")

    # Set random seed
    np.random.seed(42)

    # Example 1: Significant paired difference
    logger.info("\n=== Example 1: Significant paired difference ===")

    before = np.random.normal(10, 2, 30)
    after = before + np.random.normal(2, 1, 30)  # Correlated increase

    test_ttest_rel(before, after, var_x="Before", var_y="After")

    # Example 2: With visualization
    logger.info("\n=== Example 2: With visualization ===")

    test_ttest_rel(before, after, plot=True)
    stx.io.save(plt.gcf(), "./.dev/ttest_rel_example2.jpg")
    plt.close()

    # Example 3: DataFrame output
    logger.info("\n=== Example 3: DataFrame output ===")

    df_result = test_ttest_rel(before, after, return_as="dataframe")
    logger.info(f"\n{df_result.T}")  # type: ignore[union-attr]

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demonstrate paired samples t-test")
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
