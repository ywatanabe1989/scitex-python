#!/usr/bin/env python3
# Timestamp: "2025-10-01 17:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/normality/_test_ks_2samp.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------


r"""
Functionalities:
  - Perform two-sample Kolmogorov-Smirnov test (compare two empirical distributions)
  - Generate CDF comparison plots
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


def test_ks_2samp(
    x: Union[np.ndarray, pd.Series, str],
    y: Union[np.ndarray, pd.Series, str],
    var_x: str = "x",
    var_y: str = "y",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
    plot: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
    data: Union[pd.DataFrame, str, None] = None,
    return_as: Literal["dict", "dataframe"] = "dict",
    decimals: int = 3,
    verbose: bool = False,
) -> Union[dict, pd.DataFrame]:
    r"""
    Perform two-sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    x, y : arrays or Series
        Two samples to compare
    var_x, var_y : str
        Labels for samples
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Alternative hypothesis
    alpha : float, default 0.05
        Significance level
    plot : bool, default False
        Whether to generate CDF comparison plot
    data : DataFrame, str, or None, optional
        DataFrame or CSV path. When provided, string values for x/y
        are resolved as column names (seaborn-style).
    return_as : {'dict', 'dataframe'}, default 'dict'
        Output format
    decimals : int, default 3
        Number of decimal places for rounding

    Returns
    -------
    results : dict or DataFrame
        Test results including:
        - test_method: 'Kolmogorov-Smirnov test (2-sample)'
        - statistic_name: 'D'
        - statistic: KS D-statistic
        - pvalue: p-value
        - pstars: Significance stars
        - rejected: Whether null hypothesis is rejected
        - n_x, n_y: Sample sizes
        - var_x, var_y: Variable labels
        - H0: Null hypothesis description
    fig : matplotlib.figure.Figure, optional
        Figure with CDF comparison (only if plot=True)

    Notes
    -----
    The two-sample Kolmogorov-Smirnov test compares the ECDFs of two samples.

    **Null Hypothesis (H0)**: Both samples come from the same distribution

    **Test Statistic D**:

    .. math::
        D = \\sup_x |F_{n_1}(x) - F_{n_2}(x)|

    Where F_{n_1} and F_{n_2} are the empirical CDFs.

    **Advantages**:
    - Distribution-free (non-parametric)
    - Tests entire distribution, not just location
    - Can detect differences in location, scale, or shape

    **Disadvantages**:
    - Less powerful than t-test when assumptions are met
    - Most sensitive to differences near the center of distributions
    - Less sensitive to tail differences

    **When to use**:
    - Comparing two independent samples
    - No assumptions about distribution shape
    - Want to test overall distribution equality (not just means)
    - Alternative to t-test when normality violated

    **Comparison with other tests**:
    - vs t-test: More robust, less powerful
    - vs Mann-Whitney U: Tests different hypotheses (distribution vs median)
    - vs Brunner-Munzel: KS tests full distribution, BM tests P(X>Y)

    Examples
    --------
    >>> # Two samples from same distribution
    >>> x = np.random.normal(0, 1, 100)
    >>> y = np.random.normal(0, 1, 100)
    >>> result = test_ks_2samp(x, y)
    >>> result['rejected']
    False

    >>> # Two samples from different distributions
    >>> x = np.random.normal(0, 1, 100)
    >>> y = np.random.normal(2, 1, 100)
    >>> result = test_ks_2samp(x, y)
    >>> result['rejected']
    True
    """
    # Resolve column names from DataFrame (seaborn-style data= parameter)
    if data is not None:
        from scitex.stats._utils._csv_support import resolve_columns

        resolved = resolve_columns(data, x=x, y=y)
        x, y = resolved["x"], resolved["y"]

    from scitex.stats._utils._formatters import p2stars
    from scitex.stats._utils._normalizers import convert_results, force_dataframe

    # Convert to numpy arrays and remove NaN
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n_x = len(x)
    n_y = len(y)

    # Perform two-sample KS test
    ks_result = stats.ks_2samp(x, y, alternative=alternative)
    d_stat = float(ks_result.statistic)
    pvalue = float(ks_result.pvalue)

    # Determine rejection
    rejected = pvalue < alpha

    # Compile results
    result = {
        "test_method": "Kolmogorov-Smirnov test (2-sample)",
        "statistic": round(d_stat, decimals),
        "stat_symbol": "D",
        "n_x": n_x,
        "n_y": n_y,
        "var_x": var_x,
        "var_y": var_y,
        "pvalue": round(pvalue, decimals),
        "stars": p2stars(pvalue),
        "alpha": alpha,
        "significant": rejected,
        "same_distribution": not rejected,
        "H0": "Both samples come from the same distribution",
    }

    # Log results if verbose
    if verbose:
        logger.info(
            f"KS test (2-sample): D = {d_stat:.3f}, p = {pvalue:.4f} {p2stars(pvalue)}"
        )
        logger.info(f"Same distribution: {not rejected}")

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            fig, axes = stx.plt.subplots(1, 2, figsize=(14, 6))
            _plot_ks_2samp_full(x, y, var_x, var_y, result, axes)
        else:
            _plot_ks_2samp_simple(x, y, var_x, var_y, result, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)
    elif return_as not in ["dict", "dataframe"]:
        return convert_results(result, return_as=return_as)

    return result


def _plot_ks_2samp_full(x, y, var_x, var_y, result, axes):
    """Create 2-panel CDF comparison plot for two-sample KS test."""
    from scitex.stats._plot_helpers import stats_text_box

    # Plot 1: CDF comparison
    ax = axes[0]

    # Compute ECDFs
    x_sorted = np.sort(x)
    ecdf_x = np.arange(1, len(x) + 1) / len(x)

    y_sorted = np.sort(y)
    ecdf_y = np.arange(1, len(y) + 1) / len(y)

    # Plot both ECDFs
    ax.step(x_sorted, ecdf_x, where="post", label=var_x)
    ax.step(y_sorted, ecdf_y, where="post", label=var_y)

    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("KS Test (2-sample)")
    ax.legend()

    # Add text with results
    stats_text_box(
        ax,
        [
            fmt_stat("D", result["statistic"]),
            fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
            f"Same dist: {result['same_distribution']}",
            f"{fmt_sym('n_x')} = {result['n_x']}, {fmt_sym('n_y')} = {result['n_y']}",
        ],
    )

    # Plot 2: Overlapping histograms
    ax = axes[1]

    ax.hist(
        x,
        bins="auto",
        density=True,
        label=var_x,
    )
    ax.hist(
        y,
        bins="auto",
        density=True,
        label=var_y,
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram")
    ax.legend()


def _plot_ks_2samp_simple(x, y, var_x, var_y, result, ax):
    """Create single CDF comparison plot on provided axes."""
    from scitex.stats._plot_helpers import stats_text_box

    # Compute ECDFs
    x_sorted = np.sort(x)
    ecdf_x = np.arange(1, len(x) + 1) / len(x)

    y_sorted = np.sort(y)
    ecdf_y = np.arange(1, len(y) + 1) / len(y)

    # Plot both ECDFs
    ax.step(x_sorted, ecdf_x, where="post", label=var_x)
    ax.step(y_sorted, ecdf_y, where="post", label=var_y)

    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("KS Test (2-sample)")
    ax.legend()

    # Add text with results
    stats_text_box(
        ax,
        [
            fmt_stat("D", result["statistic"]),
            fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
            f"Same dist: {result['same_distribution']}",
            f"{fmt_sym('n_x')} = {result['n_x']}, {fmt_sym('n_y')} = {result['n_y']}",
        ],
    )


"""Main function"""


def main(args):
    """Demonstrate two-sample Kolmogorov-Smirnov test functionality."""
    logger.info("Demonstrating two-sample Kolmogorov-Smirnov test")

    # Set random seed
    np.random.seed(42)

    # Example 1: Two-sample test - same distribution
    logger.info("\n=== Example 1: Two-sample KS test (same distribution) ===")

    x1 = np.random.normal(0, 1, 100)
    y1 = np.random.normal(0, 1, 100)

    result1 = test_ks_2samp(x1, y1, var_x="Sample 1", var_y="Sample 2", verbose=True)

    # Example 2: Two-sample test - different means
    logger.info("\n=== Example 2: Two-sample KS test (different means) ===")

    x2 = np.random.normal(0, 1, 100)
    y2 = np.random.normal(2, 1, 100)

    result2 = test_ks_2samp(x2, y2, var_x="Group A", var_y="Group B", verbose=True)

    # Example 3: Two-sample test with visualization
    logger.info("\n=== Example 3: Two-sample KS test with visualization ===")

    x3 = np.random.normal(5, 1, 80)
    y3 = np.random.exponential(2, 80)

    result3 = test_ks_2samp(
        x3, y3, var_x="Normal", var_y="Exponential", plot=True, verbose=True
    )
    stx.io.save(plt.gcf(), "./ks_2samp_example.jpg")
    plt.close()

    # Example 4: Export results
    logger.info("\n=== Example 4: Export results ===")

    from scitex.stats._utils._normalizers import convert_results, force_dataframe

    test_results = [result1, result2, result3]

    df = force_dataframe(test_results)
    logger.info(f"\nDataFrame shape: {df.shape}")

    convert_results(test_results, return_as="excel", path="./ks_2samp_results.xlsx")  # type: ignore[arg-type]
    logger.info("Results exported to ./ks_2samp_results.xlsx")

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate two-sample Kolmogorov-Smirnov test"
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
