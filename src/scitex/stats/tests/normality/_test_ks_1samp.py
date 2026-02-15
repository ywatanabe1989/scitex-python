#!/usr/bin/env python3
# Timestamp: "2025-10-01 17:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/normality/_test_ks_1samp.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------


r"""
Functionalities:
  - Perform one-sample Kolmogorov-Smirnov test (compare to reference distribution)
  - Generate CDF comparison plots
  - Support flexible output formats (dict or DataFrame)

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: One sample (array or Series)
  - output: Test results (dict or DataFrame) and optional figure
"""

"""Imports"""
import argparse  # noqa: E402
from typing import Callable, Literal, Optional, Union  # noqa: E402

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


def test_ks_1samp(  # noqa: C901
    x: Union[np.ndarray, pd.Series, str],
    cdf: Union[str, Callable] = "norm",
    args: tuple = (),
    var_x: str = "x",
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
    Perform one-sample Kolmogorov-Smirnov test.

    Parameters
    ----------
    x : array or Series
        Sample to test
    cdf : str or callable, default 'norm'
        Reference distribution. Either:
        - String: 'norm', 'uniform', 'expon', etc. (scipy.stats distribution name)
        - Callable: CDF function
    args : tuple, default ()
        Distribution parameters (e.g., (loc, scale) for normal)
    var_x : str, default 'x'
        Label for sample
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Alternative hypothesis
    alpha : float, default 0.05
        Significance level
    plot : bool, default False
        Whether to generate CDF comparison plot
    data : DataFrame, str, or None, optional
        DataFrame or CSV path. When provided, string value for x
        is resolved as a column name (seaborn-style).
    return_as : {'dict', 'dataframe'}, default 'dict'
        Output format
    decimals : int, default 3
        Number of decimal places for rounding

    Returns
    -------
    results : dict or DataFrame
        Test results including:
        - test_method: 'Kolmogorov-Smirnov test (1-sample)'
        - statistic_name: 'D'
        - statistic: KS D-statistic (maximum CDF difference)
        - pvalue: p-value
        - pstars: Significance stars
        - rejected: Whether null hypothesis is rejected
        - n_x: Sample size
        - var_x: Variable label
        - reference_distribution: Name of reference distribution
        - H0: Null hypothesis description
    fig : matplotlib.figure.Figure, optional
        Figure with CDF comparison (only if plot=True)

    Notes
    -----
    The one-sample Kolmogorov-Smirnov test compares the empirical cumulative
    distribution function (ECDF) of the sample against a reference CDF.

    **Null Hypothesis (H0)**: Data follow the specified distribution

    **Test Statistic D**:

    .. math::
        D = \\sup_x |F_n(x) - F(x)|

    Where:
    - F_n(x): Empirical CDF of sample
    - F(x): Reference CDF

    **Advantages**:
    - Distribution-free (no assumptions about data)
    - Can test against any continuous distribution
    - More general than Shapiro-Wilk (not limited to normality)

    **Disadvantages**:
    - Less powerful than Shapiro-Wilk for normality testing
    - Sensitive to sample size (large n → high power, may detect trivial deviations)
    - Assumes continuous distribution (not suitable for discrete data)

    **When to use**:
    - Testing goodness-of-fit to any continuous distribution
    - Comparing sample to theoretical distribution
    - When Shapiro-Wilk is not applicable (non-normal distributions)
    - Large sample sizes (n > 50)

    References
    ----------
    .. [1] Kolmogorov, A. (1933). "Sulla determinazione empirica di una legge
           di distribuzione". Giornale dell'Istituto Italiano degli Attuari, 4, 83-91.
    .. [2] Smirnov, N. (1948). "Table for estimating the goodness of fit of
           empirical distributions". Annals of Mathematical Statistics, 19(2), 279-281.

    Examples
    --------
    >>> # Test if data are normally distributed
    >>> x = np.random.normal(0, 1, 100)
    >>> result = test_ks_1samp(x, cdf='norm', args=(0, 1))
    >>> result['rejected']
    False

    >>> # Test if data are uniformly distributed
    >>> x = np.random.uniform(0, 1, 100)
    >>> result = test_ks_1samp(x, cdf='uniform', args=(0, 1))
    """
    from scitex.stats._utils._formatters import p2stars
    from scitex.stats._utils._normalizers import convert_results, force_dataframe

    # Resolve column names from DataFrame (seaborn-style data= parameter)
    if data is not None:
        from scitex.stats._utils._csv_support import resolve_columns

        resolved = resolve_columns(data, x=x)
        x = resolved["x"]

    # Convert to numpy array and remove NaN
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    n_x = len(x)

    # Check sample size
    if n_x < 3:
        raise ValueError("KS test requires at least 3 observations")

    # Get reference distribution
    if isinstance(cdf, str):
        ref_dist_name = cdf
        # Get scipy distribution
        ref_dist = getattr(stats, cdf)
        if args:

            def cdf_func(t):  # noqa: E731
                return ref_dist.cdf(t, *args)

        else:
            cdf_func = ref_dist.cdf
    else:
        ref_dist_name = "custom"
        cdf_func = cdf

    # Perform KS test
    ks_result = stats.ks_1samp(x, cdf_func, alternative=alternative)
    d_stat = float(ks_result.statistic)
    pvalue = float(ks_result.pvalue)

    # Determine if distribution matches
    rejected = pvalue < alpha
    matches = not rejected

    # Compile results
    result = {
        "test_method": "Kolmogorov-Smirnov test (1-sample)",
        "statistic": round(d_stat, decimals),
        "stat_symbol": "D",
        "n": n_x,
        "var_x": var_x,
        "pvalue": round(pvalue, decimals),
        "stars": p2stars(pvalue),
        "alpha": alpha,
        "significant": rejected,
        "matches_distribution": matches,
        "reference_distribution": ref_dist_name,
        "H0": f"Data follow {ref_dist_name} distribution",
    }

    # Log results if verbose
    if verbose:
        logger.info(
            f"KS test (1-sample): D = {d_stat:.3f}, p = {pvalue:.4f} {p2stars(pvalue)}"
        )
        logger.info(f"Matches {ref_dist_name}: {matches}")

    # Auto-enable plotting if ax is provided
    if ax is not None:
        plot = True

    # Generate plot if requested
    if plot:
        if ax is None:
            fig, axes = stx.plt.subplots(1, 2, figsize=(14, 6))
            _plot_ks_1samp_full(x, cdf_func, var_x, result, ref_dist_name, axes)
        else:
            _plot_ks_1samp_simple(x, cdf_func, var_x, result, ref_dist_name, ax)

    # Convert to requested format
    if return_as == "dataframe":
        result = force_dataframe(result)
    elif return_as not in ["dict", "dataframe"]:
        return convert_results(result, return_as=return_as)

    return result


def _plot_ks_1samp_full(x, cdf_func, var_x, result, ref_dist_name, axes):
    """Create 2-panel CDF comparison plot for one-sample KS test."""
    from scitex.stats._plot_helpers import stats_text_box

    # Plot 1: CDF comparison
    ax = axes[0]

    # Compute ECDF
    x_sorted = np.sort(x)
    ecdf = np.arange(1, len(x) + 1) / len(x)

    # Compute reference CDF
    ref_cdf = cdf_func(x_sorted)

    # Plot both CDFs
    ax.step(
        x_sorted,
        ecdf,
        where="post",
        label=f"Empirical ({var_x})",
    )
    ax.plot(
        x_sorted,
        ref_cdf,
        label=f"Reference ({ref_dist_name})",
    )

    # Mark maximum difference
    diff = np.abs(ecdf - ref_cdf)
    max_idx = np.argmax(diff)
    ax.vlines(
        x_sorted[max_idx],
        ecdf[max_idx],
        ref_cdf[max_idx],
        linestyles="dashed",
        label=f"D = {result['statistic']:.3f}",
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("KS Test (1-sample)")
    ax.legend()

    # Add text with results
    stats_text_box(
        ax,
        [
            fmt_stat("D", result["statistic"]),
            fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
            f"Matches: {result['matches_distribution']}",
            f"{fmt_sym('n')} = {result['n']}",
        ],
    )

    # Plot 2: Histogram with reference PDF
    ax = axes[1]

    ax.hist(x, bins="auto", density=True, label="Data")

    # If reference is a known distribution, plot PDF
    if ref_dist_name != "custom":
        x_range = np.linspace(np.min(x), np.max(x), 200)
        ref_dist = getattr(stats, ref_dist_name)
        # Try to get PDF
        try:
            if hasattr(ref_dist, "pdf"):
                pdf_vals = ref_dist.pdf(x_range)
                ax.plot(x_range, pdf_vals, label=f"{ref_dist_name} PDF")
        except Exception:  # noqa: E722
            pass

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram")
    ax.legend()


def _plot_ks_1samp_simple(x, cdf_func, var_x, result, ref_dist_name, ax):
    """Create single CDF comparison plot on provided axes."""
    from scitex.stats._plot_helpers import stats_text_box

    # Compute ECDF
    x_sorted = np.sort(x)
    ecdf = np.arange(1, len(x) + 1) / len(x)

    # Compute reference CDF
    ref_cdf = cdf_func(x_sorted)

    # Plot both CDFs
    ax.step(
        x_sorted,
        ecdf,
        where="post",
        label=f"Empirical ({var_x})",
    )
    ax.plot(
        x_sorted,
        ref_cdf,
        label=f"Reference ({ref_dist_name})",
    )

    # Mark maximum difference
    diff = np.abs(ecdf - ref_cdf)
    max_idx = np.argmax(diff)
    ax.vlines(
        x_sorted[max_idx],
        ecdf[max_idx],
        ref_cdf[max_idx],
        linestyles="dashed",
        label=f"D = {result['statistic']:.3f}",
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("KS Test (1-sample)")
    ax.legend()

    # Add text with results
    stats_text_box(
        ax,
        [
            fmt_stat("D", result["statistic"]),
            fmt_stat("p", result["pvalue"], fmt=".4f", stars=result["stars"]),
            f"Matches: {result['matches_distribution']}",
            f"{fmt_sym('n')} = {result['n']}",
        ],
    )


"""Main function"""


def main(args):
    """Demonstrate one-sample Kolmogorov-Smirnov test functionality."""
    logger.info("Demonstrating one-sample Kolmogorov-Smirnov test")

    # Set random seed
    np.random.seed(42)

    # Example 1: One-sample test - normal data
    logger.info("\n=== Example 1: One-sample KS test (normal data) ===")

    x_normal = np.random.normal(0, 1, 100)
    result1 = test_ks_1samp(
        x_normal, cdf="norm", args=(0, 1), var_x="Normal data", verbose=True
    )

    # Example 2: One-sample test - exponential data tested against normal
    logger.info("\n=== Example 2: One-sample KS test (exponential data vs normal) ===")

    x_exp = np.random.exponential(2, 100)
    result2 = test_ks_1samp(
        x_exp,
        cdf="norm",
        args=(np.mean(x_exp), np.std(x_exp)),
        var_x="Exponential data",
        verbose=True,
    )

    # Example 3: One-sample test with visualization
    logger.info("\n=== Example 3: One-sample KS test with visualization ===")

    x_mixed = np.concatenate([np.random.normal(0, 1, 90), np.random.normal(3, 1, 10)])
    result3 = test_ks_1samp(
        x_mixed,
        cdf="norm",
        args=(0, 1),
        var_x="Mixed data",
        plot=True,
        verbose=True,
    )
    stx.io.save(plt.gcf(), "./ks_1samp_example.jpg")
    plt.close()

    # Example 4: Export results
    logger.info("\n=== Example 4: Export results ===")

    from scitex.stats._utils._normalizers import convert_results, force_dataframe

    test_results = [result1, result2, result3]

    df = force_dataframe(test_results)
    logger.info(f"\nDataFrame shape: {df.shape}")

    convert_results(test_results, return_as="excel", path="./ks_1samp_results.xlsx")  # type: ignore[arg-type]
    logger.info("Results exported to ./ks_1samp_results.xlsx")

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate one-sample Kolmogorov-Smirnov test"
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
