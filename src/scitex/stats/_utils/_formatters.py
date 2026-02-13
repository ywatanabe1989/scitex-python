#!/usr/bin/env python3
# Timestamp: "2025-10-01 14:45:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/utils/_formatters.py

"""Statistical result formatters.

Functionalities:
  - Convert p-values to significance stars for scientific reporting
  - Format p-values according to publication standards
  - Support pandas DataFrames and numpy arrays

Dependencies:
  - packages: numpy, pandas

IO:
  - input: p-values (float, array, or DataFrame)
  - output: formatted strings with significance indicators
"""

"""Imports"""
import argparse  # noqa: E402
import re  # noqa: E402
from typing import Union  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scitex as stx  # noqa: E402
from scitex.logging import getLogger  # noqa: E402

logger = getLogger(__name__)

"""Functions"""

# Matplotlib mathtext mapping for publication stat symbols.
# Latin letters use \mathit{} for italic (APA style) because SCITEX sets
# mathtext.default=regular which renders plain $t$ upright.
# Greek letters are naturally upright in APA, so no \mathit{} needed.
_MATHTEXT = {
    "t": "$\\mathit{t}$",
    "F": "$\\mathit{F}$",
    "p": "$\\mathit{p}$",
    "r": "$\\mathit{r}$",
    "d": "$\\mathit{d}$",
    "D": "$\\mathit{D}$",
    "U": "$\\mathit{U}$",
    "W": "$\\mathit{W}$",
    "H": "$\\mathit{H}$",
    "Z": "$\\mathit{Z}$",
    "BM": "$\\mathit{BM}$",
    "n": "$\\mathit{n}$",
    "n_x": "$\\mathit{n}_x$",
    "n_y": "$\\mathit{n}_y$",
    "n_1": "$\\mathit{n}_1$",
    "n_2": "$\\mathit{n}_2$",
    "X": "$\\mathit{X}$",
    "Y": "$\\mathit{Y}$",
    "chi2": "$\\chi^2$",
    "delta": "$\\delta$",
    "epsilon2": "$\\epsilon^2$",
    "eta2": "$\\eta^2$",
    "eta_p2": "$\\eta_p^2$",
    "R2": "$\\mathit{R}^2$",
    "rho": "$\\rho$",
    "tau": "$\\tau$",
}


# Latin stat symbols that need \mathit{} for APA italic rendering.
# Single letters only — multi-char like "ns", Greek, or already-decorated skipped.
_ITALIC_LATIN = frozenset("tFprdDUWHZnR")


def italicize_stats(text: str) -> str:
    r"""Auto-convert plain $X$ to $\mathit{X}$ for known Latin stat symbols.

    Users can write simple ``$t$`` or ``$p$`` and this function ensures
    they render italic under SCITEX's ``mathtext.default=regular`` setting.

    Already-decorated expressions (containing ``\mathit``, ``\chi``, ``\eta``,
    ``\rho``, ``\tau``) are left untouched.

    Parameters
    ----------
    text : str
        Text potentially containing plain mathtext like ``$t$``, ``$F$``.

    Returns
    -------
    str
        Text with Latin stat symbols wrapped in ``\mathit{}``.

    Examples
    --------
    >>> italicize_stats('$t$ = 2.31')
    '$\\mathit{t}$ = 2.31'
    >>> italicize_stats('$\\chi^2$ = 5.99')
    '$\\chi^2$ = 5.99'
    >>> italicize_stats('$\\mathit{F}$ = 3.14')
    '$\\mathit{F}$ = 3.14'
    """

    def _replace(m):
        inner = m.group(1)
        # Skip already-decorated or Greek
        if "\\" in inner:
            return m.group(0)
        # Single Latin stat letter
        if inner in _ITALIC_LATIN:
            return f"$\\mathit{{{inner}}}$"
        # Subscripted forms like n_x, n_1
        if re.match(r"^[A-Za-z]_[A-Za-z0-9]$", inner):
            base = inner[0]
            sub = inner[2]
            if base in _ITALIC_LATIN:
                return f"$\\mathit{{{base}}}_{sub}$"
        # R^2 pattern
        if re.match(r"^([A-Za-z])\^(\d)$", inner):
            base = inner[0]
            exp = inner[2]
            if base in _ITALIC_LATIN:
                return f"$\\mathit{{{base}}}^{exp}$"
        return m.group(0)

    return re.sub(r"\$([^$]+)\$", _replace, text)


def fmt_sym(symbol: str) -> str:
    """Return mathtext-italic stat symbol for matplotlib annotations.

    Parameters
    ----------
    symbol : str
        Plain symbol name (e.g. 't', 'F', 'p', 'chi2', 'eta_p2').

    Returns
    -------
    str
        Mathtext string (e.g. '$t$', '$F$', '$\\\\chi^2$').

    Examples
    --------
    >>> fmt_sym('t')
    '$\\\\mathit{t}$'
    >>> fmt_sym('chi2')
    '$\\\\chi^2$'
    """
    return _MATHTEXT.get(symbol, f"${symbol}$")


def fmt_stat(
    symbol: str,
    value,
    fmt: str = ".3f",
    df=None,
    stars: str = None,
) -> str:
    """Format a stat line with italic symbol for matplotlib text.

    Parameters
    ----------
    symbol : str
        Stat symbol name (e.g. 't', 'F', 'p').
    value : float or str
        Statistic value.
    fmt : str
        Format spec for the value (default '.3f').
    df : str or tuple, optional
        Degrees of freedom. Shown as symbol(df) = value.
    stars : str, optional
        Significance stars to append.

    Returns
    -------
    str
        Formatted line, e.g. '$t$(28) = 2.310 ***'.

    Examples
    --------
    >>> fmt_stat('t', -10.521, stars='***')
    '$\\\\mathit{t}$ = -10.521 ***'
    >>> fmt_stat('F', 119.265, df='2, 147', stars='***')
    '$\\\\mathit{F}$(2, 147) = 119.265 ***'
    >>> fmt_stat('p', 0.0001, fmt='.4f')
    '$\\\\mathit{p}$ = 0.0001'
    """
    sym = fmt_sym(symbol)
    val = f"{value:{fmt}}" if isinstance(value, (int, float)) else str(value)
    # Replace hyphen-minus with proper Unicode minus sign for display
    val = val.replace("-", "\u2212")
    if df is not None:
        line = f"{sym}({df}) = {val}"
    else:
        line = f"{sym} = {val}"
    if stars:
        s = stars.replace("ns", "$ns$")
        line += f" {s}"
    return line


def p2stars(
    pvalue: Union[float, np.ndarray, pd.Series, pd.DataFrame],
    thresholds: tuple = (0.001, 0.01, 0.05),
    symbols: tuple = ("***", "**", "*", "ns"),
    ns_symbol: bool = True,
) -> Union[str, np.ndarray, pd.Series, pd.DataFrame]:
    """
    Convert p-values to significance stars.

    Parameters
    ----------
    pvalue : float, array, Series, or DataFrame
        P-value(s) to convert to stars
    thresholds : tuple of float, default (0.001, 0.01, 0.05)
        Significance thresholds (must be in ascending order)
    symbols : tuple of str, default ('***', '**', '*', 'ns')
        Symbols for each threshold level
        Length must be len(thresholds) + 1
    ns_symbol : bool, default True
        Whether to include 'ns' for non-significant results
        If False, returns empty string for non-significant

    Returns
    -------
    str, array, Series, or DataFrame
        Significance symbols matching input type

    Notes
    -----
    Default thresholds and symbols:
    - p < 0.001: '***' (highly significant)
    - p < 0.01:  '**'  (very significant)
    - p < 0.05:  '*'   (significant)
    - p >= 0.05: 'ns'  (not significant)

    Examples
    --------
    >>> p2stars(0.001)
    '***'

    >>> p2stars(0.023)
    '*'

    >>> p2stars(0.15)
    'ns'

    >>> p2stars(0.15, ns_symbol=False)
    ''

    >>> # Works with arrays
    >>> pvals = np.array([0.0001, 0.005, 0.045, 0.15])
    >>> p2stars(pvals)
    array(['***', '**', '*', 'ns'], dtype='<U3')

    >>> # Works with DataFrames
    >>> df = pd.DataFrame({'pvalue': [0.001, 0.05, 0.15]})
    >>> df['pstars'] = p2stars(df['pvalue'])
    """
    # Validate inputs
    if len(symbols) != len(thresholds) + 1:
        raise ValueError(
            f"symbols must have {len(thresholds) + 1} elements "
            f"(one more than thresholds)"
        )

    if not all(thresholds[i] < thresholds[i + 1] for i in range(len(thresholds) - 1)):
        raise ValueError("thresholds must be in ascending order")

    # Handle different input types
    if isinstance(pvalue, (pd.DataFrame, pd.Series)):
        # Apply to DataFrame/Series
        return pvalue.apply(
            lambda p: _p2stars_scalar(p, thresholds, symbols, ns_symbol)
        )
    elif isinstance(pvalue, np.ndarray):
        # Vectorized for numpy arrays
        return np.vectorize(
            lambda p: _p2stars_scalar(p, thresholds, symbols, ns_symbol)
        )(pvalue)
    else:
        # Single value
        return _p2stars_scalar(pvalue, thresholds, symbols, ns_symbol)


def _p2stars_scalar(
    pvalue: float, thresholds: tuple, symbols: tuple, ns_symbol: bool
) -> str:
    """Convert single p-value to stars (internal function)."""
    # Handle NaN
    if pd.isna(pvalue):
        return "NaN"

    # Handle invalid p-values
    if pvalue < 0 or pvalue > 1:
        logger.warning(f"Invalid p-value: {pvalue}. Should be between 0 and 1.")
        return "invalid"

    # Find appropriate symbol (use <= to include boundary values)
    for i, threshold in enumerate(thresholds):
        if pvalue <= threshold:
            return symbols[i]

    # Non-significant
    return symbols[-1] if ns_symbol else ""


"""Main function"""


def main(args):
    """Demonstrate p2stars functionality."""
    logger.info("Demonstrating p2stars functionality")

    # Example 1: Single p-values
    logger.info("\n=== Example 1: Single p-values ===")
    test_pvalues = [0.0001, 0.005, 0.023, 0.049, 0.051, 0.15]

    for p in test_pvalues:
        stars = p2stars(p)
        logger.info(f"p = {p:6.4f} → {stars:3s}")

    # Example 2: Array of p-values
    logger.info("\n=== Example 2: Array of p-values ===")
    pvals_array = np.array([0.0001, 0.005, 0.023, 0.049, 0.051, 0.15])
    stars_array = p2stars(pvals_array)

    logger.info(f"P-values: {pvals_array}")
    logger.info(f"Stars:    {stars_array}")

    # Example 3: DataFrame
    logger.info("\n=== Example 3: DataFrame ===")
    df = pd.DataFrame(
        {
            "test": ["Test 1", "Test 2", "Test 3", "Test 4"],
            "pvalue": [0.0001, 0.023, 0.051, 0.15],
        }
    )

    df["pstars"] = p2stars(df["pvalue"])
    logger.info(f"\n{df}")

    # Example 4: Custom thresholds
    logger.info("\n=== Example 4: Custom thresholds (more stringent) ===")
    custom_stars = p2stars(
        0.01, thresholds=(0.0001, 0.001, 0.01), symbols=("****", "***", "**", "ns")
    )
    logger.info(f"p = 0.01 with custom thresholds → {custom_stars}")

    # Example 5: Without 'ns' symbol
    logger.info("\n=== Example 5: Without 'ns' symbol ===")
    for p in [0.001, 0.05, 0.15]:
        stars = p2stars(p, ns_symbol=False)
        logger.info(f"p = {p:6.4f} → '{stars}'")

    # Create visualization
    logger.info("\n=== Creating visualization ===")
    fig, ax = stx.plt.subplots(figsize=(10, 6))

    # Generate range of p-values
    pvals = np.logspace(-4, 0, 100)  # 0.0001 to 1.0
    stars = p2stars(pvals)

    # Color map for stars
    color_map = {"***": "red", "**": "orange", "*": "yellow", "ns": "lightgray"}
    colors = [color_map.get(s, "gray") for s in stars]

    # Plot
    ax.scatter(pvals, range(len(pvals)), c=colors, alpha=0.6, s=50)
    ax.set_xscale("log")
    ax.set_xlabel("P-value")
    ax.set_ylabel("Test index")
    ax.set_title("P-value to Stars Conversion")

    # Add vertical lines for thresholds
    for threshold in [0.001, 0.01, 0.05]:
        ax.axvline(threshold, color="black", linestyle="--", alpha=0.3)
        ax.text(
            threshold,
            len(pvals) * 0.95,
            f"{threshold}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", label="p < 0.001 (***)"),
        Patch(facecolor="orange", label="p < 0.01 (**)"),
        Patch(facecolor="yellow", label="p < 0.05 (*)"),
        Patch(facecolor="lightgray", label="p ≥ 0.05 (ns)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Save
    stx.io.save(fig, "./p2stars_demo.jpg")
    logger.info("Visualization saved")

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate p-value to stars conversion"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Initialize SciTeX framework and run main."""
    global CONFIG, sys, plt, rng

    import sys

    import matplotlib.pyplot as plt

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=args.verbose,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=args.verbose,
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
