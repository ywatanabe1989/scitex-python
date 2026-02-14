#!/usr/bin/env python3
# Time-stamp: "2025-01-15 00:00:00 (ywatanabe)"
# File: ./scitex_repo/src/scitex/stats/tests/categorical/_demo_fisher.py
# ----------------------------------------

"""
Demo script for Fisher's exact test examples.

Run with: python -m scitex.stats.tests.categorical._demo_fisher
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import scitex as stx
from scitex.logging import getLogger
from scitex.stats._utils._normalizers import force_dataframe

from ._test_fisher import test_fisher

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)


def main(args):
    """Run demonstration."""
    logger.info("=" * 70)
    logger.info("Fisher's Exact Test - Examples")
    logger.info("=" * 70)

    # Example 1: Small sample treatment study
    logger.info("\nExample 1: Small sample treatment study")
    logger.info("-" * 70)
    observed1 = [[8, 2], [1, 5]]  # Treatment: Success/Failure
    result1 = test_fisher(
        observed1,
        var_row="Treatment",
        var_col="Outcome",
        plot=True,
        verbose=True,
    )
    logger.info(force_dataframe(result1))
    stx.io.save(stx.plt.gcf(), "fisher_example1.jpg")
    stx.plt.close()

    # Example 2: Case-control study (exposure × disease)
    logger.info("\nExample 2: Case-control study")
    logger.info("-" * 70)
    observed2 = [[12, 5], [8, 20]]  # Exposure: Cases/Controls
    result2 = test_fisher(
        observed2,
        var_row="Exposure",
        var_col="Disease",
        plot=True,
        verbose=True,
    )
    logger.info(force_dataframe(result2))
    stx.io.save(stx.plt.gcf(), "fisher_example2.jpg")
    stx.plt.close()

    # Example 3: One-tailed test (expect positive association)
    logger.info("\nExample 3: One-tailed test (alternative='greater')")
    logger.info("-" * 70)
    observed3 = [[10, 2], [3, 8]]
    logger.info("Two-tailed:")
    test_fisher(observed3, alternative="two-sided", verbose=True)
    logger.info("\nOne-tailed (greater):")
    test_fisher(observed3, alternative="greater", verbose=True)

    # Example 4: Using pandas DataFrame with labels
    print("\nExample 4: Using pandas DataFrame")
    print("-" * 70)
    df4 = pd.DataFrame(
        [[15, 5], [3, 10]],
        index=["Group A", "Group B"],
        columns=["Success", "Failure"],
    )
    df4.index.name = "Group"
    df4.columns.name = "Outcome"
    result4 = test_fisher(df4, plot=True)
    print(force_dataframe(result4))
    stx.io.save(stx.plt.gcf(), "example4_dataframe.jpg")
    stx.plt.close()

    # Example 5: Compare Fisher vs Chi-square
    print("\nExample 5: Compare Fisher's exact vs Chi-square")
    print("-" * 70)
    observed5 = [[5, 10], [10, 5]]
    fisher_result = test_fisher(observed5, plot=False)

    from ._test_chi2 import test_chi2

    chi2_result = test_chi2(observed5, plot=False)

    print(f"Fisher's exact test: p = {fisher_result['pvalue']:.4f} (exact)")
    print(f"Chi-square test:     p = {chi2_result['pvalue']:.4f} (approximation)")
    print("→ Fisher's exact provides exact p-value, chi-square is approximation")

    # Example 6: Very small sample
    print("\nExample 6: Very small sample (chi-square not recommended)")
    print("-" * 70)
    observed6 = [[2, 3], [1, 4]]
    result6 = test_fisher(observed6, var_row="Group", var_col="Response", plot=True)
    print(force_dataframe(result6))
    print("Fisher's exact test is ideal for small samples")
    stx.io.save(stx.plt.gcf(), "example6_small_sample.jpg")
    stx.plt.close()

    # Example 7: Strong association
    print("\nExample 7: Strong positive association")
    print("-" * 70)
    observed7 = [[20, 2], [3, 18]]
    result7 = test_fisher(observed7, var_row="Factor A", var_col="Factor B", plot=True)
    print(force_dataframe(result7))
    print(f"Very strong association: OR = {result7['statistic']:.1f}")
    stx.io.save(stx.plt.gcf(), "example7_strong_association.jpg")
    stx.plt.close()

    # Example 8: No association (OR ≈ 1)
    print("\nExample 8: No association")
    print("-" * 70)
    observed8 = [[10, 10], [10, 10]]
    result8 = test_fisher(observed8, plot=True)
    print(force_dataframe(result8))
    print(f"OR = {result8['statistic']:.2f} ≈ 1 (no association)")
    stx.io.save(stx.plt.gcf(), "example8_no_association.jpg")
    stx.plt.close()

    # Example 9: Negative association (OR < 1)
    print("\nExample 9: Negative association (OR < 1)")
    print("-" * 70)
    observed9 = [[2, 15], [12, 8]]
    result9 = test_fisher(
        observed9, var_row="Treatment", var_col="Adverse Event", plot=True
    )
    print(force_dataframe(result9))
    print(f"OR = {result9['statistic']:.3f} < 1 (negative association)")
    stx.io.save(stx.plt.gcf(), "example9_negative_association.jpg")
    stx.plt.close()

    # Example 10: Export to various formats
    print("\nExample 10: Export to various formats")
    print("-" * 70)
    result10 = test_fisher(
        observed2, var_row="Exposure", var_col="Disease", return_as="dataframe"
    )
    stx.io.save(result10, "fisher_demo.csv")
    stx.io.save(result10, "fisher_demo.tex")
    print("Exported to CSV and LaTeX formats")
    print(result10)

    logger.info(f"\n{'=' * 70}")
    logger.info("All examples completed")
    logger.info(f"{'=' * 70}")

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Initialize SciTeX framework and run main."""
    import sys  # noqa: E402

    global CONFIG

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(  # type: ignore[name-defined]
        sys,
        stx.plt,
        args=args,
        file=__FILE__,
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
