#!/usr/bin/env python3
# Timestamp: "2025-10-01 22:43:58 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/nonparametric/_demo_friedman.py

"""
Demo script for Friedman test.

Demonstrates various use cases of test_friedman().
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import scitex as stx
from scitex.logging import getLogger

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)


def main(args):  # noqa: C901
    """Demonstrate Friedman test functionality."""
    from ._test_friedman import test_friedman

    # Example 1: Pain ratings (ordinal data)
    logger.info("\n[Example 1] Pain ratings across 4 time points (ordinal)")
    logger.info("-" * 70)

    np.random.seed(42)
    # Simulate decreasing pain over time
    pain_data = np.array(
        [
            [7, 6, 5, 4],
            [8, 7, 6, 5],
            [6, 5, 4, 3],
            [9, 8, 7, 6],
            [7, 6, 5, 4],
            [8, 7, 6, 5],
            [6, 5, 5, 4],
            [7, 6, 5, 5],
        ]
    )

    result = test_friedman(
        pain_data,
        condition_names=["Baseline", "Week 1", "Week 2", "Week 3"],
        plot=True,
        verbose=True,
    )
    stx.io.save(stx.plt.gcf(), "./friedman_example1.jpg")

    # Example 2: Likert scale ratings
    logger.info("\n[Example 2] Likert scale ratings (1-5) for 4 products")
    logger.info("-" * 70)

    likert_data = np.array(
        [
            [3, 4, 5, 3],
            [2, 3, 4, 2],
            [4, 5, 5, 4],
            [3, 4, 4, 3],
            [2, 3, 5, 2],
            [3, 4, 4, 3],
            [4, 5, 5, 4],
            [3, 3, 4, 3],
            [2, 4, 5, 3],
            [3, 4, 4, 2],
        ]
    )

    result_likert = test_friedman(
        likert_data,
        condition_names=["Product A", "Product B", "Product C", "Product D"],
        plot=True,
        verbose=True,
    )
    stx.io.save(stx.plt.gcf(), "./friedman_example2.jpg")
    stx.plt.close()

    logger.info(f"χ²({result_likert['df']}) = {result_likert['statistic']:.3f}")
    logger.info(f"p-value = {result_likert['pvalue']:.4f}")
    logger.info(f"Kendall's W = {result_likert['kendall_w']:.3f}")

    # Example 3: Long format DataFrame
    logger.info("\n[Example 3] Long format DataFrame input")
    logger.info("-" * 70)

    subjects = np.repeat(np.arange(8), 4)
    conditions = np.tile(["Pre", "Mid1", "Mid2", "Post"], 8)
    values = np.random.randint(1, 11, 32)  # Random scores 1-10

    df_long = pd.DataFrame(
        {"Subject": subjects, "TimePoint": conditions, "Score": values}
    )

    result_long = test_friedman(
        df_long,
        subject_col="Subject",
        condition_col="TimePoint",
        value_col="Score",
        plot=True,
        verbose=True,
    )
    stx.io.save(stx.plt.gcf(), "./friedman_example3.jpg")
    stx.plt.close()

    logger.info(f"χ² = {result_long['statistic']:.3f}, p = {result_long['pvalue']:.4f}")

    # Example 4: Comparison with RM-ANOVA
    logger.info("\n[Example 4] Comparison: Friedman vs RM-ANOVA")
    logger.info("-" * 70)

    from ..parametric import test_anova_rm

    # Data with outliers
    data_outlier = np.random.normal(5, 1, (10, 4))
    data_outlier[0, 0] = 20  # Add outlier

    test_friedman(data_outlier, verbose=True)
    result_rm_anova = test_anova_rm(data_outlier, verbose=True)

    logger.info(
        f"RM-ANOVA: F = {result_rm_anova['statistic']:.3f}, p = {result_rm_anova['pvalue']:.4f}"  # type: ignore[call-overload]
    )
    logger.info("Note: Friedman is more robust to outliers")

    # Example 5: Export results
    logger.info("\n[Example 5] Export results")
    logger.info("-" * 70)

    stx.io.save(result, "./friedman_results.xlsx")

    # EOF

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Initialize SciTeX framework and run main."""
    import sys

    import matplotlib.pyplot as plt  # noqa: F401

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, _CC, _rng_manager = stx.session.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
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
