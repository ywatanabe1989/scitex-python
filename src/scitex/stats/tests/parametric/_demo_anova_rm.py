#!/usr/bin/env python3
# Timestamp: "2025-10-01 17:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/parametric/_demo_anova_rm.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Demo script for repeated measures ANOVA examples.

Run with: python -m scitex.stats.tests.parametric._demo_anova_rm
"""

import argparse

import numpy as np
import pandas as pd

import scitex as stx
from scitex.logging import getLogger

from ._test_anova_rm import test_anova_rm

logger = getLogger(__name__)


def main(args):
    """Run demonstration."""
    logger.info("=" * 70)
    logger.info("Repeated Measures ANOVA Examples")
    logger.info("=" * 70)

    # Example 1: Basic repeated measures (4 time points)
    logger.info("\n[Example 1] Basic repeated measures - 4 time points")
    logger.info("-" * 70)

    np.random.seed(42)
    n_subjects = 12
    # Simulate increasing trend over time
    time_effects = np.array([0, 0.5, 1.0, 0.8])
    data = np.random.normal(5, 1, (n_subjects, 4)) + time_effects

    result = test_anova_rm(
        data,
        condition_names=["Baseline", "Week 1", "Week 2", "Week 3"],
        plot=True,
        verbose=True,
    )
    stx.io.save(stx.plt.gcf(), "./.dev/anova_rm_example1.jpg")
    stx.plt.close()

    logger.info(
        f"F({result['df_effect']:.1f}, {result['df_error']:.1f}) = {result['statistic']:.3f}"  # type: ignore[call-overload]
    )
    logger.info(f"p-value = {result['pvalue']:.4f} {result['stars']}")  # type: ignore[call-overload]
    logger.info(
        f"Partial η² = {result['effect_size']:.3f} ({result['effect_size_interpretation']})"  # type: ignore[call-overload]
    )
    if "sphericity_met" in result:
        logger.info(f"Sphericity met: {result['sphericity_met']}")  # type: ignore[call-overload]

    # Example 2: Sphericity violation
    logger.info("\n[Example 2] Data with sphericity violation")
    logger.info("-" * 70)

    # Create data that violates sphericity
    data_spher = np.random.normal(0, 1, (15, 4))
    data_spher[:, 1] += np.random.normal(0, 2, 15)  # High variance for condition 2
    data_spher[:, 2] += np.random.normal(0.5, 0.5, 15)

    result_spher = test_anova_rm(
        data_spher,
        condition_names=["T1", "T2", "T3", "T4"],
        correction="auto",
        plot=True,
        verbose=True,
    )
    stx.io.save(stx.plt.gcf(), "./.dev/anova_rm_example2.jpg")
    stx.plt.close()

    logger.info(f"Sphericity W = {result_spher.get('sphericity_W', 'N/A')}")  # type: ignore[union-attr]
    logger.info(f"Sphericity p = {result_spher.get('sphericity_pvalue', 'N/A')}")  # type: ignore[union-attr]
    logger.info(f"Correction applied: {result_spher.get('correction_applied', 'none')}")  # type: ignore[union-attr]
    logger.info(
        f"Adjusted F({result_spher['df_effect']:.2f}, {result_spher['df_error']:.2f}) = {result_spher['statistic']:.3f}"  # type: ignore[call-overload]
    )
    logger.info(f"p-value = {result_spher['pvalue']:.4f}")  # type: ignore[call-overload]

    # Example 3: Long format DataFrame
    logger.info("\n[Example 3] Long format DataFrame input")
    logger.info("-" * 70)

    # Create long format data
    subjects = np.repeat(np.arange(10), 3)
    conditions = np.tile(["Pre", "Mid", "Post"], 10)
    values = np.random.normal(10, 2, 30) + np.tile([0, 1, 1.5], 10)

    df_long = pd.DataFrame(
        {"Subject": subjects, "TimePoint": conditions, "Score": values}
    )

    result_long = test_anova_rm(
        df_long,
        subject_col="Subject",
        condition_col="TimePoint",
        value_col="Score",
        plot=True,
        verbose=True,
    )
    stx.io.save(stx.plt.gcf(), "./.dev/anova_rm_example3.jpg")
    stx.plt.close()

    logger.info(f"F = {result_long['statistic']:.3f}, p = {result_long['pvalue']:.4f}")  # type: ignore[call-overload]
    logger.info(f"Conditions: {result_long['condition_names']}")  # type: ignore[call-overload]

    # Example 4: Wide format DataFrame
    logger.info("\n[Example 4] Wide format DataFrame")
    logger.info("-" * 70)

    df_wide = pd.DataFrame(
        np.random.normal(50, 10, (20, 5)),
        columns=[
            "Drug_0mg",
            "Drug_5mg",
            "Drug_10mg",
            "Drug_15mg",
            "Drug_20mg",
        ],
    )
    # Add dose-response trend
    for i, dose in enumerate([0, 5, 10, 15, 20]):
        df_wide.iloc[:, i] += dose * 0.5

    result_wide = test_anova_rm(df_wide, plot=True, verbose=True)
    stx.io.save(stx.plt.gcf(), "./.dev/anova_rm_example4.jpg")
    stx.plt.close()

    logger.info(f"F = {result_wide['statistic']:.3f}, p = {result_wide['pvalue']:.4f}")  # type: ignore[call-overload]
    logger.info(f"Partial η² = {result_wide['effect_size']:.3f}")  # type: ignore[call-overload]

    # Example 5: Export results
    logger.info("\n[Example 5] Export results")
    logger.info("-" * 70)

    from scitex.stats._utils._normalizers import convert_results

    convert_results(result, return_as="excel", path="./.dev/anova_rm_results.xlsx")  # type: ignore[arg-type]
    logger.info("Saved to: ./.dev/anova_rm_results.xlsx")

    # EOF

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Initialize SciTeX framework and run main."""
    global CONFIG, CC, sys, plt

    import sys

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(  # type: ignore[name-defined]
        sys,  # type: ignore[name-defined]
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
