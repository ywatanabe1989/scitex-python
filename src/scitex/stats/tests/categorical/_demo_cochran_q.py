#!/usr/bin/env python3
# Timestamp: "2025-10-01 19:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/categorical/_demo_cochran_q.py
# ----------------------------------------

"""
Demo script for Cochran's Q test examples.

Run with: python -m scitex.stats.tests.categorical._demo_cochran_q
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import scitex as stx
from scitex.logging import getLogger

from ._test_cochran_q import test_cochran_q

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)


def main(args):
    """Demonstrate Cochran's Q test functionality."""
    logger.info("=" * 70)
    logger.info("Cochran's Q Test Examples")
    logger.info("=" * 70)

    # Example 1: Treatment success over time
    logger.info("\n[Example 1] Treatment success (0/1) across 4 visits")
    logger.info("-" * 70)

    np.random.seed(42)
    # Simulate improving success rate over time
    data = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 1],
        ]
    )

    result, _ = test_cochran_q(
        data,
        condition_names=["Visit 1", "Visit 2", "Visit 3", "Visit 4"],
        plot=True,
    )

    logger.info(
        f"Q = {result['statistic']:.3f}, p = {result['pvalue']:.4f} {result['stars']}"
    )
    logger.info(
        f"Effect size (W) = {result['effect_size']:.3f} ({result['effect_size_interpretation']})"
    )
    logger.info(f"Proportions: {[f'{p:.1%}' for p in result['proportions']]}")
    stx.io.save(stx.plt.gcf(), "./.dev/cochran_q_example1.jpg")
    stx.plt.close()

    # Example 2: Symptom presence (binary)
    logger.info("\n[Example 2] Symptom presence across 3 time points")
    logger.info("-" * 70)

    symptom_data = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]
    )

    result_symptom, _ = test_cochran_q(
        symptom_data,
        condition_names=["Baseline", "Week 2", "Week 4"],
        plot=True,
    )

    logger.info(f"Q({result_symptom['df']}) = {result_symptom['statistic']:.3f}")
    logger.info(f"p-value = {result_symptom['pvalue']:.4f}")
    stx.io.save(stx.plt.gcf(), "./.dev/cochran_q_example2.jpg")
    stx.plt.close()

    # Example 3: Comparison with Friedman test
    logger.info("\n[Example 3] Comparison: Cochran Q vs Friedman")
    logger.info("-" * 70)

    from scitex.stats.tests.nonparametric import test_friedman

    result_cochran = test_cochran_q(data)
    result_friedman = test_friedman(data.astype(float))

    logger.info(
        f"Cochran's Q:    Q = {result_cochran['statistic']:.3f}, p = {result_cochran['pvalue']:.4f}"  # type: ignore[call-overload]
    )
    logger.info(
        f"Friedman test:  χ² = {result_friedman['statistic']:.3f}, p = {result_friedman['pvalue']:.4f}"
    )
    logger.info("Note: For binary data, both tests are similar")

    # Example 4: Long format DataFrame
    logger.info("\n[Example 4] Long format DataFrame input")
    logger.info("-" * 70)

    subjects = np.repeat(np.arange(10), 3)
    conditions = np.tile(["Pre", "Mid", "Post"], 10)
    values = np.random.binomial(1, [0.3, 0.5, 0.7] * 10)

    df_long = pd.DataFrame(
        {"Subject": subjects, "TimePoint": conditions, "Success": values}
    )

    result_long, _ = test_cochran_q(
        df_long,
        subject_col="Subject",
        condition_col="TimePoint",
        value_col="Success",
        plot=True,
    )

    logger.info(f"Q = {result_long['statistic']:.3f}, p = {result_long['pvalue']:.4f}")
    stx.io.save(stx.plt.gcf(), "./.dev/cochran_q_example4.jpg")
    stx.plt.close()

    # Example 5: Export results
    logger.info("\n[Example 5] Export results")
    logger.info("-" * 70)

    from scitex.stats._utils._normalizers import convert_results

    df_result = convert_results(result, return_as="dataframe")
    df_result.to_excel("./cochran_q_results.xlsx", index=False)  # type: ignore[union-attr]
    logger.info("Saved to: ./cochran_q_results.xlsx")

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

    CONFIG, sys.stdout, sys.stderr, plt, CC, _rng_manager = stx.session.start(  # type: ignore[name-defined]
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
