#!/usr/bin/env python3
# Timestamp: "2025-10-01 16:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/categorical/_demo_mcnemar.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Demo script for McNemar's test examples.

Run with: python -m scitex.stats.tests.categorical._demo_mcnemar
"""

import argparse

import pandas as pd

import scitex as stx
from scitex.logging import getLogger

from ._test_mcnemar import test_mcnemar

logger = getLogger(__name__)


def main(args):
    """Run demonstration."""
    # Parse empty args

    logger.info("=" * 70)
    logger.info("McNemar's Test Examples")
    logger.info("=" * 70)

    # Example 1: Treatment effectiveness
    logger.info("\n[Example 1] Treatment effectiveness (before/after)")
    logger.info("-" * 70)

    observed = [
        [
            59,
            6,
        ],  # No disease before: stayed negative (59), became positive (6)
        [16, 19],
    ]  # Disease before: became negative (16), stayed positive (19)

    result = test_mcnemar(
        observed,
        var_before="Before Treatment",
        var_after="After Treatment",
        plot=True,
    )

    logger.info("Contingency table:")
    logger.info(f"  [[{observed[0][0]}, {observed[0][1]}],")
    logger.info(f"   [{observed[1][0]}, {observed[1][1]}]]")
    logger.info(
        f"\nχ² = {result['statistic']:.3f}, p = {result['pvalue']:.4f} {result['stars']}"
    )
    logger.info(f"Discordant pairs: b={result['b']}, c={result['c']}")
    logger.info(
        f"Odds Ratio = {result['odds_ratio']:.3f} ({result['effect_size_interpretation']})"
    )
    logger.info(f"Significant: {result['significant']}")

    # Example 2: No change (null case)
    logger.info("\n[Example 2] No change (equal discordant pairs)")
    logger.info("-" * 70)

    observed_null = [[40, 10], [10, 40]]

    result_null = test_mcnemar(observed_null, plot=True)

    logger.info(f"b = {result_null['b']}, c = {result_null['c']}")
    logger.info(f"Odds Ratio = {result_null['odds_ratio']:.3f}")
    logger.info(f"p-value = {result_null['pvalue']:.4f}")
    logger.info("Result: No significant change (as expected)")

    # Example 3: Strong effect
    logger.info("\n[Example 3] Strong treatment effect")
    logger.info("-" * 70)

    observed_strong = [
        [50, 25],  # Many improved (0→1)
        [2, 23],
    ]  # Few relapsed (1→0)

    result_strong = test_mcnemar(observed_strong, plot=True)

    logger.info(f"Improvement: {result_strong['b']} patients")
    logger.info(f"Relapse: {result_strong['c']} patients")
    logger.info(f"Odds Ratio = {result_strong['odds_ratio']:.3f}")
    logger.info(f"p-value = {result_strong['pvalue']:.4f} {result_strong['stars']}")

    # Example 4: With and without correction
    logger.info("\n[Example 4] Effect of continuity correction")
    logger.info("-" * 70)

    observed_small = [[40, 6], [2, 12]]

    result_with = test_mcnemar(observed_small, correction=True)
    result_without = test_mcnemar(observed_small, correction=False)

    logger.info(
        f"With correction:    χ² = {result_with['statistic']:.3f}, p = {result_with['pvalue']:.4f}"
    )
    logger.info(
        f"Without correction: χ² = {result_without['statistic']:.3f}, p = {result_without['pvalue']:.4f}"
    )
    logger.info("Difference: Correction makes test more conservative")

    # Example 5: DataFrame output
    logger.info("\n[Example 5] DataFrame output format")
    logger.info("-" * 70)

    results_list = []
    for i in range(3):
        obs = [[40 + i * 5, 10 + i], [8 - i, 42 + i * 3]]
        r = test_mcnemar(obs, var_before=f"Time_{i}", var_after=f"Time_{i + 1}")
        results_list.append(r)

    df_results = pd.DataFrame(results_list)
    logger.info(
        f"\n{df_results[['var_before', 'var_after', 'statistic', 'pvalue', 'odds_ratio', 'significant']].to_string()}"
    )

    # Example 6: Export results
    logger.info("\n[Example 6] Export results to Excel")
    logger.info("-" * 70)

    df_results.to_excel("./mcnemar_results.xlsx", index=False)
    logger.info("Saved to: ./mcnemar_results.xlsx")

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

    CONFIG, sys.stdout, sys.stderr, plt, _CC, _rng_manager = stx.session.start(  # type: ignore[name-defined]
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
